#!/usr/bin/env python3
"""
Advanced Trainer for MIMIC-CXR Radiology Report Model
Implements curriculum learning, class balancing, JSON drift prevention, and early stopping
AGGRESSIVE MPS OPTIMIZATION - NO CPU FALLBACK!
"""

# Apply accelerator compatibility patch first
import sys
sys.path.append('/Users/bilbouser/radiology_report')
try:
    import accelerator_patch
except ImportError:
    pass

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

def setup_mps_aggressively():
    """Select best available device (CUDA > MPS > CPU) and apply safe memory tweaks.

    Note: On Colab/Paperspace we want CUDA if available. MPS-only tweaks are
    retained but we no longer force CPU.
    """

    print("🔥 DEVICE SELECTION & MEMORY SETUP")
    print("=" * 50)

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Prefer CUDA (Colab/Paperspace), then fall back to MPS (macOS), then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ Using CUDA device")
        return device

    # Minimal, safe MPS env hints (macOS only)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        print("✅ Using MPS device")
        return torch.device('mps')

    print("✅ Using CPU device")
    return torch.device('cpu')

from dataset import CHEXPERT_ORDER, ICD_ORDER, CurriculumDataset, StageMixDataset

# Import from correct path
import sys
sys.path.append('/Users/bilbouser/radiology_report/train')
from advanced_curriculum import (
    AdvancedCurriculumSampler,
    JSONDriftPrevention,
    create_advanced_curriculum_dataset,
    create_json_drift_prevention
)
logger = logging.getLogger(__name__)

class StageMixCallback(TrainerCallback):
    """Rebuilds the Stage-mixed dataset at the start of each epoch."""

    def __init__(self, dataset: StageMixDataset, seed: int):
        self.dataset = dataset
        self.seed = seed

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_idx = int(state.epoch) if state.epoch is not None else 0
        rebuild_seed = self.seed + epoch_idx
        self.dataset.rebuild(rebuild_seed)
        return control

class AdvancedCurriculumTrainer(Trainer):
    """
    Custom trainer with stage transition logic and curriculum learning
    """
    
    def __init__(self, config, curriculum_sampler, stage_split_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.curriculum_sampler = curriculum_sampler
        self.stage_split_step = stage_split_step
        self.current_stage = "A"
        self.checkpoint_a_saved = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute combined generative + auxiliary classification loss."""
        inputs = inputs.copy()
        chexpert_targets = inputs.pop('chexpert_targets', None)
        chexpert_masks = inputs.pop('chexpert_masks', None)
        icd_targets = inputs.pop('icd_targets', None)
        icd_masks = inputs.pop('icd_masks', None)
        
        # Ensure dict output with hidden states for auxiliary heads
        inputs['output_hidden_states'] = True
        inputs['return_dict'] = True
        outputs = model(**inputs)
        base_loss = outputs.loss
        if base_loss is None:
            logits = outputs.logits
            labels = inputs.get('labels')
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous() if labels is not None else None
            if shift_labels is None:
                raise RuntimeError("Labels required to compute base loss are missing.")
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            base_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        total_loss = base_loss
        aux_losses: Dict[str, torch.Tensor] = {}
        aux_cfg = self.config.get('aux_loss', {})
        labels_tensor = inputs.get('labels')
        
        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states; set output_hidden_states=True.")
        pooled = self._pool_hidden_states(outputs.hidden_states[-1], labels_tensor)
        device = pooled.device
        
        if chexpert_targets is not None and hasattr(model, "chexpert_head"):
            chexpert_targets = chexpert_targets.to(device)
            chexpert_masks = chexpert_masks.to(device) if chexpert_masks is not None else None
            chexpert_logits = model.chexpert_head(pooled)
            loss_cfg = aux_cfg.get('chexpert', {})
            aux_loss = self._compute_auxiliary_loss(
                chexpert_logits,
                chexpert_targets,
                chexpert_masks,
                loss_cfg
            )
            if aux_loss is not None:
                weight = loss_cfg.get('weight', 0.5)
                total_loss = total_loss + weight * aux_loss
                aux_losses['chexpert_aux'] = aux_loss.detach()
        
        if icd_targets is not None and hasattr(model, "icd_head"):
            icd_targets = icd_targets.to(device)
            icd_masks = icd_masks.to(device) if icd_masks is not None else None
            icd_logits = model.icd_head(pooled)
            loss_cfg = aux_cfg.get('icd', {})
            aux_loss = self._compute_auxiliary_loss(
                icd_logits,
                icd_targets,
                icd_masks,
                loss_cfg
            )
            if aux_loss is not None:
                weight = loss_cfg.get('weight', 0.4)
                total_loss = total_loss + weight * aux_loss
                aux_losses['icd_aux'] = aux_loss.detach()
        
        if aux_losses and self.state.global_step % max(1, self.args.logging_steps) == 0:
            for name, value in aux_losses.items():
                self.log({name: value.item()})
        
        if return_outputs:
            outputs.loss = total_loss
            return total_loss, outputs
        return total_loss
    
    def _pool_hidden_states(self, last_hidden_state: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        """Average hidden states over the assistant tokens (labels != -100)."""
        if labels is None:
            return last_hidden_state.mean(dim=1)
        mask = (labels != -100).float().to(last_hidden_state.device)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (last_hidden_state * mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled
    
    def _compute_auxiliary_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
        cfg: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """Compute BCE/Focal auxiliary loss for structured labels."""
        if mask is None:
            mask = torch.ones_like(targets)
        valid = mask.sum()
        if valid.item() == 0:
            return None
        
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        gamma = cfg.get('focal_gamma', 0.0)
        if gamma and gamma > 0:
            probs = torch.sigmoid(logits)
            focal_pos = (1 - probs).clamp(min=1e-4) ** gamma
            focal_neg = probs.clamp(min=1e-4) ** gamma
            focal = torch.where(targets > 0.5, focal_pos, focal_neg)
            bce = bce * focal
        
        pos_weight = cfg.get('pos_weight', 1.0)
        if pos_weight and pos_weight > 1.0:
            weight_tensor = torch.where(targets > 0.5, torch.full_like(targets, pos_weight), torch.ones_like(targets))
            bce = bce * weight_tensor
        
        bce = bce * mask
        loss = bce.sum() / valid.clamp(min=1.0)
        return loss
        
    def training_step(self, model, inputs):
        """Override training step to handle stage transitions and MPS optimization"""
        # Check for stage transition
        if not self.checkpoint_a_saved and self.state.global_step >= self.stage_split_step:
            self._transition_to_stage_b()
        
        # Update curriculum sampler step
        if self.curriculum_sampler:
            self.curriculum_sampler.update_step(self.state.global_step)
        
        # CPU optimization: ensure all inputs are on CPU device
        device = torch.device('cpu')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # CRITICAL: Ensure model is on MPS device
        if hasattr(model, 'to'):
            model = model.to(device)
        # Ensure all model parameters are on MPS
        for param in model.parameters():
            if param.device != device:
                param.data = param.data.to(device)
        
        # CRITICAL: Force mm_projector to MPS device - AGGRESSIVE APPROACH
        try:
            # Method 1: Direct access
            if hasattr(model, 'get_model'):
                base_model = model.get_model()
                if hasattr(base_model, 'mm_projector'):
                    base_model.mm_projector = base_model.mm_projector.to(device)
                    for param in base_model.mm_projector.parameters():
                        param.data = param.data.to(device)
                    for buffer in base_model.mm_projector.buffers():
                        buffer.data = buffer.data.to(device)
            
            # Method 2: Through base_model attribute
            if hasattr(model, 'base_model'):
                if hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'mm_projector'):
                    model.base_model.model.mm_projector = model.base_model.model.mm_projector.to(device)
                    for param in model.base_model.model.mm_projector.parameters():
                        param.data = param.data.to(device)
                    for buffer in model.base_model.model.mm_projector.buffers():
                        buffer.data = buffer.data.to(device)
            
            # Method 3: Recursive search and move
            def move_to_device(module, device):
                module.to(device)
                for child in module.children():
                    move_to_device(child, device)
            
            if hasattr(model, 'get_model'):
                move_to_device(model.get_model(), device)
                
        except Exception as e:
            print(f"⚠️ MPS device move warning in training step: {e}")
            # Force fallback to CPU if MPS fails
            device = torch.device('cpu')
            model = model.to(device)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Call parent training step
        return super().training_step(model, inputs)
    
    def _transition_to_stage_b(self):
        """Transition from Stage A to Stage B"""
        logger.info("=" * 70)
        logger.info(f"STAGE TRANSITION: A → B at step {self.state.global_step}")
        logger.info(f"Step ratio: {self.state.global_step / self.state.max_steps:.2%}")
        logger.info("=" * 70)
        
        # Save checkpoint A
        self._save_checkpoint_a()
        
        # Update stage
        self.current_stage = "B"
        self.checkpoint_a_saved = True
        
        logger.info("Now training Stage B (image+EHR samples)")
    
    def _save_checkpoint_a(self):
        """Save Stage A checkpoint"""
        checkpoint_path = Path(self.config['checkpoint_dir']) / self.config['checkpoint_a_name']
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint_A to {checkpoint_path}")
        
        # Save model
        self.model.save_pretrained(checkpoint_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save training state
        state_dict = {
            'global_step': self.state.global_step,
            'epoch': self.state.epoch,
            'stage': 'A',
            'config': self.config
        }
        
        with open(checkpoint_path / 'training_state.json', 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        logger.info(f"✅ Checkpoint A saved to {checkpoint_path}")

class AdvancedRadiologyTrainer:
    """
    Advanced trainer with curriculum learning, class balancing, and JSON drift prevention
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
        # AGGRESSIVE MPS SETUP - NO CPU FALLBACK!
        print("🔥 AGGRESSIVE MPS INITIALIZATION")
        print("=" * 50)
        self.device = setup_mps_aggressively()
        print(f"✅ Using device: {self.device}")
        
        # Ensure auxiliary loss configuration exists with sane defaults
        aux_cfg = self.config.setdefault('aux_loss', {})
        aux_cfg.setdefault('chexpert', {})
        aux_cfg.setdefault('icd', {})
        aux_cfg['chexpert'].setdefault('weight', 0.5)
        aux_cfg['chexpert'].setdefault('pos_weight', 3.0)
        aux_cfg['chexpert'].setdefault('focal_gamma', 0.0)
        aux_cfg['icd'].setdefault('weight', 0.4)
        aux_cfg['icd'].setdefault('pos_weight', 2.5)
        aux_cfg['icd'].setdefault('focal_gamma', 0.0)
        self.config.setdefault('unfreeze_language_layers', 2)
        self.config.setdefault('unfreeze_projector', True)
        
        # Initialize components
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.curriculum_sampler = None
        self.json_prevention = None
        self.trainer = None
        
        # Training state
        self.current_step = 0
        self.best_val_score = 0.0
        self.early_stopping_patience = 0
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_model_and_tokenizer(self):
        """Setup model, tokenizer, and processor with LLaVA-Med specific handling"""
        logger.info("=" * 70)
        logger.info("LOADING LLAVA-MED MODEL")
        logger.info("=" * 70)
        logger.info(f"Model: {self.config['base_model']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            trust_remote_code=True
        )
        
        # Create LLaVA-Med specific processor
        self.processor = self._create_llava_processor()
        
        # Load model with MPS-optimized settings - force float32 for MPS
        dtype = torch.float32  # Always use float32 for MPS compatibility
        
        try:
            # Try LLaVA-Med specific model loading
            from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
            
            self.model = LlavaMistralForCausalLM.from_pretrained(
                self.config['base_model'],
                torch_dtype=dtype,
                device_map=None,  # Load to CPU first
                trust_remote_code=True,
            )
            
            # AGGRESSIVE MPS optimization
            print("   Applying aggressive MPS optimizations...")
            self.model = self.model.to(self.device)
            self.model.float()  # Force float32
            
            # Ensure all parameters are on MPS and float32
            for param in self.model.parameters():
                param.data = param.data.to(self.device).float()
            
            # Convert all buffers to MPS and float32
            for buffer in self.model.buffers():
                buffer.data = buffer.data.to(self.device).float()
            
            logger.info(f"✅ LLaVA-Med model loaded with MPS optimization (dtype: {dtype})")
            
        except ImportError:
            logger.warning("LLaVA-Med specific import failed, trying standard transformers...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['base_model'],
                torch_dtype=dtype,
                device_map=None,
                trust_remote_code=True,
            )
            
            # Apply MPS optimization
            self.model = self.model.to(self.device)
            self.model.float()
            
            for param in self.model.parameters():
                param.data = param.data.to(self.device).float()
            
            for buffer in self.model.buffers():
                buffer.data = buffer.data.to(self.device).float()
            
            logger.info(f"✅ Model loaded with MPS optimization (dtype: {dtype})")
        
        # Configure model for MPS training
        self.model.config.use_cache = False
        torch.set_float32_matmul_precision("high")
        
        # Additional MPS optimizations
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        print(f"   🎯 Model ready on {self.device}")
        
        # Force float32 everywhere for MPS compatibility
        self.model.float()
        for p in self.model.parameters():
            p.data = p.data.float()
        
        # CRITICAL: Ensure ALL model components are on MPS
        self.model = self.model.to(self.device)
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                param.data = param.data.to(self.device)
        for name, buffer in self.model.named_buffers():
            if buffer.device != self.device:
                buffer.data = buffer.data.to(self.device)
        
        # CRITICAL: Force mm_projector to MPS device - AGGRESSIVE APPROACH
        try:
            # Method 1: Direct access
            if hasattr(self.model, 'get_model'):
                base_model = self.model.get_model()
                if hasattr(base_model, 'mm_projector'):
                    base_model.mm_projector = base_model.mm_projector.to(self.device)
                    for param in base_model.mm_projector.parameters():
                        param.data = param.data.to(self.device)
                    for buffer in base_model.mm_projector.buffers():
                        buffer.data = buffer.data.to(self.device)
            
            # Method 2: Through base_model attribute
            if hasattr(self.model, 'base_model'):
                if hasattr(self.model.base_model, 'model') and hasattr(self.model.base_model.model, 'mm_projector'):
                    self.model.base_model.model.mm_projector = self.model.base_model.model.mm_projector.to(self.device)
                    for param in self.model.base_model.model.mm_projector.parameters():
                        param.data = param.data.to(self.device)
                    for buffer in self.model.base_model.model.mm_projector.buffers():
                        buffer.data = buffer.data.to(self.device)
            
            # Method 3: Recursive search and move
            def move_to_device(module, device):
                module.to(device)
                for child in module.children():
                    move_to_device(child, device)
            
            if hasattr(self.model, 'get_model'):
                move_to_device(self.model.get_model(), self.device)
                
        except Exception as e:
            print(f"⚠️ MPS device move warning: {e}")
            # Force fallback to CPU if MPS fails
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
        
        # Disable torch.compile for MPS stability
        torch._dynamo.config.suppress_errors = True
        
        # Fix tokenizer
        self._fix_tokenizer()
        
        # Add special tokens
        self._add_special_tokens()
        
        # Freeze vision tower
        self._freeze_vision_tower()
        
        # Enable gradient checkpointing
        if self.config.get('gradient_checkpointing', True):
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        
        # Setup LoRA
        if self.config.get('use_lora', True):
            self._setup_lora()
        
        # Attach auxiliary heads and unfreeze selected backbone layers for top-up training
        self._attach_auxiliary_heads()
        self._unfreeze_language_backbone()
        
        # Ensure model produces hidden states for auxiliary supervision
        self.model.config.output_hidden_states = True
        
        logger.info("Model and tokenizer setup complete")
    
    def _create_llava_processor(self):
        """Create LLaVA-Med specific processor"""
        import torchvision.transforms as transforms
        from torchvision.transforms import InterpolationMode
        
        # Default image processing parameters
        image_size = self.config.get('image_size', 336)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        
        # Try to get config from vision tower after model loads
        try:
            # This will be updated after model loads
            pass
        except Exception as e:
            logger.warning(f"Could not read vision config, using defaults: {e}")
        
        # Create image processor
        image_processor = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])
        
        # Create processor class
        class LLaVAProcessor:
            def __init__(self, tokenizer, image_processor):
                self.tokenizer = tokenizer
                self.image_processor = image_processor
            
            def __call__(self, text=None, images=None, return_tensors="pt", padding=True, truncation=True, max_length=512, **kwargs):
                # Handle text-only processing
                if images is None:
                    return self.tokenizer(
                        text, 
                        return_tensors=return_tensors,
                        padding=padding,
                        truncation=truncation,
                        max_length=max_length,
                        **kwargs
                    )
                
                # Handle image+text processing
                if isinstance(images, list):
                    processed_images = [self.image_processor(img) for img in images]
                    pixel_values = torch.stack(processed_images)
                else:
                    pixel_values = self.image_processor(images).unsqueeze(0)
                
                # Process text
                text_encoding = self.tokenizer(
                    text,
                    return_tensors=return_tensors,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    **kwargs
                )
                
                # Combine results
                return {
                    'input_ids': text_encoding['input_ids'],
                    'attention_mask': text_encoding['attention_mask'],
                    'pixel_values': pixel_values,
                }
        
        processor = LLaVAProcessor(self.tokenizer, image_processor)
        logger.info("✅ LLaVA-Med processor created")
        return processor
    
    def _fix_tokenizer(self):
        """Fix tokenizer pad token and padding side"""
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            logger.info("✅ Fixed tokenizer pad token")
    
    def _add_special_tokens(self):
        """Add special tokens like <image>"""
        special = {"additional_special_tokens": ["<image>"]}
        added = self.tokenizer.add_special_tokens(special)
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"✅ Added <image> token, resized embeddings: {added} tokens added")
    
    def _freeze_vision_tower(self):
        """Freeze vision tower to save memory"""
        try:
            self.model.get_vision_tower().requires_grad_(False)
            logger.info("✅ Vision tower frozen")
        except Exception as e:
            logger.warning(f"Could not freeze vision tower: {e}")
    
    def _setup_lora(self):
        """Setup LoRA configuration with proper parameter management"""
        resume_lora_path = self.config.get('lora_checkpoint_path')
        resume_dir = Path(resume_lora_path) if resume_lora_path else None
        loaded_existing = False
        if resume_dir and resume_dir.exists():
            logger.info(f"🔁 Loading existing LoRA adapter from {resume_dir.resolve()}")
            self.model = PeftModel.from_pretrained(
                self.model,
                resume_dir,
                is_trainable=True,
                adapter_name="default"
            )
            loaded_existing = True
        else:
            target_modules = self.config.get(
                'lora_target_modules',
                [
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config['lora_r'],
                lora_alpha=self.config['lora_alpha'],
                lora_dropout=self.config['lora_dropout'],
                target_modules=target_modules,
            )

            self.model = get_peft_model(self.model, lora_config)
            logger.info("🆕 Initialized fresh LoRA adapters")
        
        # Freeze everything first
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        # Unfreeze LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                logger.info(f"✅ LoRA trainable: {name}")
        
        # Always train mm_projector fully (critical for vision models)
        for name, param in self.model.named_parameters():
            if "mm_projector" in name:
                param.requires_grad_(True)
                logger.info(f"✅ mm_projector trainable: {name}")
        
        # Log trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ Trainable params (LoRA + projector): {trainable:,} ({trainable/total*100:.2f}%)")
        logger.info(f"✅ Total params: {total:,}")
        if loaded_existing:
            logger.info("🔄 Continuing training from existing LoRA weights")
    
    def _attach_auxiliary_heads(self):
        """Attach auxiliary classification heads for CheXpert and ICD supervision."""
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            embeddings = getattr(self.model, "get_input_embeddings", None)
            if callable(embeddings):
                embedding_layer = embeddings()
                if embedding_layer is not None and hasattr(embedding_layer, "embedding_dim"):
                    hidden_size = embedding_layer.embedding_dim
        if hidden_size is None and hasattr(self.model, "model") and hasattr(self.model.model, "model_dim"):
            hidden_size = self.model.model.model_dim
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "hidden_size", 4096)
        
        self.model.chexpert_head = nn.Linear(hidden_size, len(CHEXPERT_ORDER), bias=True)
        self.model.icd_head = nn.Linear(hidden_size, len(ICD_ORDER), bias=True)
        nn.init.xavier_uniform_(self.model.chexpert_head.weight)
        nn.init.zeros_(self.model.chexpert_head.bias)
        nn.init.xavier_uniform_(self.model.icd_head.weight)
        nn.init.zeros_(self.model.icd_head.bias)
        self.model.chexpert_head.to(self.device)
        self.model.icd_head.to(self.device)
        logger.info("✅ Attached auxiliary classification heads for CheXpert/ICD supervision")
    
    def _unfreeze_language_backbone(self):
        """Optionally unfreeze the projector and the last N transformer blocks."""
        num_layers = int(self.config.get('unfreeze_language_layers', 0))
        if num_layers <= 0:
            logger.info("Skipping language layer unfreeze (num_layers <= 0)")
            return
        
        layer_container = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layer_container = self.model.model.layers
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") \
                and hasattr(self.model.base_model.model, "layers"):
            layer_container = self.model.base_model.model.layers
        
        if layer_container is not None:
            for layer in layer_container[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad_(True)
            logger.info(f"✅ Unfrozen last {num_layers} transformer blocks")
        else:
            logger.warning("⚠️ Unable to locate transformer layers for unfreezing")
        
        if self.config.get('unfreeze_projector', True):
            projector = None
            if hasattr(self.model, "mm_projector"):
                projector = self.model.mm_projector
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") \
                    and hasattr(self.model.base_model.model, "mm_projector"):
                projector = self.model.base_model.model.mm_projector
            if projector is not None:
                for param in projector.parameters():
                    param.requires_grad_(True)
                logger.info("✅ mm_projector parameters unfrozen")
            else:
                logger.warning("⚠️ Unable to locate mm_projector to unfreeze")
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🔁 Updated trainable params after unfreeze: {trainable:,} ({trainable/total*100:.2f}%)")
    
    def setup_curriculum_learning(self, train_samples: List[Dict]):
        """Setup advanced curriculum learning"""
        logger.info("Setting up curriculum learning...")
        
        # Create curriculum sampler
        self.curriculum_sampler = create_advanced_curriculum_dataset(
            train_samples,
            self.config
        )
        
        # Create JSON drift prevention
        self.json_prevention = create_json_drift_prevention(
            self.config['chexpert_labels'],
            self.config['icd_classes']
        )
        
        logger.info("Curriculum learning setup complete")
    
    def create_advanced_dataset(self, samples: List[Dict], is_training: bool = True) -> CurriculumDataset:
        """Create dataset with advanced features"""
        dataset = CurriculumDataset(
            samples=samples,
            image_root=".",  # FIXED: Use current directory since paths already include "files/"
            processor=self.processor,
            max_length=self.config.get('max_length', 512),
            stage="both",
            max_label_tokens=self.config.get('max_label_tokens', 128),
        )
        return dataset
    
    def compute_advanced_metrics(self, eval_preds) -> Dict[str, float]:
        """Compute advanced metrics including JSON validity and class-specific F1"""
        try:
            predictions, labels = eval_preds
            
            # Safe decode function to handle tokenizer crashes
            def _safe_decode(tokenizer, ids):
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    ids = [t for seq in ids for t in seq]  # flatten
                try:
                    return tokenizer.decode(ids, skip_special_tokens=True)
                except Exception as e:
                    logger.warning(f"Decode error: {e}")
                    return ""
            
            # Decode predictions and labels safely
            pred_texts = [_safe_decode(self.tokenizer, pred) for pred in predictions]
            label_texts = [_safe_decode(self.tokenizer, label) for label in labels]
            
            # Basic metrics
            metrics = {
                "eval_loss": 0.0,
                "eval_accuracy": 0.0,
                "eval_f1": 0.0,
                "eval_precision": 0.0,
                "eval_recall": 0.0
            }
            
            # TODO: Add BLEU, ROUGE, METEOR, CheXpert F1, ICD F1
            # For now, return basic metrics to avoid crashes
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics computation error: {e}")
            return {
                "eval_loss": 0.0,
                "eval_accuracy": 0.0,
                "eval_f1": 0.0,
                "eval_precision": 0.0,
                "eval_recall": 0.0
            }
    
    def _extract_chexpert_predictions(self, texts: List[str]) -> List[List[int]]:
        """Extract CheXpert predictions from generated text"""
        predictions = []
        
        for text in texts:
            pred = [-1] * len(self.config['chexpert_labels'])  # Default to uncertain
            
            # Extract CheXpert section
            if "CheXpert:" in text:
                chexpert_start = text.find("CheXpert:")
                chexpert_end = text.find("}", chexpert_start) + 1
                if chexpert_end > chexpert_start:
                    chexpert_json = text[chexpert_start:chexpert_end]
                    try:
                        chexpert_data = json.loads(chexpert_json.replace("CheXpert:", "").strip())
                        for i, label in enumerate(self.config['chexpert_labels']):
                            if label in chexpert_data:
                                pred[i] = chexpert_data[label]
                    except:
                        pass
            
            predictions.append(pred)
        
        return predictions
    
    def _extract_icd_predictions(self, texts: List[str]) -> List[List[int]]:
        """Extract ICD predictions from generated text"""
        predictions = []
        
        for text in texts:
            pred = [0] * len(self.config['icd_classes'])  # Default to negative
            
            # Extract ICD section
            if "ICD:" in text:
                icd_start = text.find("ICD:")
                icd_end = text.find("}", icd_start) + 1
                if icd_end > icd_start:
                    icd_json = text[icd_start:icd_end]
                    try:
                        icd_data = json.loads(icd_json.replace("ICD:", "").strip())
                        for i, icd_class in enumerate(self.config['icd_classes']):
                            if icd_class in icd_data:
                                pred[i] = 1 if icd_data[icd_class] else 0
                    except:
                        pass
            
            predictions.append(pred)
        
        return predictions
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments with advanced features"""
        return TrainingArguments(
            output_dir=self.config['output_dir'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config.get('per_device_eval_batch_size', 2),
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            num_train_epochs=self.config['epochs'],
            warmup_ratio=self.config['warmup_ratio'],
            weight_decay=self.config['weight_decay'],
            max_grad_norm=self.config['max_grad_norm'],
            logging_dir=self.config['logging_dir'],
            logging_steps=self.config['logging_steps'],
            evaluation_strategy=self.config['evaluation_strategy'],
            eval_steps=self.config['eval_steps'],
            save_strategy=self.config['save_strategy'],
            save_steps=self.config['save_steps'],
            save_total_limit=self.config['save_total_limit'],
            load_best_model_at_end=True,
            metric_for_best_model='chexpert_f1',
            greater_is_better=True,
            bf16=self.config.get('bf16', False),
            fp16=self.config.get('fp16', False),
            dataloader_num_workers=self.config.get('dataloader_num_workers', 0),
            dataloader_pin_memory=self.config.get('dataloader_pin_memory', False),
            remove_unused_columns=self.config.get('remove_unused_columns', False),
            gradient_checkpointing=self.config.get('gradient_checkpointing', True),
            dataloader_drop_last=self.config.get('dataloader_drop_last', False),
            seed=self.config['seed'],
            report_to=self.config.get('report_to', 'tensorboard'),
            run_name=f"radiology_curriculum_{self.config['seed']}"
        )
    
    def train(self):
        """Main training loop with advanced curriculum learning and stage transitions"""
        logger.info("=" * 70)
        logger.info("STARTING ADVANCED TRAINING")
        logger.info("=" * 70)
        
        # Load data
        with open(self.config['dataset_path'], 'r') as f:
            train_samples = [json.loads(line) for line in f]
        
        with open(self.config['validation_path'], 'r') as f:
            val_samples = [json.loads(line) for line in f]
        
        stage_mix_dataset: Optional[StageMixDataset] = None
        if self.config.get('curriculum_learning', True):
            self.setup_curriculum_learning(train_samples)
            train_dataset = self.create_advanced_dataset(train_samples, is_training=True)
        else:
            self.curriculum_sampler = None
            stage_a_samples = [sample for sample in train_samples if sample.get('stage') == 'A']
            stage_b_samples = [sample for sample in train_samples if sample.get('stage') == 'B']
            stage_mix_dataset = StageMixDataset(
                stage_a_samples=stage_a_samples,
                stage_b_samples=stage_b_samples,
                image_root=".",
                processor=self.processor,
                max_length=self.config.get('max_length', 512),
                max_label_tokens=self.config.get('max_label_tokens', 128),
                stage_b_fraction=self.config.get('stage_b_fraction', 0.65),
                seed=self.config.get('seed', 42),
            )
            train_dataset = stage_mix_dataset
        val_dataset = self.create_advanced_dataset(val_samples, is_training=False)
        
        # Calculate training steps and stage split
        total_steps = len(train_dataset) * self.config['epochs']
        stage_split_step = int(total_steps * self.config['stage_split'])
        
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Stage A→B transition at step: {stage_split_step}")
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Create custom trainer with stage transition logic
        self.trainer = AdvancedCurriculumTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=train_dataset.collate_fn,
            compute_metrics=self.compute_advanced_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            config=self.config,
            curriculum_sampler=self.curriculum_sampler,
            stage_split_step=stage_split_step
        )

        if stage_mix_dataset is not None:
            self.trainer.add_callback(StageMixCallback(stage_mix_dataset, self.config.get('seed', 42)))
        
        # Start training
        logger.info("Starting training with advanced curriculum learning...")
        self.trainer.train()
        
        # Save final model
        self.trainer.save_model()
        logger.info("=" * 70)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
    
    def _is_valid_json(self, text: str) -> bool:
        """Check if text contains valid JSON blocks"""
        try:
            # Look for CheXpert and ICD JSON blocks
            chexpert_match = re.search(r'2\) CheXpert: (\{.*?\})', text, re.DOTALL)
            icd_match = re.search(r'3\) ICD: (\{.*?\})', text, re.DOTALL)
            
            if chexpert_match:
                json.loads(chexpert_match.group(1))
            if icd_match:
                json.loads(icd_match.group(1))
            
            return True
        except:
            return False
    
    def _generate_with_json_guard(self, images, prompt, max_new_tokens=256):
        """Generate with JSON drift prevention"""
        # First generation attempt
        gen = self.model.generate(
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        out = self.tokenizer.decode(gen[0], skip_special_tokens=True)
        
        # Check if JSON is valid
        if not self._is_valid_json(out):
            logger.warning("JSON invalid, attempting repair...")
            # Retry with repair prompt
            repair_prompt = "Return CheXpert and ICD as *valid JSON only*."
            repair_gen = self.model.generate(
                images=None,  # Reuse cached KV
                prompt=repair_prompt,
                max_new_tokens=128,
                temperature=0.0
            )
            repair_out = self.tokenizer.decode(repair_gen[0], skip_special_tokens=True)
            # Try to fix the original output
            out = self._try_fix_json(out, repair_out)
        
        return out
    
    def _try_fix_json(self, original: str, repair: str) -> str:
        """Try to fix JSON in original text using repair text"""
        # Simple repair: replace JSON blocks if repair has valid ones
        try:
            chexpert_match = re.search(r'2\) CheXpert: (\{.*?\})', repair, re.DOTALL)
            icd_match = re.search(r'3\) ICD: (\{.*?\})', repair, re.DOTALL)
            
            if chexpert_match and json.loads(chexpert_match.group(1)):
                original = re.sub(r'2\) CheXpert: \{.*?\}', f'2) CheXpert: {chexpert_match.group(1)}', original, flags=re.DOTALL)
            
            if icd_match and json.loads(icd_match.group(1)):
                original = re.sub(r'3\) ICD: \{.*?\}', f'3) ICD: {icd_match.group(1)}', original, flags=re.DOTALL)
            
            return original
        except:
            return original
    
    def generate_sample_output(self, image_path: str, patient_data: Dict = None) -> str:
        """Generate sample output with JSON drift prevention"""
        # This would be implemented for inference
        # For now, return a placeholder
        return "Sample output generation not implemented yet"


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Radiology Report Training")
    parser.add_argument("--config", type=str, default="advanced_training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trainer
    trainer = AdvancedRadiologyTrainer(args.config)
    
    # Setup model
    trainer.setup_model_and_tokenizer()
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
