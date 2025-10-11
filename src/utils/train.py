#!/usr/bin/env python3.10
"""
Main training script for MIMIC-CXR radiology report generation.

Implements curriculum learning with LoRA fine-tuning:
- Stage A: Image-only warm-up (35% of steps)
- Stage B: Image+EHR training (65% of steps)

Usage:
    # Single GPU/MPS
    python3.10 train.py --config train/config.yaml
    
    # Multi-GPU with accelerate
    accelerate launch train.py --config train/config.yaml
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train.dataset import CurriculumDataset, create_dataloader
from train.trainer import CurriculumTrainer, setup_lora, load_config
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


def setup_device(config: Dict[str, Any]) -> str:
    """
    Setup training device (CUDA/MPS/CPU).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Device string
    """
    if config.get('device') == 'auto':
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = config.get('device', 'cpu')
    
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device


def load_model_and_processor(config: Dict[str, Any], device: str):
    """
    Load base model and processor for LLaVA-Med.
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info("=" * 70)
    logger.info("LOADING LLAVA-MED MODEL")
    logger.info("=" * 70)
    logger.info(f"Model: {config['base_model']}")
    
    # Add LLaVA-Med to path for custom model loading
    llava_med_path = Path(__file__).parent / "LLaVA-Med"
    if llava_med_path.exists():
        sys.path.append(str(llava_med_path))
        logger.info("✅ Added LLaVA-Med to Python path")
    
    # Load processor for LLaVA-Med
    try:
        # LLaVA-Med uses separate tokenizer and image processor
        from transformers import AutoTokenizer
        from PIL import Image
        import torchvision.transforms as transforms
        
        tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
        
        # Get image processor config from model's vision tower
        image_size = 336
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        
        # Try to get config from vision tower after model loads
        logger.info("Will extract image processor config after model loads...")
        
        # Create a processor-like object
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
                # Process images
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
        
        # Create temporary processor (will be updated after model loads)
        processor = LLaVAProcessor(tokenizer, None)
        logger.info("✅ LLaVA-Med processor loaded (tokenizer + image transforms)")
        logger.info(f"   Vocab size: {len(tokenizer)}")
        
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise
    
    # Load model using LLaVA-Med specific method
    dtype = torch.float16 if config.get('fp16', True) else torch.float32
    
    try:
        # Import LLaVA-Med specific model class
        from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
        
        model = LlavaMistralForCausalLM.from_pretrained(
            config['base_model'],
            torch_dtype=dtype,
            device_map=None,  # Load to CPU first, then move to device
            trust_remote_code=True,
        )
        
        # Move model to device
        model.to(device)
        
        logger.info(f"✅ LLaVA-Med model loaded (dtype: {dtype})")
        logger.info(f"   Parameters: {model.num_parameters():,}")
        
    except ImportError:
        logger.warning("LLaVA-Med specific import failed, trying standard transformers...")
        try:
            # Fallback to standard transformers
            model = AutoModelForCausalLM.from_pretrained(
                config['base_model'],
                torch_dtype=dtype,
                device_map=None,
                trust_remote_code=True,
            )
            model.to(device)
            logger.info(f"✅ Model loaded with AutoModelForCausalLM (dtype: {dtype})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    # Configure model for training
    model.config.use_cache = False
    torch.set_float32_matmul_precision("high")
    
    # Extract image processor config from vision tower
    try:
        vt = model.get_vision_tower()
        # Try different ways to access the vision model
        vision_model = None
        if hasattr(vt, "vision_tower"):
            vision_model = vt.vision_tower
        elif hasattr(vt, "vision_model"):
            vision_model = vt.vision_model
        elif hasattr(vt, "model"):
            vision_model = vt.model
        
        if vision_model is not None and hasattr(vision_model, "config"):
            config = vision_model.config
            if hasattr(config, "image_size"):
                image_size = config.image_size
            if hasattr(config, "image_mean"):
                image_mean = config.image_mean
            if hasattr(config, "image_std"):
                image_std = config.image_std
            logger.info(f"✅ Using vision tower image proc: size={image_size}, mean={image_mean}, std={image_std}")
        else:
            logger.info(f"✅ Using fallback image proc: size={image_size}, mean={image_mean}, std={image_std}")
    except Exception as e:
        logger.warning(f"Could not read vision image processor, using fallback: {e}")
    
    # Create proper image processor with extracted config
    from torchvision.transforms import InterpolationMode
    image_processor = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])
    
    # Update processor with proper image processor
    processor.image_processor = image_processor
    
    # Fix tokenizer pad token and padding side
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        logger.info("✅ Fixed tokenizer pad token")
    
    # Add <image> token if not present
    special = {"additional_special_tokens": ["<image>"]}
    added = processor.tokenizer.add_special_tokens(special)
    if added > 0:
        model.resize_token_embeddings(len(processor.tokenizer))
        logger.info(f"✅ Added <image> token, resized embeddings: {added} tokens added")
    
    # Freeze vision tower
    try:
        model.get_vision_tower().requires_grad_(False)
        logger.info("✅ Vision tower frozen")
    except Exception as e:
        logger.warning(f"Could not freeze vision tower: {e}")
    
    # Enable gradient checkpointing for memory efficiency
    if config.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        logger.info("✅ Gradient checkpointing enabled")
    
    return model, processor


def setup_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict[str, Any],
    num_training_steps: int,
):
    """
    Setup optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        num_training_steps: Total number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    logger.info("Setting up optimizer and scheduler...")
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
        eps=config.get('adam_epsilon', 1e-8),
        weight_decay=config.get('weight_decay', 0.01),
    )
    
    # Create scheduler
    num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.03))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    logger.info(f"✅ Optimizer: AdamW (lr={config['learning_rate']})")
    logger.info(f"✅ Scheduler: Linear warmup ({num_warmup_steps} steps) + decay")
    
    return optimizer, scheduler


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    logger.info("=" * 70)
    logger.info("MIMIC-CXR RADIOLOGY REPORT TRAINING")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    
    # Create output directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['logging_dir']).mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device(config)
    
    # Load model and processor
    model, processor = load_model_and_processor(config, device)
    
    # Setup LoRA
    if config.get('use_lora', True):
        model = setup_lora(model, config)
        
        # Freeze everything first
        for p in model.parameters():
            p.requires_grad_(False)
        
        # Unfreeze LoRA (PEFT handles attached modules)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                logger.info(f"✅ LoRA trainable: {name}")
        
        # Always train mm_projector fully
        for name, param in model.named_parameters():
            if "mm_projector" in name:
                param.requires_grad_(True)
                logger.info(f"✅ mm_projector trainable: {name}")
        
        # Log actual trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Trainable params (LoRA + projector): {trainable:,} ({trainable/total*100:.2f}%)")
        logger.info(f"✅ Total params: {total:,}")
    
    # Create dataloaders
    logger.info("=" * 70)
    logger.info("CREATING DATALOADERS")
    logger.info("=" * 70)
    
    train_dataloader = create_dataloader(
        data_path=config['dataset_path'],
        processor=processor,
        batch_size=config['batch_size'],
        image_root=config.get('image_root', '.'),
        max_length=config.get('max_length', 512),
        stage="both",  # Load all samples
        shuffle=True,
        num_workers=config.get('dataloader_num_workers', 2),
        pin_memory=config.get('dataloader_pin_memory', False),
    )
    
    val_dataloader = create_dataloader(
        data_path=config['validation_path'],
        processor=processor,
        batch_size=config.get('per_device_eval_batch_size', 2),
        image_root=config.get('image_root', '.'),
        max_length=config.get('max_length', 512),
        stage="both",
        shuffle=False,
        num_workers=config.get('dataloader_num_workers', 2),
        pin_memory=False,
    )
    
    logger.info(f"✅ Train dataloader: {len(train_dataloader)} batches")
    logger.info(f"✅ Val dataloader: {len(val_dataloader)} batches")
    
    # Calculate total training steps
    num_training_steps = len(train_dataloader) * config['epochs']
    logger.info(f"Total training steps: {num_training_steps}")
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model, config, num_training_steps
    )
    
    # Create trainer
    trainer = CurriculumTrainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    
    # Start training
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    
    try:
        trainer.train()
        logger.info("=" * 70)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        logger.info("Saving current state...")
        trainer._save_checkpoint_b()
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIMIC-CXR radiology report model")
    parser.add_argument(
        "--config",
        type=str,
        default="train/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    main(args)

