#!/usr/bin/env python3.10
"""
Trainer for MIMIC-CXR curriculum learning with LoRA.

Implements staged training: Stage A (image-only) → Stage B (image+EHR)
with automatic checkpoint saving at stage transition.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json

logger = logging.getLogger(__name__)


class CurriculumTrainer:
    """
    Trainer for curriculum learning with automatic stage transition.
    
    Implements:
    - Stage A training (image-only) for first 35% of steps
    - Automatic checkpoint_A saving at transition
    - Stage B training (image+EHR) for remaining 65% of steps
    - Final checkpoint_B saving
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: str = "cuda",
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Calculate total steps and stage split
        self.total_steps = len(train_dataloader) * config['epochs']
        self.stage_split_step = int(self.total_steps * config['stage_split'])
        
        # Stage tracking
        self.current_stage = "A"
        self.checkpoint_a_saved = False
        
        logger.info(f"Total training steps: {self.total_steps}")
        logger.info(f"Stage A→B transition at step: {self.stage_split_step}")
        logger.info(f"Training device: {device}")
    
    def _should_skip_sample(self, batch: Dict[str, Any]) -> bool:
        """
        Determine if sample should be skipped based on current stage.
        
        During Stage A (warmup), skip image+EHR samples.
        
        Args:
            batch: Current batch
            
        Returns:
            True if sample should be skipped
        """
        if self.current_stage == "A":
            # Skip image+EHR samples during Stage A
            if self.config.get('skip_ehr_during_warmup', True):
                # Check if any sample in batch has EHR context
                has_ehr = batch.get('has_ehr', None)
                if has_ehr is not None:
                    # If has_ehr is a tensor, check if any are True
                    if isinstance(has_ehr, torch.Tensor):
                        if has_ehr.any():
                            return True
                    # If has_ehr is a list/array, check if any are True
                    elif any(has_ehr):
                        return True
                else:
                    # Fallback to mode checking
                    modes = batch.get('mode', [])
                    if any(mode == 'image_ehr' for mode in modes):
                        return True
        
        return False
    
    def _filter_batch_for_stage(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter batch to keep only samples appropriate for current stage.
        
        Args:
            batch: Current batch
            
        Returns:
            Filtered batch (may be empty)
        """
        if self.current_stage == "A" and self.config.get('skip_ehr_during_warmup', True):
            # Keep only image-only samples during Stage A
            has_ehr = batch.get('has_ehr', None)
            if has_ehr is not None:
                if isinstance(has_ehr, torch.Tensor):
                    # Keep samples where has_ehr is False
                    keep_indices = (~has_ehr).nonzero().squeeze(-1)
                    if keep_indices.numel() == 0:
                        return {}  # Empty batch
                    
                    # Filter all tensor fields
                    filtered_batch = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor) and value.size(0) == has_ehr.size(0):
                            filtered_batch[key] = value.index_select(0, keep_indices)
                        else:
                            filtered_batch[key] = value
                    return filtered_batch
                else:
                    # List/array case
                    keep_indices = [i for i, has_ehr_val in enumerate(has_ehr) if not has_ehr_val]
                    if not keep_indices:
                        return {}  # Empty batch
                    
                    # Filter all fields
                    filtered_batch = {}
                    for key, value in batch.items():
                        if isinstance(value, (list, tuple)) and len(value) == len(has_ehr):
                            filtered_batch[key] = [value[i] for i in keep_indices]
                        elif isinstance(value, torch.Tensor) and value.size(0) == len(has_ehr):
                            filtered_batch[key] = value[keep_indices]
                        else:
                            filtered_batch[key] = value
                    return filtered_batch
        
        return batch
    
    def _compute_step_ratio(self) -> float:
        """Compute current step ratio (0.0 to 1.0)."""
        return self.global_step / self.total_steps if self.total_steps > 0 else 0.0
    
    def _check_stage_transition(self):
        """Check if we should transition from Stage A to Stage B."""
        step_ratio = self._compute_step_ratio()
        
        if not self.checkpoint_a_saved and step_ratio >= self.config['stage_split']:
            logger.info(f"=" * 70)
            logger.info(f"STAGE TRANSITION: A → B at step {self.global_step}")
            logger.info(f"Step ratio: {step_ratio:.2%}")
            logger.info(f"=" * 70)
            
            # Save checkpoint A
            self._save_checkpoint_a()
            
            # Update stage
            self.current_stage = "B"
            self.checkpoint_a_saved = True
            
            logger.info(f"Now training Stage B (image+EHR samples)")
    
    def _save_checkpoint_a(self):
        """Save Stage A checkpoint."""
        checkpoint_path = Path(self.config['checkpoint_dir']) / self.config['checkpoint_a_name']
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint_A to {checkpoint_path}")
        
        # Save LoRA weights
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path / "pytorch_model.bin")
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        with open(checkpoint_path / "trainer_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"✅ Checkpoint A saved successfully")
    
    def _save_checkpoint_b(self):
        """Save final Stage B checkpoint."""
        checkpoint_path = Path(self.config['checkpoint_dir']) / self.config['checkpoint_b_name']
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint_B (FINAL) to {checkpoint_path}")
        
        # Save LoRA weights
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path / "pytorch_model.bin")
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        with open(checkpoint_path / "trainer_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"✅ Checkpoint B (FINAL) saved successfully")
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary with loss and metrics
        """
        # Filter batch for current stage
        batch = self._filter_batch_for_stage(batch)
        
        # Skip if batch becomes empty after filtering
        if not batch or (isinstance(batch.get('input_ids'), torch.Tensor) and batch['input_ids'].size(0) == 0):
            return {'loss': 0.0, 'skipped': True}
        
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch.get('input_ids'),
            attention_mask=batch.get('attention_mask'),
            images=batch.get('pixel_values'),
            labels=batch.get('labels'),
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.get('max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.config['max_grad_norm']
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update global step
        self.global_step += 1
        
        # Check for stage transition
        self._check_stage_transition()
        
        return {
            'loss': loss.item(),
            'skipped': False,
            'stage': self.current_stage,
            'step': self.global_step,
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        num_skipped = 0
        
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch} [Stage {self.current_stage}]"
        )
        
        for batch in pbar:
            # Train step
            metrics = self.train_step(batch)
            
            if metrics['skipped']:
                num_skipped += 1
                continue
            
            total_loss += metrics['loss']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'stage': self.current_stage,
                'step': f"{self.global_step}/{self.total_steps}",
            })
            
            # Log every N steps
            if self.global_step % self.config.get('logging_steps', 10) == 0:
                logger.info(
                    f"Step {self.global_step}/{self.total_steps} | "
                    f"Stage {self.current_stage} | "
                    f"Loss: {metrics['loss']:.4f}"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'num_batches': num_batches,
            'num_skipped': num_skipped,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_dataloader, desc="Validation")
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch.get('input_ids'),
                pixel_values=batch.get('pixel_values'),
                attention_mask=batch.get('attention_mask'),
                labels=batch.get('labels'),
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            pbar.set_postfix({'val_loss': f"{outputs.loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
        }
    
    def train(self):
        """Run complete training loop."""
        logger.info("=" * 70)
        logger.info("STARTING CURRICULUM TRAINING")
        logger.info("=" * 70)
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            logger.info(f"\n{'='*70}")
            logger.info(f"EPOCH {epoch + 1}/{self.config['epochs']}")
            logger.info(f"{'='*70}")
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            logger.info(f"\nEpoch {epoch + 1} training complete:")
            logger.info(f"  Train loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Batches: {train_metrics['num_batches']}")
            logger.info(f"  Skipped: {train_metrics['num_skipped']}")
            
            # Validation
            if (epoch + 1) % self.config.get('eval_epochs', 1) == 0:
                val_metrics = self.validate()
                logger.info(f"  Val loss: {val_metrics['val_loss']:.4f}")
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    logger.info(f"  ✅ New best validation loss!")
        
        # Save final checkpoint B
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"{'='*70}")
        self._save_checkpoint_b()
        
        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Checkpoint A: {Path(self.config['checkpoint_dir']) / self.config['checkpoint_a_name']}")
        logger.info(f"   Checkpoint B: {Path(self.config['checkpoint_dir']) / self.config['checkpoint_b_name']}")


def setup_lora(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """
    Setup LoRA for efficient fine-tuning with proper target modules.
    
    Args:
        model: Base model
        config: Configuration dictionary
        
    Returns:
        Model with LoRA adapters
    """
    logger.info("Setting up LoRA...")
    
    # Remove mm_projector from LoRA targets (we'll fully fine-tune it)
    lora_targets = [m for m in config['lora_target_modules'] if m != "mm_projector"]
    
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=lora_targets,
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Fully fine-tune the mm_projector (vision-to-text projection)
    for name, param in model.named_parameters():
        if "mm_projector" in name:
            param.requires_grad_(True)
            logger.info(f"✅ Fully fine-tuning: {name}")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"LoRA setup complete:")
    logger.info(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  LoRA targets: {lora_targets}")
    
    return model


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Test trainer setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = load_config("train/config.yaml")
    print("Config loaded successfully:")
    print(json.dumps(config, indent=2))

