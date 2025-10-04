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
    AutoModelForVision2Seq,
    AutoModel,
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
    Load base model and processor.
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info("=" * 70)
    logger.info("LOADING BASE MODEL")
    logger.info("=" * 70)
    logger.info(f"Model: {config['base_model']}")
    
    # Load processor/tokenizer
    try:
        processor = AutoProcessor.from_pretrained(config['base_model'], trust_remote_code=True)
        logger.info("✅ Processor loaded")
    except Exception as e:
        logger.warning(f"Could not load AutoProcessor: {e}")
        logger.info("Trying AutoTokenizer...")
        processor = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
        logger.info("✅ Tokenizer loaded")
    
    # Load model
    dtype = torch.bfloat16 if config.get('bf16', False) else torch.float32
    
    try:
        # Try AutoModelForVision2Seq first (standard LLaVA)
        model = AutoModelForVision2Seq.from_pretrained(
            config['base_model'],
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        logger.info(f"✅ Model loaded with AutoModelForVision2Seq (dtype: {dtype})")
    except Exception as e:
        logger.warning(f"AutoModelForVision2Seq failed: {e}")
        logger.info("Trying AutoModel with trust_remote_code...")
        
        # Try AutoModel with trust_remote_code for custom architectures
        model = AutoModel.from_pretrained(
            config['base_model'],
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        logger.info(f"✅ Model loaded with AutoModel (dtype: {dtype})")
    
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
        num_workers=config.get('dataloader_num_workers', 4),
        pin_memory=config.get('dataloader_pin_memory', True),
    )
    
    val_dataloader = create_dataloader(
        data_path=config['validation_path'],
        processor=processor,
        batch_size=config.get('per_device_eval_batch_size', 2),
        image_root=config.get('image_root', '.'),
        max_length=config.get('max_length', 512),
        stage="both",
        shuffle=False,
        num_workers=config.get('dataloader_num_workers', 4),
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

