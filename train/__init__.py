"""
Training module for MIMIC-CXR radiology report generation.

This module contains all components needed for curriculum learning with LoRA.
"""

from .dataset import CurriculumDataset, StageAwareDataset, create_dataloader
from .trainer import CurriculumTrainer, setup_lora, load_config
from .metrics import MetricsCalculator

__all__ = [
    'CurriculumDataset',
    'StageAwareDataset',
    'create_dataloader',
    'CurriculumTrainer',
    'setup_lora',
    'load_config',
    'MetricsCalculator',
]

