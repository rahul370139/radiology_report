#!/usr/bin/env python3.10
"""
Test training setup before running full training.

This script verifies:
1. Config loads correctly
2. Dataset loads and processes samples
3. Model components are accessible
4. No import errors

Usage:
    python3.10 test_training_setup.py
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config():
    """Test configuration loading."""
    logger.info("=" * 70)
    logger.info("TEST 1: Configuration Loading")
    logger.info("=" * 70)
    
    try:
        from train.trainer import load_config
        config = load_config("train/config.yaml")
        
        logger.info("✅ Config loaded successfully")
        logger.info(f"   Base model: {config['base_model']}")
        logger.info(f"   Epochs: {config['epochs']}")
        logger.info(f"   LoRA rank: {config['lora_r']}")
        logger.info(f"   Stage split: {config['stage_split']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Config loading failed: {e}")
        return False


def test_dataset():
    """Test dataset loading."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Dataset Loading")
    logger.info("=" * 70)
    
    try:
        from train.dataset import CurriculumDataset
        
        dataset = CurriculumDataset(
            data_path="data/processed/curriculum_train.jsonl",
            image_root=".",
            stage="both",
        )
        
        logger.info(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Test loading a sample
        sample = dataset[0]
        logger.info(f"✅ Sample loaded:")
        logger.info(f"   Study ID: {sample['study_id']}")
        logger.info(f"   Mode: {sample['mode']}")
        logger.info(f"   Stage: {sample['stage']}")
        logger.info(f"   Image size: {sample['image'].size}")
        
        # Test filtering by stage
        dataset_a = CurriculumDataset(
            data_path="data/processed/curriculum_train.jsonl",
            image_root=".",
            stage="stage_a",
        )
        logger.info(f"✅ Stage A dataset: {len(dataset_a)} samples")
        
        dataset_b = CurriculumDataset(
            data_path="data/processed/curriculum_train.jsonl",
            image_root=".",
            stage="stage_b",
        )
        logger.info(f"✅ Stage B dataset: {len(dataset_b)} samples")
        
        return True
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics calculation."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Metrics Calculation")
    logger.info("=" * 70)
    
    try:
        from train.metrics import MetricsCalculator
        
        calculator = MetricsCalculator()
        
        # Test extraction functions
        test_text = """Impression: NO ACUTE CARDIOPULMONARY PROCESS.

CheXpert:
{
  "No Finding": "Positive",
  "Pleural Effusion": "Uncertain"
}"""
        
        impression = calculator.extract_impression(test_text)
        chexpert = calculator.extract_chexpert_json(test_text)
        
        logger.info("✅ Metrics calculator initialized")
        logger.info(f"   Impression extracted: {len(impression)} chars")
        logger.info(f"   CheXpert parsed: {chexpert is not None}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test all required imports."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Required Imports")
    logger.info("=" * 70)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT (LoRA)'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm'),
    ]
    
    all_ok = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {name} installed")
        except ImportError:
            logger.error(f"❌ {name} NOT installed")
            all_ok = False
    
    return all_ok


def test_data_files():
    """Test that all required data files exist."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Data Files")
    logger.info("=" * 70)
    
    required_files = [
        "data/processed/curriculum_train.jsonl",
        "data/processed/curriculum_val.jsonl",
        "data/processed/ehr_context.jsonl",
        "train/config.yaml",
    ]
    
    all_ok = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            logger.info(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            logger.error(f"❌ {file_path} NOT FOUND")
            all_ok = False
    
    # Check images directory
    image_dir = Path("files/p10")
    if image_dir.exists():
        image_count = len(list(image_dir.rglob("*.jpg")))
        logger.info(f"✅ files/p10/ ({image_count:,} images)")
    else:
        logger.error(f"❌ files/p10/ NOT FOUND")
        all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    logger.info("MIMIC-CXR Training Setup Verification")
    logger.info("=" * 70)
    
    results = {
        'imports': test_imports(),
        'data_files': test_data_files(),
        'config': test_config(),
        'dataset': test_dataset(),
        'metrics': test_metrics(),
    }
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    if all_passed:
        logger.info("\n" + "=" * 70)
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Ready to start training")
        logger.info("=" * 70)
        logger.info("\nTo start training, run:")
        logger.info("  python3.10 train.py --config train/config.yaml")
        return 0
    else:
        logger.error("\n" + "=" * 70)
        logger.error("❌ SOME TESTS FAILED")
        logger.error("Please fix the issues above before training")
        logger.error("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

