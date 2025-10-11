#!/usr/bin/env python3
"""
Quick Compatibility Check for LLaVA-Med Training
Tests data loading and basic setup without downloading the model
"""

import json
import sys
import traceback
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading from our clean datasets"""
    logger.info("🔍 TESTING DATA LOADING")
    logger.info("=" * 50)
    
    try:
        # Test training data
        train_file = Path("data/processed/curriculum_train_final_clean.jsonl")
        if not train_file.exists():
            logger.error(f"❌ Training file not found: {train_file}")
            return False
        
        with open(train_file, 'r') as f:
            train_data = [json.loads(line) for line in f]
        
        logger.info(f"✅ Training data loaded: {len(train_data):,} samples")
        
        # Test validation data
        val_file = Path("data/processed/curriculum_val_final_clean.jsonl")
        if not val_file.exists():
            logger.error(f"❌ Validation file not found: {val_file}")
            return False
        
        with open(val_file, 'r') as f:
            val_data = [json.loads(line) for line in f]
        
        logger.info(f"✅ Validation data loaded: {len(val_data):,} samples")
        
        # Test data structure
        sample = train_data[0]
        required_fields = ['image_path', 'impression', 'chexpert_labels', 'stage']
        
        for field in required_fields:
            if field not in sample:
                logger.error(f"❌ Missing required field: {field}")
                return False
        
        logger.info("✅ Data structure validation passed")
        
        # Test Stage A samples
        stage_a = [s for s in train_data if s.get('stage') == 'A']
        stage_b = [s for s in train_data if s.get('stage') == 'B']
        
        logger.info(f"✅ Stage A samples: {len(stage_a):,}")
        logger.info(f"✅ Stage B samples: {len(stage_b):,}")
        
        # Test Stage B has patient_data
        stage_b_with_ehr = [s for s in stage_b if 'patient_data' in s]
        logger.info(f"✅ Stage B with EHR: {len(stage_b_with_ehr):,}")
        
        # Test vitals coverage
        vitals_count = sum(1 for s in stage_b if s.get('patient_data', {}).get('Vitals'))
        labs_count = sum(1 for s in stage_b if s.get('patient_data', {}).get('Labs'))
        
        logger.info(f"✅ Vitals coverage: {vitals_count}/{len(stage_b)} ({vitals_count/len(stage_b)*100:.1f}%)")
        logger.info(f"✅ Labs coverage: {labs_count}/{len(stage_b)} ({labs_count/len(stage_b)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading"""
    logger.info("\n🔍 TESTING CONFIG LOADING")
    logger.info("=" * 50)
    
    try:
        import yaml
        
        config_file = Path("train/config.yaml")
        if not config_file.exists():
            logger.error(f"❌ Config file not found: {config_file}")
            return False
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required config fields
        required_fields = ['dataset_path', 'validation_path', 'base_model', 'output_dir']
        for field in required_fields:
            if field not in config:
                logger.error(f"❌ Missing config field: {field}")
                return False
        
        logger.info("✅ Config file loaded successfully")
        logger.info(f"✅ Dataset path: {config['dataset_path']}")
        logger.info(f"✅ Validation path: {config['validation_path']}")
        logger.info(f"✅ Model: {config['base_model']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test required dependencies"""
    logger.info("\n🔍 TESTING DEPENDENCIES")
    logger.info("=" * 50)
    
    try:
        # Test core dependencies
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        
        import transformers
        logger.info(f"✅ Transformers: {transformers.__version__}")
        
        import peft
        logger.info(f"✅ PEFT: {peft.__version__}")
        
        import accelerate
        logger.info(f"✅ Accelerate: {accelerate.__version__}")
        
        # Test MPS availability
        if torch.backends.mps.is_available():
            logger.info("✅ MPS (Metal Performance Shaders) available")
        else:
            logger.warning("⚠️ MPS not available (will use CPU)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dependency check failed: {e}")
        traceback.print_exc()
        return False

def test_dataset_module():
    """Test dataset module loading"""
    logger.info("\n🔍 TESTING DATASET MODULE")
    logger.info("=" * 50)
    
    try:
        from train.dataset import CurriculumDataset
        logger.info("✅ Dataset class imported successfully")
        
        # Test dataset initialization (without model)
        dataset = CurriculumDataset(
            data_path="data/processed/curriculum_train_final_clean.jsonl",
            image_root=".",
            processor=None,  # Skip processor for now
            max_length=512
        )
        
        logger.info(f"✅ Dataset initialized: {len(dataset)} samples")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset module test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all compatibility checks"""
    logger.info("🚀 QUICK COMPATIBILITY CHECK")
    logger.info("=" * 70)
    logger.info("Testing data loading and basic setup (no model download)")
    logger.info("=" * 70)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Config Loading", test_config_loading),
        ("Dependencies", test_dependencies),
        ("Dataset Module", test_dataset_module),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"📊 COMPATIBILITY CHECK RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL CHECKS PASSED! Ready for fine-tuning on server!")
        return True
    else:
        logger.error(f"❌ {total - passed} checks failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
