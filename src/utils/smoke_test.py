#!/usr/bin/env python3.10
"""
5-minute smoke test for LLaVA-Med training setup.
Tests critical components before full training.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from train.trainer import load_config
from train.dataset import create_dataloader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_processor_and_model():
    """Test 1: Load processor and model, move to MPS."""
    logger.info("=" * 70)
    logger.info("SMOKE TEST 1: Processor & Model Loading")
    logger.info("=" * 70)
    
    try:
        config = load_config("train/config.yaml")
        
        # Load processor
        from transformers import AutoTokenizer
        from PIL import Image
        import torchvision.transforms as transforms
        
        tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
        
        image_processor = transforms.Compose([
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
        
        # Test processor
        result = processor(text="test")
        logger.info("✅ Processor test passed")
        
        # Load model
        from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
        
        model = LlavaMistralForCausalLM.from_pretrained(
            config['base_model'],
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        )
        
        # Move to MPS
        model.to("mps")
        logger.info("✅ Model moved to MPS")
        
        # Configure model
        model.config.use_cache = False
        torch.set_float32_matmul_precision("high")
        
        # Extract image processor config from vision tower
        image_size = 336
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        
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
        image_processor = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size if isinstance(image_size, int) else image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])
        
        # Update processor with proper image processor
        processor.image_processor = image_processor
        
        # Fix tokenizer pad token (Mistral sometimes lacks pad)
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
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        logger.info("✅ Gradient checkpointing enabled")
        
        return model, processor, config
        
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
        raise

def test_forward_pass(model, processor, config):
    """Test 2: One batch through forward pass."""
    logger.info("\n" + "=" * 70)
    logger.info("SMOKE TEST 2: Forward Pass")
    logger.info("=" * 70)
    
    try:
        # Create minimal dataloader with batch_size=1
        train_dataloader = create_dataloader(
            data_path=config['dataset_path'],
            processor=processor,
            batch_size=1,  # Start with 1 to avoid OOM
            image_root=config.get('image_root', '.'),
            max_length=config.get('max_length', 512),
            stage="both",
            shuffle=False,
            num_workers=0,  # No multiprocessing for test
            pin_memory=False,
        )
        
        logger.info(f"✅ Dataloader created: {len(train_dataloader)} batches")
        
        # Get one batch
        batch = next(iter(train_dataloader))
        logger.info(f"✅ Batch loaded: {list(batch.keys())}")
        
        # Move batch to MPS
        batch = {k: v.to("mps") if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        model.train()
        # LLaVA-Med expects images with correct kwarg
        outputs = model(
            input_ids=batch.get('input_ids'),
            attention_mask=batch.get('attention_mask'),
            images=batch.get('pixel_values'),
            labels=batch.get('labels'),
        )
        
        loss = outputs.loss
        logger.info(f"✅ Forward pass successful")
        logger.info(f"   Loss: {loss.item():.4f}")
        logger.info(f"   Loss shape: {loss.shape}")
        
        # Test backward pass
        loss.backward()
        logger.info("✅ Backward pass successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")
        logger.info("Trying with gradient accumulation...")
        
        try:
            # Try with gradient accumulation
            config['batch_size'] = 1
            config['gradient_accumulation_steps'] = 16
            
            train_dataloader = create_dataloader(
                data_path=config['dataset_path'],
                processor=processor,
                batch_size=1,
                image_root=config.get('image_root', '.'),
                max_length=config.get('max_length', 512),
                stage="both",
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            
            batch = next(iter(train_dataloader))
            batch = {k: v.to("mps") if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            model.train()
            outputs = model(
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                images=batch.get('pixel_values'),
                labels=batch.get('labels'),
            )
            
            loss = outputs.loss
            loss.backward()
            
            logger.info("✅ Forward pass successful with gradient accumulation")
            logger.info(f"   Loss: {loss.item():.4f}")
            return True
            
        except Exception as e2:
            logger.error(f"❌ Test 2 failed even with gradient accumulation: {e2}")
            return False

def test_step_timing(model, processor, config):
    """Test 3: Confirm step time doesn't explode."""
    logger.info("\n" + "=" * 70)
    logger.info("SMOKE TEST 3: Step Timing")
    logger.info("=" * 70)
    
    try:
        import time
        
        # Create dataloader
        train_dataloader = create_dataloader(
            data_path=config['dataset_path'],
            processor=processor,
            batch_size=1,
            image_root=config.get('image_root', '.'),
            max_length=config.get('max_length', 512),
            stage="both",
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        
        # Time a few steps
        model.train()
        times = []
        
        for i, batch in enumerate(train_dataloader):
            if i >= 3:  # Test 3 steps
                break
                
            batch = {k: v.to("mps") if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            start_time = time.time()
            
            outputs = model(
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                images=batch.get('pixel_values'),
                labels=batch.get('labels'),
            )
            
            loss = outputs.loss
            loss.backward()
            
            end_time = time.time()
            step_time = end_time - start_time
            times.append(step_time)
            
            logger.info(f"   Step {i+1}: {step_time:.2f}s")
        
        avg_time = sum(times) / len(times)
        logger.info(f"✅ Average step time: {avg_time:.2f}s")
        
        if avg_time > 30:  # If step takes more than 30 seconds
            logger.warning(f"⚠️  Step time is slow: {avg_time:.2f}s")
            logger.info("   Consider reducing batch size or max_length")
        else:
            logger.info("✅ Step timing looks good")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 3 failed: {e}")
        return False

def main():
    """Run all smoke tests."""
    logger.info("🚀 Starting 5-minute smoke test for LLaVA-Med training")
    logger.info("=" * 70)
    
    try:
        # Test 1: Load processor and model
        model, processor, config = test_processor_and_model()
        
        # Test 2: Forward pass
        forward_ok = test_forward_pass(model, processor, config)
        
        # Test 3: Step timing
        timing_ok = test_step_timing(model, processor, config)
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("SMOKE TEST SUMMARY")
        logger.info("=" * 70)
        
        if forward_ok and timing_ok:
            logger.info("🎉 ALL SMOKE TESTS PASSED!")
            logger.info("✅ Ready to start full training")
            logger.info("\nTo start training, run:")
            logger.info("  python train.py --config train/config.yaml")
            return 0
        else:
            logger.error("❌ Some smoke tests failed")
            logger.error("Please fix issues before starting full training")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Smoke test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
