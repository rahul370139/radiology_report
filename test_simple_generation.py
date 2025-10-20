#!/usr/bin/env python3
"""
Simple test to debug LLaVA model generation
"""
import os
import sys
import torch
from PIL import Image

# Add src to path
sys.path.append('src')

def test_simple_generation():
    print("🧪 Testing simple LLaVA generation...")
    
    # Set environment variables
    os.environ['BASE_MODEL_PATH'] = '/Users/bilbouser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/f2f72301dc934e74948b5802c87dbc83d100e6bd'
    os.environ['LORA_DIR'] = 'checkpoints'
    
    try:
        from src.utils.load_finetuned_model import load_finetuned_llava
        from src.inference.pipeline import RadiologyInferencePipeline
        
        print("✅ Imports successful")
        
        # Load model
        print("🤖 Loading model...")
        pipeline = RadiologyInferencePipeline(device="cpu")
        print("✅ Model loaded successfully")
        
        # Test with a simple image
        test_image = "evaluation/demo_images/demo_a_001.jpg"
        if os.path.exists(test_image):
            print(f"📸 Testing with image: {test_image}")
            
            # Try simple generation
            result = pipeline.generate(test_image, None)
            print(f"✅ Generation result: {result}")
        else:
            print(f"❌ Test image not found: {test_image}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_generation()
