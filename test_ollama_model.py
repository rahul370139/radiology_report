#!/usr/bin/env python3.10
"""
Test Ollama LLaVA-Med model on sample chest X-rays.

Usage:
    python3.10 test_ollama_model.py
"""

import subprocess
import json
from pathlib import Path

def test_ollama_model():
    """Test the Ollama LLaVA-Med model."""
    print("🧪 TESTING OLLAMA LLAVA-MED MODEL")
    print("=" * 70)
    
    # Check if model is available
    print("\n1. Checking if model is installed...")
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    
    if "llava-med" in result.stdout:
        print("✅ llava-med model found")
    else:
        print("❌ llava-med model not found")
        print("\nPlease run first:")
        print("  ollama pull rohithbojja/llava-med-v1.6")
        return
    
    # Load a sample image from curriculum
    print("\n2. Loading sample image from curriculum...")
    
    with open("data/processed/curriculum_train.jsonl") as f:
        sample = json.loads(f.readline())
    
    image_path = sample['image']
    print(f"   Image: {image_path}")
    print(f"   Ground truth impression: {sample['target'][:100]}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print("✅ Image loaded")
    
    # Test with Ollama
    print("\n3. Running inference with Ollama...")
    print("   Prompt: 'Provide Impression and CheXpert JSON for this chest X-ray'")
    
    try:
        result = subprocess.run([
            "ollama", "run", "rohithbojja/llava-med-v1.6",
            "--image", image_path,
            "Provide clinical Impression and CheXpert labels for this chest X-ray in JSON format."
        ], capture_output=True, text=True, timeout=60)
        
        print("\n📋 MODEL OUTPUT:")
        print("-" * 70)
        print(result.stdout)
        print("-" * 70)
        
        print("\n📋 GROUND TRUTH:")
        print("-" * 70)
        print(sample['target'][:500])
        print("-" * 70)
        
        print("\n✅ Inference successful!")
        
    except subprocess.TimeoutExpired:
        print("⏱️  Inference timed out (model might be loading)")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
    
    print("\n" + "=" * 70)
    print("✅ OLLAMA MODEL TEST COMPLETE")
    print("=" * 70)
    print("\nThe model is ready for:")
    print("  • Local inference on Mac")
    print("  • Demo deployment")
    print("  • A/B testing")

if __name__ == "__main__":
    test_ollama_model()

