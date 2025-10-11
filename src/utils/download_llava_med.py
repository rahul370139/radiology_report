#!/usr/bin/env python3
"""
LLaVA-Med Model Download Script
Downloads and verifies the microsoft/llava-med-v1.5-mistral-7b model for fine-tuning.

Usage:
    python download_llava_med.py

Requirements:
    - llava_med package installed
    - Sufficient disk space (~14GB)
    - Internet connection
"""

import sys
import os
import torch
from pathlib import Path

def check_llava_med_installation():
    """Check if LLaVA-Med is properly installed."""
    try:
        import llava_med
        print("✅ LLaVA-Med package is installed")
        return True
    except ImportError:
        print("❌ LLaVA-Med package not found. Installing...")
        os.system("pip install git+https://github.com/microsoft/LLaVA-Med.git")
        try:
            import llava_med
            print("✅ LLaVA-Med package installed successfully")
            return True
        except ImportError:
            print("❌ Failed to install LLaVA-Med package")
            return False

def download_model():
    """Download the LLaVA-Med model."""
    print("🚀 Starting LLaVA-Med model download...")
    
    # Add LLaVA-Med to path
    llava_med_path = Path(__file__).parent / "LLaVA-Med"
    if llava_med_path.exists():
        sys.path.append(str(llava_med_path))
    
    try:
        from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
        from transformers import AutoTokenizer
        
        model_name = "microsoft/llava-med-v1.5-mistral-7b"
        
        print(f"📥 Downloading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Tokenizer downloaded: {len(tokenizer)} tokens")
        
        print(f"📥 Downloading model weights from {model_name}...")
        model = LlavaMistralForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )
        
        print(f"✅ Model downloaded: {model.num_parameters():,} parameters")
        print("🎉 Model ready for fine-tuning!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None, None

def verify_model():
    """Verify the model is working correctly."""
    print("🔍 Verifying model functionality...")
    
    try:
        from transformers import AutoTokenizer
        from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
        
        model_name = "microsoft/llava-med-v1.5-mistral-7b"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = LlavaMistralForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )
        
        # Test basic functionality
        test_text = "Hello, this is a test."
        inputs = tokenizer(test_text, return_tensors="pt")
        
        print("✅ Model verification successful!")
        print(f"   - Model parameters: {model.num_parameters():,}")
        print(f"   - Tokenizer vocab size: {len(tokenizer)}")
        print(f"   - Model device: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def main():
    """Main function to download and verify the model."""
    print("=" * 60)
    print("LLaVA-Med Model Download Script")
    print("=" * 60)
    
    # Check installation
    if not check_llava_med_installation():
        sys.exit(1)
    
    # Download model
    model, tokenizer = download_model()
    if model is None:
        sys.exit(1)
    
    # Verify model
    if not verify_model():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS: LLaVA-Med model is ready for fine-tuning!")
    print("=" * 60)
    print(f"Model: microsoft/llava-med-v1.5-mistral-7b")
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Tokenizer: {len(tokenizer)} tokens")
    print(f"Device: {next(model.parameters()).device}")
    print("=" * 60)

if __name__ == "__main__":
    main()
