"""
One-liner loader for the fine-tuned model (CPU or GPU)
Returns (tokenizer, model, image_processor, context_len) ready for inference.
"""

import torch
import os
from pathlib import Path
from llava.model.builder import load_pretrained_model
from peft import PeftModel
from llava.mm_utils import get_model_name_from_path

def load_finetuned_llava(base_model="microsoft/llava-med-v1.5-mistral-7b",
                         lora_dir="checkpoints",
                         device="cpu"):
    """
    Returns (tokenizer, model, image_processor, context_len) ready for inference.
    
    Args:
        base_model: Base model path or name
        lora_dir: Directory containing LoRA weights
        device: "cpu", "mps", or "cuda"
    
    Returns:
        tuple: (tokenizer, model, image_processor, context_len)
    """
    print(f"🤖 Loading fine-tuned LLaVA model on {device}...")
    
    # Allow environment overrides
    env_base = os.getenv("BASE_MODEL_PATH")
    env_lora = os.getenv("LORA_DIR")
    if env_base:
        base_model = env_base
        print(f"📦 Using BASE_MODEL_PATH from env: {base_model}")
    if env_lora:
        lora_dir = env_lora
        print(f"📦 Using LORA_DIR from env: {lora_dir}")
    
    # Set CUDA environment
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Mock CUDA availability for CPU
        torch.cuda.is_available = lambda: False
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Get model name
    model_name = get_model_name_from_path(base_model)
    
    # Load base model
    print("📥 Loading base model...")
    # Force CPU and float32 for compatibility
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.cuda.is_available = lambda: False
        # Monkey patch torch.float16 to use float32 on CPU
        original_float16 = torch.float16
        torch.float16 = torch.float32
    
    tokenizer, model, image_proc, ctx_len = load_pretrained_model(
        model_path=base_model, 
        model_base=None, 
        model_name=model_name,
        device_map="cpu", 
        load_8bit=False, 
        load_4bit=False
    )
    
    # Ensure image processor is loaded correctly
    if image_proc is None:
        print("⚠️ Image processor is None, trying to load manually...")
        from transformers import CLIPImageProcessor
        try:
            # Try loading from the base model path first
            image_proc = CLIPImageProcessor.from_pretrained(base_model)
            print("✅ Image processor loaded from base model")
        except Exception as e:
            print(f"⚠️ Failed to load from base model: {e}")
            try:
                # Fallback to the original model name
                image_proc = CLIPImageProcessor.from_pretrained("microsoft/llava-med-v1.5-mistral-7b")
                print("✅ Image processor loaded from HuggingFace hub")
            except Exception as e2:
                print(f"❌ Failed to load image processor: {e2}")
                # Create a basic CLIPImageProcessor as last resort
                image_proc = CLIPImageProcessor()
                print("⚠️ Using basic CLIPImageProcessor as fallback")
    
    # Move model to device
    model = model.to(device)
    
    # Attach LoRA weights if available
    lora_path = Path(lora_dir)
    if lora_path.exists() and any(lora_path.iterdir()):
        print("🔗 Loading LoRA weights...")
        try:
            model = PeftModel.from_pretrained(model, lora_dir, device_map="cpu" if device == "cpu" else device)
            print("✅ LoRA weights loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load LoRA weights: {e}")
            print("   Proceeding with base model only")
    else:
        print("⚠️ Warning: No LoRA weights found, using base model only")
    
    # Set to evaluation mode
    model.eval()
    
    # Restore original float16 if we patched it
    if device == "cpu" and 'original_float16' in locals():
        torch.float16 = original_float16
    
    print("✅ Model loaded successfully")
    return tokenizer, model, image_proc, ctx_len

# Convenience function for quick loading
def quick_load(device="cpu"):
    """Quick load with default settings"""
    return load_finetuned_llava(device=device)
