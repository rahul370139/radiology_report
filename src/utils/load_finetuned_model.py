"""
Utility helpers to load the fine-tuned LLaVA model (CPU or GPU).
Returns (tokenizer, model, image_processor, context_len) ready for inference.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer
from llava.model import LlavaMistralForCausalLM
from peft import PeftModel

DEFAULT_MODEL_ID = "microsoft/llava-med-v1.5-mistral-7b"


def _resolve_base_model(base_model: Optional[str]) -> str:
    """Resolve the base model identifier or path, falling back to the default HF ID."""
    if not base_model:
        return DEFAULT_MODEL_ID
    if os.path.isdir(base_model):
        preproc_file = Path(base_model) / "preprocessor_config.json"
        if not preproc_file.exists():
            print("⚠️ preprocessor_config.json missing in provided path; using default model ID")
            return DEFAULT_MODEL_ID
    return base_model


def load_finetuned_llava(
    base_model: str = DEFAULT_MODEL_ID,
    lora_dir: str = "checkpoints",
    device: str = "cpu",
) -> Tuple[AutoTokenizer, torch.nn.Module, object, int]:
    """
    Load the fine-tuned LLaVA model and associated tokenizer / processor.

    Args:
        base_model: Base model path or Hugging Face ID.
        lora_dir: Directory containing LoRA adapter weights.
        device: Target device ("cpu", "cuda", or "mps").

    Returns:
        (tokenizer, model, image_processor, context_length)
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

    base_model = _resolve_base_model(base_model)

    dtype = torch.float32 if device == "cpu" else torch.float16

    use_merged = os.getenv("USE_MERGED_WEIGHTS", "false").lower() == "true"
    merged_path_env = os.getenv("MERGED_WEIGHTS_PATH")
    merged_candidate = None
    if use_merged:
        merged_candidate = Path(merged_path_env) if merged_path_env else Path(lora_dir) / "merged"
        if merged_candidate.exists():
            print(f"🗄️ Loading merged weights from {merged_candidate}")
            tokenizer_source = merged_candidate if (merged_candidate / "tokenizer_config.json").exists() else base_model
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
            model_kwargs = {
                "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
                "low_cpu_mem_usage": device == "cpu",
                "use_flash_attention_2": False,
                "trust_remote_code": True,
            }
            if device != "cpu":
                model_kwargs["device_map"] = {"": device}
            model = LlavaMistralForCausalLM.from_pretrained(str(merged_candidate), **model_kwargs)
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=device, dtype=model_kwargs["torch_dtype"])
            model.model.mm_projector.to(device=device, dtype=model_kwargs["torch_dtype"])
            model.to(device)
            image_processor = vision_tower.image_processor
            if image_processor is None:
                raise RuntimeError("Vision tower did not provide an image processor")
            context_len = getattr(model.config, "max_sequence_length", 2048)
            print("✅ Merged model loaded successfully")
            return tokenizer, model, image_processor, context_len
        else:
            print("⚠️ Requested merged weights but path not found; falling back to LoRA loading.")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": device == "cpu",
        "use_flash_attention_2": False,
        "trust_remote_code": True,
    }
    if device != "cpu":
        model_kwargs["device_map"] = {"": device}

    model = LlavaMistralForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Ensure tokenizer includes multimodal tokens (matches original builder behaviour)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # Load and move the vision tower / projector
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=dtype)
    model.model.mm_projector.to(device=device, dtype=dtype)
    model.to(device)
    image_processor = vision_tower.image_processor
    if image_processor is None:
        raise RuntimeError("Vision tower did not provide an image processor")

    # Attach LoRA weights if present
    lora_path = Path(lora_dir)
    if lora_path.exists() and any(lora_path.iterdir()):
        print("🔗 Loading LoRA weights...")
        lora_device_map = None if device == "cpu" else {"": device}
        model = PeftModel.from_pretrained(model, lora_dir, device_map=lora_device_map)
        model = model.to(device)
        print("✅ LoRA weights loaded successfully")
        if os.getenv("MERGE_LORA_ON_LOAD", "false").lower() == "true":
            print("🧮 Merging LoRA adapters into the base model...")
            merged_model = model.merge_and_unload()
            save_path = os.getenv("SAVE_MERGED_WEIGHTS_PATH")
            if save_path:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                merged_model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"💾 Saved merged weights to {save_dir}")
            model = merged_model.to(device)
    else:
        print("⚠️ Warning: No LoRA weights found, using base model only")

    model.eval()

    context_len = getattr(model.config, "max_sequence_length", 2048)
    print("✅ Model loaded successfully")
    return tokenizer, model, image_processor, context_len


def quick_load(device: str = "cpu"):
    """Convenience wrapper with defaults."""
    return load_finetuned_llava(device=device)
