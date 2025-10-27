"""
⭐ MODEL LOADER: Load fine-tuned LLaVA v1.6 model for inference

WHAT IT DOES:
- Loads base LLaVA v1.6 model from HuggingFace
- Loads LoRA adapters from checkpoints/ (if not using merged weights)
- Merges LoRA weights with base model (optional)
- Returns ready-to-use model for inference

WHY THIS IS CRITICAL:
- This is the ONE WAY to load the fine-tuned model correctly
- Handles both CPU and GPU inference
- Manages LoRA weight merging
- Handles different model architectures (LLaVA-Next vs classic LLaVA)

KEY CHALLENGES:
- Model trained on A100 GPU but needs to run on CPU
- LoRA adapters must be properly merged
- Different architectures (LLaVA-Next) require different loaders
- Memory constraints on CPU require careful loading

USAGE:
    tokenizer, model, processor, context_len = load_finetuned_llava(device="cpu")
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

DEFAULT_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"  # Updated to v1.6


def _get_model_type(path_or_id: str) -> Optional[str]:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path_or_id, trust_remote_code=True)
        return getattr(cfg, "model_type", None)
    except Exception:
        config_path = Path(path_or_id) / "config.json" if os.path.isdir(path_or_id) else None
        if config_path and config_path.exists():
            try:
                with config_path.open('r') as cf:
                    return json.load(cf).get('model_type')
            except Exception:
                return None
    return None


def _load_llava_next(base_path: str, device: str, dtype: torch.dtype):
    """
    Load LLaVA-Next model architecture.
    
    WHY LLaVA-Next:
    - Newer architecture with better vision-text alignment
    - Supports "anyres" (any resolution) image processing
    - Better handling of variable image sizes
    - Improved multimodal understanding
    
    HOW IT WORKS:
    1. Load AutoProcessor (handles image+text preprocessing)
    2. Load LlavaNextForConditionalGeneration model
    3. Apply device placement (CPU/GPU)
    4. Return processor + model
    
    CPU vs GPU:
    - CPU: Full precision (float32), memory efficient
    - GPU: Half precision (float16), 8-bit quantization optional
    - 8-bit only works on CUDA, not CPU/MPS
    """
    # Try to load processor, but fallback if not available
    try:
        processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=False)
        image_processor = getattr(processor, 'image_processor', None)
        if image_processor is None:
            raise RuntimeError("Processor did not provide an image processor")
    except Exception as e:
        print(f"⚠️ Could not load processor: {e}")
        print("📦 Loading tokenizer and vision processor separately...")
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=False)
        # Try to load vision processor separately
        try:
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained(base_path, trust_remote_code=True)
        except:
            raise RuntimeError("Could not load vision processor")
        
        # Create a minimal processor
        class MinimalProcessor:
            def __init__(self, tokenizer, image_processor):
                self.tokenizer = tokenizer
                self.image_processor = image_processor
            def __call__(self, **kwargs):
                return self.tokenizer(**kwargs)
        
        processor = MinimalProcessor(tokenizer, image_processor)

    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,  # Always use low memory mode
        "trust_remote_code": True,
    }
    
    # 8-bit quantization only works with CUDA, not CPU
    # Skip it on CPU/MPS to avoid errors
    use_8bit = os.getenv("USE_8BIT_QUANTIZATION", "false").lower() == "true"
    if use_8bit and device in ["cuda", "gpu"]:
        try:
            from transformers import BitsAndBytesConfig
            print("⚡ Using 8-bit quantization for faster GPU inference")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            model_kwargs["quantization_config"] = bnb_config
        except ImportError:
            print("⚠️ bitsandbytes not available, using full precision")
    else:
        print("ℹ️ Using full precision (8-bit only works on CUDA)")
    
    # Only use device_map for GPU
    if device in ["cuda", "gpu"]:
        model_kwargs["device_map"] = {"": device}
    # For CPU/MPS, don't use device_map to avoid accelerate dependency
    
    # Load checkpoint directly - the config says it's llava_next but we'll load as any model that works
    print(f"📦 Loading merged checkpoint from {base_path}...")
    try:
        # First, try to create a temporary config that transformers can understand
        import json
        import shutil
        
        # Read the checkpoint config and modify it temporarily
        config_path = Path(base_path) / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Backup original config
        backup_path = Path(base_path) / "config.json.backup"
        if not backup_path.exists():
            shutil.copy(config_path, backup_path)
        
        # Change model_type to mistral (a supported type) temporarily
        original_model_type = config_dict.get('model_type', '')
        config_dict['model_type'] = 'mistral'  # Use a known model type
        
        # Save the modified config
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Now try loading the model
        from transformers import AutoModelForCausalLM
        print("📦 Loading model...")
        # Ensure trust_remote_code is in model_kwargs, not duplicate
        if 'trust_remote_code' not in model_kwargs:
            model_kwargs['trust_remote_code'] = True
        model = AutoModelForCausalLM.from_pretrained(base_path, **model_kwargs)
        print("✅ Model loaded successfully")
        
        # Restore original config
        if original_model_type:
            config_dict['model_type'] = original_model_type
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        # Save model to cache directory for faster subsequent loads
        cache_dir = Path(base_path).parent / "cached_model"
        cache_dir.mkdir(exist_ok=True)
        print(f"💾 Caching model to {cache_dir} for faster future loads...")
        try:
            model.save_pretrained(str(cache_dir))
            print(f"✅ Model cached successfully")
        except Exception as save_err:
            print(f"⚠️ Could not cache model: {save_err}")
            
    except Exception as e:
        # If loading failed, try loading from cache
        cache_dir = Path(base_path).parent / "cached_model"
        if cache_dir.exists():
            print(f"⚠️ Loading failed, trying cached model at {cache_dir}")
            try:
                from transformers import AutoModelForCausalLM
                # Add trust_remote_code to model_kwargs if not present
                if 'trust_remote_code' not in model_kwargs:
                    model_kwargs['trust_remote_code'] = True
                model = AutoModelForCausalLM.from_pretrained(str(cache_dir), **model_kwargs)
                print("✅ Loaded from cache")
            except Exception as cache_error:
                print(f"❌ Cache load failed: {cache_error}")
                print(f"❌ Original error: {e}")
                import traceback
                traceback.print_exc()
                raise e
        else:
            print(f"❌ Failed to load: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Only move to device if not using device_map
    if "device_map" not in model_kwargs or model_kwargs["device_map"] not in ["cpu", {"": device}]:
        model.to(device)

    context_len = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'max_sequence_length', 2048)
    return tokenizer, model, processor, context_len


def _load_classic_llava(base_path: str, tokenizer, device: str, dtype: torch.dtype):
    """
    Load classic LLaVA model architecture.
    
    WHY CLASSIC LLaVA:
    - Original Microsoft architecture (llava-hf/llava-v1.6-mistral-7b)
    - Stable and well-tested
    - Better CPU compatibility
    - Simpler processing pipeline
    
    KEY DIFFERENCE FROM LLaVA-Next:
    - Classic LLaVA has fixed image processing
    - LLaVA-Next has dynamic "anyres" processing
    - Classic requires manual image preprocessing
    - LLaVA-Next handles it automatically
    
    HOW IT WORKS:
    1. Load AutoProcessor for image+text handling
    2. Load LlavaNextForConditionalGeneration model
    3. Apply device placement (CPU/GPU)
    4. Return FULL processor (image + text processing)
    
    IMPORTANT:
    This returns the full AutoProcessor, which handles both image preprocessing
    and text encoding automatically. Use processor.tokenizer for text-only operations.
    """
    print("📦 Loading LlavaNextForConditionalGeneration directly from checkpoint...")
    
    try:
        from transformers import AutoProcessor
        from transformers.models.llava_next import LlavaNextForConditionalGeneration
        
        # Load processor from checkpoint
        processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
        print("✅ Loaded processor from checkpoint")
        
        # Load model directly - this is the ONLY correct way
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        print("✅ Loaded LlavaNextForConditionalGeneration from checkpoint")
        
        # Move to device
        model.to(device)
        
        # Return the FULL processor
        context_len = getattr(model.config, "max_sequence_length", 2048)
        return model, processor, context_len
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        raise


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
    ⭐ MAIN FUNCTION: Load fine-tuned LLaVA model for inference
    
    HOW IT WORKS:
    1. Check for merged weights (complete model, no LoRA needed)
    2. If merged weights exist → load directly
    3. If not → load base model + LoRA adapters
    4. Merge LoRA weights (optional, can be skipped for faster loading)
    5. Return model ready for inference
    
    ARCHITECTURES SUPPORTED:
    - LLaVA-Next: Newer architecture with improved vision-text alignment
    - Classic LLaVA: Original architecture from Microsoft
    - Auto-detects based on config.json
    
    CPU vs GPU:
    - CPU: Float32 precision, slower but compatible
    - GPU: Float16 precision, faster but requires CUDA
    - Auto-selection based on device parameter
    
    Args:
        base_model: Base model path (default: llava-v1.6-mistral-7b)
        lora_dir: Directory with LoRA adapters (checkpoints/)
        device: Device to load on ("cpu", "cuda", "mps")
        
    Returns:
        Tuple of (tokenizer, model, processor, context_len)
        - tokenizer: Text tokenizer for prompt encoding
        - model: Loaded fine-tuned model
        - processor: Image+text processor
        - context_len: Maximum sequence length
    """
    print(f"🤖 Loading fine-tuned LLaVA model on {device}...")

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
    
    # Check for radiology_checkpoints first (merged v16 model)
    radiology_checkpoint = Path.cwd() / "radiology_checkpoints" / "merged" / "main_merged_v16"
    merged_candidate = Path(merged_path_env) if merged_path_env else (radiology_checkpoint if radiology_checkpoint.exists() else Path(lora_dir) / "merged")
    
    # Auto-enable merged if radiology_checkpoint exists
    if radiology_checkpoint.exists():
        use_merged = True
        print(f"✅ Found radiology_checkpoints, auto-enabling merged model loading")
    
    if use_merged and merged_candidate.exists():
        print(f"🗄️ Loading merged weights from {merged_candidate}")
        model_source = str(merged_candidate)
    elif use_merged:
        print("⚠️ Requested merged weights but path not found; falling back to base/LoRA loading.")
        model_source = base_model
    else:
        model_source = base_model

    model_type = (_get_model_type(model_source) or "").lower()
    is_llava_next = "llava_next" in model_type
    
    # For merged checkpoints, ALWAYS use classic LLaVA loader
    # The "llava_next" checkpoint is actually just a MistralForCausalLM without vision
    # We need to force classic LLaVA loading to get the full architecture
    
    # Check the actual architecture in config.json
    config_path = Path(model_source) / "config.json" if Path(model_source).exists() else None
    if config_path and config_path.exists():
        import json
        with config_path.open('r') as f:
            cfg = json.load(f)
            architecture = cfg.get('architectures', [])
            if architecture and 'LlavaNext' in architecture[0]:
                print(f"⚠️ Detected LLaVA-Next architecture but switching to classic loader")
                # Force classic LLaVA for merged checkpoints
                is_llava_next = False
    
    # Force classic LLaVA for merged checkpoints to ensure multimodal support
    if Path(model_source).exists() and is_llava_next:
        # Use LLaVA-Next loader
        processor = None
        tokenizer, model, processor, context_len = _load_llava_next(model_source, device, dtype)
        print("✅ Loaded LLaVA-Next model")
    else:
        # Load as classic LLaVA
        print(f"🤖 Loading as classic LLaVA from {model_source}")
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=False)
        model, processor, context_len = _load_classic_llava(model_source, tokenizer, device, dtype)
        # For LlavaNext, _load_classic_llava returns the FULL AutoProcessor
        # Use processor.tokenizer as the tokenizer
        tokenizer = processor.tokenizer
        print("✅ Loaded classic LLaVA model")

    lora_path = Path(lora_dir)
    should_load_lora = (not use_merged) and lora_path.exists() and any(lora_path.iterdir())
    if should_load_lora:
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
    elif not use_merged:
        print("⚠️ Warning: No LoRA weights found, using base model only")

    model.eval()
    print("✅ Model loaded successfully")
    return tokenizer, model, processor, context_len


def quick_load(device: str = "cpu"):
    """Convenience wrapper with defaults."""
    return load_finetuned_llava(device=device)
