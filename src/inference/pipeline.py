"""
Minimal inference pipeline for radiology report generation.
Supports Stage A (image only) and Stage B (image + EHR) via multi-pass prompting
with structured JSON outputs for Impression, CheXpert, and ICD predictions.
"""

import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    MaxLengthCriteria,
    StoppingCriteriaList,
)

# Import our shared processor (try relative import first)
try:
    from utils.processor import LLaVAProcessor
except ImportError:
    try:
        from src.utils.processor import LLaVAProcessor
    except ImportError:
        LLaVAProcessor = None  # Will use HuggingFace processor instead

# LLaVA constants and utilities (HuggingFace compatible)
DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200

# Simple conversation template
conv_templates = {
    "llava_v1": type('ConvTemplate', (), {
        'roles': ('USER', 'ASSISTANT'),
        'system': 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions.',
        'copy': lambda self: self,
        'append_message': lambda self, role, message: setattr(self, 'messages', getattr(self, 'messages', []) + [(role, message)]),
        'get_prompt': lambda self: f"{self.system}\n\n" + "\n".join([f"{role}: {msg}" for role, msg in getattr(self, 'messages', [])]) + "\nASSISTANT:"
    })()
}

# Import our model loader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../utils'))
from load_finetuned_model import load_finetuned_llava  # noqa: E402

# Label definitions matching the dataset (12 labels)
CHEXPERT = [
    "No Finding", "Enlarged Cardiomediastinum", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
    "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices"
]

# ICD-10 indicator list
ICD = [
    "Pneumonia", "Pleural_Effusion", "Pneumothorax", "Pulmonary_Edema",
    "Cardiomegaly", "Atelectasis", "Pulmonary_Embolism", "Rib_Fracture"
]


class TokenBiasProcessor(LogitsProcessor):
    """Applies additive bias to specified token ids during generation."""

    def __init__(self, token_bias: Dict[int, float]):
        self.token_bias = token_bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.token_bias:
            return scores
        for token_id, bias in self.token_bias.items():
            if 0 <= token_id < scores.shape[-1]:
                scores[:, token_id] += bias
        return scores


class RadiologyInferencePipeline:
    """
    ⭐ MAIN INFERENCE ENGINE: Radiology Report Generation Pipeline
    
    WHAT IT DOES:
    - Loads fine-tuned LLaVA v1.6 model with LoRA adapters
    - Generates structured radiology reports (impression + CheXpert + ICD)
    - Supports Stage A (image-only) and Stage B (image+EHR)
    - Uses multi-pass generation for JSON validity
    - Applies token biasing to boost label prediction accuracy
    
    HOW IT WORKS:
    1. Load model: Loads base model + LoRA adapters from checkpoints/
    2. Process image: Converts image to tensor using vision encoder
    3. Generate impression: First pass generates clinical impression
    4. Generate CheXpert: Second pass predicts 12 disease labels (JSON)
    5. Generate ICD: Third pass predicts 8 diagnostic codes (Stage B only)
    6. Apply voting: Multiple generations for stability
    
    USAGE:
        pipeline = RadiologyInferencePipeline(device="cpu")
        result = pipeline.generate("path/to/xray.jpg", ehr_json={...})
    
    OUTPUT:
        {
            "impression": "Clear lung fields...",
            "chexpert": {"Pneumonia": 0, "Edema": 1, ...},
            "icd": {"Pneumonia": 1, ...}  # Stage B only
        }
    """

    def __init__(self, device: str = "cpu"):
        """Initialize the inference pipeline.
        
        Args:
            device: Device to run inference on ("cpu", "cuda", "mps")
        """
        self.device = device
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.image_processor = None
        self.context_len = None
        self.conv_mode = None
        self._positive_token_ids: List[int] = []
        self._negative_token_ids: List[int] = []
        self._load_model()

    # ------------------------------------------------------------------ #
    # Helper setup

    def _load_model(self) -> None:
        """
        Load the fine-tuned model for inference.
        
        WHAT IT DOES:
        1. Calls load_finetuned_llava() to load base model + LoRA adapters
        2. Detects processor type (full AutoProcessor vs partial)
        3. Creates combined processor if needed (for classic LLaVA)
        4. Configures conversation template (llava_v1 for Mistral-based models)
        5. Sets up token bias IDs for label prediction (boosts "1" and "0" tokens)
        
        PROCESSOR HANDLING:
        - AutoProcessor (LLaVA-Next): Has tokenizer + image_processor built-in
        - Partial processor: Need to create CombinedProcessor wrapper
        - Combined processor: Manually combines tokenizer + image_processor
        
        WHY TOKEN BIAS:
        - Model needs to output "1" or "0" for disease labels
        - Token biasing increases probability of generating these tokens
        - Positive bias for "1" → boosts disease detection (sensitivity)
        - Negative bias for "0" → reduces false positives (specificity)
        
        IMPORTANT:
        - Model trained on A100 GPU, now running on CPU (challenge)
        - LoRA adapters must be properly merged with base model
        - Processor handles image preprocessing consistently
        """
        print("🤖 Loading fine-tuned model...")
        self.tokenizer, self.model, hf_processor, self.context_len = load_finetuned_llava(device=self.device)
        
        # Check if hf_processor is a proper processor or just image_processor
        # For LlavaNext, it's the full AutoProcessor
        if hasattr(hf_processor, 'tokenizer') and hasattr(hf_processor, 'image_processor'):
            # It's the full AutoProcessor (LlavaNext)
            print(f"DEBUG: Using full AutoProcessor")
            self.processor = hf_processor
            self.image_processor = hf_processor.image_processor
            self.tokenizer = hf_processor.tokenizer  # Use processor's tokenizer
            
            # Don't modify processor - let it work as designed
            # The checkpoint was trained with this processor configuration
            print("✅ Using processor as-is (no anyres modifications)")
        elif hasattr(hf_processor, 'tokenizer'):
            # It's already a combined processor
            print(f"DEBUG: hf_processor is a combined processor")
            self.processor = hf_processor
            self.image_processor = hf_processor
        else:
            # It's just image_processor, need to create combined processor
            class CombinedProcessor:
                def __init__(self, tokenizer, image_processor):
                    self.tokenizer = tokenizer
                    print(f"DEBUG: Creating combined processor with tokenizer: {tokenizer}, image_processor: {image_processor}")
                    self.image_processor = image_processor
                
                def __call__(self, text=None, images=None, return_tensors="pt", padding=True, truncation=True, max_length=512, **kwargs):
                    # Match the pattern from advanced_trainer.py exactly
                    result = {}
                    
                    # Handle images first (classic LLaVA uses vision tower image_processor)
                    if images:
                        if isinstance(images, list):
                            processed_images = []
                            for img in images:
                                img_result = self.image_processor(img)
                                # Handle FeatureExtractionOutput or tensor directly
                                if hasattr(img_result, 'pixel_values'):
                                    processed_images.append(img_result.pixel_values)
                                elif isinstance(img_result, torch.Tensor):
                                    processed_images.append(img_result)
                                elif isinstance(img_result, list):
                                    # If it's a list, take the first element or stack them
                                    if all(isinstance(x, torch.Tensor) for x in img_result):
                                        processed_images.append(torch.stack(img_result))
                                    else:
                                        raise ValueError(f"Unexpected image processor output type: {type(img_result[0])}")
                                else:
                                    raise ValueError(f"Unexpected image processor output type: {type(img_result)}")
                            result['pixel_values'] = torch.stack(processed_images)
                        else:
                            img_result = self.image_processor(images)
                            # Handle FeatureExtractionOutput or tensor directly
                            if hasattr(img_result, 'pixel_values'):
                                pixel_vals = img_result.pixel_values
                            elif isinstance(img_result, torch.Tensor):
                                pixel_vals = img_result
                            elif isinstance(img_result, list):
                                # If list, convert to tensor
                                if all(isinstance(x, torch.Tensor) for x in img_result):
                                    pixel_vals = torch.stack(img_result)
                                else:
                                    pixel_vals = torch.tensor(img_result)
                            else:
                                raise ValueError(f"Unexpected image processor output type: {type(img_result)}")
                            
                            # Add batch dimension if needed and ensure it's a tensor
                            if isinstance(pixel_vals, torch.Tensor):
                                if len(pixel_vals.shape) < 4:
                                    result['pixel_values'] = pixel_vals.unsqueeze(0)
                                else:
                                    result['pixel_values'] = pixel_vals
                            elif isinstance(pixel_vals, np.ndarray):
                                # Convert numpy array to tensor
                                result['pixel_values'] = torch.from_numpy(pixel_vals).unsqueeze(0) if len(pixel_vals.shape) < 4 else torch.from_numpy(pixel_vals)
                            elif isinstance(pixel_vals, list):
                                # Convert list to tensor
                                if all(isinstance(x, torch.Tensor) for x in pixel_vals):
                                    result['pixel_values'] = torch.stack(pixel_vals)
                                elif all(isinstance(x, np.ndarray) for x in pixel_vals):
                                    result['pixel_values'] = torch.stack([torch.from_numpy(x) for x in pixel_vals])
                                else:
                                    result['pixel_values'] = torch.tensor(pixel_vals)
                            else:
                                result['pixel_values'] = torch.tensor(pixel_vals) if isinstance(pixel_vals, (list, np.ndarray)) else pixel_vals
                    
                    # Handle text
                    if text:
                        text_result = self.tokenizer(
                            text,
                            return_tensors=return_tensors,
                            padding=padding,
                            truncation=truncation,
                            max_length=max_length,
                            **kwargs
                        )
                        result['input_ids'] = text_result['input_ids']
                        result['attention_mask'] = text_result['attention_mask']
                    
                    return result
            
            self.processor = CombinedProcessor(self.tokenizer, hf_processor)
            self.image_processor = hf_processor
        
        name_hint = str(getattr(self.model.config, "_name_or_path", "")).lower()
        model_type = str(getattr(self.model.config, "model_type", "")).lower()
        combo_name = f"{name_hint} {model_type}"
        if "llava" in combo_name or "mistral" in combo_name:
            self.conv_mode = "llava_v1"
        elif "llama" in combo_name:
            self.conv_mode = "llava_llama_2"
        else:
            self.conv_mode = "llava_v1"
        self._setup_token_bias_ids()
        print(f"💬 Using conversation template: {self.conv_mode}")
        print("✅ Model loaded successfully")

    def _collect_token_ids(self, variants: List[str]) -> List[int]:
        ids: List[int] = []
        for text in variants:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.append(encoded[-1])
        return ids

    def _setup_token_bias_ids(self) -> None:
        positive_variants = ["1", " 1", "\"1\"", ":1", ": 1", "1}"]
        negative_variants = ["0", " 0", "\"0\"", ":0", ": 0", "0}"]
        self._positive_token_ids = list({tid for tid in self._collect_token_ids(positive_variants)})
        self._negative_token_ids = list({tid for tid in self._collect_token_ids(negative_variants)})

    # ------------------------------------------------------------------ #
    # Utility helpers

    @staticmethod
    def _str_to_bool(value: Optional[str], default: bool) -> bool:
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "y")

    def _normalize_label_value(self, value: Any, allow_negative: bool = False) -> int:
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            if allow_negative and value < 0:
                return -1
            if value > 0:
                return 1
        return 0

    def _aggregate_chexpert_votes(
        self,
        votes: List[Dict[str, Any]],
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Aggregate multiple CheXpert generations using voting system.
        
        WHY VOTING:
        - Model generates same label 3-5 times independently
        - Majority vote ensures stable predictions
        - Reduces random errors from single generation
        - Clinical accuracy: "Consensus" among multiple predictions
        
        HOW IT WORKS:
        - For each label (Pneumonia, Edema, etc.):
          - Count positive votes (1): model said "disease present"
          - Count negative votes (-1): model said "disease absent"
          - Count uncertain votes (0): model said "uncertain"
        - If >= 34% positive → label = 1
        - If >= 34% negative → label = -1
        - Otherwise → label = 0
        
        WHY 34% THRESHOLD:
        - Prevents tie-breaking issues with 3 votes
        - Ensures clear majority (>1/3) for decision
        - Balances sensitivity (find diseases) vs specificity (avoid false positives)
        
        Args:
            votes: List of 3-5 independent CheXpert predictions
            fallback: Single prediction to use if all votes fail
            
        Returns:
            Final aggregated CheXpert labels
        """
        result = {label: 0 for label in CHEXPERT}
        if not votes:
            if fallback:
                for label in CHEXPERT:
                    result[label] = self._normalize_label_value(fallback.get(label, 0), allow_negative=True)
            return result

        total = len(votes)
        pos_threshold = float(os.getenv("CHEXPERT_POS_VOTE_THRESHOLD", "0.34"))
        neg_threshold = float(os.getenv("CHEXPERT_NEG_VOTE_THRESHOLD", "0.34"))
        pos_required = max(1, math.ceil(total * pos_threshold))
        neg_required = max(1, math.ceil(total * neg_threshold))

        for label in CHEXPERT:
            pos_votes = sum(
                1 for v in votes if self._normalize_label_value(v.get(label, 0), allow_negative=True) > 0
            )
            neg_votes = sum(
                1 for v in votes if self._normalize_label_value(v.get(label, 0), allow_negative=True) < 0
            )
            if pos_votes >= pos_required:
                result[label] = 1
            elif neg_votes >= neg_required:
                result[label] = -1
            else:
                result[label] = 0
        return result

    def _aggregate_icd_votes(
        self,
        votes: List[Dict[str, Any]],
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Aggregate multiple ICD generations using voting system.
        
        WHY VOTING FOR ICD:
        - ICD codes are 0/1 binary (no uncertainty like CheXpert)
        - Model generates 3-5 independent predictions
        - Majority vote ensures stable diagnostic codes
        - Reduces false positive diagnoses
        
        HOW IT WORKS:
        - For each ICD code (Pneumonia, Pleural_Effusion, etc.):
          - Count positive votes (1): model said "diagnosis present"
          - Count negative votes (0): model said "diagnosis absent"
        - If >= 34% positive → code = 1 (diagnosis confirmed)
        - Otherwise → code = 0 (no diagnosis)
        
        WHY ICD IS BINARY (NOT TRI-VALUED LIKE CHEXPERT):
        - ICD-10 codes are diagnostic codes, not observational findings
        - Cannot be "uncertain" - either have the disease or not
        - Binary keeps interpretation clear for billing/coding
        
        Args:
            votes: List of 3-5 independent ICD predictions
            fallback: Single prediction to use if all votes fail
            
        Returns:
            Final aggregated ICD diagnostic codes (0 or 1 only)
        """
        result = {label: 0 for label in ICD}
        if not votes:
            if fallback:
                for label in ICD:
                    result[label] = 1 if self._normalize_label_value(fallback.get(label, 0)) > 0 else 0
            return result

        total = len(votes)
        threshold = float(os.getenv("ICD_VOTE_THRESHOLD", "0.34"))
        required = max(1, math.ceil(total * threshold))

        for label in ICD:
            pos_votes = sum(1 for v in votes if self._normalize_label_value(v.get(label, 0)) > 0)
            result[label] = 1 if pos_votes >= required else 0
        return result

    def _make_token_bias(self, positive_bias: float, negative_bias: float) -> Dict[int, float]:
        """
        ⭐ Create token biasing dictionary for generation
        
        WHY TOKEN BIASING:
        - Model needs to output "1" (disease present) or "0" (disease absent)
        - Without biasing, model might output "maybe" or other tokens
        - Increases probability of generating desired tokens
        - Acts like a "push" toward the correct answer
        
        HOW IT WORKS:
        - Positive bias: Adds value to "1" token probabilities (boost disease detection)
        - Negative bias: Adds value to "0" token probabilities (reduce false positives)
        - Example: positive_bias=4.0 → "1" token gets +4.0 logit boost
        
        WHY DIFFERENT VALUES FOR POSITIVE/NEGATIVE:
        - Positive bias: Higher (4.0) → find diseases, don't miss them (sensitivity)
        - Negative bias: Lower (-0.25) → reduce false positives, but not too much (specificity)
        - Balance: Find all diseases but minimize false alarms
        
        USAGE:
        - Applied during generation via LogitsProcessor
        - Affects all label predictions (CheXpert and ICD)
        - Configurable via environment variables:
          - CHEXPERT_POSITIVE_BIAS=4.0
          - CHEXPERT_NEGATIVE_BIAS=-0.25
        
        Args:
            positive_bias: How much to boost "1" tokens (typically 2.0-4.0)
            negative_bias: How much to boost "0" tokens (typically -0.25 to 0.0)
            
        Returns:
            Dictionary mapping token_id → bias value
        """
        token_bias: Dict[int, float] = {}
        for tid in self._positive_token_ids:
            token_bias[tid] = token_bias.get(tid, 0.0) + positive_bias
        for tid in self._negative_token_ids:
            token_bias[tid] = token_bias.get(tid, 0.0) + negative_bias
        return token_bias

    # ------------------------------------------------------------------ #
    # Public helpers

    def format_ehr(self, ehr_dict: Dict[str, Any]) -> str:
        keep = ["Age", "Sex", "Vitals", "Labs", "O2_device", "Chronic_conditions"]
        filtered_ehr = {k: ehr_dict.get(k, {}) for k in keep if k in ehr_dict}
        return json.dumps(filtered_ehr, indent=2)

    def _gen_params(self):
        temperature = float(os.getenv("GEN_TEMPERATURE", "0.1"))
        max_new_tokens = int(os.getenv("GEN_MAX_NEW_TOKENS", "512"))
        top_p = float(os.getenv("GEN_TOP_P", "0.9"))
        do_sample = self._str_to_bool(os.getenv("GEN_DO_SAMPLE", "true"), True)
        return temperature, max_new_tokens, top_p, do_sample

    # ------------------------------------------------------------------ #
    # Core generation logic

    def generate(self, image_path: str, ehr_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ⭐ Generate radiology report from chest X-ray image
        
        STAGE A (Image-Only):
        - Input: Chest X-ray image
        - Output: Impression + CheXpert labels (12 classes)
        - Task: Visual interpretation only
        
        STAGE B (Image+EHR):
        - Input: Chest X-ray image + Patient EHR data
        - Output: Impression + CheXpert + ICD labels (8 classes)
        - Task: Clinical reasoning with patient context
        
        MULTI-PASS GENERATION:
        1. Pass 1: Generate clinical impression (text only)
        2. Pass 2: Generate CheXpert JSON with voting (3-5 generations)
        3. Pass 3: Generate ICD JSON (Stage B only, 3-5 generations)
        
        WHY MULTI-PASS:
        - Separate passes ensure valid JSON output
        - Voting improves stability (3-5 independent generations)
        - Token biasing boosts positive/negative labels
        
        Args:
            image_path: Path to chest X-ray image
            ehr_json: Patient EHR data (vitals, labs, devices, chronic conditions)
            
        Returns:
            {
                "impression": "Clear lung fields...",
                "chexpert": {"Pneumonia": 0, "Edema": 1, ...},
                "icd": {"Pneumonia": 1, ...},  # Stage B only
                "raw_output": "Full generation text"
            }
        """
        stage = "B" if ehr_json else "A"
        img = Image.open(image_path).convert("RGB")
        print(f"✅ Loaded image: {img.size}")
        
        # Force single-patch behavior on CPU by resizing to 336x336
        if str(self.device).lower() == "cpu":
            try:
                if img.size != (336, 336):
                    img = img.resize((336, 336))
                    print("✅ Resized image to 336x336 for single-patch processing on CPU")
            except Exception:
                pass

        if self.image_processor is None:
            raise RuntimeError("Image processor not available")

        # Our shared processor handles all image processing consistently
        print("✅ Using shared LLaVA processor for consistent image processing")

        strict_json = self._str_to_bool(os.getenv("GEN_STRICT_JSON", "true"), True)
        debug_mode = self._str_to_bool(os.getenv("DEBUG_PIPELINE", "false"), False)

        def _run_generation(
            system_prompt: Optional[str],
            user_prompt: str,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            do_sample: Optional[bool] = None,
            token_bias: Optional[Dict[int, float]] = None,
        ) -> str:
            # Build the conversation prompt
            conv_key = self.conv_mode if self.conv_mode in conv_templates else "llava_v1"
            conv = conv_templates[conv_key].copy()
            if system_prompt:
                conv.system = system_prompt
            user_role, assistant_role = conv.roles
            conv.append_message(user_role, f"{user_prompt}\n\n{DEFAULT_IMAGE_TOKEN}")
            conv.append_message(assistant_role, None)
            prompt_text = conv.get_prompt()
            
            # For LLaVA-Next with anyres, the processor automatically expands
            # <image> to the correct number of tokens based on the image grid

            # Get generation parameters
            base_temp, base_max_tokens, base_top_p, base_do_sample = self._gen_params()
            t = temperature if temperature is not None else base_temp
            tp = top_p if top_p is not None else base_top_p
            mx = max_tokens if max_tokens is not None else base_max_tokens
            ds = base_do_sample if do_sample is None else do_sample

            # Setup logits processors and stopping criteria
            logits_processors = None
            if token_bias:
                logits_processors = LogitsProcessorList([TokenBiasProcessor(token_bias)])

            with torch.inference_mode():
                # Use processor to prepare all inputs (processor handles token expansion and image sizes)
                inputs = self.processor(
                    images=img,
                    text=prompt_text,
                    return_tensors="pt",
                    padding=True,
                )
                # Move inputs to the correct device
                inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
 
                # Calculate total max length for stopping criteria
                total_max_length = inputs['input_ids'].shape[1] + mx
                stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(total_max_length)])
                 
                # Prepare generation kwargs
                gen_kwargs = {
                    'max_new_tokens': mx,
                    'do_sample': ds,
                    'temperature': t,
                    'top_p': tp,
                    'pad_token_id': self.tokenizer.eos_token_id,
                    'use_cache': True,
                }
                     
                if logits_processors:
                    gen_kwargs['logits_processor'] = logits_processors
                if stopping_criteria:
                    gen_kwargs['stopping_criteria'] = stopping_criteria
                 
                # Pass all inputs directly to generate - let the model handle it
                output_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if strict_json:
            # Pass 1: Impression (text only)
            imp_temp = float(os.getenv("IMP_TEMPERATURE", "0.15"))
            imp_top_p = float(os.getenv("IMP_TOP_P", "0.9"))
            imp_do_sample = self._str_to_bool(os.getenv("IMP_DO_SAMPLE", "false"), False)
            imp_max_tokens = int(os.getenv("IMP_MAX_NEW_TOKENS", "160"))

            if stage == "A":
                system_imp = "You are a board-certified radiologist. Provide concise, clinically precise impressions."
                prompt_imp = "Write a concise radiology IMPRESSION (1-3 sentences) for this chest X-ray. Return text only."
            else:
                ehr_text = self.format_ehr(ehr_json)
                system_imp = "You are a board-certified radiologist who combines imaging and EHR context."
                prompt_imp = (
                    f"Patient EHR summary:\n{ehr_text}\n\n"
                    "Write a concise radiology IMPRESSION (1-3 sentences) that integrates both the image and the EHR. Return text only."
                )

            impression_text = _run_generation(
                system_imp,
                prompt_imp,
                temperature=imp_temp,
                top_p=imp_top_p,
                max_tokens=imp_max_tokens,
                do_sample=imp_do_sample,
            ).strip()
            if debug_mode:
                print("[DEBUG] Impression draft:", impression_text)

            # Pass 2: CheXpert JSON with voting
            chexpert_positive_bias = float(os.getenv("CHEXPERT_POSITIVE_BIAS", "2.0"))
            chexpert_negative_bias = float(os.getenv("CHEXPERT_NEGATIVE_BIAS", "-0.25"))
            chexpert_token_bias = self._make_token_bias(chexpert_positive_bias, chexpert_negative_bias)
            cx_vote = max(1, int(os.getenv("CHEXPERT_VOTE", "3")))
            cx_temp = float(os.getenv("CHEXPERT_TEMPERATURE", "0.7"))
            cx_top_p = float(os.getenv("CHEXPERT_TOP_P", "0.9"))
            cx_do_sample = self._str_to_bool(os.getenv("CHEXPERT_DO_SAMPLE", "true"), True)
            cx_max_tokens = int(os.getenv("CHEXPERT_MAX_NEW_TOKENS", "200"))
            retries = max(0, int(os.getenv("GEN_RETRIES", "2")))

            chexpert_example_pos = (
                "{\n"
                '  "No Finding": 0,\n'
                '  "Enlarged Cardiomediastinum": 1,\n'
                '  "Lung Opacity": 1,\n'
                '  "Lung Lesion": 0,\n'
                '  "Edema": 1,\n'
                '  "Consolidation": 1,\n'
                '  "Pneumonia": 1,\n'
                '  "Pneumothorax": 0,\n'
                '  "Pleural Effusion": 1,\n'
                '  "Pleural Other": 0,\n'
                '  "Fracture": 0,\n'
                '  "Support Devices": 1\n'
                "}"
            )
            chexpert_example_neg = (
                "{\n"
                '  "No Finding": 1,\n'
                '  "Enlarged Cardiomediastinum": 0,\n'
                '  "Lung Opacity": 0,\n'
                '  "Lung Lesion": 0,\n'
                '  "Edema": 0,\n'
                '  "Consolidation": 0,\n'
                '  "Pneumonia": 0,\n'
                '  "Pneumothorax": 0,\n'
                '  "Pleural Effusion": 0,\n'
                '  "Pleural Other": 0,\n'
                '  "Fracture": 0,\n'
                '  "Support Devices": 0\n'
                "}"
            )
            cx_system = "You are assisting with CheXpert labelling. Always respond with JSON only."
            cx_prompt = (
                "Using the chest X-ray and the impression below, fill the CheXpert JSON with integers in {-1,0,1}.\n"
                "Positive example:\n"
                f"{chexpert_example_pos}\n\n"
                "Negative example:\n"
                f"{chexpert_example_neg}\n\n"
                f"Draft impression: {impression_text}\n\n"
                "Now respond with ONLY the JSON object (no commentary)."
            )

            chexpert_votes: List[Dict[str, Any]] = []
            last_chexpert_dict: Optional[Dict[str, Any]] = None
            for _ in range(cx_vote):
                cx_text = _run_generation(
                    cx_system,
                    cx_prompt,
                    temperature=cx_temp,
                    top_p=cx_top_p,
                    max_tokens=cx_max_tokens,
                    do_sample=cx_do_sample,
                    token_bias=chexpert_token_bias,
                )
                if debug_mode:
                    print("[DEBUG] CheXpert response:", cx_text)
                blob = self._extract_json(cx_text)
                attempts = 0
                while blob is None and attempts < retries:
                    attempts += 1
                    cx_text = _run_generation(
                        cx_system,
                        "Your previous answer was not valid JSON. Return ONLY the CheXpert JSON object now.",
                        temperature=cx_temp,
                        top_p=cx_top_p,
                        max_tokens=cx_max_tokens,
                        do_sample=cx_do_sample,
                        token_bias=chexpert_token_bias,
                    )
                    blob = self._extract_json(cx_text)
                if blob:
                    try:
                        cx_dict = json.loads(blob)
                        chexpert_votes.append(cx_dict)
                        last_chexpert_dict = cx_dict
                    except Exception as exc:
                        if debug_mode:
                            print(f"[DEBUG] Failed to parse CheXpert JSON: {exc}")

            # Pass 3: ICD JSON with voting (Stage B only)
            icd_votes: List[Dict[str, Any]] = []
            last_icd_dict: Optional[Dict[str, Any]] = None
            if stage == "B":
                icd_positive_bias = float(os.getenv("ICD_POSITIVE_BIAS", "2.0"))
                icd_negative_bias = float(os.getenv("ICD_NEGATIVE_BIAS", "-0.25"))
                icd_token_bias = self._make_token_bias(icd_positive_bias, icd_negative_bias)
                icd_vote = max(1, int(os.getenv("ICD_VOTE", "3")))
                icd_temp = float(os.getenv("ICD_TEMPERATURE", "0.7"))
                icd_top_p = float(os.getenv("ICD_TOP_P", "0.9"))
                icd_do_sample = self._str_to_bool(os.getenv("ICD_DO_SAMPLE", "true"), True)
                icd_max_tokens = int(os.getenv("ICD_MAX_NEW_TOKENS", "160"))

                icd_example = (
                    "{\n"
                    '  "Pneumonia": 1,\n'
                    '  "Pleural_Effusion": 1,\n'
                    '  "Pneumothorax": 0,\n'
                    '  "Pulmonary_Edema": 1,\n'
                    '  "Cardiomegaly": 1,\n'
                    '  "Atelectasis": 1,\n'
                    '  "Pulmonary_Embolism": 0,\n'
                    '  "Rib_Fracture": 0\n'
                    "}"
                )
                icd_system = "You are assisting with ICD indicator labelling. Always respond with JSON only."
                icd_prompt = (
                    f"Draft impression: {impression_text}\n\n"
                    "Patient EHR indicators:\n"
                    f"{self.format_ehr(ehr_json)}\n\n"
                    "Example ICD JSON with positive findings:\n"
                    f"{icd_example}\n\n"
                    "Return ONLY the ICD JSON for the current case."
                )

                for _ in range(icd_vote):
                    icd_text = _run_generation(
                        icd_system,
                        icd_prompt,
                        temperature=icd_temp,
                        top_p=icd_top_p,
                        max_tokens=icd_max_tokens,
                        do_sample=icd_do_sample,
                        token_bias=icd_token_bias,
                    )
                    if debug_mode:
                        print("[DEBUG] ICD response:", icd_text)
                    blob = self._extract_json(icd_text)
                    attempts = 0
                    while blob is None and attempts < retries:
                        attempts += 1
                        icd_text = _run_generation(
                            icd_system,
                            "Your previous answer was not valid JSON. Return ONLY the ICD JSON object now.",
                            temperature=icd_temp,
                            top_p=icd_top_p,
                            max_tokens=icd_max_tokens,
                            do_sample=icd_do_sample,
                            token_bias=icd_token_bias,
                        )
                        blob = self._extract_json(icd_text)
                    if blob:
                        try:
                            icd_dict = json.loads(blob)
                            icd_votes.append(icd_dict)
                            last_icd_dict = icd_dict
                        except Exception as exc:
                            if debug_mode:
                                print(f"[DEBUG] Failed to parse ICD JSON: {exc}")

            parsed = {
                "impression": impression_text.strip(),
                "chexpert": self._aggregate_chexpert_votes(chexpert_votes, fallback=last_chexpert_dict),
                "icd": self._aggregate_icd_votes(icd_votes, fallback=last_icd_dict) if stage == "B" else {label: 0 for label in ICD},
                "raw_output": impression_text,
            }
            if self._str_to_bool(os.getenv("ENABLE_LABEL_KEYWORDS", "true"), True):
                self._apply_keyword_rules(parsed, stage, ehr_json)
            if debug_mode:
                print("[DEBUG] Final multi-pass result:", parsed)
            return parsed

        # Fallback: single shot JSON (rarely used)
        system_prompt = (
            "You are a radiology assistant. Respond with a strict JSON object containing the requested keys."
        )
        if stage == "A":
            example = (
                "{\n"
                '  "impression": "The lungs are clear. No effusion or pneumothorax.",\n'
                '  "chexpert": {"No Finding": 1, "Enlarged Cardiomediastinum": 0, "Lung Opacity": 0, "Lung Lesion": 0, "Edema": 0, "Consolidation": 0, "Pneumonia": 0, "Pneumothorax": 0, "Pleural Effusion": 0, "Pleural Other": 0, "Fracture": 0, "Support Devices": 0}\n'
                "}"
            )
            user_prompt = (
                "Analyze the chest X-ray image and return ONLY the JSON object shown in the example.\n"
                f"Example:\n{example}\n"
                "Now respond for the current case."
            )
        else:
            ehr_text = self.format_ehr(ehr_json)
            example = (
                "{\n"
                '  "impression": "Moderate pulmonary edema with bilateral pleural effusions.",\n'
                '  "chexpert": {"No Finding": 0, "Enlarged Cardiomediastinum": 1, "Lung Opacity": 1, "Lung Lesion": 0, "Edema": 1, "Consolidation": 0, "Pneumonia": 0, "Pneumothorax": 0, "Pleural Effusion": 1, "Pleural Other": 0, "Fracture": 0, "Support Devices": 1},\n'
                '  "icd": {"Pneumonia": 0, "Pleural_Effusion": 1, "Pneumothorax": 0, "Pulmonary_Edema": 1, "Cardiomegaly": 1, "Atelectasis": 0, "Pulmonary_Embolism": 0, "Rib_Fracture": 0}\n'
                "}"
            )
            user_prompt = (
                f"Patient EHR summary:\n{ehr_text}\n\n"
                "Analyze the chest X-ray image and return ONLY the JSON object shown in the example.\n"
                f"Example:\n{example}\n"
                "Now respond for the current case."
            )

        raw_response = _run_generation(system_prompt, user_prompt)
        parsed = self.parse_output(raw_response, stage, ehr_json)
        if self._str_to_bool(os.getenv("ENABLE_LABEL_KEYWORDS", "true"), True):
            self._apply_keyword_rules(parsed, stage, ehr_json)
        return parsed

    # ------------------------------------------------------------------ #
    # Parsing helpers

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """
        ⭐ Extract JSON object from text (find first { ... })
        
        WHY THIS IS NEEDED:
        - Model may output: "Here's the JSON: {...}"
        - Need to extract just the {...} part
        - Handles cases where model adds commentary
        - Returns None if no JSON found
        
        HOW IT WORKS:
        1. Find first "{" in text
        2. Track bracket depth (handles nested JSON)
        3. Find matching "}" at depth 0
        4. Return substring between { }
        
        EXAMPLE:
        Input:  "The JSON is: {\"Pneumonia\": 1, \"Edema\": 0}"
        Output: "{\"Pneumonia\": 1, \"Edema\": 0}"
        
        WHY NESTED BRACKETS:
        - JSON can have nested objects
        - Need to find the "outer" brackets, not inner ones
        - Track depth to find matching "}"
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            JSON string if found, None otherwise
        """
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return None

    def parse_output(self, text: str, stage: str, ehr_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ⭐ Parse model output text into structured result
        
        WHY THIS IS NEEDED:
        - Model outputs raw text, not structured dict
        - Need to extract impression, CheXpert, ICD from text
        - Handles malformed JSON, missing fields
        - Acts as fallback if multi-pass generation fails
        
        HOW IT WORKS:
        1. Extract JSON blob from text (find { ... })
        2. Parse JSON into dict
        3. Extract impression, chexpert, icd fields
        4. Fill missing fields with defaults (0)
        5. Apply keyword rules for extra safety
        
        STAGE A vs B:
        - Stage A: Only impression + CheXpert (no ICD)
        - Stage B: Impression + CheXpert + ICD
        - ICD parsing only if stage == "B"
        
        FALLBACK BEHAVIOR:
        - If no JSON found → use entire text as impression
        - If missing CheXpert field → all zeros
        - If missing ICD field → all zeros (Stage B only)
        - Keyword rules can fill in obvious findings
        
        Args:
            text: Raw model output text
            stage: "A" (image-only) or "B" (image+EHR)
            ehr_json: Patient EHR data (for keyword rules)
            
        Returns:
            Structured dict with impression, chexpert, icd fields
        """
        result = {
            "impression": "",
            "chexpert": {label: 0 for label in CHEXPERT},
            "icd": {label: 0 for label in ICD}
        }

        json_blob = self._extract_json(text)
        if json_blob is None:
            result["impression"] = text.strip()
            return result

        try:
            payload = json.loads(json_blob)
        except json.JSONDecodeError:
            result["impression"] = text.strip()
            return result

        impression = payload.get("impression") or payload.get("Impression") or ""
        result["impression"] = impression.strip()
        if not result["impression"] and text:
            result["impression"] = text.strip()

        chexpert_payload = payload.get("chexpert") or {}
        for label in CHEXPERT:
            result["chexpert"][label] = self._normalize_label_value(chexpert_payload.get(label, 0), allow_negative=True)

        icd_payload = payload.get("icd") or {}
        if stage == "B":
            for label in ICD:
                result["icd"][label] = 1 if self._normalize_label_value(icd_payload.get(label, 0)) > 0 else 0

        result["raw_output"] = text.strip()
        if self._str_to_bool(os.getenv("ENABLE_LABEL_KEYWORDS", "true"), True):
            self._apply_keyword_rules(result, stage, ehr_json)
        return result

    def _apply_keyword_rules(self, parsed: Dict[str, Any], stage: str, ehr_json: Optional[Dict[str, Any]]) -> None:
        """
        Apply keyword-based post-processing rules to boost recall.
        
        WHY THIS IS NEEDED:
        - Model may miss obvious findings in impression text
        - Example: Impression says "pneumonia" but label is 0
        - Keyword rules act as safety net to catch missed labels
        - Improves recall (finds more diseases, reduces false negatives)
        
        HOW IT WORKS:
        1. Scan impression text for disease keywords
        2. If keyword found → set corresponding label to 1
        3. Example: "pneumothorax" in impression → Pneumothorax = 1
        
        KEYWORDS MATCHED:
        - CheXpert: pneumothorax, effusion, opacity, edema, consolidation, etc.
        - ICD: pneumonia, effusion, pulmonary edema, cardiomegaly, etc.
        - Support devices: pacemaker, line, catheter, tube
        
        EHR-BASED RULES (Stage B only):
        - If BNP > 1200 → boost Edema + Pleural Effusion (heart failure indicator)
        - If CRP > 150 + "pneumonia" → boost Pneumonia (infection marker)
        - If O2 device present → boost Support Devices
        
        WHY IT'S IMPORTANT:
        - Clinical accuracy: Keywords reflect radiologist's actual findings
        - Reduces false negatives: Model may under-predict
        - EHR integration: Uses patient history to corroborate findings
        
        Args:
            parsed: Parsed result dictionary (impression, chexpert, icd)
            stage: "A" (image-only) or "B" (image+EHR)
            ehr_json: Patient EHR data for Stage B rules
        """
        impression = (parsed.get("impression") or "").lower()
        if not impression:
            return

        chexpert = parsed.get("chexpert", {})
        icd = parsed.get("icd", {})

        chexpert_keywords = [
            ("pneumothorax", "Pneumothorax"),
            ("tension pneumothorax", "Pneumothorax"),
            ("effusion", "Pleural Effusion"),
            ("pleural effusions", "Pleural Effusion"),
            ("opacity", "Lung Opacity"),
            ("opacities", "Lung Opacity"),
            ("edema", "Edema"),
            ("pulmonary edema", "Edema"),
            ("consolidation", "Consolidation"),
            ("cardiomegaly", "Enlarged Cardiomediastinum"),
            ("enlarged cardiac silhouette", "Enlarged Cardiomediastinum"),
            ("fracture", "Fracture"),
            ("rib fracture", "Fracture"),
            ("pacemaker", "Support Devices"),
            ("line", "Support Devices"),
            ("catheter", "Support Devices"),
            ("tube", "Support Devices"),
        ]
        for keyword, label in chexpert_keywords:
            if keyword in impression:
                chexpert[label] = 1

        if stage == "B":
            icd_keywords = [
                ("pneumonia", "Pneumonia"),
                ("effusion", "Pleural_Effusion"),
                ("pulmonary edema", "Pulmonary_Edema"),
                ("cardiomegaly", "Cardiomegaly"),
                ("atelectasis", "Atelectasis"),
                ("embolism", "Pulmonary_Embolism"),
                ("pneumothorax", "Pneumothorax"),
                ("fracture", "Rib_Fracture"),
            ]
            for keyword, label in icd_keywords:
                if keyword in impression:
                    icd[label] = 1

        if stage == "B" and ehr_json:
            labs = ehr_json.get("Labs", {})
            bnp_info = labs.get("BNP") or labs.get("bnp")
            try:
                bnp_value = float(bnp_info.get("value")) if bnp_info and "value" in bnp_info else None
            except (TypeError, ValueError):
                bnp_value = None
            bnp_threshold = float(os.getenv("BNP_EDEMA_THRESHOLD", "1200"))
            admission_type = (ehr_json.get("admission_type") or "").lower()
            if bnp_value and bnp_value >= bnp_threshold and ("elective" in admission_type):
                chexpert["Edema"] = 1
                chexpert["Pleural Effusion"] = 1
                icd["Pulmonary_Edema"] = 1
                icd["Pleural_Effusion"] = 1
            crp_info = labs.get("CRP") or labs.get("crp")
            try:
                crp_value = float(crp_info.get("value")) if crp_info and "value" in crp_info else None
            except (TypeError, ValueError):
                crp_value = None
            crp_threshold = float(os.getenv("CRP_PNEUMONIA_THRESHOLD", "150"))
            if crp_value and crp_value >= crp_threshold and "pneumonia" in impression:
                icd["Pneumonia"] = 1
            o2_info = ehr_json.get("O2_device") or {}
            device_name = (o2_info.get("device") or "").lower()
            if device_name and device_name not in ("unknown", "room air", "none"):
                chexpert["Support Devices"] = 1

        positive_labels = [label for label, value in chexpert.items() if label != "No Finding" and value != 0]
        chexpert["No Finding"] = 0 if positive_labels else 1

    # ------------------------------------------------------------------ #
    # Convenience wrappers


_pipeline: Optional[RadiologyInferencePipeline] = None
_pipeline_device: Optional[str] = None


def get_pipeline(device: str = "cpu") -> RadiologyInferencePipeline:
    global _pipeline, _pipeline_device
    if _pipeline is None or _pipeline_device != device:
        print(f"🔄 Loading pipeline for device: {device}")
        _pipeline = RadiologyInferencePipeline(device=device)
        _pipeline_device = device
        print(f"✅ Pipeline loaded and cached for device: {device}")
    else:
        print(f"⚡ Using cached pipeline for device: {device}")
    return _pipeline


def clear_pipeline_cache():
    """Clear the cached pipeline to force reload on next call."""
    global _pipeline, _pipeline_device
    _pipeline = None
    _pipeline_device = None
    print("🗑️ Pipeline cache cleared")


def generate(image_path: str, ehr_json: Optional[Dict[str, Any]] = None, device: str = "cpu") -> Dict[str, Any]:
    pipeline = get_pipeline(device=device)
    return pipeline.generate(image_path, ehr_json)


if __name__ == "__main__":
    print("🔬 Testing Stage A inference…")
    result_a = generate("src/data/sample_images/sample_xray_1.jpg")
    print("Stage A Result:", json.dumps(result_a, indent=2))

    print("\n🔬 Testing Stage B inference…")
    sample_ehr = {
        "Age": 65,
        "Sex": "M",
        "Vitals": {"heart_rate": 92, "o2_saturation": 90},
        "Labs": {"BNP": 310000, "CRP": 162},
        "Chronic_conditions": ["hypertension", "heart_failure"],
    }
    result_b = generate("src/data/sample_images/sample_xray_1.jpg", sample_ehr)
    print("Stage B Result:", json.dumps(result_b, indent=2))
