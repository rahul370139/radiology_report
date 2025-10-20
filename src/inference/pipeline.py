"""
Minimal I/O pipeline for radiology report generation
Supports both Stage A (image-only) and Stage B (image+EHR) inference
"""

import torch
import json
import os
import re
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates

# Import our model loader
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from load_finetuned_model import load_finetuned_llava

# Label definitions matching the dataset (12 labels)
CHEXPERT = [
    "No Finding", "Enlarged Cardiomediastinum", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
    "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices"
]

ICD = [
    "Pneumonia", "Pleural_Effusion", "Pneumothorax", "Pulmonary_Edema",
    "Cardiomegaly", "Atelectasis", "Pulmonary_Embolism", "Rib_Fracture"
]

class RadiologyInferencePipeline:
    """Minimal I/O pipeline for radiology report generation"""
    
    def __init__(self, device="cpu"):
        """Initialize the pipeline with model loading"""
        self.device = device
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.conv_mode = None
        self.target_image_size = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model"""
        print("🤖 Loading fine-tuned model...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_finetuned_llava(device=self.device)
        name_hint = str(getattr(self.model.config, "_name_or_path", "")).lower()
        model_type = str(getattr(self.model.config, "model_type", "")).lower()
        combo_name = f"{name_hint} {model_type}"
        if "mistral" in combo_name:
            self.conv_mode = "mistral_instruct"
        elif "llama" in combo_name:
            self.conv_mode = "llava_llama_2"
        elif "llava" in combo_name or "v1" in combo_name:
            self.conv_mode = "llava_v1"
        else:
            self.conv_mode = "llava_v0"
        if self.conv_mode not in conv_templates:
            self.conv_mode = "llava_v1"
        vision_module = getattr(self.model, "get_vision_tower", None)
        vision_tower = vision_module() if callable(vision_module) else None
        tower_core = getattr(vision_tower, "vision_tower", vision_tower)
        tower_config = getattr(tower_core, "config", None)
        image_size = getattr(tower_config, "image_size", None)
        patch_size = getattr(tower_config, "patch_size", None)
        processor_size = getattr(self.image_processor, "size", None)
        processor_crop = getattr(self.image_processor, "crop_size", None)
        self.target_image_size = image_size
        if self.target_image_size:
            try:
                if hasattr(self.image_processor, "size"):
                    if isinstance(self.image_processor.size, dict):
                        for key in ("height", "width", "shortest_edge", "longest_edge"):
                            if key in self.image_processor.size:
                                self.image_processor.size[key] = self.target_image_size
                    else:
                        self.image_processor.size = self.target_image_size
                if hasattr(self.image_processor, "crop_size"):
                    if isinstance(self.image_processor.crop_size, dict):
                        for key in ("height", "width"):
                            if key in self.image_processor.crop_size:
                                self.image_processor.crop_size[key] = self.target_image_size
                    else:
                        self.image_processor.crop_size = self.target_image_size
            except Exception as img_cfg_err:
                print(f"⚠️ Unable to adjust image processor size: {img_cfg_err}")
        print(f"🖼️ Vision tower image_size={image_size}, patch_size={patch_size}, processor size={getattr(self.image_processor, 'size', None)}, crop_size={getattr(self.image_processor, 'crop_size', None)}")
        print(f"💬 Using conversation template: {self.conv_mode}")
        print("✅ Model loaded successfully")
    
    def format_ehr(self, ehr_dict: Dict[str, Any]) -> str:
        """Format EHR data for prompt"""
        keep = ["Age", "Sex", "Vitals", "Labs", "O2_device", "Chronic_conditions"]
        filtered_ehr = {k: ehr_dict.get(k, {}) for k in keep if k in ehr_dict}
        return json.dumps(filtered_ehr, indent=2)
    
    def _gen_params(self):
        # Allow tuning via env vars
        temperature = float(os.getenv("GEN_TEMPERATURE", "0.1"))
        max_new_tokens = int(os.getenv("GEN_MAX_NEW_TOKENS", "384"))
        top_p = float(os.getenv("GEN_TOP_P", "0.95"))
        do_sample = os.getenv("GEN_DO_SAMPLE", "true").lower() == "true"
        return temperature, max_new_tokens, top_p, do_sample

    def generate(self, image_path: str, ehr_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate radiology report from image and optional EHR data
        """
        stage = "B" if ehr_json else "A"
        img = Image.open(image_path).convert("RGB")
        if self.target_image_size and img.size != (self.target_image_size, self.target_image_size):
            img = img.resize((self.target_image_size, self.target_image_size), resample=Image.BICUBIC)
        print(f"✅ Loaded image: {img.size}")
        
        if self.image_processor is None:
            print("❌ Image processor is None, cannot process image")
            return {"impression": "Error: Image processor not available", "chexpert": {}, "icd": {}}
            
        dtype = torch.float32 if self.model.device.type == 'cpu' else torch.float16
        img_tensor = process_images([img], self.image_processor, self.model.config).to(self.model.device, dtype=dtype)
        print(f"✅ Processed image tensor shape: {img_tensor.shape}")
        
        chexpert_json_template = (
            '{"No Finding": 0, "Enlarged Cardiomediastinum": 0, "Lung Opacity": 0, '
            '"Lung Lesion": 0, "Edema": 0, "Consolidation": 0, "Pneumonia": 0, '
            '"Pneumothorax": 0, "Pleural Effusion": 0, "Pleural Other": 0, '
            '"Fracture": 0, "Support Devices": 0}'
        )
        icd_json_template = (
            '{"Pneumonia": 0, "Pleural_Effusion": 0, "Pneumothorax": 0, '
            '"Pulmonary_Edema": 0, "Cardiomegaly": 0, "Atelectasis": 0, '
            '"Pulmonary_Embolism": 0, "Rib_Fracture": 0}'
        )

        if stage == "A":
            question = (
                "You are a radiology assistant. Given a chest X-ray image, first write a concise "
                "Impression (1-3 sentences). Then output CheXpert labels as a JSON object with keys "
                "exactly matching the following 12 labels and integer values in {-1,0,1}.\n"
                "1) Impression:\n"
                f"2) CheXpert: {chexpert_json_template}"
            )
        else:
            ehr_text = self.format_ehr(ehr_json)
            question = (
                "You are a radiology assistant. Consider the patient's EHR JSON:\n"
                f"{ehr_text}\n"
                "Given the chest X-ray image, write a concise Impression (1-3 sentences), then output "
                "CheXpert labels as JSON (same 12 keys, integer values in {-1,0,1}), and ICD as JSON "
                "with keys: Pneumonia, Pleural_Effusion, Pneumothorax, Pulmonary_Edema, Cardiomegaly, "
                "Atelectasis, Pulmonary_Embolism, Rib_Fracture with values in {0,1}.\n"
                "1) Impression:\n"
                f"2) CheXpert: {chexpert_json_template}\n"
                f"3) ICD: {icd_json_template}"
            )

        conv_key = self.conv_mode if self.conv_mode in conv_templates else "llava_v1"
        conv = conv_templates[conv_key].copy()
        user_role, assistant_role = conv.roles
        if getattr(self.model.config, "mm_use_im_start_end", False):
            user_message = f"{question}\n{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
        else:
            user_message = f"{question}\n{DEFAULT_IMAGE_TOKEN}"
        conv.append_message(user_role, user_message)
        conv.append_message(assistant_role, None)
        prompt_text = conv.get_prompt()
        print(f"📝 Prompt preview: {prompt_text[:200]}...")

        # Tokenize the prompt with image token
        input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        if input_ids is None:
            print("❌ Failed to tokenize prompt")
            return {"impression": "Error: Failed to tokenize prompt", "chexpert": {}, "icd": {}}
        
        # Ensure input_ids has the correct shape for LLaVA model
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension if missing
        
        input_ids = input_ids.to(self.model.device)
        print(f"✅ Tokenized input_ids shape: {input_ids.shape}")
        
        # Ensure the tensor is contiguous and has the right dtype
        input_ids = input_ids.contiguous()
        print(f"✅ Final input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        
        temperature, max_new_tokens, top_p, do_sample = self._gen_params()
        with torch.inference_mode():
            # Debug: Check input_ids before generation
            print(f"🔍 Debug: input_ids type: {type(input_ids)}, shape: {input_ids.shape if input_ids is not None else 'None'}")
            print(f"🔍 Debug: img_tensor type: {type(img_tensor)}, shape: {img_tensor.shape if img_tensor is not None else 'None'}")
            try:
                encoded = self.model.encode_images(img_tensor)
                print(f"🔍 Encoded image features shape: {encoded.shape}")
                if isinstance(encoded, torch.Tensor):
                    print(f"🔍 Encoded image features sample shape: {encoded[0].shape}")
                token_embeds = self.model.get_model().embed_tokens(input_ids[0])
                print(f"🔍 Token embeddings shape: {token_embeds.shape}")
            except Exception as enc_err:
                print(f"❌ encode/token embedding debug failed: {enc_err}")
            try:
                debug_prepared = self.model.prepare_inputs_labels_for_multimodal(
                    input_ids.clone(),
                    None,
                    None,
                    None,
                    None,
                    img_tensor,
                    image_sizes=None
                )
                print("🔍 prepare_inputs_labels_for_multimodal succeeded")
            except Exception as prep_err:
                print(f"❌ prepare_inputs_labels_for_multimodal failed: {prep_err}")
                return {"impression": "Error: Model preparation failed", "chexpert": {}, "icd": {}}
            
            # For LLaVA models, we need to use the specific generation approach
            # Try using the model's forward method first to prepare inputs
            try:
                print("🔄 Attempting LLaVA-specific generation...")
                
                # Prepare inputs for LLaVA model
                from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
                
                # Use the model's specific generation method
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=img_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
                print("✅ LLaVA-specific generate call successful")
            except Exception as e:
                print(f"⚠️ LLaVA-specific generate failed: {e}")
                print("🔄 Trying standard generate call...")
                try:
                    output_ids = self.model.generate(
                        inputs=input_ids,
                        images=img_tensor,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    print("✅ Standard generate call successful")
                except Exception as e2:
                    print(f"⚠️ Standard generate failed: {e2}")
                    print("🔄 Trying minimal generate call...")
                    try:
                        output_ids = self.model.generate(
                            inputs=input_ids,
                            max_new_tokens=max_new_tokens
                        )
                        print("✅ Minimal generate call successful")
                    except Exception as e3:
                        print(f"❌ All generate attempts failed: {e3}")
                        return {"impression": "Error: Model generation failed", "chexpert": {}, "icd": {}}
        
        output_text = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        if assistant_role and assistant_role in output_text:
            output_text = output_text.split(assistant_role, maxsplit=1)[-1].strip()
        output_text = output_text.replace("</s>", "").strip()
        return self.parse_output(output_text)
    
    def parse_output(self, text: str) -> Dict[str, Any]:
        """Parse generated text into structured output"""
        result = {
            "impression": "",
            "chexpert": {label: 0 for label in CHEXPERT},
            "icd": {label: 0 for label in ICD}
        }
        
        # Extract impression (case-insensitive, tolerate variations)
        impression_match = re.search(r"1\)\s*Impression:\s*(.*?)(?=\n2\)|$)", text, re.IGNORECASE | re.DOTALL)
        if impression_match:
            result["impression"] = impression_match.group(1).strip()
        
        # Extract CheXpert JSON
        chexpert_match = re.search(r"2\)\s*CheXpert:\s*(\{.*?\})", text, re.IGNORECASE | re.DOTALL)
        if chexpert_match:
            try:
                chexpert_dict = json.loads(chexpert_match.group(1))
                for label in CHEXPERT:
                    if label in chexpert_dict:
                        result["chexpert"][label] = chexpert_dict[label]
            except json.JSONDecodeError:
                print("⚠️ Failed to parse CheXpert JSON")
        
        # Extract ICD JSON
        icd_match = re.search(r"3\)\s*ICD:\s*(\{.*?\})", text, re.IGNORECASE | re.DOTALL)
        if icd_match:
            try:
                icd_dict = json.loads(icd_match.group(1))
                for label in ICD:
                    if label in icd_dict:
                        result["icd"][label] = icd_dict[label]
            except json.JSONDecodeError:
                print("⚠️ Failed to parse ICD JSON")
        
        return result

# Global pipeline instance for easy import
_pipeline = None

def get_pipeline(device="cpu"):
    """Get or create global pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RadiologyInferencePipeline(device=device)
    return _pipeline


def generate(image_path: str, ehr_json: Optional[Dict[str, Any]] = None, device="cpu") -> Dict[str, Any]:
    """
    Convenience function for single inference
    """
    pipeline = get_pipeline(device=device)
    return pipeline.generate(image_path, ehr_json)

if __name__ == "__main__":
    print("🔬 Testing Stage A inference...")
    result_a = generate("src/data/sample_images/sample_xray_1.jpg")
    print("Stage A Result:")
    print(f"Impression: {result_a['impression']}")
    print(f"CheXpert: {result_a['chexpert']}")
    print("\n🔬 Testing Stage B inference...")
    sample_ehr = {
        "Age": 65,
        "Sex": "M",
        "Vitals": {"heart_rate": 85, "o2_saturation": 95},
        "Labs": {"Creatinine": 1.2, "BNP": 500},
        "Chronic_conditions": ["hypertension", "diabetes"]
    }
    result_b = generate("src/data/sample_images/sample_xray_1.jpg", sample_ehr)
    print("Stage B Result:")
    print(f"Impression: {result_b['impression']}")
    print(f"CheXpert: {result_b['chexpert']}")
    print(f"ICD: {result_b['icd']}")
