#!/usr/bin/env python3
"""
Single Sample Evaluation Script for Radiology Report Model
Loads trained model + LoRA adapter and runs inference on single image
"""

import json
import argparse
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import re
import os

# Import LLaVA components
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

class RadiologyModelEvaluator:
    """Single sample evaluator for radiology report generation"""
    
    def __init__(self, model_path: str, lora_path: str):
        """
        Initialize evaluator with base model and LoRA adapter
        
        Args:
            model_path: Path to base LLaVA-Med model
            lora_path: Path to trained LoRA adapter
        """
        self.model_path = model_path
        self.lora_path = lora_path
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        
        # CheXpert labels
        self.chexpert_labels = [
            "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture",
            "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
        ]
        
        # ICD classes
        self.icd_classes = [
            "Pneumonia", "Pleural_Effusion", "Pneumothorax", "Pulmonary_Edema",
            "Cardiomegaly", "Atelectasis", "Pulmonary_Embolism", "Rib_Fracture"
        ]
    
    def load_model(self):
        """Load base model and LoRA adapter"""
        print(f"🤖 Loading base model from {self.model_path}...")
        
        # Force CPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.cuda.is_available = lambda: False
        
        # Disable torch init for faster loading
        disable_torch_init()
        
        # Load model and tokenizer
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device_map="cpu"
        )
        
        print(f"✅ Base model loaded successfully")
        
        # Ensure model is on CPU
        self.model = self.model.to(torch.device("cpu"))
        self.model = self.model.float()
        
        # Load LoRA adapter
        print(f"🔧 Loading LoRA adapter from {self.lora_path}...")
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            print(f"✅ LoRA adapter loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load LoRA adapter: {e}")
            raise
    
    def format_ehr_data(self, patient_data: Dict[str, Any]) -> str:
        """Format patient data into compact EHR JSON"""
        ehr = {}
        
        # Basic demographics
        if 'Age' in patient_data:
            ehr['Age'] = patient_data['Age']
        if 'Sex' in patient_data:
            ehr['Sex'] = patient_data['Sex']
        
        # Vitals with time deltas
        if 'Vitals' in patient_data and patient_data['Vitals']:
            vitals = {}
            for vital_name, vital_info in patient_data['Vitals'].items():
                if isinstance(vital_info, dict) and 'value' in vital_info:
                    # Map vital names to standard abbreviations
                    vital_key = self._map_vital_name(vital_name)
                    if vital_key:
                        vitals[vital_key] = vital_info['value']
                        # Add time delta if available
                        if 'delta_hours_from_cxr' in vital_info:
                            vitals[f"{vital_key}_delta_hours"] = vital_info['delta_hours_from_cxr']
            if vitals:
                ehr['Vitals'] = vitals
        
        # Labs with time deltas
        if 'Labs' in patient_data and patient_data['Labs']:
            labs = {}
            for lab_name, lab_info in patient_data['Labs'].items():
                if isinstance(lab_info, dict) and 'value' in lab_info and 'unit' in lab_info:
                    # Use canonical name and include unit
                    canonical_name = lab_info.get('canonical_name', lab_name)
                    labs[canonical_name] = {
                        'value': lab_info['value'],
                        'unit': lab_info['unit']
                    }
                    # Add time delta if available
                    if 'delta_hours_from_cxr' in lab_info:
                        labs[canonical_name]['delta_hours'] = lab_info['delta_hours_from_cxr']
            if labs:
                ehr['Labs'] = labs
        
        # Chronic conditions
        if 'Chronic_conditions' in patient_data and patient_data['Chronic_conditions']:
            ehr['Chronic'] = patient_data['Chronic_conditions']
        
        return json.dumps(ehr, separators=(',', ':'))
    
    def _map_vital_name(self, vital_name: str) -> Optional[str]:
        """Map vital name to standard abbreviation"""
        mapping = {
            'heart_rate': 'HR',
            'systolic_bp': 'SBP',
            'diastolic_bp': 'DBP',
            'mean_bp': 'MBP',
            'blood_pressure': 'BP',
            'o2_saturation': 'SpO2',
            'respiratory_rate': 'RR',
            'temperature_f': 'TempF',
            'temperature_c': 'TempC',
            'weight': 'Weight',
            'height': 'Height',
            'bmi': 'BMI'
        }
        return mapping.get(vital_name.lower())
    
    def get_stage_a_prompt(self) -> str:
        """Get Stage A prompt template (image-only)"""
        return """You are a radiology assistant. Analyze the chest X-ray and produce:
1) IMPRESSION: A single concise paragraph.
2) CheXpert: A strict JSON with keys [Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices].

<image>"""
    
    def get_stage_b_prompt(self, ehr_data: str) -> str:
        """Get Stage B prompt template (image + EHR)"""
        return f"""EHR:
{ehr_data}

Now analyze the chest X-ray and produce:

1) IMPRESSION: A single concise paragraph.
2) CheXpert: A strict JSON with keys [Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices].
3) ICD: A strict JSON with keys [Pneumonia, Pleural_Effusion, Pneumothorax, Pulmonary_Edema, Cardiomegaly, Atelectasis, Pulmonary_Embolism, Rib_Fracture] and 0/1 values.

<image>"""
    
    def generate_response(self, image_path: str, ehr_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate radiology report response
        
        Args:
            image_path: Path to chest X-ray image
            ehr_data: Optional patient EHR data for Stage B
            
        Returns:
            Generated response text
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config)
        
        # Prepare prompt
        if ehr_data is None:
            # Stage A: Image only
            prompt = self.get_stage_a_prompt()
        else:
            # Stage B: Image + EHR
            ehr_json = self.format_ehr_data(ehr_data)
            prompt = self.get_stage_b_prompt(ehr_json)
        
        # Tokenize
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_tokens = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_tokens, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                top_p=0.7,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True
            )
        
        # Decode response
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Extract only the assistant's response
        if conv.roles[1] in response:
            response = response.split(conv.roles[1])[-1].strip()
        
        return response
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse generated response to extract structured outputs
        
        Args:
            response: Generated response text
            
        Returns:
            Dictionary with parsed impression, chexpert, and icd
        """
        result = {
            'impression': '',
            'chexpert': {},
            'icd': {}
        }
        
        # Extract impression
        impression_match = re.search(r'1\) IMPRESSION:\s*(.+?)(?=\n\n|\n2\)|$)', response, re.DOTALL)
        if impression_match:
            result['impression'] = impression_match.group(1).strip()
        
        # Extract CheXpert JSON
        chexpert_match = re.search(r'2\) CheXpert:\s*(\{.*?\})', response, re.DOTALL)
        if chexpert_match:
            try:
                chexpert_json = json.loads(chexpert_match.group(1))
                for label in self.chexpert_labels:
                    result['chexpert'][label] = chexpert_json.get(label, 0)
            except json.JSONDecodeError:
                print("⚠️ Failed to parse CheXpert JSON")
        
        # Extract ICD JSON (Stage B only)
        icd_match = re.search(r'3\) ICD:\s*(\{.*?\})', response, re.DOTALL)
        if icd_match:
            try:
                icd_json = json.loads(icd_match.group(1))
                for label in self.icd_classes:
                    result['icd'][label] = icd_json.get(label, 0)
            except json.JSONDecodeError:
                print("⚠️ Failed to parse ICD JSON")
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Single Sample Evaluation for Radiology Report Model')
    parser.add_argument('--image', required=True, help='Path to chest X-ray image')
    parser.add_argument('--ehr_json', help='Path to EHR JSON file (optional, for Stage B)')
    parser.add_argument('--model_path', default='microsoft/llava-med-v1.5-mistral-7b', help='Base model path')
    parser.add_argument('--lora_path', default='checkpoints', help='LoRA adapter path')
    parser.add_argument('--output', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    print("🔬 SINGLE SAMPLE EVALUATION")
    print("=" * 40)
    
    # Initialize evaluator
    evaluator = RadiologyModelEvaluator(args.model_path, args.lora_path)
    
    # Load model
    evaluator.load_model()
    
    # Load EHR data if provided
    ehr_data = None
    if args.ehr_json:
        with open(args.ehr_json, 'r') as f:
            ehr_data = json.load(f)
        print(f"📋 Loaded EHR data from {args.ehr_json}")
    
    # Generate response
    print(f"🖼️ Processing image: {args.image}")
    response = evaluator.generate_response(args.image, ehr_data)
    
    # Parse response
    parsed = evaluator.parse_response(response)
    
    # Display results
    print("\n📊 RESULTS:")
    print("=" * 40)
    print(f"IMPRESSION:\n{parsed['impression']}\n")
    
    print("CHEXPERT LABELS:")
    for label, value in parsed['chexpert'].items():
        print(f"  {label}: {value}")
    
    if parsed['icd']:
        print("\nICD PREDICTIONS:")
        for label, value in parsed['icd'].items():
            print(f"  {label}: {value}")
    
    # Save output if requested
    if args.output:
        output_data = {
            'image_path': args.image,
            'ehr_data': ehr_data,
            'response': response,
            'parsed': parsed
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

if __name__ == "__main__":
    main()
