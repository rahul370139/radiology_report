#!/usr/bin/env python3.10
"""
Dataset class for MIMIC-CXR curriculum training.

Handles loading images, prompts, and targets for both Stage A and Stage B training.
"""

import copy
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

CHEXPERT_ORDER = [
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

ICD_ORDER = [
    "Pneumonia",
    "Pleural_Effusion",
    "Pneumothorax",
    "Pulmonary_Edema",
    "Cardiomegaly",
    "Atelectasis",
    "Pulmonary_Embolism",
    "Rib_Fracture",
]

ICD_CODE_MAP = {
    "J18": "Pneumonia",
    "J12": "Pneumonia",
    "J13": "Pneumonia",
    "J14": "Pneumonia",
    "J15": "Pneumonia",
    "J16": "Pneumonia",
    "J17": "Pneumonia",
    "J94": "Pleural_Effusion",
    "J93": "Pneumothorax",
    "J81": "Pulmonary_Edema",
    "I51.7": "Cardiomegaly",
    "I10": "Cardiomegaly",
    "J98.1": "Atelectasis",
    "I26": "Pulmonary_Embolism",
    "S22": "Rib_Fracture",
}

def _is_low_ehr(sample: Dict[str, Any]) -> bool:
    patient = sample.get('patient_data') or {}
    vitals = patient.get('Vitals') or {}
    labs = patient.get('Labs') or {}
    return len(vitals) <= 2 and len(labs) <= 2

def convert_stage_b_to_stage_a(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Create a Stage-A style sample from Stage-B by stripping EHR context."""
    clone = copy.deepcopy(sample)
    clone['stage'] = 'A'
    patient = clone.get('patient_data') or {}
    minimal = {}
    if 'Age' in patient:
        minimal['Age'] = patient['Age']
    if 'Sex' in patient:
        minimal['Sex'] = patient['Sex']
    clone['patient_data'] = minimal if minimal else None
    clone['ehr_context'] = ""
    return clone

def build_stage_mix_samples(
    stage_a_samples: List[Dict[str, Any]],
    stage_b_samples: List[Dict[str, Any]],
    seed: int,
    stage_b_fraction: float = 0.65,
) -> List[Dict[str, Any]]:
    """Construct a mixed Stage-A/Stage-B epoch sample list."""
    rng = random.Random(seed)
    stage_a_block = [copy.deepcopy(s) for s in stage_a_samples]

    low_ehr_candidates = [s for s in stage_b_samples if _is_low_ehr(s)]
    synthetic_source = low_ehr_candidates if len(low_ehr_candidates) >= len(stage_a_samples) else stage_b_samples
    synthetic_count = min(len(synthetic_source), len(stage_a_samples))
    synthetic_stage_a = [
        convert_stage_b_to_stage_a(copy.deepcopy(s))
        for s in rng.sample(synthetic_source, synthetic_count)
    ]

    stage_b_count = max(1, int(len(stage_b_samples) * stage_b_fraction))
    sampled_stage_b = [
        copy.deepcopy(s) for s in rng.sample(stage_b_samples, stage_b_count)
    ]

    combined = stage_a_block + synthetic_stage_a + sampled_stage_b
    rng.shuffle(combined)
    return combined

class CurriculumDataset(Dataset):
    """
    Dataset for curriculum learning with chest X-ray images.
    
    Supports both Stage A (image-only) and Stage B (image+EHR) training.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
        image_root: str = ".",
        processor: Optional[Any] = None,
        max_length: int = 512,
        stage: str = "both",  # "stage_a", "stage_b", or "both"
        max_label_tokens: int = 128,
    ):
        """
        Initialize curriculum dataset.
        
        Args:
            data_path: Path to curriculum JSONL file
            image_root: Root directory for image paths
            processor: HuggingFace processor for model
            max_length: Maximum sequence length
            stage: Which stage to load ("stage_a", "stage_b", or "both")
        """
        if data_path is None and samples is None:
            raise ValueError("Either data_path or samples must be provided to CurriculumDataset")
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_length = max_length
        self.stage = stage
        self.max_label_tokens = max_label_tokens
        
        # Load curriculum data
        if samples is not None:
            self.samples = [
                sample for sample in samples
                if self.stage == "both"
                or (self.stage == "stage_a" and sample.get('stage') == 'A')
                or (self.stage == "stage_b" and sample.get('stage') == 'B')
            ]
        else:
            self.samples = self._load_curriculum(data_path)
        
        logger.info(f"Loaded {len(self.samples)} samples"
                    f"{'' if data_path is None else f' from {data_path}'}")
        logger.info(f"Stage filter: {stage}")
        
        # Count samples by mode
        self._log_statistics()
    
    def _load_curriculum(self, data_path: str) -> List[Dict[str, Any]]:
        """Load and filter curriculum samples."""
        samples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    
                    # Filter by stage if specified
                    if self.stage == "stage_a" and sample['stage'] != 'A':
                        continue
                    elif self.stage == "stage_b" and sample['stage'] != 'B':
                        continue
                    
                    samples.append(sample)
        
        return samples
    
    def _log_statistics(self):
        """Log dataset statistics."""
        stage_a_count = sum(1 for s in self.samples if s['stage'] == 'A')
        stage_b_count = sum(1 for s in self.samples if s['stage'] == 'B')
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Stage A (image-only): {stage_a_count}")
        logger.info(f"  Stage B (image+EHR): {stage_b_count}")
        logger.info(f"  Total: {len(self.samples)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - image: PIL Image
                - prompt: Text prompt
                - target: Target text
                - mode: "image_only" or "image_ehr"
                - stage: "A" or "B"
                - study_id: Study identifier
        """
        sample = self.samples[idx]
        
        # Load image (keep as PIL Image for collate_fn processing)
        image_path = self.image_root / sample['image_path']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (336, 336), color='black')
        
        # Build prompt and target using structured templates
        ehr_context = ""
        if sample['stage'] == 'A':
            # Stage A: Image-only with structured prompt
            prompt = """You are a radiology assistant. Analyze the chest X-ray and produce:

TASK:
1) IMPRESSION: A single concise paragraph.
2) CheXpert: A strict JSON with keys [Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices].

<image>"""
            
            # Format target with CheXpert labels
            chexpert_labels = sample.get('chexpert_labels', {})
            chexpert_vec, chexpert_mask = self._build_chexpert_vector(chexpert_labels)
            icd_vec = [0.0 for _ in ICD_ORDER]
            icd_mask = [0.0 for _ in ICD_ORDER]
            target = self._format_stage_a_target(sample['impression'], chexpert_labels)
            mode = "image_only"
        else:
            # Stage B: Image + EHR with structured prompt
            patient_data = sample.get('patient_data', {})
            ehr_data = self._format_ehr_data(patient_data)
            ehr_context = ehr_data
            prompt = f"""EHR:
{ehr_data}

Now analyze the chest X-ray and produce:

TASK:
1) IMPRESSION: A single concise paragraph.
2) CheXpert: A strict JSON with keys [Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices].
3) ICD: A strict JSON with keys [Pneumonia, Pleural_Effusion, Pneumothorax, Pulmonary_Edema, Cardiomegaly, Atelectasis, Pulmonary_Embolism, Rib_Fracture] and 0/1 values.

<image>"""
            
            # Format target with CheXpert and ICD labels
            chexpert_labels = sample.get('chexpert_labels', {})
            icd_labels = self._normalize_icd_labels(sample.get('icd_labels'))
            chexpert_vec, chexpert_mask = self._build_chexpert_vector(chexpert_labels)
            icd_vec, icd_mask = self._build_icd_vector(icd_labels)
            target = self._format_stage_b_target(sample['impression'], chexpert_labels, icd_labels)
            mode = "image_ehr"
        
        # Extract study_id from image_path if available
        study_id = None
        if 'image_path' in sample:
            # Extract study_id from path like "files/p10/p10088966/s56754337/..."
            path_parts = sample['image_path'].split('/')
            if len(path_parts) >= 4 and path_parts[3].startswith('s'):
                study_id = path_parts[3]
        
        return {
            'image': image,  # Return PIL Image for collate_fn to process
            'prompt': prompt,
            'target': target,
            'mode': mode,
            'stage': sample['stage'],
            'study_id': study_id,
            'image_path': str(image_path),
            'ehr_context': ehr_context,
            'chexpert_target': chexpert_vec,
            'chexpert_mask': chexpert_mask,
            'icd_target': icd_vec,
            'icd_mask': icd_mask,
        }

    def _build_chexpert_vector(self, chexpert_labels: Dict[str, int]) -> Tuple[List[float], List[float]]:
        """Build dense target vector and supervision mask for CheXpert labels."""
        values: List[float] = []
        mask: List[float] = []
        for label in CHEXPERT_ORDER:
            raw = chexpert_labels.get(label, 0)
            if raw == -1:
                values.append(0.0)
                mask.append(0.0)
            else:
                values.append(1.0 if raw in (1, True) else 0.0)
                mask.append(1.0)
        return values, mask

    def _normalize_icd_labels(self, icd_labels: Any) -> Dict[str, int]:
        """Normalize ICD annotations into a dict keyed by ICD_ORDER with 0/1 values."""
        normalized = {label: 0 for label in ICD_ORDER}
        if not icd_labels:
            return normalized
        if isinstance(icd_labels, dict):
            for label in ICD_ORDER:
                value = icd_labels.get(label, 0)
                normalized[label] = 1 if value in (1, True) else 0
            return normalized
        if isinstance(icd_labels, list):
            for item in icd_labels:
                if not isinstance(item, dict):
                    continue
                code = str(item.get('code', '')).upper()
                for prefix, mapped_label in ICD_CODE_MAP.items():
                    if code.startswith(prefix):
                        normalized[mapped_label] = 1
                        break
        return normalized

    def _build_icd_vector(self, icd_labels: Dict[str, int]) -> Tuple[List[float], List[float]]:
        """Build dense target vector and supervision mask for ICD indicators."""
        values: List[float] = []
        mask: List[float] = []
        for label in ICD_ORDER:
            raw = icd_labels.get(label, 0)
            values.append(1.0 if raw in (1, True) else 0.0)
            mask.append(1.0)
        return values, mask
    
    def _format_ehr_data(self, patient_data: Dict[str, Any]) -> str:
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
    
    def _format_stage_a_target(self, impression: str, chexpert_labels: Dict[str, int]) -> str:
        """Format Stage A target response"""
        chexpert_json = {}
        for label in CHEXPERT_ORDER:
            chexpert_json[label] = chexpert_labels.get(label, 0)
        
        return f"""1) IMPRESSION: {impression}

2) CheXpert: {json.dumps(chexpert_json)}"""
    
    def _format_stage_b_target(self, impression: str, chexpert_labels: Dict[str, int], 
                              icd_labels: Dict[str, int]) -> str:
        """Format Stage B target response"""
        chexpert_json = {}
        for label in CHEXPERT_ORDER:
            chexpert_json[label] = chexpert_labels.get(label, 0)
        
        icd_json = {}
        for label in ICD_ORDER:
            icd_json[label] = icd_labels.get(label, 0)
        
        return f"""1) IMPRESSION: {impression}

2) CheXpert: {json.dumps(chexpert_json)}

3) ICD: {json.dumps(icd_json)}"""
    
    def create_stratified_sampler(self, rare_boost: float = 5.0) -> WeightedRandomSampler:
        """Create stratified sampler to boost rare positive labels"""
        
        # Rare labels that need boosting
        rare_chexpert = ['Pneumothorax', 'Fracture', 'Lung Lesion']
        rare_icd = ['Pulmonary_Embolism', 'Rib_Fracture', 'Pneumothorax']
        
        weights = []
        for i, sample in enumerate(self.samples):
            weight = 1.0
            
            # Boost rare CheXpert positives
            if 'chexpert_labels' in sample:
                for label in rare_chexpert:
                    if sample['chexpert_labels'].get(label, 0) == 1:
                        weight *= rare_boost
            
            # Boost rare ICD positives
            if 'icd_labels' in sample:
                for label in rare_icd:
                    if sample['icd_labels'].get(label, 0) == 1:
                        weight *= rare_boost
            
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.samples),
            replacement=True
        )
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function that builds multimodal prompts for HF LLaVA processors."""
        images = [item['image'] for item in batch]
        targets = [item['target'] for item in batch]
        chexpert_targets = [torch.tensor(item['chexpert_target'], dtype=torch.float32) for item in batch]
        chexpert_masks = [torch.tensor(item['chexpert_mask'], dtype=torch.float32) for item in batch]
        icd_targets = [torch.tensor(item['icd_target'], dtype=torch.float32) for item in batch]
        icd_masks = [torch.tensor(item['icd_mask'], dtype=torch.float32) for item in batch]

        if self.processor is not None:
            chat_texts: List[str] = []
            for item in batch:
                user_entries: List[Dict[str, Any]] = []
                if item['mode'] == 'image_ehr' and item.get('ehr_context'):
                    user_entries.append({"type": "text", "text": item['ehr_context'] + "\n"})
                user_entries.append({"type": "image"})
                user_entries.append({"type": "text", "text": "Provide:\nImpression:\nCheXpert:\nICD:"})
                messages = [
                    {"role": "user", "content": user_entries},
                    {"role": "assistant", "content": [{"type": "text", "text": item['target']}]} ,
                ]
                chat_texts.append(
                    self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=False,
                        tokenize=False,
                    )
                )

            processed = self.processor(
                text=chat_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )

            labels = torch.full_like(processed['input_ids'], -100)
            for i, chat_text in enumerate(chat_texts):
                assistant_start = chat_text.find("assistant")
                if assistant_start != -1:
                    tokens = self.processor.tokenizer.encode(chat_text, add_special_tokens=False)
                    assistant_tokens = self.processor.tokenizer.encode("assistant", add_special_tokens=False)
                    for j in range(len(tokens) - len(assistant_tokens) + 1):
                        if tokens[j:j+len(assistant_tokens)] == assistant_tokens:
                            labels[i, j+len(assistant_tokens):] = processed['input_ids'][i, j+len(assistant_tokens):]
                            break
            processed['labels'] = labels
            if 'pixel_values' in processed and 'images' not in processed:
                processed['images'] = processed['pixel_values']
        else:
            processed = {
                'pixel_values': torch.stack([self.processor.image_processor(img) for img in images]) if self.processor else None,
                'labels': None,
            }

        processed['chexpert_targets'] = torch.stack(chexpert_targets)
        processed['chexpert_masks'] = torch.stack(chexpert_masks)
        processed['icd_targets'] = torch.stack(icd_targets)
        processed['icd_masks'] = torch.stack(icd_masks)

        return processed


class StageMixDataset(CurriculumDataset):
    """Dataset that rebuilds Stage-A/B mixture each epoch without full curriculum sampler."""

    def __init__(
        self,
        stage_a_samples: List[Dict[str, Any]],
        stage_b_samples: List[Dict[str, Any]],
        image_root: str = ".",
        processor: Optional[Any] = None,
        max_length: int = 512,
        max_label_tokens: int = 128,
        stage_b_fraction: float = 0.65,
        seed: int = 42,
    ):
        self.stage_b_fraction = stage_b_fraction
        self.seed = seed
        self.stage_a_base = [copy.deepcopy(s) for s in stage_a_samples]
        self.stage_b_base = [copy.deepcopy(s) for s in stage_b_samples]
        initial_samples = build_stage_mix_samples(
            self.stage_a_base,
            self.stage_b_base,
            seed=self.seed,
            stage_b_fraction=self.stage_b_fraction,
        )
        super().__init__(
            samples=initial_samples,
            image_root=image_root,
            processor=processor,
            max_length=max_length,
            stage="both",
            max_label_tokens=max_label_tokens,
        )

    def rebuild(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self.samples = build_stage_mix_samples(
            self.stage_a_base,
            self.stage_b_base,
            seed=self.seed,
            stage_b_fraction=self.stage_b_fraction,
        )
        self._log_statistics()


class StageAwareDataset(CurriculumDataset):
    """
    Dataset that can dynamically switch between Stage A and Stage B.
    
    Used during curriculum training where we first train on Stage A samples,
    then continue with Stage B samples.
    """
    
    def __init__(
        self,
        data_path: str,
        image_root: str = ".",
        processor: Optional[Any] = None,
        max_length: int = 512,
        current_stage: str = "stage_a",
    ):
        """
        Initialize stage-aware dataset.
        
        Args:
            data_path: Path to curriculum JSONL file
            image_root: Root directory for image paths
            processor: HuggingFace processor
            max_length: Maximum sequence length
            current_stage: Current training stage ("stage_a" or "stage_b")
        """
        # Load ALL samples
        super().__init__(
            data_path=data_path,
            image_root=image_root,
            processor=processor,
            max_length=max_length,
            stage="both",  # Load all samples
        )
        
        self.current_stage = current_stage
        self._update_active_samples()
    
    def _update_active_samples(self):
        """Update active samples based on current stage."""
        if self.current_stage == "stage_a":
            # Stage A: only image-only samples
            self.active_indices = [
                i for i, s in enumerate(self.samples)
                if s['stage'] == 'A'
            ]
        elif self.current_stage == "stage_b":
            # Stage B: only image+EHR samples
            self.active_indices = [
                i for i, s in enumerate(self.samples)
                if s['stage'] == 'B'
            ]
        else:  # "both"
            self.active_indices = list(range(len(self.samples)))
        
        logger.info(f"Active samples for {self.current_stage}: {len(self.active_indices)}")
    
    def set_stage(self, stage: str):
        """
        Switch training stage.
        
        Args:
            stage: New stage ("stage_a", "stage_b", or "both")
        """
        if stage != self.current_stage:
            logger.info(f"Switching from {self.current_stage} to {stage}")
            self.current_stage = stage
            self._update_active_samples()
    
    def __len__(self) -> int:
        return len(self.active_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Map to actual sample index
        actual_idx = self.active_indices[idx]
        return super().__getitem__(actual_idx)


def create_dataloader(
    data_path: str,
    processor: Any,
    batch_size: int,
    image_root: str = ".",
    max_length: int = 512,
    stage: str = "both",
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for curriculum training.
    
    Args:
        data_path: Path to curriculum JSONL file
        processor: HuggingFace processor
        batch_size: Batch size
        image_root: Root directory for images
        max_length: Maximum sequence length
        stage: Training stage filter
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader instance
    """
    dataset = CurriculumDataset(
        data_path=data_path,
        image_root=image_root,
        processor=processor,
        max_length=max_length,
        stage=stage,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn,
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    dataset = CurriculumDataset(
        data_path="data/processed/curriculum_train.jsonl",
        image_root=".",
        stage="both",
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Study ID: {sample['study_id']}")
    print(f"  Mode: {sample['mode']}")
    print(f"  Stage: {sample['stage']}")
    print(f"  Image size: {sample['image'].size}")
    print(f"  Prompt: {sample['prompt'][:100]}...")
    print(f"  Target: {sample['target'][:100]}...")
