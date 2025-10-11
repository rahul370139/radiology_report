#!/usr/bin/env python3
"""
Advanced Curriculum Learning Implementation
Implements Stage-A oversampling, synthetic Stage-A, and stratified sampling
"""

import json
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedCurriculumSampler:
    """
    Advanced curriculum sampler that implements:
    1. Stage-A oversampling in early training
    2. Synthetic Stage-A (Stage-B with empty EHR)
    3. Stratified sampling for class imbalance
    """
    
    def __init__(
        self,
        samples: List[Dict],
        stage_a_oversample_ratio: float = 0.5,
        synthetic_stage_a_ratio: float = 0.3,
        rare_icd_boost: float = 2.0,
        chexpert_positive_boost: float = 1.5,
        current_step: int = 0,
        total_steps: int = 1000,
        stage_split: float = 0.3
    ):
        self.samples = samples
        self.stage_a_oversample_ratio = stage_a_oversample_ratio
        self.synthetic_stage_a_ratio = synthetic_stage_a_ratio
        self.rare_icd_boost = rare_icd_boost
        self.chexpert_positive_boost = chexpert_positive_boost
        self.current_step = current_step
        self.total_steps = total_steps
        self.stage_split = stage_split
        
        # Separate samples by stage
        self.stage_a_samples = [s for s in samples if s['stage'] == 'A']
        self.stage_b_samples = [s for s in samples if s['stage'] == 'B']
        
        # Build class frequency maps
        self._build_class_maps()
        
        logger.info(f"Advanced Curriculum Sampler initialized:")
        logger.info(f"  Stage A samples: {len(self.stage_a_samples)}")
        logger.info(f"  Stage B samples: {len(self.stage_b_samples)}")
        logger.info(f"  Stage split at: {int(total_steps * stage_split)} steps")
    
    def _build_class_maps(self):
        """Build frequency maps for stratified sampling"""
        # ICD frequency map
        self.icd_freq = defaultdict(int)
        self.chexpert_freq = defaultdict(int)
        
        for sample in self.stage_b_samples:
            # Count ICD codes
            if 'patient_data' in sample and 'Chronic_conditions' in sample['patient_data']:
                for icd in sample['patient_data']['Chronic_conditions']:
                    self.icd_freq[icd] += 1
            
            # Count CheXpert positives
            if 'chexpert_labels' in sample:
                for label, value in sample['chexpert_labels'].items():
                    if value == 1:  # Positive
                        self.chexpert_freq[label] += 1
        
        # Calculate sampling weights
        self.icd_weights = self._calculate_weights(self.icd_freq, self.rare_icd_boost)
        self.chexpert_weights = self._calculate_weights(self.chexpert_freq, self.chexpert_positive_boost)
        
        logger.info(f"Class frequency maps built:")
        logger.info(f"  ICD classes: {len(self.icd_freq)}")
        logger.info(f"  CheXpert labels: {len(self.chexpert_freq)}")
    
    def _calculate_weights(self, freq_map: Dict, boost_factor: float) -> Dict:
        """Calculate sampling weights with boost for rare classes"""
        if not freq_map:
            return {}
        
        max_freq = max(freq_map.values())
        weights = {}
        
        for class_name, freq in freq_map.items():
            # Inverse frequency weighting with boost
            weight = (max_freq / freq) ** 0.5  # Square root to moderate the boost
            if freq < max_freq * 0.1:  # Rare class threshold
                weight *= boost_factor
            weights[class_name] = weight
        
        return weights
    
    def get_curriculum_ratio(self) -> Tuple[float, float, float]:
        """
        Get current curriculum mixing ratios based on training progress
        Returns: (stage_a_ratio, synthetic_stage_a_ratio, stage_b_ratio)
        """
        progress = self.current_step / self.total_steps
        
        if progress < self.stage_split:
            # Early training: Stage-A heavy
            stage_a_ratio = self.stage_a_oversample_ratio
            synthetic_ratio = self.synthetic_stage_a_ratio
            stage_b_ratio = 1.0 - stage_a_ratio - synthetic_ratio
        else:
            # Later training: Stage-B dominant
            stage_a_ratio = 0.2
            synthetic_ratio = 0.1
            stage_b_ratio = 0.7
        
        return stage_a_ratio, synthetic_ratio, stage_b_ratio
    
    def create_synthetic_stage_a(self, sample: Dict) -> Dict:
        """Create synthetic Stage-A sample by removing EHR data"""
        synthetic = sample.copy()
        synthetic['stage'] = 'A'
        synthetic['patient_data'] = {
            'subject_id': sample['patient_data'].get('subject_id', 0),
            'Age': sample['patient_data'].get('Age', 0),
            'Sex': sample['patient_data'].get('Sex', 'Unknown'),
            'Vitals': {},
            'Labs': {},
            'O2_device': None,
            'Chronic_conditions': []
        }
        return synthetic
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample a batch with curriculum learning and stratified sampling"""
        stage_a_ratio, synthetic_ratio, stage_b_ratio = self.get_curriculum_ratio()
        
        batch = []
        
        # Calculate sample counts
        stage_a_count = int(batch_size * stage_a_ratio)
        synthetic_count = int(batch_size * synthetic_ratio)
        stage_b_count = batch_size - stage_a_count - synthetic_count
        
        # Sample Stage-A
        if stage_a_count > 0:
            stage_a_batch = random.sample(self.stage_a_samples, min(stage_a_count, len(self.stage_a_samples)))
            batch.extend(stage_a_batch)
        
        # Sample synthetic Stage-A
        if synthetic_count > 0:
            synthetic_samples = random.sample(self.stage_b_samples, min(synthetic_count, len(self.stage_b_samples)))
            synthetic_batch = [self.create_synthetic_stage_a(s) for s in synthetic_samples]
            batch.extend(synthetic_batch)
        
        # Sample Stage-B with stratified sampling
        if stage_b_count > 0:
            stage_b_batch = self._stratified_sample_stage_b(stage_b_count)
            batch.extend(stage_b_batch)
        
        # Shuffle the batch
        random.shuffle(batch)
        
        return batch
    
    def _stratified_sample_stage_b(self, count: int) -> List[Dict]:
        """Sample Stage-B samples with stratified sampling for class balance"""
        if not self.icd_weights and not self.chexpert_weights:
            return random.sample(self.stage_b_samples, min(count, len(self.stage_b_samples)))
        
        # Calculate sampling probabilities
        sample_weights = []
        for sample in self.stage_b_samples:
            weight = 1.0
            
            # Boost for rare ICDs
            if 'patient_data' in sample and 'Chronic_conditions' in sample['patient_data']:
                for icd in sample['patient_data']['Chronic_conditions']:
                    if icd in self.icd_weights:
                        weight *= self.icd_weights[icd]
            
            # Boost for positive CheXpert labels
            if 'chexpert_labels' in sample:
                for label, value in sample['chexpert_labels'].items():
                    if value == 1 and label in self.chexpert_weights:
                        weight *= self.chexpert_weights[label]
            
            sample_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(sample_weights)
        if total_weight > 0:
            sample_weights = [w / total_weight for w in sample_weights]
        else:
            sample_weights = [1.0 / len(self.stage_b_samples)] * len(self.stage_b_samples)
        
        # Sample with replacement
        indices = np.random.choice(
            len(self.stage_b_samples),
            size=min(count, len(self.stage_b_samples)),
            replace=True,
            p=sample_weights
        )
        
        return [self.stage_b_samples[i] for i in indices]
    
    def update_step(self, step: int):
        """Update current training step"""
        self.current_step = step
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current sampling statistics"""
        stage_a_ratio, synthetic_ratio, stage_b_ratio = self.get_curriculum_ratio()
        
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress': self.current_step / self.total_steps,
            'stage_a_ratio': stage_a_ratio,
            'synthetic_ratio': synthetic_ratio,
            'stage_b_ratio': stage_b_ratio,
            'stage_a_samples': len(self.stage_a_samples),
            'stage_b_samples': len(self.stage_b_samples),
            'icd_classes': len(self.icd_freq),
            'chexpert_labels': len(self.chexpert_freq)
        }


class JSONDriftPrevention:
    """
    JSON drift prevention with constrained output and self-repair
    """
    
    def __init__(self, chexpert_labels: List[str], icd_classes: List[str]):
        self.chexpert_labels = chexpert_labels
        self.icd_classes = icd_classes
        
        # Build regex patterns for constrained decoding
        self.chexpert_pattern = self._build_chexpert_pattern()
        self.icd_pattern = self._build_icd_pattern()
    
    def _build_chexpert_pattern(self) -> str:
        """Build regex pattern for CheXpert JSON validation"""
        labels_str = "|".join(self.chexpert_labels)
        return rf'CheXpert:\s*\{{[^}}]*"(?:{labels_str})"[^}}]*\}}'
    
    def _build_icd_pattern(self) -> str:
        """Build regex pattern for ICD JSON validation"""
        classes_str = "|".join([cls.split('(')[0].strip() for cls in self.icd_classes])
        return rf'ICD:\s*\{{[^}}]*"(?:{classes_str})"[^}}]*\}}'
    
    def validate_json(self, text: str) -> bool:
        """Validate JSON structure in generated text"""
        try:
            # Extract CheXpert section
            if "CheXpert:" in text:
                chexpert_start = text.find("CheXpert:")
                chexpert_end = text.find("}", chexpert_start) + 1
                if chexpert_end > chexpert_start:
                    chexpert_json = text[chexpert_start:chexpert_end]
                    # Try to parse as JSON
                    json.loads(chexpert_json.replace("CheXpert:", "").strip())
            
            # Extract ICD section
            if "ICD:" in text:
                icd_start = text.find("ICD:")
                icd_end = text.find("}", icd_start) + 1
                if icd_end > icd_start:
                    icd_json = text[icd_start:icd_end]
                    # Try to parse as JSON
                    json.loads(icd_json.replace("ICD:", "").strip())
            
            return True
        except:
            return False
    
    def create_repair_prompt(self, invalid_text: str) -> str:
        """Create self-repair prompt for invalid JSON"""
        return f"""
The following text contains invalid JSON. Please fix it and return only the corrected JSON sections:

{invalid_text}

Please provide:
1. Valid CheXpert JSON with all required keys
2. Valid ICD JSON with proper structure
3. No additional text or explanations
"""
    
    def get_constrained_prompt(self) -> str:
        """Get prompt with JSON structure constraints"""
        chexpert_keys = ", ".join([f'"{label}"' for label in self.chexpert_labels])
        icd_keys = ", ".join([f'"{cls}"' for cls in self.icd_classes])
        
        return f"""
<image>
Patient Data: {{EHR_JSON}}
Task: Provide (1) Impression, (2) CheXpert JSON, (3) ICD JSON.

Impression:
<one concise paragraph>

CheXpert:
{{ {chexpert_keys} }}

ICD:
{{ {icd_keys} }}
"""


def create_advanced_curriculum_dataset(
    samples: List[Dict],
    config: Dict[str, Any]
) -> AdvancedCurriculumSampler:
    """Create advanced curriculum sampler from configuration"""
    return AdvancedCurriculumSampler(
        samples=samples,
        stage_a_oversample_ratio=config.get('stage_a_oversample_ratio', 0.5),
        synthetic_stage_a_ratio=config.get('synthetic_stage_a_ratio', 0.3),
        rare_icd_boost=config.get('rare_icd_boost', 2.0),
        chexpert_positive_boost=config.get('chexpert_positive_boost', 1.5),
        total_steps=config.get('total_steps', 1000),
        stage_split=config.get('stage_split', 0.3)
    )


def create_json_drift_prevention(
    chexpert_labels: List[str],
    icd_classes: List[str]
) -> JSONDriftPrevention:
    """Create JSON drift prevention system"""
    return JSONDriftPrevention(chexpert_labels, icd_classes)
