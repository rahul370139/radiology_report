#!/usr/bin/env python3.10
"""
Dataset class for MIMIC-CXR curriculum training.

Handles loading images, prompts, and targets for both Stage A and Stage B training.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CurriculumDataset(Dataset):
    """
    Dataset for curriculum learning with chest X-ray images.
    
    Supports both Stage A (image-only) and Stage B (image+EHR) training.
    """
    
    def __init__(
        self,
        data_path: str,
        image_root: str = ".",
        processor: Optional[Any] = None,
        max_length: int = 512,
        stage: str = "both",  # "stage_a", "stage_b", or "both"
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
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_length = max_length
        self.stage = stage
        
        # Load curriculum data
        self.samples = self._load_curriculum(data_path)
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
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
        
        # Build prompt and target from available data
        if sample['stage'] == 'A':
            # Stage A: Image-only
            prompt = "<image>\nTask: Provide (1) Impression, (2) CheXpert JSON, (3) ICD JSON.\nImpression:"
            target = sample['impression']
            mode = "image_only"
        else:
            # Stage B: Image + EHR
            patient_data = sample.get('patient_data', {})
            ehr_json = json.dumps(patient_data, indent=2) if patient_data else "{}"
            prompt = f"<image>\nPatient Data: {ehr_json}\nTask: Provide (1) Impression, (2) CheXpert JSON, (3) ICD JSON.\nImpression:"
            target = sample['impression']
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
        }
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched data ready for model input
        """
        images = [item['image'] for item in batch]  # PIL Images from __getitem__
        prompts = [item['prompt'] for item in batch]
        targets = [item['target'] for item in batch]
        
        # Process with HuggingFace processor if available
        if self.processor is not None:
            # Build chat template messages for each sample
            chat_texts = []
            for i, (prompt, target, mode) in enumerate(zip(prompts, targets, [item['mode'] for item in batch])):
                # Extract EHR context if present
                has_ehr = mode == 'image_ehr'
                ehr_block = ""
                if has_ehr and 'ehr_context' in batch[i]:
                    ehr_block = batch[i]['ehr_context'] + "\n"
                
                # Build messages using chat template (must alternate user/assistant only)
                messages = [
                    {"role": "user", "content": (ehr_block + "\n" if has_ehr else "") + "<image>\n"
                                              "Provide:\nImpression:\nCheXpert:\nICD:"},
                    {"role": "assistant", "content": target}
                ]
                
                # Apply chat template
                text = self.processor.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False
                )
                chat_texts.append(text)
            
            # Process images and text together
            processed = self.processor(
                text=chat_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            
            # Create proper labels with masking
            # Mask everything except the assistant's response
            labels = torch.full_like(processed['input_ids'], -100)
            
            for i, (chat_text, target) in enumerate(zip(chat_texts, targets)):
                # Find where the assistant response starts
                # Look for the assistant role marker in the tokenized text
                assistant_start = chat_text.find("assistant")
                if assistant_start != -1:
                    # Tokenize to find the position
                    tokens = self.processor.tokenizer.encode(chat_text, add_special_tokens=False)
                    assistant_tokens = self.processor.tokenizer.encode("assistant", add_special_tokens=False)
                    
                    # Find assistant token position
                    for j in range(len(tokens) - len(assistant_tokens) + 1):
                        if tokens[j:j+len(assistant_tokens)] == assistant_tokens:
                            # Mask everything before assistant response
                            labels[i, j+len(assistant_tokens):] = processed['input_ids'][i, j+len(assistant_tokens):]
                            break
                else:
                    # Fallback: mask first 80% of tokens
                    seq_len = processed['input_ids'][i].size(0)
                    mask_start = int(seq_len * 0.8)
                    if mask_start < seq_len:
                        labels[i, mask_start:] = processed['input_ids'][i, mask_start:]
            
            processed['labels'] = labels
            
            # Rename pixel_values to images for LLaVA-Med compatibility
            if 'pixel_values' in processed:
                processed['images'] = processed.pop('pixel_values')
        else:
            # Return raw data if no processor
            processed = {
                'images': images,
                'prompts': prompts,
                'targets': targets,
            }
        
        # Note: mode, stage, study_id are metadata fields that should not be passed to the model
        # They can be extracted from the batch items if needed for logging/curriculum control
        
        return processed


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

