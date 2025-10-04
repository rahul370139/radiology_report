#!/usr/bin/env python3.10
"""
Create curriculum training file with image-only and image+EHR samples.

This script creates a unified curriculum file that contains:
1. Image-only prompts (for Stage A training)
2. Image+EHR prompts (for Stage B training)

Each study produces TWO rows: one without EHR, one with EHR context.

Usage:
    python3.10 13_create_curriculum.py

Output:
    data/processed/curriculum.jsonl - Unified training curriculum
    data/processed/curriculum_train.jsonl - Training split
    data/processed/curriculum_val.jsonl - Validation split
"""

import json
import pathlib as pl
import random
from typing import List, Dict, Any
import pandas as pd

# Configuration
MANIFEST_FILE = pl.Path("data/processed/phaseA_manifest.jsonl")
EHR_CONTEXT_FILE = pl.Path("data/processed/ehr_context.jsonl")
TRAIN_SPLIT_FILE = pl.Path("data/processed/phaseA_train.jsonl")
VAL_SPLIT_FILE = pl.Path("data/processed/phaseA_val.jsonl")

OUTPUT_CURRICULUM = pl.Path("data/processed/curriculum.jsonl")
OUTPUT_TRAIN = pl.Path("data/processed/curriculum_train.jsonl")
OUTPUT_VAL = pl.Path("data/processed/curriculum_val.jsonl")

RANDOM_SEED = 42

def load_data() -> tuple:
    """Load all required data files."""
    print("📂 Loading data files...")
    print("=" * 70)
    
    # Load manifest (images, impressions, chexpert)
    manifest = []
    with MANIFEST_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                manifest.append(json.loads(line))
    print(f"Loaded {len(manifest)} records from manifest")
    
    # Load EHR context
    ehr_context = {}
    with EHR_CONTEXT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                ehr_context[str(record['study_id'])] = {
                    'ehr_json': record['ehr_json'],
                    'icd_json': record['icd_json']
                }
    print(f"Loaded {len(ehr_context)} EHR context records")
    
    # Load train/val splits
    train_studies = set()
    with TRAIN_SPLIT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                train_studies.add(str(record['study_id']))
    print(f"Train split: {len(train_studies)} studies")
    
    val_studies = set()
    with VAL_SPLIT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                val_studies.add(str(record['study_id']))
    print(f"Val split: {len(val_studies)} studies")
    
    return manifest, ehr_context, train_studies, val_studies

def format_chexpert_json(chexpert: Dict[str, int]) -> str:
    """Format CheXpert labels as JSON string."""
    # Convert to readable format
    formatted = {}
    for label, value in chexpert.items():
        if value == 1:
            formatted[label] = "Positive"
        elif value == 0:
            formatted[label] = "Negative"
        else:  # -1
            formatted[label] = "Uncertain"
    
    return json.dumps(formatted, indent=2)

def format_icd_json(icd_json: Dict[str, bool]) -> str:
    """Format ICD flags as JSON string."""
    # Only include positive findings
    positive_findings = {k: v for k, v in icd_json.items() if v}
    if not positive_findings:
        return json.dumps({"findings": "None"}, indent=2)
    return json.dumps(positive_findings, indent=2)

def format_ehr_json(ehr_json: Dict[str, Any]) -> str:
    """Format EHR data as compact JSON string."""
    # Create compact representation
    compact_ehr = {
        "Age": ehr_json.get("Age"),
        "Sex": ehr_json.get("Sex"),
        "Vitals": ehr_json.get("Vitals", {}),
        "Labs": ehr_json.get("Labs", {}),
        "O2_device": ehr_json.get("O2_device"),
        "Chronic_conditions": ehr_json.get("Chronic_conditions", [])
    }
    
    return json.dumps(compact_ehr, indent=2)

def create_image_only_sample(record: Dict[str, Any]) -> Dict[str, Any]:
    """Create image-only training sample (Stage A)."""
    impression = record['impression']
    chexpert = format_chexpert_json(record['chexpert'])
    
    return {
        "study_id": record['study_id'],
        "image": record['image_path'],
        "prompt": "Image:<image>\nAnswer Impression & CheXpert.",
        "target": f"Impression: {impression}\n\nCheXpert:\n{chexpert}",
        "mode": "image_only",
        "stage": "A"
    }

def create_image_ehr_sample(record: Dict[str, Any], ehr_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create image+EHR training sample (Stage B)."""
    impression = record['impression']
    chexpert = format_chexpert_json(record['chexpert'])
    ehr_json_str = format_ehr_json(ehr_data['ehr_json'])
    icd_json_str = format_icd_json(ehr_data['icd_json'])
    
    return {
        "study_id": record['study_id'],
        "image": record['image_path'],
        "prompt": f"Patient Data:\n{ehr_json_str}\n\nImage:<image>\nAnswer Impression, CheXpert & ICD.",
        "target": f"Impression: {impression}\n\nCheXpert:\n{chexpert}\n\nICD:\n{icd_json_str}",
        "mode": "image_ehr",
        "stage": "B"
    }

def create_curriculum() -> None:
    """Create curriculum training file."""
    print("\n🎓 CREATING CURRICULUM")
    print("=" * 70)
    
    # Load data
    manifest, ehr_context, train_studies, val_studies = load_data()
    
    # Create curriculum samples
    all_samples = []
    train_samples = []
    val_samples = []
    
    samples_with_ehr = 0
    samples_without_ehr = 0
    
    print("\n🔄 Processing samples...")
    
    for record in manifest:
        study_id = str(record['study_id'])
        
        # Create image-only sample (for all studies)
        image_only_sample = create_image_only_sample(record)
        all_samples.append(image_only_sample)
        
        # Add to appropriate split
        if study_id in train_studies:
            train_samples.append(image_only_sample)
        elif study_id in val_studies:
            val_samples.append(image_only_sample)
        
        # Create image+EHR sample (if EHR data available)
        if study_id in ehr_context:
            image_ehr_sample = create_image_ehr_sample(record, ehr_context[study_id])
            all_samples.append(image_ehr_sample)
            samples_with_ehr += 1
            
            # Add to appropriate split
            if study_id in train_studies:
                train_samples.append(image_ehr_sample)
            elif study_id in val_studies:
                val_samples.append(image_ehr_sample)
        else:
            samples_without_ehr += 1
    
    print(f"Created {len(all_samples)} total samples")
    print(f"  - Image-only samples: {len(manifest)}")
    print(f"  - Image+EHR samples: {samples_with_ehr}")
    print(f"  - Studies without EHR: {samples_without_ehr}")
    
    # Shuffle samples
    print("\n🔀 Shuffling samples...")
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    # Save curriculum files
    print("\n💾 Saving curriculum files...")
    OUTPUT_CURRICULUM.parent.mkdir(parents=True, exist_ok=True)
    
    # Save complete curriculum
    with OUTPUT_CURRICULUM.open("w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✅ Saved complete curriculum: {OUTPUT_CURRICULUM}")
    print(f"   {len(all_samples)} samples")
    
    # Save train split
    with OUTPUT_TRAIN.open("w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✅ Saved training curriculum: {OUTPUT_TRAIN}")
    print(f"   {len(train_samples)} samples")
    
    # Save val split
    with OUTPUT_VAL.open("w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✅ Saved validation curriculum: {OUTPUT_VAL}")
    print(f"   {len(val_samples)} samples")

def analyze_curriculum() -> None:
    """Analyze the generated curriculum."""
    print("\n📊 CURRICULUM ANALYSIS")
    print("=" * 70)
    
    # Load curriculum files
    train_samples = []
    with OUTPUT_TRAIN.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_samples.append(json.loads(line))
    
    val_samples = []
    with OUTPUT_VAL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_samples.append(json.loads(line))
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Total: {len(train_samples) + len(val_samples)}")
    
    # Analyze by mode
    train_image_only = sum(1 for s in train_samples if s['mode'] == 'image_only')
    train_image_ehr = sum(1 for s in train_samples if s['mode'] == 'image_ehr')
    val_image_only = sum(1 for s in val_samples if s['mode'] == 'image_only')
    val_image_ehr = sum(1 for s in val_samples if s['mode'] == 'image_ehr')
    
    print(f"\nTraining breakdown:")
    print(f"  Image-only (Stage A): {train_image_only}")
    print(f"  Image+EHR (Stage B): {train_image_ehr}")
    
    print(f"\nValidation breakdown:")
    print(f"  Image-only (Stage A): {val_image_only}")
    print(f"  Image+EHR (Stage B): {val_image_ehr}")
    
    # Show sample records
    print(f"\n📋 SAMPLE RECORDS")
    print("=" * 70)
    
    print("\n1. Image-Only Sample:")
    image_only_sample = next(s for s in train_samples if s['mode'] == 'image_only')
    print(f"Study ID: {image_only_sample['study_id']}")
    print(f"Prompt: {image_only_sample['prompt'][:100]}...")
    print(f"Target: {image_only_sample['target'][:200]}...")
    
    print("\n2. Image+EHR Sample:")
    image_ehr_sample = next(s for s in train_samples if s['mode'] == 'image_ehr')
    print(f"Study ID: {image_ehr_sample['study_id']}")
    print(f"Prompt: {image_ehr_sample['prompt'][:200]}...")
    print(f"Target: {image_ehr_sample['target'][:200]}...")

def main():
    """Main function."""
    print("MIMIC-CXR Curriculum Creator")
    print("=" * 70)
    
    try:
        # Create curriculum
        create_curriculum()
        
        # Analyze curriculum
        analyze_curriculum()
        
        print(f"\n🎉 Curriculum creation complete!")
        print(f"📁 Files created:")
        print(f"   - {OUTPUT_CURRICULUM}")
        print(f"   - {OUTPUT_TRAIN}")
        print(f"   - {OUTPUT_VAL}")
        
    except Exception as e:
        print(f"❌ Error creating curriculum: {e}")
        raise

if __name__ == "__main__":
    main()
