#!/usr/bin/env python3
"""
Create Comprehensive Curriculum for Radiology Report Generation

This script creates a comprehensive curriculum dataset that combines:
1. Enhanced EHR context with comprehensive vitals and labs
2. Proper deduplication at patient-admission level
3. Stage A (image-only) and Stage B (image + EHR) samples
4. Medical accuracy without fake defaults

Author: AI Assistant
Date: 2025-10-08
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_enhanced_ehr_context(ehr_file: str) -> Dict[str, Dict]:
    """Load the enhanced EHR context data."""
    logger.info("📋 Loading enhanced EHR context...")
    
    ehr_data = {}
    with open(ehr_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            study_id = data.get('study_id')
            if study_id:
                ehr_data[study_id] = data
    
    logger.info(f"   Loaded {len(ehr_data):,} enhanced EHR records")
    return ehr_data

def load_phaseA_manifest(manifest_file: str) -> List[Dict]:
    """Load the Phase A manifest data."""
    logger.info("📋 Loading Phase A manifest...")
    
    manifest_data = []
    with open(manifest_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            manifest_data.append(data)
    
    logger.info(f"   Loaded {len(manifest_data):,} manifest records")
    return manifest_data

def deduplicate_ehr_context(ehr_data: Dict[str, Dict]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Deduplicate EHR context at patient-admission level.
    Returns deduplicated EHR data and study-to-EHR mapping.
    """
    logger.info("🔄 Deduplicating EHR context at patient-admission level...")
    
    # Group by patient-admission key
    patient_admission_groups = defaultdict(list)
    
    for study_id, ehr_record in ehr_data.items():
        patient_key = (
            ehr_record.get('subject_id'),
            ehr_record.get('hadm_id'),
            ehr_record.get('Age'),
            ehr_record.get('Sex'),
            str(ehr_record.get('Vitals', {})),
            str(ehr_record.get('Labs', {})),
            str(ehr_record.get('Chronic_conditions', []))
        )
        patient_admission_groups[patient_key].append(study_id)
    
    # Create deduplicated EHR data
    deduplicated_ehr = {}
    study_to_ehr_mapping = {}
    
    for patient_key, study_ids in patient_admission_groups.items():
        # Use the first study as the representative
        representative_study = study_ids[0]
        ehr_record = ehr_data[representative_study]
        
        # Store the deduplicated record
        deduplicated_ehr[representative_study] = ehr_record
        
        # Map all studies to this representative
        for study_id in study_ids:
            study_to_ehr_mapping[study_id] = representative_study
    
    logger.info(f"   Deduplicated from {len(ehr_data):,} to {len(deduplicated_ehr):,} unique patient-admissions")
    logger.info(f"   Created mapping for {len(study_to_ehr_mapping):,} studies")
    
    return deduplicated_ehr, study_to_ehr_mapping

def create_stageA_samples(manifest_data: List[Dict]) -> List[Dict]:
    """Create Stage A samples (image-only)."""
    logger.info("🎯 Creating Stage A samples (image-only)...")
    
    stageA_samples = []
    
    for record in manifest_data:
        stageA_sample = {
            "image_path": record.get("image_path"),
            "impression": record.get("impression"),
            "chexpert_labels": record.get("chexpert", {}),
            "stage": "A"
        }
        stageA_samples.append(stageA_sample)
    
    logger.info(f"   Created {len(stageA_samples):,} Stage A samples")
    return stageA_samples

def create_stageB_samples(manifest_data: List[Dict], ehr_data: Dict[str, Dict], study_mapping: Dict[str, str]) -> List[Dict]:
    """Create Stage B samples (image + EHR)."""
    logger.info("🎯 Creating Stage B samples (image + EHR)...")
    
    stageB_samples = []
    
    for record in manifest_data:
        study_id = record.get("study_id")
        if not study_id:
            continue
            
        # Get the representative EHR record for this study
        representative_study = study_mapping.get(study_id)
        if not representative_study or representative_study not in ehr_data:
            continue
            
        ehr_record = ehr_data[representative_study]
        
        # Create Stage B sample
        ehr_json = ehr_record.get("ehr_json", {})
        icd_json = ehr_record.get("icd_json", {})
        
        # Extract chronic conditions from ICD flags
        chronic_conditions = []
        for condition, has_condition in icd_json.items():
            if has_condition and condition in ["diabetes", "copd", "ckd", "hypertension", "coronary_artery_disease", "atrial_fibrillation", "stroke", "liver_disease"]:
                chronic_conditions.append(condition)
        
        stageB_sample = {
            "image_path": record.get("image_path"),
            "impression": record.get("impression"),
            "chexpert_labels": record.get("chexpert", {}),
            "stage": "B",
            "patient_data": {
                "subject_id": ehr_record.get("subject_id"),
                "Age": ehr_json.get("Age"),
                "Sex": ehr_json.get("Sex"),
                "hadm_id": ehr_json.get("hadm_id"),
                "admission_type": ehr_json.get("admission_type"),
                "Vitals": ehr_json.get("Vitals", {}),
                "Labs": ehr_json.get("Labs", {}),
                "O2_device": ehr_json.get("O2_device"),
                "Chronic_conditions": chronic_conditions
            }
        }
        stageB_samples.append(stageB_sample)
    
    logger.info(f"   Created {len(stageB_samples):,} Stage B samples")
    return stageB_samples

def create_train_val_split(samples: List[Dict], val_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict]]:
    """Create train/validation split ensuring no patient overlap."""
    logger.info(f"🔄 Creating train/validation split (val_ratio={val_ratio})...")
    
    # Group samples by subject_id to ensure no patient overlap
    patient_groups = defaultdict(list)
    for sample in samples:
        if sample.get("stage") == "B" and "patient_data" in sample:
            subject_id = sample["patient_data"].get("subject_id")
        else:
            # For Stage A samples, use a hash of image path as pseudo-subject_id
            subject_id = hash(sample.get("image_path", ""))
        
        patient_groups[subject_id].append(sample)
    
    # Shuffle patient groups
    patient_ids = list(patient_groups.keys())
    random.shuffle(patient_ids)
    
    # Split patients
    n_val_patients = int(len(patient_ids) * val_ratio)
    val_patient_ids = set(patient_ids[:n_val_patients])
    train_patient_ids = set(patient_ids[n_val_patients:])
    
    # Create splits
    train_samples = []
    val_samples = []
    
    for patient_id, patient_samples in patient_groups.items():
        if patient_id in val_patient_ids:
            val_samples.extend(patient_samples)
        else:
            train_samples.extend(patient_samples)
    
    logger.info(f"   Train: {len(train_samples):,} samples from {len(train_patient_ids):,} patients")
    logger.info(f"   Validation: {len(val_samples):,} samples from {len(val_patient_ids):,} patients")
    
    return train_samples, val_samples

def analyze_curriculum_quality(samples: List[Dict], stage: str) -> None:
    """Analyze the quality of the curriculum samples."""
    logger.info(f"📊 Analyzing {stage} curriculum quality...")
    
    if stage == "A":
        logger.info(f"   Total Stage A samples: {len(samples):,}")
        return
    
    # Analyze Stage B samples
    total_samples = len(samples)
    if total_samples == 0:
        logger.info("   No Stage B samples to analyze")
        return
        
    samples_with_vitals = sum(1 for s in samples if s.get("patient_data", {}).get("Vitals"))
    samples_with_labs = sum(1 for s in samples if s.get("patient_data", {}).get("Labs"))
    samples_with_chronic = sum(1 for s in samples if s.get("patient_data", {}).get("Chronic_conditions"))
    
    logger.info(f"   Total Stage B samples: {total_samples:,}")
    logger.info(f"   Samples with vitals: {samples_with_vitals:,} ({samples_with_vitals/total_samples*100:.1f}%)")
    logger.info(f"   Samples with labs: {samples_with_labs:,} ({samples_with_labs/total_samples*100:.1f}%)")
    logger.info(f"   Samples with chronic conditions: {samples_with_chronic:,} ({samples_with_chronic/total_samples*100:.1f}%)")
    
    # Analyze vitals coverage
    vitals_coverage = defaultdict(int)
    for sample in samples:
        vitals = sample.get("patient_data", {}).get("Vitals", {})
        for vital_name in vitals.keys():
            vitals_coverage[vital_name] += 1
    
    logger.info("   Top vitals coverage:")
    for vital, count in sorted(vitals_coverage.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"     {vital}: {count:,} ({count/total_samples*100:.1f}%)")
    
    # Analyze labs coverage
    labs_coverage = defaultdict(int)
    for sample in samples:
        labs = sample.get("patient_data", {}).get("Labs", {})
        for lab_name in labs.keys():
            labs_coverage[lab_name] += 1
    
    logger.info("   Top labs coverage:")
    for lab, count in sorted(labs_coverage.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"     {lab}: {count:,} ({count/total_samples*100:.1f}%)")

def main():
    """Main function to create comprehensive curriculum."""
    logger.info("🎓 CREATING COMPREHENSIVE CURRICULUM")
    logger.info("=" * 70)
    
    # File paths
    ehr_file = "data/processed/ehr_context.jsonl"
    manifest_file = "data/processed/phaseA_manifest.jsonl"
    output_dir = Path("data/processed")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ehr_data = load_enhanced_ehr_context(ehr_file)
    manifest_data = load_phaseA_manifest(manifest_file)
    
    # Deduplicate EHR context
    deduplicated_ehr, study_mapping = deduplicate_ehr_context(ehr_data)
    
    # Create Stage A samples
    stageA_samples = create_stageA_samples(manifest_data)
    
    # Create Stage B samples
    stageB_samples = create_stageB_samples(manifest_data, deduplicated_ehr, study_mapping)
    
    # Create train/validation splits
    stageA_train, stageA_val = create_train_val_split(stageA_samples, val_ratio=0.15)
    stageB_train, stageB_val = create_train_val_split(stageB_samples, val_ratio=0.15)
    
    # Combine all samples
    all_train = stageA_train + stageB_train
    all_val = stageA_val + stageB_val
    
    # Shuffle combined samples
    random.shuffle(all_train)
    random.shuffle(all_val)
    
    # Save curriculum files
    logger.info("💾 Saving curriculum files...")
    
    # Save individual stage files
    with open(output_dir / "curriculum_stageA_train.jsonl", 'w') as f:
        for sample in stageA_train:
            f.write(json.dumps(sample) + '\n')
    
    with open(output_dir / "curriculum_stageA_val.jsonl", 'w') as f:
        for sample in stageA_val:
            f.write(json.dumps(sample) + '\n')
    
    with open(output_dir / "curriculum_stageB_train.jsonl", 'w') as f:
        for sample in stageB_train:
            f.write(json.dumps(sample) + '\n')
    
    with open(output_dir / "curriculum_stageB_val.jsonl", 'w') as f:
        for sample in stageB_val:
            f.write(json.dumps(sample) + '\n')
    
    # Save combined curriculum files
    with open(output_dir / "curriculum_train_comprehensive.jsonl", 'w') as f:
        for sample in all_train:
            f.write(json.dumps(sample) + '\n')
    
    with open(output_dir / "curriculum_val_comprehensive.jsonl", 'w') as f:
        for sample in all_val:
            f.write(json.dumps(sample) + '\n')
    
    # Save deduplicated EHR context
    with open(output_dir / "ehr_context_deduplicated.jsonl", 'w') as f:
        for ehr_record in deduplicated_ehr.values():
            f.write(json.dumps(ehr_record) + '\n')
    
    # Save study mapping
    with open(output_dir / "study_to_ehr_mapping.json", 'w') as f:
        json.dump(study_mapping, f, indent=2)
    
    # Analyze quality
    logger.info("📊 CURRICULUM QUALITY ANALYSIS")
    logger.info("=" * 50)
    
    analyze_curriculum_quality(stageA_train, "Stage A Train")
    analyze_curriculum_quality(stageA_val, "Stage A Val")
    analyze_curriculum_quality(stageB_train, "Stage B Train")
    analyze_curriculum_quality(stageB_val, "Stage B Val")
    
    # Final summary
    logger.info("🎉 COMPREHENSIVE CURRICULUM CREATION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"📁 Output files:")
    logger.info(f"   - curriculum_stageA_train.jsonl: {len(stageA_train):,} samples")
    logger.info(f"   - curriculum_stageA_val.jsonl: {len(stageA_val):,} samples")
    logger.info(f"   - curriculum_stageB_train.jsonl: {len(stageB_train):,} samples")
    logger.info(f"   - curriculum_stageB_val.jsonl: {len(stageB_val):,} samples")
    logger.info(f"   - curriculum_train_comprehensive.jsonl: {len(all_train):,} samples")
    logger.info(f"   - curriculum_val_comprehensive.jsonl: {len(all_val):,} samples")
    logger.info(f"   - ehr_context_deduplicated.jsonl: {len(deduplicated_ehr):,} unique patient-admissions")
    logger.info(f"   - study_to_ehr_mapping.json: {len(study_mapping):,} study mappings")
    
    logger.info("\n✅ Ready for model finetuning!")

if __name__ == "__main__":
    main()
