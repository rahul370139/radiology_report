#!/usr/bin/env python3.10
"""
Build Phase A manifest for MIMIC-CXR dataset.

This script creates a manifest for Phase A training that includes:
- Image paths for chest X-ray images
- Corresponding impressions
- CheXpert labels
- Filters for frontal views only (PA/AP)

Usage:
    python3.10 06_build_phaseA_manifest.py

Output:
    data/processed/phaseA_manifest.jsonl - Phase A training manifest
"""

import pandas as pd
import json
import pathlib as pl
from typing import Dict, List, Any, Optional
import re

# Configuration
STUDY_LIST_FILE = pl.Path("cxr-study-list.csv")
RECORD_LIST_FILE = pl.Path("cxr-record-list.csv")
CHEXPERT_DICT_FILE = pl.Path("data/processed/chexpert_dict.json")
IMPRESSIONS_FILE = pl.Path("data/processed/impressions.jsonl")
OUTPUT_FILE = pl.Path("data/processed/phaseA_manifest.jsonl")

def load_data_sources() -> tuple:
    """Load all required data sources."""
    print("Loading data sources...")
    
    # Load study list
    if not STUDY_LIST_FILE.exists():
        raise FileNotFoundError(f"Study list file not found: {STUDY_LIST_FILE}")
    studies_df = pd.read_csv(STUDY_LIST_FILE)
    print(f"Loaded {len(studies_df)} studies")
    
    # Load record list (maps images to studies)
    if not RECORD_LIST_FILE.exists():
        raise FileNotFoundError(f"Record list file not found: {RECORD_LIST_FILE}")
    records_df = pd.read_csv(RECORD_LIST_FILE)
    print(f"Loaded {len(records_df)} records")
    
    # Load CheXpert labels
    if not CHEXPERT_DICT_FILE.exists():
        raise FileNotFoundError(f"CheXpert dict file not found: {CHEXPERT_DICT_FILE}")
    with CHEXPERT_DICT_FILE.open("r", encoding="utf-8") as f:
        chexpert_dict = json.load(f)
    print(f"Loaded {len(chexpert_dict)} CheXpert records")
    
    # Load impressions
    if not IMPRESSIONS_FILE.exists():
        raise FileNotFoundError(f"Impressions file not found: {IMPRESSIONS_FILE}")
    impressions_dict = {}
    with IMPRESSIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                impressions_dict[record["study_id"]] = record["impression"]
    print(f"Loaded {len(impressions_dict)} impressions")
    
    return studies_df, records_df, chexpert_dict, impressions_dict

def infer_view_from_path(image_path: str) -> Optional[str]:
    """
    Infer view type from image path or filename.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        View type ('PA', 'AP', 'LATERAL', etc.) or None if cannot determine
    """
    path_lower = image_path.lower()
    
    # Common patterns for view inference
    if 'pa' in path_lower or 'posteroanterior' in path_lower:
        return 'PA'
    elif 'ap' in path_lower or 'anteroposterior' in path_lower:
        return 'AP'
    elif 'lateral' in path_lower or 'lat' in path_lower:
        return 'LATERAL'
    elif 'portable' in path_lower:
        return 'PORTABLE'  # Often AP
    else:
        # Default to PA for frontal views if no specific indicator
        return 'PA'

def create_image_study_mapping(records_df: pd.DataFrame) -> Dict[str, str]:
    """
    Create mapping from image ID to study ID.
    
    Args:
        records_df: DataFrame with record information
        
    Returns:
        Dictionary mapping image_id to study_id
    """
    image_to_study = {}
    
    for _, row in records_df.iterrows():
        # Extract image ID from DICOM path
        dicom_path = row['path']
        if dicom_path.endswith('.dcm'):
            # Convert DICOM path to JPG path
            jpg_path = dicom_path.replace('.dcm', '.jpg')
            image_id = pl.Path(jpg_path).stem
            image_to_study[image_id] = str(row['study_id'])
    
    return image_to_study

def build_phaseA_manifest() -> List[Dict[str, Any]]:
    """Build Phase A manifest with image paths, impressions, and CheXpert labels."""
    print("Building Phase A manifest...")
    
    # Load all data sources
    studies_df, records_df, chexpert_dict, impressions_dict = load_data_sources()
    
    # Create image to study mapping
    image_to_study = create_image_study_mapping(records_df)
    print(f"Created mapping for {len(image_to_study)} images")
    
    manifest = []
    skipped_reasons = {
        'no_impression': 0,
        'no_chexpert': 0,
        'no_image': 0,
        'non_frontal': 0,
        'missing_file': 0
    }
    
    print(f"Processing {len(studies_df)} studies...")
    
    for idx, row in studies_df.iterrows():
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(studies_df)} studies...")
        
        study_id = str(row['study_id'])
        study_id_with_s = 's' + study_id  # Add 's' prefix for impression lookup
        
        # Check if we have impression
        if study_id_with_s not in impressions_dict:
            skipped_reasons['no_impression'] += 1
            continue
        
        # Check if we have CheXpert labels
        if study_id not in chexpert_dict:
            skipped_reasons['no_chexpert'] += 1
            continue
        
        # Find corresponding image
        image_found = False
        for image_id, mapped_study_id in image_to_study.items():
            if mapped_study_id == study_id:
                # Check if image file exists - use the full path from record list
                # Find the original DICOM path and convert to JPG
                matching_records = records_df[records_df['study_id'] == int(study_id)]
                if len(matching_records) > 0:
                    # Use the first matching record
                    dicom_path = matching_records.iloc[0]['path']
                    image_path = dicom_path.replace('.dcm', '.jpg')
                    
                    if not pl.Path(image_path).exists():
                        skipped_reasons['missing_file'] += 1
                        continue
                
                # Extract image ID from path
                image_id = pl.Path(image_path).stem
                
                # Infer view type
                view = infer_view_from_path(image_path)
                
                # Filter for frontal views only (PA/AP)
                if view not in ['PA', 'AP']:
                    skipped_reasons['non_frontal'] += 1
                    continue
                
                # Add to manifest
                manifest.append({
                    "image_path": image_path,
                    "image_id": image_id,
                    "study_id": study_id,
                    "subject_id": int(row['subject_id']),
                    "view": view,
                    "impression": impressions_dict[study_id_with_s],
                    "chexpert": chexpert_dict[study_id]
                })
                
                image_found = True
                break
        
        if not image_found:
            skipped_reasons['no_image'] += 1
    
    print(f"\nManifest creation complete!")
    print(f"Total records in manifest: {len(manifest)}")
    print(f"\nSkipped reasons:")
    for reason, count in skipped_reasons.items():
        print(f"  {reason}: {count}")
    
    return manifest

def save_manifest(manifest: List[Dict[str, Any]]) -> None:
    """Save manifest to JSONL file."""
    print(f"Saving manifest to: {OUTPUT_FILE}")
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in manifest:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(manifest)} records to manifest")

def analyze_manifest(manifest: List[Dict[str, Any]]) -> None:
    """Analyze the created manifest."""
    print(f"\n📊 MANIFEST ANALYSIS")
    print(f"=" * 50)
    
    print(f"Total records: {len(manifest)}")
    
    if not manifest:
        print("No records in manifest!")
        return
    
    # Analyze views
    views = [record['view'] for record in manifest]
    view_counts = pd.Series(views).value_counts()
    print(f"\nView distribution:")
    for view, count in view_counts.items():
        print(f"  {view}: {count}")
    
    # Analyze CheXpert labels
    print(f"\nCheXpert label distribution:")
    all_labels = set()
    for record in manifest:
        all_labels.update(record['chexpert'].keys())
    
    for label in sorted(all_labels):
        values = [record['chexpert'][label] for record in manifest]
        value_counts = pd.Series(values).value_counts()
        print(f"  {label}: {dict(value_counts)}")
    
    # Analyze impression lengths
    lengths = [len(record['impression']) for record in manifest]
    print(f"\nImpression statistics:")
    print(f"  Average length: {sum(lengths)/len(lengths):.1f} characters")
    print(f"  Min length: {min(lengths)} characters")
    print(f"  Max length: {max(lengths)} characters")
    
    # Show sample records
    print(f"\n📋 Sample manifest records:")
    for i, record in enumerate(manifest[:3]):
        print(f"\n{i+1}. Study ID: {record['study_id']}")
        print(f"   Image: {record['image_path']}")
        print(f"   View: {record['view']}")
        print(f"   Impression: {record['impression'][:100]}...")
        print(f"   CheXpert: {record['chexpert']}")

def main():
    """Main function."""
    print("MIMIC-CXR Phase A Manifest Builder")
    print("=" * 50)
    
    try:
        # Build manifest
        manifest = build_phaseA_manifest()
        
        # Save manifest
        save_manifest(manifest)
        
        # Analyze manifest
        analyze_manifest(manifest)
        
        print(f"\n✅ Phase A manifest created successfully!")
        print(f"📁 Manifest saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Error creating manifest: {e}")
        raise

if __name__ == "__main__":
    main()
