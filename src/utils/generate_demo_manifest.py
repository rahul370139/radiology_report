#!/usr/bin/env python3
"""
Generate demo manifest for evaluation
Creates a balanced set of Demo A (image-only) and Demo B (image+EHR) samples
"""

import json
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Any


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file"""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def select_demo_samples(samples: List[Dict[str, Any]],
                       n_demo_a: int = 40,
                       n_demo_b: int = 40,
                       seed: int = 42) -> List[Dict[str, Any]]:
    """Select balanced demo samples"""
    random.seed(seed)
    
    # Separate by stage
    stage_a_samples = [s for s in samples if s.get('stage') == 'A']
    stage_b_samples = [s for s in samples if s.get('stage') == 'B']
    
    print(f"📊 Dataset stats:")
    print(f"   Stage A samples: {len(stage_a_samples)}")
    print(f"   Stage B samples: {len(stage_b_samples)}")
    
    # Select Demo A samples (Stage A - image only)
    demo_a_samples = random.sample(stage_a_samples, min(n_demo_a, len(stage_a_samples)))
    
    # Select Demo B samples (Stage B - image + EHR)
    demo_b_samples = random.sample(stage_b_samples, min(n_demo_b, len(stage_b_samples)))
    
    # Create manifest entries
    manifest = []
    
    # Demo A entries
    for i, sample in enumerate(demo_a_samples):
        manifest.append({
            'study_id': f"demo_a_{i+1:03d}",
            'image_path': sample['image_path'],
            'stage': 'A',
            'ehr_json_path': None,
            'ground_truth_impression': sample.get('impression', ''),
            'ground_truth_chexpert': sample.get('chexpert_labels', {}),
            'ground_truth_icd': None,
            'subject_id': sample.get('patient_data', {}).get('subject_id', 'unknown')
        })
    
    # Demo B entries
    for i, sample in enumerate(demo_b_samples):
        # Create EHR JSON file path
        ehr_json_path = f"evaluation/demo_ehr/demo_b_{i+1:03d}_ehr.json"
        
        manifest.append({
            'study_id': f"demo_b_{i+1:03d}",
            'image_path': sample['image_path'],
            'stage': 'B',
            'ehr_json_path': ehr_json_path,
            'ground_truth_impression': sample.get('impression', ''),
            'ground_truth_chexpert': sample.get('chexpert_labels', {}),
            'ground_truth_icd': sample.get('icd_labels', []),
            'subject_id': sample.get('patient_data', {}).get('subject_id', 'unknown')
        })
    
    return manifest, demo_b_samples


def save_ehr_json_files(demo_b_samples: List[Dict[str, Any]], output_dir: str):
    """Save EHR JSON files for Demo B samples"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(demo_b_samples):
        ehr_data = sample.get('patient_data', {})
        ehr_file = output_path / f"demo_b_{i+1:03d}_ehr.json"
        
        with open(ehr_file, 'w') as f:
            json.dump(ehr_data, f, indent=2)


def main():
    print("🎯 GENERATING DEMO MANIFEST")
    print("=" * 40)
    
    # Paths
    train_file = "src/data/processed/curriculum_train_final_clean.jsonl"
    val_file = "src/data/processed/curriculum_val_final_clean.jsonl"
    output_dir = "evaluation"
    
    # Load datasets
    print("📂 Loading datasets...")
    train_samples = load_dataset(train_file)
    val_samples = load_dataset(val_file)
    all_samples = train_samples + val_samples
    
    print(f"   Total samples: {len(all_samples)}")
    
    # Select demo samples
    print("\n🎲 Selecting demo samples...")
    manifest, demo_b_samples = select_demo_samples(
        all_samples,
        n_demo_a=40,
        n_demo_b=40,
        seed=42
    )
    
    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_path = Path(output_dir) / "demo_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"✅ Saved manifest: {manifest_path}")
    print(f"   Demo A samples: {len([m for m in manifest if m['stage'] == 'A'])}")
    print(f"   Demo B samples: {len([m for m in manifest if m['stage'] == 'B'])}")
    
    # Save EHR JSON files for Demo B
    print("\n💾 Saving EHR JSON files...")
    save_ehr_json_files(demo_b_samples, f"{output_dir}/demo_ehr")
    
    print(f"✅ Saved EHR files to: {output_dir}/demo_ehr/")
    
    # Show sample manifest entries
    print("\n📋 Sample manifest entries:")
    print(manifest_df.head(3).to_string(index=False))
    
    print(f"\n🎉 Demo manifest generation complete!")
    print(f"   Manifest: {manifest_path}")
    print(f"   EHR files: {output_dir}/demo_ehr/")


if __name__ == "__main__":
    main()
