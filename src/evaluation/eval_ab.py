#!/usr/bin/env python3
"""
A-B Evaluation Script for Radiology Report Model
Compares Stage B performance with and without EHR data using the inference pipeline
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from inference.pipeline import generate as pipeline_generate


def load_validation_data(val_path: str) -> List[Dict]:
    """Load validation data (expects JSONL)"""
    samples = []
    with open(val_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples

ICD_LABELS = [
    "Pneumonia", "Pleural_Effusion", "Pneumothorax", "Pulmonary_Edema",
    "Cardiomegaly", "Atelectasis", "Pulmonary_Embolism", "Rib_Fracture"
]


def extract_icd_labels(sample: Dict) -> Dict[str, int]:
    """Extract ground truth ICD labels from sample (dict of label->0/1)"""
    labels = {label: 0 for label in ICD_LABELS}
    if 'icd_labels' in sample and isinstance(sample['icd_labels'], dict):
        for label in ICD_LABELS:
            labels[label] = 1 if sample['icd_labels'].get(label, 0) else 0
    return labels


def evaluate_icd_with_toggle(samples: List[Dict], use_ehr: bool = True, device: str = 'cpu') -> Dict[str, float]:
    """Evaluate ICD performance with EHR on/off using the inference pipeline"""
    y_true_per_label: Dict[str, List[int]] = {label: [] for label in ICD_LABELS}
    y_pred_per_label: Dict[str, List[int]] = {label: [] for label in ICD_LABELS}

    for sample in samples:
        if sample.get('stage') != 'B':
            continue
        labels = extract_icd_labels(sample)
        for label, val in labels.items():
            y_true_per_label[label].append(val)

        ehr = sample.get('patient_data', {}) if use_ehr else {}
        parsed = pipeline_generate(sample['image_path'], ehr if ehr else None, device=device)
        pred_icd = parsed.get('icd', {})
        for label in ICD_LABELS:
            y_pred_per_label[label].append(1 if pred_icd.get(label, 0) else 0)

    # Compute metrics per label and macro averages
    per_class_f1, per_class_p, per_class_r = {}, {}, {}
    for label in ICD_LABELS:
        y_true = y_true_per_label[label]
        y_pred = y_pred_per_label[label]
        per_class_f1[label] = f1_score(y_true, y_pred, zero_division=0)
        per_class_p[label] = precision_score(y_true, y_pred, zero_division=0)
        per_class_r[label] = recall_score(y_true, y_pred, zero_division=0)

    macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0
    macro_p = float(np.mean(list(per_class_p.values()))) if per_class_p else 0.0
    macro_r = float(np.mean(list(per_class_r.values()))) if per_class_r else 0.0

    return {
        'macro_f1': macro_f1,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_p,
        'per_class_recall': per_class_r,
    }


def save_ab_results(output_dir: Path, results_off: Dict[str, float], results_on: Dict[str, float]):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save macro results
    macro_df = pd.DataFrame([{
        'macro_f1_off': results_off['macro_f1'],
        'macro_f1_on': results_on['macro_f1'],
        'macro_precision_off': results_off['macro_precision'],
        'macro_precision_on': results_on['macro_precision'],
        'macro_recall_off': results_off['macro_recall'],
        'macro_recall_on': results_on['macro_recall'],
        'macro_f1_delta': results_on['macro_f1'] - results_off['macro_f1']
    }])
    macro_df.to_csv(output_dir / 'ab_macro_metrics.csv', index=False)

    # Save per-class metrics
    per_label_rows = []
    for label in ICD_LABELS:
        per_label_rows.append({
            'label': label,
            'f1_off': results_off['per_class_f1'].get(label, 0.0),
            'f1_on': results_on['per_class_f1'].get(label, 0.0),
            'precision_off': results_off['per_class_precision'].get(label, 0.0),
            'precision_on': results_on['per_class_precision'].get(label, 0.0),
            'recall_off': results_off['per_class_recall'].get(label, 0.0),
            'recall_on': results_on['per_class_recall'].get(label, 0.0),
            'delta_f1': results_on['per_class_f1'].get(label, 0.0) - results_off['per_class_f1'].get(label, 0.0)
        })
    pd.DataFrame(per_label_rows).to_csv(output_dir / 'ab_per_label_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='A-B Evaluation for Radiology Report Model')
    parser.add_argument('--val', required=True, help='Path to validation JSONL file')
    parser.add_argument('--device', default='cpu', help='Device: cpu|cuda|mps')
    parser.add_argument('--output_dir', default='evaluation/results', help='Directory for A/B results')
    args = parser.parse_args()

    print("🔬 A-B EVALUATION FOR RADIOLOGY REPORT MODEL")
    print("=" * 50)

    samples = load_validation_data(args.val)
    stage_b_samples = [s for s in samples if s.get('stage') == 'B']
    print(f"   Found {len(stage_b_samples)} Stage B samples")

    print("\n📉 Evaluating B-off (without EHR data)...")
    results_off = evaluate_icd_with_toggle(stage_b_samples, use_ehr=False, device=args.device)

    print("📈 Evaluating B-on (with EHR data)...")
    results_on = evaluate_icd_with_toggle(stage_b_samples, use_ehr=True, device=args.device)

    print("\n📊 RESULTS:")
    print("=" * 50)
    print(f"{'ICD':<20} {'F1_off':<10} {'F1_on':<10} {'Δ':<10}")
    print("-" * 50)

    for label in ICD_LABELS:
        f1_off = results_off['per_class_f1'][label]
        f1_on = results_on['per_class_f1'][label]
        delta = f1_on - f1_off
        print(f"{label:<20} {f1_off:<10.3f} {f1_on:<10.3f} {delta:+.3f}")

    macro_delta = results_on['macro_f1'] - results_off['macro_f1']
    print("-" * 50)
    print(f"{'macro_F1':<20} {results_off['macro_f1']:<10.3f} {results_on['macro_f1']:<10.3f} {macro_delta:+.3f}")

    # Save results
    save_ab_results(Path(args.output_dir), results_off, results_on)

    print("\n✅ EVALUATION COMPLETE")
    print(f"   EHR data improves macro F1 by {macro_delta:+.3f}")

if __name__ == '__main__':
    main()
