"""
Simple Batch Evaluation Script for Radiology Report Model
Reads demo_manifest.csv, runs inference using inference.pipeline, and computes metrics.
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics import f1_score, precision_score, recall_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Use the real inference pipeline
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from inference.pipeline import generate as pipeline_generate


def list_to_dict(lst):
    """Convert ICD list format to dict format for evaluation"""
    if not lst:
        return {}
    if isinstance(lst, dict):
        return lst
    out = {l["code"]: l.get("confidence", 0) for l in lst}
    return out

class SimpleBatchEvaluator:
    """Simple Batch evaluator for radiology report generation"""
    
    def __init__(self, config_path: str, device: str = "cpu"):
        self.config_path = config_path
        self.device = device
        
        # 12-label CheXpert schema
        self.chexpert_labels = [
            "No Finding", "Enlarged Cardiomediastinum", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
            "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
        ]
        self.icd_classes = [
            "Pneumonia", "Pleural_Effusion", "Pneumothorax", "Pulmonary_Edema",
            "Cardiomegaly", "Atelectasis", "Pulmonary_Embolism", "Rib_Fracture"
        ]
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth_function = SmoothingFunction().method1

    def evaluate_batch(self, manifest_path: str, output_dir: Path) -> List[Dict]:
        """
        Reads manifest, runs inference for each sample, and saves results.
        """
        manifest_df = pd.read_csv(manifest_path)
        results = []
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_jsonl_path = output_dir / "batch_predictions.jsonl"
        
        with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
            for idx, row in manifest_df.iterrows():
                print(f"Processing sample {idx+1}/{len(manifest_df)}: {row['study_id']}")
                image_path = Path(row['image_path'])
                ehr_json_path = Path(row['ehr_json_path']) if pd.notna(row['ehr_json_path']) and row['ehr_json_path'] else None
                
                ehr_data = None
                if ehr_json_path and ehr_json_path.exists():
                    with open(ehr_json_path, 'r') as f:
                        ehr_data = json.load(f)
                
                # Generate structured output using pipeline
                parsed = pipeline_generate(str(image_path), ehr_data, device=self.device)
                
                sample_result = {
                    'image_path': str(image_path),
                    'ehr_data': ehr_data,
                    'response': json.dumps(parsed),
                    'parsed': parsed,
                    'study_id': row['study_id'],
                    'stage': row.get('stage', 'A')
                }
                
                # Add ground truth to the result for metric computation
                sample_result['ground_truth_impression'] = row['ground_truth_impression']
                sample_result['ground_truth_chexpert'] = json.loads(row['ground_truth_chexpert'].replace("'", '"'))
                # Handle ICD ground truth - convert list to dict if needed
                if pd.notna(row['ground_truth_icd']):
                    icd_data = json.loads(row['ground_truth_icd'].replace("'", '"'))
                    sample_result['ground_truth_icd'] = list_to_dict(icd_data)
                else:
                    sample_result['ground_truth_icd'] = {}
                
                # Map parsed results to expected field names for metrics computation
                sample_result['generated_impression'] = sample_result['parsed'].get('impression', '')
                sample_result['generated_chexpert'] = sample_result['parsed'].get('chexpert', {})
                sample_result['generated_icd'] = sample_result['parsed'].get('icd', {})
                
                results.append(sample_result)
                outfile.write(json.dumps(sample_result) + '\n')
        
        print(f"✅ Batch predictions saved to {output_jsonl_path}")
        return results

    def _compute_text_metrics(self, gts: List[str], preds: List[str]) -> Dict[str, float]:
        if not gts or not preds:
            return {}
        rouge_scores = [self.rouge_scorer.score(g, p) for g, p in zip(gts, preds)]
        metrics = {
            'rougeL_fmeasure': float(np.mean([s['rougeL'].fmeasure for s in rouge_scores])),
            'rouge1_fmeasure': float(np.mean([s['rouge1'].fmeasure for s in rouge_scores])),
            'rouge2_fmeasure': float(np.mean([s['rouge2'].fmeasure for s in rouge_scores]))
        }
        tokenized_gt = [g.split() for g in gts]
        tokenized_gen = [p.split() for p in preds]
        bleu_scores = [sentence_bleu([gt], gen, smoothing_function=self.smooth_function) for gt, gen in zip(tokenized_gt, tokenized_gen)]
        metrics['bleu_score'] = float(np.mean(bleu_scores))
        return metrics

    def _compute_label_metrics(self, gt_list: List[List[int]], pred_list: List[List[int]]) -> Dict[str, float]:
        if not gt_list or not pred_list:
            return {}
        y_true = np.array(gt_list)
        y_pred = np.array(pred_list)
        return {
            'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'micro_precision': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'micro_recall': recall_score(y_true, y_pred, average='micro', zero_division=0)
        }

    def compute_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:
        def split_by_stage(preds):
            a = [p for p in preds if p.get('stage') == 'A']
            b = [p for p in preds if p.get('stage') == 'B']
            return a, b
        
        def collect_blocks(preds):
            gti, geni = [], []
            gt_cx, gen_cx = [], []
            gt_icd, gen_icd = [], []
            for p in preds:
                if p.get('ground_truth_impression') and p.get('generated_impression'):
                    gti.append(p['ground_truth_impression'])
                    geni.append(p['generated_impression'])
                if p.get('ground_truth_chexpert') and p.get('generated_chexpert'):
                    gt_cx.append([p['ground_truth_chexpert'].get(l, 0) for l in self.chexpert_labels])
                    gen_cx.append([p['generated_chexpert'].get(l, 0) for l in self.chexpert_labels])
                if p.get('ground_truth_icd') and p.get('generated_icd') and p['ground_truth_icd']:
                    gt_icd.append([p['ground_truth_icd'].get(l, 0) for l in self.icd_classes])
                    gen_icd.append([p['generated_icd'].get(l, 0) for l in self.icd_classes])
            return gti, geni, gt_cx, gen_cx, gt_icd, gen_icd
        
        metrics: Dict[str, Any] = {}
        # Overall
        gti, geni, gt_cx, gen_cx, gt_icd, gen_icd = collect_blocks(predictions)
        metrics.update({f"overall_{k}": v for k, v in self._compute_text_metrics(gti, geni).items()})
        cx = self._compute_label_metrics(gt_cx, gen_cx)
        for k, v in cx.items():
            metrics[f"overall_chexpert_{k}"] = v
        icd = self._compute_label_metrics(gt_icd, gen_icd)
        for k, v in icd.items():
            metrics[f"overall_icd_{k}"] = v
        
        # By stage
        a_preds, b_preds = split_by_stage(predictions)
        for label, subset in (('stageA', a_preds), ('stageB', b_preds)):
            gti, geni, gt_cx, gen_cx, gt_icd, gen_icd = collect_blocks(subset)
            txt = self._compute_text_metrics(gti, geni)
            for k, v in txt.items():
                metrics[f"{label}_{k}"] = v
            cx = self._compute_label_metrics(gt_cx, gen_cx)
            for k, v in cx.items():
                metrics[f"{label}_chexpert_{k}"] = v
            icd = self._compute_label_metrics(gt_icd, gen_icd)
            for k, v in icd.items():
                metrics[f"{label}_icd_{k}"] = v
        
        return metrics

    def save_metrics_to_csv(self, metrics: Dict[str, Any], output_csv_path: Path):
        """Saves computed metrics to a CSV file."""
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_csv_path, index=False)
        print(f"✅ Metrics saved to {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Simple Batch Evaluation for Radiology Report Model')
    parser.add_argument('--config', type=str, default="configs/advanced_training_config.yaml", help='Path to the training config file (unused)')
    parser.add_argument('--manifest', type=str, default="evaluation/demo_manifest.csv", help='Path to the demo manifest CSV file')
    parser.add_argument('--output_dir', type=str, default="evaluation/results", help='Directory to save batch predictions and metrics')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu|cuda|mps')
    
    args = parser.parse_args()
    
    print("📊 SIMPLE BATCH EVALUATION")
    print("=" * 50)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_evaluator = SimpleBatchEvaluator(args.config, device=args.device)
    
    print(f"📂 Evaluating samples from manifest: {args.manifest}")
    predictions = batch_evaluator.evaluate_batch(args.manifest, output_dir)
    
    print("\n🧮 Computing metrics...")
    metrics = batch_evaluator.compute_metrics(predictions)
    
    print("\n--- EVALUATION METRICS ---")
    for key, value in metrics.items():
        try:
            print(f"{key}: {value:.4f}")
        except Exception:
            print(f"{key}: {value}")
    
    batch_evaluator.save_metrics_to_csv(metrics, output_dir / "demo_metrics.csv")
    
    print("\n✅ BATCH EVALUATION COMPLETE")

if __name__ == '__main__':
    main()