#!/usr/bin/env python3
"""
Batch Evaluation Script for Radiology Report Model
Processes demo manifest and computes comprehensive metrics
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

# Import our single evaluator
import sys
import os
sys.path.append(os.path.dirname(__file__))
from eval_single import RadiologyModelEvaluator

class BatchEvaluator:
    """Batch evaluator for radiology report generation"""
    
    def __init__(self, model_path: str, lora_path: str):
        """Initialize batch evaluator"""
        self.evaluator = RadiologyModelEvaluator(model_path, lora_path)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.bleu_smoothing = SmoothingFunction().method1
        
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
        """Load the model"""
        self.evaluator.load_model()
    
    def load_manifest(self, manifest_path: str) -> pd.DataFrame:
        """Load demo manifest"""
        return pd.read_csv(manifest_path)
    
    def evaluate_sample(self, row: pd.Series) -> Dict[str, Any]:
        """Evaluate a single sample"""
        try:
            # Load EHR data if available
            ehr_data = None
            if pd.notna(row['ehr_json_path']) and row['ehr_json_path']:
                with open(row['ehr_json_path'], 'r') as f:
                    ehr_data = json.load(f)
            
            # Generate response
            response = self.evaluator.generate_response(row['image_path'], ehr_data)
            
            # Parse response
            parsed = self.evaluator.parse_response(response)
            
            # Compute metrics
            metrics = self.compute_metrics(row, parsed)
            
            return {
                'study_id': row['study_id'],
                'stage': row['stage'],
                'response': response,
                'parsed': parsed,
                'metrics': metrics,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'study_id': row['study_id'],
                'stage': row['stage'],
                'response': '',
                'parsed': {'impression': '', 'chexpert': {}, 'icd': {}},
                'metrics': {},
                'success': False,
                'error': str(e)
            }
    
    def compute_metrics(self, row: pd.Series, parsed: Dict[str, Any]) -> Dict[str, float]:
        """Compute evaluation metrics for a single sample"""
        metrics = {}
        
        # Impression metrics (ROUGE-L, BLEU)
        if pd.notna(row['ground_truth_impression']) and parsed['impression']:
            gt_impression = str(row['ground_truth_impression']).strip()
            pred_impression = parsed['impression'].strip()
            
            if gt_impression and pred_impression:
                # ROUGE-L
                rouge_scores = self.rouge_scorer.score(gt_impression, pred_impression)
                metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
                
                # BLEU
                try:
                    gt_tokens = gt_impression.split()
                    pred_tokens = pred_impression.split()
                    bleu_score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=self.bleu_smoothing)
                    metrics['bleu'] = bleu_score
                except:
                    metrics['bleu'] = 0.0
            else:
                metrics['rouge_l'] = 0.0
                metrics['bleu'] = 0.0
        else:
            metrics['rouge_l'] = 0.0
            metrics['bleu'] = 0.0
        
        # CheXpert metrics
        if pd.notna(row['ground_truth_chexpert']):
            try:
                gt_chexpert = eval(row['ground_truth_chexpert']) if isinstance(row['ground_truth_chexpert'], str) else row['ground_truth_chexpert']
                pred_chexpert = parsed['chexpert']
                
                # Per-class metrics
                for label in self.chexpert_labels:
                    gt_val = gt_chexpert.get(label, 0)
                    pred_val = pred_chexpert.get(label, 0)
                    metrics[f'chexpert_{label.lower().replace(" ", "_")}'] = 1.0 if gt_val == pred_val else 0.0
                
                # Overall CheXpert accuracy
                correct = sum(1 for label in self.chexpert_labels 
                            if gt_chexpert.get(label, 0) == pred_chexpert.get(label, 0))
                metrics['chexpert_accuracy'] = correct / len(self.chexpert_labels)
                
                # CheXpert F1 (micro)
                y_true = [gt_chexpert.get(label, 0) for label in self.chexpert_labels]
                y_pred = [pred_chexpert.get(label, 0) for label in self.chexpert_labels]
                metrics['chexpert_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
                
            except Exception as e:
                print(f"⚠️ Error computing CheXpert metrics: {e}")
                for label in self.chexpert_labels:
                    metrics[f'chexpert_{label.lower().replace(" ", "_")}'] = 0.0
                metrics['chexpert_accuracy'] = 0.0
                metrics['chexpert_f1'] = 0.0
        else:
            for label in self.chexpert_labels:
                metrics[f'chexpert_{label.lower().replace(" ", "_")}'] = 0.0
            metrics['chexpert_accuracy'] = 0.0
            metrics['chexpert_f1'] = 0.0
        
        # ICD metrics (Stage B only)
        if row['stage'] == 'B' and pd.notna(row['ground_truth_icd']):
            try:
                gt_icd = eval(row['ground_truth_icd']) if isinstance(row['ground_truth_icd'], str) else row['ground_truth_icd']
                pred_icd = parsed['icd']
                
                # Convert ICD labels to binary format
                gt_icd_binary = {}
                if isinstance(gt_icd, list):
                    # Convert from list of dicts format
                    for item in gt_icd:
                        if isinstance(item, dict) and 'code' in item:
                            # Map ICD codes to our classes (simplified mapping)
                            code = item['code']
                            confidence = item.get('confidence', 0)
                            if confidence > 0.5:  # Threshold for positive
                                if 'J12' in code or 'J13' in code or 'J14' in code or 'J15' in code or 'J16' in code or 'J17' in code or 'J18' in code:
                                    gt_icd_binary['Pneumonia'] = 1
                                elif 'J94' in code:
                                    gt_icd_binary['Pleural_Effusion'] = 1
                                elif 'J93' in code:
                                    gt_icd_binary['Pneumothorax'] = 1
                                elif 'J81' in code:
                                    gt_icd_binary['Pulmonary_Edema'] = 1
                                elif 'I51' in code:
                                    gt_icd_binary['Cardiomegaly'] = 1
                                elif 'J98' in code:
                                    gt_icd_binary['Atelectasis'] = 1
                                elif 'I26' in code:
                                    gt_icd_binary['Pulmonary_Embolism'] = 1
                                elif 'S22' in code:
                                    gt_icd_binary['Rib_Fracture'] = 1
                elif isinstance(gt_icd, dict):
                    gt_icd_binary = gt_icd
                
                # Per-class metrics
                for label in self.icd_classes:
                    gt_val = gt_icd_binary.get(label, 0)
                    pred_val = pred_icd.get(label, 0)
                    metrics[f'icd_{label.lower()}'] = 1.0 if gt_val == pred_val else 0.0
                
                # Overall ICD accuracy
                correct = sum(1 for label in self.icd_classes 
                            if gt_icd_binary.get(label, 0) == pred_icd.get(label, 0))
                metrics['icd_accuracy'] = correct / len(self.icd_classes)
                
                # ICD F1 (micro)
                y_true = [gt_icd_binary.get(label, 0) for label in self.icd_classes]
                y_pred = [pred_icd.get(label, 0) for label in self.icd_classes]
                metrics['icd_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
                
            except Exception as e:
                print(f"⚠️ Error computing ICD metrics: {e}")
                for label in self.icd_classes:
                    metrics[f'icd_{label.lower()}'] = 0.0
                metrics['icd_accuracy'] = 0.0
                metrics['icd_f1'] = 0.0
        else:
            for label in self.icd_classes:
                metrics[f'icd_{label.lower()}'] = 0.0
            metrics['icd_accuracy'] = 0.0
            metrics['icd_f1'] = 0.0
        
        return metrics
    
    def evaluate_batch(self, manifest_path: str, output_dir: str = "evaluation/outputs") -> Dict[str, Any]:
        """Evaluate all samples in manifest"""
        print("🔬 BATCH EVALUATION")
        print("=" * 40)
        
        # Load manifest
        manifest = self.load_manifest(manifest_path)
        print(f"📊 Loaded {len(manifest)} samples from {manifest_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Evaluate each sample
        results = []
        successful = 0
        failed = 0
        
        for idx, row in manifest.iterrows():
            print(f"🔄 Processing {row['study_id']} ({idx+1}/{len(manifest)})...")
            
            result = self.evaluate_sample(row)
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
                print(f"   ❌ Failed: {result['error']}")
        
        print(f"\n📊 Evaluation complete: {successful} successful, {failed} failed")
        
        # Save detailed results
        results_file = output_path / "detailed_results.jsonl"
        with open(results_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"💾 Detailed results saved to {results_file}")
        
        # Compute aggregate metrics
        aggregate_metrics = self.compute_aggregate_metrics(results)
        
        # Save metrics
        metrics_file = output_path / "demo_metrics.csv"
        self.save_metrics(aggregate_metrics, metrics_file)
        print(f"📈 Metrics saved to {metrics_file}")
        
        return aggregate_metrics
    
    def compute_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate metrics across all samples"""
        # Filter successful results
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {'error': 'No successful evaluations'}
        
        # Separate by stage
        stage_a_results = [r for r in successful_results if r['stage'] == 'A']
        stage_b_results = [r for r in successful_results if r['stage'] == 'B']
        
        aggregate = {
            'total_samples': len(results),
            'successful_samples': len(successful_results),
            'failed_samples': len(results) - len(successful_results),
            'stage_a_samples': len(stage_a_results),
            'stage_b_samples': len(stage_b_results)
        }
        
        # Overall metrics
        if successful_results:
            metrics_list = [r['metrics'] for r in successful_results]
            for metric in ['rouge_l', 'bleu', 'chexpert_accuracy', 'chexpert_f1']:
                values = [m.get(metric, 0) for m in metrics_list if metric in m]
                if values:
                    aggregate[f'overall_{metric}'] = np.mean(values)
                    aggregate[f'overall_{metric}_std'] = np.std(values)
        
        # Stage A metrics
        if stage_a_results:
            stage_a_metrics = [r['metrics'] for r in stage_a_results]
            for metric in ['rouge_l', 'bleu', 'chexpert_accuracy', 'chexpert_f1']:
                values = [m.get(metric, 0) for m in stage_a_metrics if metric in m]
                if values:
                    aggregate[f'stage_a_{metric}'] = np.mean(values)
                    aggregate[f'stage_a_{metric}_std'] = np.std(values)
        
        # Stage B metrics
        if stage_b_results:
            stage_b_metrics = [r['metrics'] for r in stage_b_results]
            for metric in ['rouge_l', 'bleu', 'chexpert_accuracy', 'chexpert_f1', 'icd_accuracy', 'icd_f1']:
                values = [m.get(metric, 0) for m in stage_b_metrics if metric in m]
                if values:
                    aggregate[f'stage_b_{metric}'] = np.mean(values)
                    aggregate[f'stage_b_{metric}_std'] = np.std(values)
        
        return aggregate
    
    def save_metrics(self, metrics: Dict[str, Any], output_file: str):
        """Save metrics to CSV"""
        # Flatten metrics for CSV
        flat_metrics = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                flat_metrics.append({'metric': key, 'value': value})
        
        df = pd.DataFrame(flat_metrics)
        df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Batch Evaluation for Radiology Report Model')
    parser.add_argument('--manifest', required=True, help='Path to demo manifest CSV')
    parser.add_argument('--model_path', default='microsoft/llava-med-v1.5-mistral-7b', help='Base model path')
    parser.add_argument('--lora_path', default='checkpoints', help='LoRA adapter path')
    parser.add_argument('--output_dir', default='evaluation/outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BatchEvaluator(args.model_path, args.lora_path)
    
    # Load model
    evaluator.load_model()
    
    # Run evaluation
    metrics = evaluator.evaluate_batch(args.manifest, args.output_dir)
    
    # Print summary
    print("\n📊 EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Total samples: {metrics.get('total_samples', 0)}")
    print(f"Successful: {metrics.get('successful_samples', 0)}")
    print(f"Failed: {metrics.get('failed_samples', 0)}")
    print(f"Stage A: {metrics.get('stage_a_samples', 0)}")
    print(f"Stage B: {metrics.get('stage_b_samples', 0)}")
    
    if 'overall_rouge_l' in metrics:
        print(f"\nOverall ROUGE-L: {metrics['overall_rouge_l']:.3f} ± {metrics.get('overall_rouge_l_std', 0):.3f}")
        print(f"Overall BLEU: {metrics['overall_bleu']:.3f} ± {metrics.get('overall_bleu_std', 0):.3f}")
        print(f"Overall CheXpert F1: {metrics['overall_chexpert_f1']:.3f} ± {metrics.get('overall_chexpert_f1_std', 0):.3f}")
    
    if 'stage_b_icd_f1' in metrics:
        print(f"Stage B ICD F1: {metrics['stage_b_icd_f1']:.3f} ± {metrics.get('stage_b_icd_f1_std', 0):.3f}")

if __name__ == "__main__":
    main()
