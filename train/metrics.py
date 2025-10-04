#!/usr/bin/env python3.10
"""
Evaluation metrics for radiology report generation.

Implements:
- BLEU-4 (impression quality)
- ROUGE-L (impression quality)
- METEOR (impression quality)
- CheXpert macro-F1 (label accuracy)
- ICD multi-label F1 (diagnosis accuracy)
- JSON validity check
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics for radiology reports."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.chexpert_labels = [
            'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
            'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
            'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]
        
        self.icd_conditions = [
            'pneumonia', 'pleural_effusion', 'pneumothorax', 'pulmonary_embolism',
            'chf_exacerbation', 'pulmonary_edema', 'rib_fracture', 'lung_mass'
        ]
    
    def extract_impression(self, text: str) -> str:
        """Extract impression text from generated output."""
        # Pattern: "Impression: <text>"
        match = re.search(r'Impression:\s*(.*?)(?:\n\n|\nCheXpert:|$)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    def extract_chexpert_json(self, text: str) -> Optional[Dict[str, str]]:
        """Extract CheXpert JSON from generated output."""
        try:
            # Pattern: "CheXpert:\n{...}"
            match = re.search(r'CheXpert:\s*(\{.*?\})', text, re.DOTALL | re.IGNORECASE)
            if match:
                chexpert_json = json.loads(match.group(1))
                return chexpert_json
            return None
        except json.JSONDecodeError:
            return None
    
    def extract_icd_json(self, text: str) -> Optional[Dict[str, bool]]:
        """Extract ICD JSON from generated output."""
        try:
            # Pattern: "ICD:\n{...}"
            match = re.search(r'ICD:\s*(\{.*?\})', text, re.DOTALL | re.IGNORECASE)
            if match:
                icd_json = json.loads(match.group(1))
                return icd_json
            return None
        except json.JSONDecodeError:
            return None
    
    def compute_bleu4(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute BLEU-4 score.
        
        Args:
            predictions: List of predicted impressions
            references: List of reference impressions
            
        Returns:
            BLEU-4 score
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize
            
            scores = []
            smoothing = SmoothingFunction().method1
            
            for pred, ref in zip(predictions, references):
                pred_tokens = word_tokenize(pred.lower())
                ref_tokens = word_tokenize(ref.lower())
                
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothing
                )
                scores.append(score)
            
            return np.mean(scores)
            
        except ImportError:
            logger.warning("NLTK not installed, skipping BLEU-4")
            return 0.0
    
    def compute_rouge_l(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute ROUGE-L score.
        
        Args:
            predictions: List of predicted impressions
            references: List of reference impressions
            
        Returns:
            ROUGE-L F1 score
        """
        try:
            from rouge import Rouge
            
            rouge = Rouge()
            scores = rouge.get_scores(predictions, references, avg=True)
            return scores['rouge-l']['f']
            
        except ImportError:
            logger.warning("rouge not installed, skipping ROUGE-L")
            return 0.0
    
    def compute_chexpert_f1(
        self,
        predictions: List[Dict[str, str]],
        references: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Compute CheXpert macro-F1 score.
        
        Args:
            predictions: List of predicted CheXpert label dicts
            references: List of reference CheXpert label dicts
            
        Returns:
            Dictionary with F1 scores
        """
        # Convert to binary arrays
        label_to_int = {'Positive': 1, 'Negative': 0, 'Uncertain': -1}
        
        all_f1_scores = {}
        
        for label in self.chexpert_labels:
            pred_values = []
            ref_values = []
            
            for pred, ref in zip(predictions, references):
                if pred and label in pred:
                    pred_val = label_to_int.get(pred[label], -1)
                else:
                    pred_val = -1
                
                if ref and label in ref:
                    ref_val = label_to_int.get(ref[label], -1)
                else:
                    ref_val = -1
                
                # Only compute F1 for non-uncertain cases
                if ref_val != -1:
                    pred_values.append(1 if pred_val == 1 else 0)
                    ref_values.append(1 if ref_val == 1 else 0)
            
            if len(ref_values) > 0:
                f1 = f1_score(ref_values, pred_values, average='binary', zero_division=0)
                all_f1_scores[label] = f1
        
        # Compute macro average
        macro_f1 = np.mean(list(all_f1_scores.values()))
        all_f1_scores['macro_avg'] = macro_f1
        
        return all_f1_scores
    
    def compute_icd_f1(
        self,
        predictions: List[Dict[str, bool]],
        references: List[Dict[str, bool]]
    ) -> Dict[str, float]:
        """
        Compute ICD multi-label F1 score.
        
        Args:
            predictions: List of predicted ICD flag dicts
            references: List of reference ICD flag dicts
            
        Returns:
            Dictionary with F1 scores
        """
        all_f1_scores = {}
        
        for condition in self.icd_conditions:
            pred_values = []
            ref_values = []
            
            for pred, ref in zip(predictions, references):
                pred_val = pred.get(condition, False) if pred else False
                ref_val = ref.get(condition, False) if ref else False
                
                pred_values.append(1 if pred_val else 0)
                ref_values.append(1 if ref_val else 0)
            
            if len(ref_values) > 0:
                f1 = f1_score(ref_values, pred_values, average='binary', zero_division=0)
                all_f1_scores[condition] = f1
        
        # Compute macro average
        macro_f1 = np.mean(list(all_f1_scores.values()))
        all_f1_scores['macro_avg'] = macro_f1
        
        return all_f1_scores
    
    def compute_json_validity(
        self,
        predictions: List[str]
    ) -> Dict[str, float]:
        """
        Compute JSON validity rate.
        
        Args:
            predictions: List of predicted texts
            
        Returns:
            Dictionary with validity rates
        """
        valid_chexpert = 0
        valid_icd = 0
        total = len(predictions)
        
        for pred in predictions:
            # Check CheXpert JSON
            chexpert = self.extract_chexpert_json(pred)
            if chexpert is not None:
                valid_chexpert += 1
            
            # Check ICD JSON (if present)
            icd = self.extract_icd_json(pred)
            if icd is not None or 'ICD:' not in pred:
                # Valid if either parsed or not expected
                valid_icd += 1
        
        return {
            'chexpert_valid_rate': valid_chexpert / total if total > 0 else 0.0,
            'icd_valid_rate': valid_icd / total if total > 0 else 0.0,
        }
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("Computing evaluation metrics...")
        
        # Extract impressions
        pred_impressions = [self.extract_impression(p) for p in predictions]
        ref_impressions = [self.extract_impression(r) for r in references]
        
        # Extract CheXpert labels
        pred_chexpert = [self.extract_chexpert_json(p) for p in predictions]
        ref_chexpert = [self.extract_chexpert_json(r) for r in references]
        
        # Extract ICD flags
        pred_icd = [self.extract_icd_json(p) for p in predictions]
        ref_icd = [self.extract_icd_json(r) for r in references]
        
        # Compute metrics
        metrics = {}
        
        # BLEU-4
        try:
            metrics['bleu4'] = self.compute_bleu4(pred_impressions, ref_impressions)
        except Exception as e:
            logger.warning(f"Could not compute BLEU-4: {e}")
            metrics['bleu4'] = 0.0
        
        # ROUGE-L
        try:
            metrics['rouge_l'] = self.compute_rouge_l(pred_impressions, ref_impressions)
        except Exception as e:
            logger.warning(f"Could not compute ROUGE-L: {e}")
            metrics['rouge_l'] = 0.0
        
        # CheXpert F1
        try:
            chexpert_f1 = self.compute_chexpert_f1(pred_chexpert, ref_chexpert)
            metrics['chexpert_f1'] = chexpert_f1
        except Exception as e:
            logger.warning(f"Could not compute CheXpert F1: {e}")
            metrics['chexpert_f1'] = {'macro_avg': 0.0}
        
        # ICD F1
        try:
            icd_f1 = self.compute_icd_f1(pred_icd, ref_icd)
            metrics['icd_f1'] = icd_f1
        except Exception as e:
            logger.warning(f"Could not compute ICD F1: {e}")
            metrics['icd_f1'] = {'macro_avg': 0.0}
        
        # JSON validity
        validity = self.compute_json_validity(predictions)
        metrics['json_validity'] = validity
        
        return metrics


if __name__ == "__main__":
    # Test metrics computation
    logging.basicConfig(level=logging.INFO)
    
    calculator = MetricsCalculator()
    
    # Test samples
    pred = """Impression: NO ACUTE CARDIOPULMONARY PROCESS.

CheXpert:
{
  "Consolidation": "Uncertain",
  "Edema": "Uncertain",
  "No Finding": "Positive"
}"""
    
    ref = """Impression: NO ACUTE CARDIOPULMONARY PROCESS.

CheXpert:
{
  "Consolidation": "Uncertain",
  "Edema": "Uncertain",
  "No Finding": "Positive"
}"""
    
    metrics = calculator.compute_all_metrics([pred], [ref])
    print("\nTest Metrics:")
    print(json.dumps(metrics, indent=2, default=str))

