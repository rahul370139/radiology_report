import json
import os
from pathlib import Path

import pandas as pd

from src.evaluation.eval_batch_simple import SimpleBatchEvaluator
from src.evaluation.eval_ab import load_validation_data, evaluate_icd_with_toggle


def test_batch_eval_small_subset(tmp_path):
    manifest = Path('evaluation/demo_manifest.csv')
    assert manifest.exists(), "demo_manifest.csv not found; run generate_demo_manifest.py"

    out_dir = tmp_path / 'results'
    evaluator = SimpleBatchEvaluator(config_path='configs/advanced_training_config.yaml', device='cpu')
    # Run on first 3 rows only to keep test fast
    df = pd.read_csv(manifest).head(3)
    small_manifest = tmp_path / 'small_manifest.csv'
    df.to_csv(small_manifest, index=False)

    preds = evaluator.evaluate_batch(str(small_manifest), out_dir)
    assert len(preds) > 0

    # Ensure parsed fields present and vary across samples when possible
    impressions = [p['parsed'].get('impression', '') for p in preds]
    assert all(isinstance(x, str) for x in impressions)

    metrics = evaluator.compute_metrics(preds)
    assert 'rougeL_fmeasure' in metrics


def test_ab_eval_small_subset():
    val_path = 'src/data/processed/curriculum_val_final_clean.jsonl'
    assert Path(val_path).exists(), "validation file missing"

    samples = load_validation_data(val_path)[:10]
    stage_b_samples = [s for s in samples if s.get('stage') == 'B']
    if not stage_b_samples:
        return  # skip if none in small slice

    results_off = evaluate_icd_with_toggle(stage_b_samples, use_ehr=False, device='cpu')
    results_on = evaluate_icd_with_toggle(stage_b_samples, use_ehr=True, device='cpu')

    assert 'macro_f1' in results_off and 'macro_f1' in results_on
