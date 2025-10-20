# Evaluation Guide and Analysis

This document consolidates evaluation instructions and the latest analysis.

## How to Run

- Single sample: `python src/evaluation/eval_simple.py --image <path> [--ehr_json <path>]`
- Batch: `python src/evaluation/eval_batch_simple.py --manifest evaluation/demo_manifest.csv`
- A/B (Stage B only): `python src/evaluation/eval_ab.py --ckpt checkpoints --val src/data/processed/curriculum_val_final_clean.jsonl`

Outputs are saved under `evaluation/results/`.

## Metrics

- ROUGE-1/2/L, BLEU for impression
- Micro-F1 for CheXpert (12 labels)
- Micro-F1 for ICD (8 curated classes)

## Latest Analysis (Summary)

- Low ROUGE/BLEU, zero CheXpert/ICD F1 were observed previously because a placeholder evaluator returned identical predictions.
- Refactored evaluators now use the real inference pipeline.
- If scores remain low, verify that LoRA weights load from `checkpoints/`.

## Troubleshooting

- Ensure `checkpoints/` contains adapter weights; otherwise base model is used (lower quality).
- Verify `evaluation/demo_manifest.csv` paths exist.
- For CPU runs, generation will be slower; reduce sample size when testing.
