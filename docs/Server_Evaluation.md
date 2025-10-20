# Manual Server Evaluation Guide

## Step 1: Update the server script with your server details

Edit `deployment/run_evaluation_on_server.sh` and replace:
```bash
SERVER="${1:-your_user@your_server_ip}"
```
with your actual server details, e.g.:
```bash
SERVER="${1:-rahul@192.168.1.100}"
```

## Step 2: Run the automated script

```bash
cd "/Users/rahul/Downloads/Code scripts/radiology_report"
./deployment/run_evaluation_on_server.sh
```

---

## Alternative: Manual Step-by-Step

If you prefer to run commands manually:

### 1. Sync your local changes to server
```bash
rsync -avz --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='venv' \
    --exclude='checkpoints' \
    --exclude='files' \
    --exclude='logs' \
    "/Users/rahul/Downloads/Code scripts/radiology_report/" \
    your_user@your_server_ip:~/radiology_report/
```

### 2. SSH into server and run evaluation
```bash
ssh your_user@your_server_ip
cd ~/radiology_report
source venv/bin/activate  # or conda activate radrep
pip install pandas numpy scikit-learn rouge-score nltk --quiet

# Set model paths
export BASE_MODEL_PATH="~/models/llava-med-v1.5-mistral-7b"
export LORA_DIR="checkpoints"

# Run evaluation
python src/evaluation/eval_batch_simple.py \
    --manifest evaluation/demo_manifest.csv \
    --output_dir evaluation/results \
    --config configs/advanced_training_config.yaml
```

### 3. Check results
```bash
# View metrics
cat evaluation/results/demo_metrics.csv

# View sample predictions
head -n 2 evaluation/results/batch_predictions.jsonl | python -m json.tool
```

### 4. Download results to your laptop
```bash
# From your laptop
scp your_user@your_server_ip:~/radiology_report/evaluation/results/* ./evaluation/results/
```

---

## What the evaluation will do:

1. Load the fine-tuned model from `checkpoints/` on the server
2. Process demo samples
3. Generate predictions for each sample
4. Compute metrics (ROUGE, BLEU, CheXpert and ICD micro metrics)
5. Save results to `evaluation/results/`

---

## Expected output files:

- `evaluation/results/batch_predictions.jsonl` - Predictions per sample
- `evaluation/results/demo_metrics.csv` - Summary metrics table

---

## Troubleshooting:

- "Module not found": ensure virtualenv is active
- "Model not found": check `checkpoints/` exists on server
- "CUDA OOM": add `--device cpu` to force CPU
- "Permission denied": `chmod +x deployment/run_evaluation_on_server.sh`
