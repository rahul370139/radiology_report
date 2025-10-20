#!/bin/bash
# Script to sync local changes to server and run evaluation
# Usage: ./deployment/run_evaluation_on_server.sh [SERVER_USER@SERVER_IP]

set -e  # Exit on any error

# Server details
SERVER="${1:-bilbouser@100.77.217.18}"
PROJECT_DIR="radiology_report"

echo "🚀 SYNCING AND RUNNING EVALUATION ON SERVER"
echo "=============================================="
echo "Server: $SERVER"
echo "Project: $PROJECT_DIR"
echo ""

# Step 0: Skip local test (model not available locally)
echo "🧪 Step 0: Skipping local test (model on server only)"
echo ""

# Step 1: Sync local changes to server
echo "📤 Step 1: Syncing local changes to server..."
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
    "$SERVER:~/radiology_report/"

echo "✅ Local changes synced to server"
echo ""

# Step 2: Run evaluation on server
echo "🔬 Step 2: Running evaluation on server..."
ssh "$SERVER" << 'EOF'
cd ~/radiology_report

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
elif [ -d "env" ]; then
    source env/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️ No virtual environment found, using system Python"
fi

# Install required packages if not present
pip install pandas numpy scikit-learn rouge-score nltk --quiet

# Set environment variables for model paths (adjust these paths for your server)
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/Users/bilbouser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/f2f72301dc934e74948b5802c87dbc83d100e6bd}"
export LORA_DIR="${LORA_DIR:-checkpoints}"
export GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.05}"
export GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-512}"
export GEN_TOP_P="${GEN_TOP_P:-0.9}"
export GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-false}"

echo "🔧 Environment variables:"
echo "   BASE_MODEL_PATH: $BASE_MODEL_PATH"
echo "   LORA_DIR: $LORA_DIR"
echo "   GEN_TEMPERATURE: $GEN_TEMPERATURE"
echo "   GEN_MAX_NEW_TOKENS: $GEN_MAX_NEW_TOKENS"
echo "   GEN_TOP_P: $GEN_TOP_P"
echo "   GEN_DO_SAMPLE: $GEN_DO_SAMPLE"

# Create results directory
mkdir -p evaluation/results

echo "🤖 Starting batch evaluation (CPU)..."
echo "Manifest: evaluation/demo_manifest.csv"
echo "Output: evaluation/results/"
echo ""

# Run the batch evaluation
python src/evaluation/eval_batch_simple.py \
    --manifest evaluation/demo_manifest.csv \
    --output_dir evaluation/results \
    --device cpu

echo ""
echo "✅ Batch evaluation completed!"
echo ""

# Run A/B evaluation for Stage B samples
echo "🔬 Starting A/B evaluation (Stage B with/without EHR)..."
python src/evaluation/eval_ab.py \
    --val src/data/processed/curriculum_val_final_clean.jsonl \
    --device cpu \
    --output_dir evaluation/results

echo ""
echo "✅ A/B evaluation completed!"
echo ""

# Show results summary
echo "📊 RESULTS SUMMARY:"
echo "==================="
if [ -f "evaluation/results/demo_metrics.csv" ]; then
    echo "Batch Metrics:"
    cat evaluation/results/demo_metrics.csv
    echo ""
fi

if [ -f "evaluation/results/ab_macro_metrics.csv" ]; then
    echo "A/B Macro Metrics:"
    cat evaluation/results/ab_macro_metrics.csv
    echo ""
fi

if [ -f "evaluation/results/batch_predictions.jsonl" ]; then
    echo "Sample predictions (first 2):"
    head -n 2 evaluation/results/batch_predictions.jsonl | python -m json.tool
    echo ""
fi

echo "📁 Output files:"
ls -la evaluation/results/
EOF

echo ""
echo "🎉 EVALUATION COMPLETE!"
echo "======================="
echo "Results are now on the server in ~/radiology_report/evaluation/results/"
echo ""
echo "To download results to your local machine:"
echo "scp $SERVER:~/radiology_report/evaluation/results/* ./evaluation/results/"
echo ""
echo "To run locally with server models (if accessible):"
echo "export BASE_MODEL_PATH=/path/to/server/models/llava-med-v1.5-mistral-7b"
echo "export LORA_DIR=/path/to/server/checkpoints"
echo "streamlit run app_demo.py"
