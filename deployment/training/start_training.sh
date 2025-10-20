#!/bin/bash
# Advanced Training Script for Mac Studio Server with Time Tracking

echo "🚀 STARTING ADVANCED RADIOLOGY TRAINING"
echo "======================================="
echo "Start time: $(date)"
echo ""

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Check GPU availability
echo "🔧 SYSTEM CHECK"
echo "==============="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
echo ""

# Create logs directory
mkdir -p logs

# Run training with time tracking
echo "🏃 STARTING TRAINING"
echo "==================="
echo "Training start: $(date)"
echo ""

# Run training and capture output
python train/advanced_trainer.py --config advanced_training_config.yaml --debug 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ TRAINING COMPLETED SUCCESSFULLY!"
    echo "End time: $(date)"
else
    echo ""
    echo "❌ TRAINING FAILED!"
    echo "End time: $(date)"
    exit 1
fi
