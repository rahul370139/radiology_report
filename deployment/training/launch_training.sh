#!/bin/bash
# Launch training with MPS/CPU safety and Mac wake prevention

echo "🚀 LAUNCHING RADIOLOGY REPORT TRAINING"
echo "======================================"

# MPS/CPU safety environment variables
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_ALLOCATOR_POLICY=expandable_segments

# Keep Mac awake during training
echo "🔄 Starting caffeinate to keep Mac awake..."
caffeinate -dimsu -w $$ &

# Change to project directory
cd "$(dirname "$0")/../.."

# Create necessary directories
mkdir -p checkpoints logs

# Launch training with auto-resume
echo "🎯 Starting training with auto-resume..."
echo "📊 Checkpoints every 10 steps"
echo "📈 Evaluation every 50 steps"
echo ""

python src/training/advanced_trainer.py \
    --config configs/advanced_training_config.yaml \
    --resume auto

echo ""
echo "✅ Training completed or stopped"
echo "📁 Checkpoints saved in: checkpoints/"
echo "📊 Logs available in: logs/"
