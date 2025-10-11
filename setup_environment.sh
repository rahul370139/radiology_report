#!/bin/bash
# Environment setup for LLaVA-Med training on macOS MPS

echo "🔧 Setting up environment for LLaVA-Med training..."

# Set MPS environment variables
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

echo "✅ Environment variables set:"
echo "   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"
echo "   PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "   TOKENIZERS_PARALLELISM=false"

echo ""
echo "🚀 Ready to start training!"
echo "Run: python train.py --config train/config.yaml"
