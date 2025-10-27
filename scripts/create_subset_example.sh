#!/bin/bash
# Example script to create a subset of images for Colab upload

echo "🚀 Creating image subset for Colab upload..."

# Set your paths here
SRC_ROOT="/Users/rahul/Downloads/Code scripts/radiology_report"  # Full dataset root
DST_ROOT="subset_images"  # Output directory
ZIP_FILE="subset_images.zip"  # Final zip file

# Create subset using symlinks (saves disk space)
python scripts/collect_referenced_images.py \
    --jsonl src/data/processed/curriculum_train_final_clean.jsonl \
             src/data/processed/curriculum_val_final_clean.jsonl \
    --src-root "$SRC_ROOT" \
    --dst-root "$DST_ROOT" \
    --mode symlink \
    --zip-out "$ZIP_FILE"

echo "✅ Subset created: $ZIP_FILE"
echo "📦 Upload this file to Colab and unzip to /content/radiology_report"
echo "🔧 Then set image_root in configs/advanced_training_v16.yaml to '/content/radiology_report'"
