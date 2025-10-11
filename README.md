# MIMIC-CXR Radiology Report Generation
Vision-Language Model with Curriculum Learning for Automated Radiology Report Generation

---

## 🚀 **PROJECT STATUS: VALIDATION PHASE** 

**Current Status**: Model training completed 100% of curriculum learning (424/424 Stage A batches + 179/179 Stage B batches). Currently running final validation phase on validation set.

**Validation Phase**: Running comprehensive evaluation on 847 validation samples to assess model performance.

**Key Achievement**: Successfully implemented advanced curriculum learning with 4,797 training samples and 847 validation samples.

---

## 📊 Project Overview

This project trains a vision-language model to generate radiology reports from chest X-ray images using curriculum learning with two stages:

- **Stage A (35%)**: Image-only warm-up → learns to generate Impression + CheXpert labels ✅ **COMPLETED**
- **Stage B (65%)**: Image+EHR training → adds clinical reasoning with patient context and ICD diagnoses 🔄 **COMPLETED**

**Key Innovation**: Single model, staged training, fair A/B testing (same model, EHR ON/OFF at inference).

---

## 🎯 Datasets

### Primary Training Data (FINAL CLEAN DATASET)
| File | Size | Records | Purpose | Location |
|------|------|---------|---------|----------|
| `curriculum_train_final_clean.jsonl` | 14.2 MB | 4,797 | **MAIN TRAINING DATA** | `src/data/processed/` |
| `curriculum_val_final_clean.jsonl` | 2.5 MB | 847 | **VALIDATION DATA** | `src/data/processed/` |

### Reference Data
| File | Size | Records | Purpose | Location |
|------|------|---------|---------|----------|
| `chexpert_dict.json` | 67.8 MB | 227,827 | CheXpert labels mapping | `src/data/processed/` |
| `impressions.jsonl` | 6.0 MB | 10,003 | Raw impressions (reference) | `src/data/processed/` |
| `phaseA_manifest.jsonl` | 3.9 MB | 10,003 | Phase A manifest (reference) | `src/data/processed/` |

### Image Data
- **Location**: `files/p10/`
- **Count**: 10,003 chest X-ray JPG images
- **Paths**: Already embedded in curriculum samples

### Sample Raw Reports
- **Location**: `src/data/raw_reports/`
- **Count**: 4 sample radiology reports
- **Purpose**: Reference examples of original MIMIC-CXR reports

### Data Quality & Deduplication
- **Original Dataset**: 9,638 samples
- **Final Clean Dataset**: 5,644 samples
- **Duplicates Removed**: 3,994 samples (41.4% waste eliminated)
- **EHR Coverage**: 42.6% vitals, 94.1% labs (Stage B)

---

## 📚 Data Structure

### Stage A: Image-Only (959 samples)
```json
{
  "image_path": "files/p10/.../image.jpg",
  "impression": "1. APPROPRIATE POSITIONING...",
  "chexpert_labels": {"Consolidation": -1, "Edema": 1, ...},
  "stage": "A"
}
```

### Stage B: Image+EHR (4,685 samples)
```json
{
  "image_path": "files/p10/.../image.jpg",
  "impression": "1. APPROPRIATE POSITIONING...",
  "chexpert_labels": {"Consolidation": -1, "Edema": 1, ...},
  "patient_data": {
    "subject_id": 10020944,
    "Age": 72,
    "Sex": "M",
    "Vitals": {"heart_rate": {...}, "o2_saturation": {...}},
    "Labs": {"Sodium": {...}, "Creatinine": {...}},
    "O2_device": "Oxygen_Device: 40",
    "Chronic_conditions": []
  },
  "stage": "B"
}
```

**Note**: Each image used only once (deduplicated) - efficient curriculum learning!

---

## 🚀 Quick Start

### Option A: Use Ollama (Easiest for Mac) ⭐ RECOMMENDED

```bash
# 1. Install Ollama (if not installed)
# Visit: https://ollama.ai or run: brew install ollama

# 2. Pull medical LLaVA model (~5 GB, pre-built for Mac)
ollama pull rohithbojja/llava-med-v1.6

# 3. Test the model
python test_ollama_model.py

# 4. Use for inference (no training needed!)
ollama run rohithbojja/llava-med-v1.6 --image path/to/xray.jpg
```

**Best for**: Local deployment, demo, inference on Mac

### Option B: Train Custom Model (For GPU)

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Verify Setup
python test_training_setup.py

# 3. Start Training
python train.py --config train/config.yaml
# Or: accelerate launch train.py --config train/config.yaml
```

**Best for**: Custom training on your data with GPU

---

## 📁 Project Structure

```
radiology_report/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── train.py                     # Main training script
├── test_training_setup.py       # Setup verification
├── download_10k.py              # Data download utility
├── train/                       # Training modules
│   ├── config.yaml              # Training configuration
│   ├── dataset.py               # Dataset handling
│   ├── trainer.py               # Training logic
│   └── metrics.py               # Evaluation metrics
├── data/processed/                           # Processed training data
│   ├── curriculum_train_final_clean.jsonl   # Main training data (4,797 samples)
│   ├── curriculum_val_final_clean.jsonl     # Validation data (847 samples)
│   ├── chexpert_dict.json                   # CheXpert labels mapping
│   ├── impressions.jsonl                    # Raw impressions (reference)
│   └── phaseA_manifest.jsonl                # Phase A manifest (reference)
├── files/p10/                   # Chest X-ray images (10,003 JPGs)
├── updates/                     # Project status updates
│   ├── PROJECT_STATUS.md         # Quick status overview
│   └── TECHNICAL_STATUS_REPORT.md # Detailed technical report
└── [data processing scripts]    # 01_*.py, 02_*.py, etc.
```

---

## 🔧 Requirements

### System Requirements
- **Python**: 3.9+
- **RAM**: 16GB+ (32GB+ recommended)
- **Storage**: 25GB+ free space
- **GPU**: CUDA-compatible (optional, for training)

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- `torch>=2.0.0` - PyTorch framework
- `transformers>=4.40.0` - Hugging Face transformers
- `peft>=0.8.0` - Parameter Efficient Fine-Tuning
- `accelerate>=0.20.0` - Training acceleration
- `datasets>=2.14.0` - Dataset handling
- `pillow>=9.0.0` - Image processing

---

## 📊 Model Architecture

**Base Model**: microsoft/llava-med-v1.5-mistral-7b
- **Vision Encoder**: CLIP ViT-L/14
- **Language Model**: Mistral-7B
- **Projection Layer**: Image-to-text mapping
- **Parameters**: ~7B total

**Training Approach**:
- **Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `mm_projector`
- **Rank**: 64
- **Alpha**: 128

---

## 🎯 Training Configuration

### Key Parameters
- **Batch Size**: 4 (per device)
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Warmup Steps**: 100
- **Gradient Accumulation**: 4
- **Mixed Precision**: FP16

### Curriculum Learning
- **Stage A**: 959 samples (image-only)
- **Stage B**: 4,685 samples (image+EHR)
- **Total**: 5,644 training samples
- **Deduplication**: 41.4% waste eliminated

---

## 📈 Evaluation Metrics

- **BLEU Score**: Text generation quality
- **ROUGE Score**: Summarization quality
- **CheXpert Accuracy**: Label prediction accuracy
- **ICD Accuracy**: Diagnosis code accuracy
- **Clinical Relevance**: Domain expert evaluation

---

## 🔄 Data Processing Pipeline

1. **Image Extraction**: Download chest X-ray images from MIMIC-CXR
2. **Report Processing**: Extract impressions and structured data
3. **CheXpert Mapping**: Map findings to standardized labels
4. **EHR Integration**: Add patient context and ICD codes
5. **Curriculum Creation**: Generate staged training samples
6. **Validation Split**: Create holdout test set

---

## 🖥️ Infrastructure Status

### Remote Training Environment
- **Hardware**: Apple M3 Ultra Mac Studio
- **CPU**: 32 cores (Currently using CPU for stability)
- **RAM**: 512 GB
- **Storage**: 20 GB project data transferred
- **Acceleration**: CPU-optimized (MPS disabled for compatibility)
- **Status**: ✅ **ACTIVE TRAINING**

### Data Transfer Status
- **Images**: 10,003 chest X-rays (18 GB) ✅ Complete
- **Training Data**: 81 MB processed datasets ✅ Complete
- **Code**: All training modules ✅ Complete
- **Environment**: Python 3.9.6 + PyTorch 2.8.0 ✅ Complete

### Training Progress
- **Stage A**: 424/424 batches (100%) ✅ **COMPLETED**
- **Stage B**: 179/179 batches (100%) ✅ **COMPLETED**
- **Total Progress**: 100% training complete, now in validation phase

---

## 🚨 Current Status & Issues

### ✅ Completed (100%)
- **Infrastructure Setup**: Remote Apple M3 Ultra Mac Studio
- **Data Processing**: Complete MIMIC-CXR dataset processing pipeline
- **Training Data**: 4,797 clean training samples with curriculum learning
- **Data Quality**: 41.4% duplicates removed, 42.6% vitals coverage, 94.1% labs coverage
- **Environment**: Python 3.9.6 + PyTorch 2.8.0 + CPU optimization
- **Code Transfer**: All essential files transferred to remote server
- **Dependencies**: All packages installed and configured
- **Stage A Training**: 424/424 batches completed (100%)
- **Stage B Training**: 179/179 batches completed 100%)

### 🔄 **CURRENTLY IN VALIDATION PHASE**
- **Stage B Training**: 179/179 batches completed (100%)
- **Validation Set**: Running evaluation on 847 validation samples
- **Performance Metrics**: Generating comprehensive evaluation report
- **Status**: All training phases completed successfully

### ⏳ Final Steps
1. Complete validation evaluation on 847 samples
2. Generate performance metrics and evaluation report
3. Deploy model for inference testing
4. Prepare production deployment package

---

## 📞 Support

### Common Issues
- **CUDA Out of Memory**: Reduce batch size or use gradient checkpointing
- **Model Download**: Ensure stable internet connection
- **Data Access**: Verify MIMIC-CXR credentials

### Getting Help
- Check `updates/PROJECT_STATUS.md` for current status
- Review `updates/TECHNICAL_STATUS_REPORT.md` for detailed technical info
- Run `python test_training_setup.py` to verify environment

---

## 📄 License

This project uses the MIMIC-CXR dataset, which requires institutional access and data use agreement.

---

## 🙏 Acknowledgments

- **MIMIC-CXR**: Chest X-ray dataset
- **LLaVA-Med**: Medical vision-language model
- **Hugging Face**: Transformers library
- **Microsoft**: Base model architecture

---

**Last Updated**: October 10, 2025  
**Status**: 100% Training Complete - In Validation Phase  
**Repository**: [https://github.com/rahul370139/radiology_report](https://github.com/rahul370139/radiology_report)