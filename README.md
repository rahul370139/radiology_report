# MIMIC-CXR Radiology Report Generation
Vision-Language Model with Curriculum Learning for Automated Radiology Report Generation

---

## 📊 Project Overview

This project trains a vision-language model to generate radiology reports from chest X-ray images using curriculum learning with two stages:

- **Stage A (35%)**: Image-only warm-up → learns to generate Impression + CheXpert labels
- **Stage B (65%)**: Image+EHR training → adds clinical reasoning with patient context and ICD diagnoses

**Key Innovation**: Single model, staged training, fair A/B testing (same model, EHR ON/OFF at inference).

---

## 🎯 Datasets

### Primary Training Data
| File | Size | Records | Purpose |
|------|------|---------|---------|
| `curriculum_train.jsonl` | 9.5 MB | 10,142 | **MAIN TRAINING DATA** |
| `curriculum_val.jsonl` | 1.1 MB | 1,125 | **VALIDATION DATA** |

### Reference Data
| File | Size | Records | Purpose |
|------|------|---------|---------|
| `ehr_context.jsonl` | 2.2 MB | 5,441 | EHR reference (embedded in curriculum) |
| `chexpert_dict.json` | 67.8 MB | 227,827 | Label lookup (embedded in curriculum) |

### Image Data
- **Location**: `files/p10/`
- **Count**: 10,003 chest X-ray JPG images
- **Paths**: Already embedded in curriculum samples

---

## 📚 Data Structure

### Stage A: Image-Only (5,243 samples)
```json
{
  "image": "files/p10/.../image.jpg",
  "prompt": "Image:<image>\nAnswer Impression & CheXpert.",
  "target": "Impression: ...\nCheXpert: {...}",
  "mode": "image_only",
  "stage": "A"
}
```

### Stage B: Image+EHR (4,899 samples)
```json
{
  "image": "files/p10/.../image.jpg",
  "prompt": "Patient Data: {Age, Sex, Vitals...}\nImage:<image>\nAnswer Impression, CheXpert & ICD.",
  "target": "Impression: ...\nCheXpert: {...}\nICD: {...}",
  "mode": "image_ehr",
  "stage": "B"
}
```

**Note**: Same images used twice (once for each stage) - this is intentional curriculum learning!

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
├── data/processed/              # Processed training data
│   ├── curriculum_train.jsonl   # Main training data
│   ├── curriculum_val.jsonl     # Validation data
│   ├── ehr_context.jsonl        # EHR reference
│   └── chexpert_dict.json       # Label lookup
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
- **Stage A**: 5,243 samples (image-only)
- **Stage B**: 4,899 samples (image+EHR)
- **Total**: 10,142 training samples

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
- **CPU**: 32 cores
- **RAM**: 512 GB
- **Storage**: 20 GB project data transferred
- **Acceleration**: MPS (Metal Performance Shaders)
- **Status**: ✅ Ready for training

### Data Transfer Status
- **Images**: 10,003 chest X-rays (18 GB) ✅ Complete
- **Training Data**: 81 MB processed datasets ✅ Complete
- **Code**: All training modules ✅ Complete
- **Environment**: Python 3.9.6 + PyTorch 2.8.0 ✅ Complete

---

## 🚨 Current Status & Issues

### ✅ Completed (95%)
- **Infrastructure Setup**: Remote Apple M3 Ultra Mac Studio
- **Data Processing**: Complete MIMIC-CXR dataset processing pipeline
- **Training Data**: 10,142 training samples with curriculum learning
- **Environment**: Python 3.9.6 + PyTorch 2.8.0 + MPS acceleration
- **Code Transfer**: All essential files transferred to remote server
- **Dependencies**: All packages installed and configured

### ⚠️ Current Issues
- **Compatibility**: transformers/PEFT version mismatch needs resolution
- **Model Download**: LLaVA-Med model weights need to be downloaded

### ⏳ Next Steps
1. Fix transformers/PEFT compatibility issues
2. Download LLaVA-Med model weights
3. Run initial training test
4. Begin fine-tuning process

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

**Last Updated**: October 4, 2024  
**Status**: 95% Complete - Ready for training  
**Repository**: [https://github.com/rahul370139/radiology_report](https://github.com/rahul370139/radiology_report)