# MIMIC-CXR Radiology Report Generation
Vision-Language Model with Curriculum Learning for Automated Radiology Report Generation

---

## 🚀 **PROJECT STATUS: EVALUATION & DEMO PREPARATION** 

**Current Status**: Model training completed successfully! Now in evaluation and demo preparation phase.

**Training Progress**: 100% complete (2 epoch, 270 steps) - Model ready for inference

**Key Achievement**: Successfully completed advanced curriculum learning with 4,360 training samples and 770 validation samples. Evaluation system built and demo dataset prepared.

---

## 📊 Project Overview

This project trains a vision-language model to generate radiology reports from chest X-ray images using curriculum learning with two stages:

- **Stage A (16.9%)**: Image-only warm-up → learns to generate Impression + CheXpert labels ✅ **COMPLETED**
- **Stage B (83.1%)**: Image+EHR training → adds clinical reasoning with patient context and ICD diagnoses ✅ **COMPLETED**

**Key Innovation**: Single model, staged training, fair A/B testing (same model, EHR ON/OFF at inference).

## 🎯 **TRAINING COMPLETED - EVALUATION PHASE**

### Training Results
- **Epochs Completed**: 2 (270 steps total)
- **Device**: CPU (MPS compatibility issues resolved)
- **Model**: LLaVA-Med v1.5-Mistral-7B with LoRA fine-tuning
- **Batch Size**: 1 (effective batch size: 32 with gradient accumulation)
- **Learning Rate**: 5.0e-5 (Stage A) → 3.0e-5 (Stage B)

### Curriculum Learning Results
- **Stage A**: 809 samples (16.9%) - Image-only training ✅ **COMPLETED**
- **Stage B**: 3,988 samples (83.1%) - Image+EHR training ✅ **COMPLETED**
- **Checkpoints**: Saved at steps 50 and 100
- **Validation**: 770 samples (150 Stage A + 620 Stage B) - Ready for evaluation

### Data Distribution
- **Training**: 4,360 samples (cleaned from original 4,797)
- **Validation**: 770 samples (cleaned from original 847)
- **Total Model Parameters**: 7.28B (41.9M trainable with LoRA)

## 🎯 **EVALUATION & DEMO STATUS**

### ✅ **Completed Components**
- **Model Training**: 100% complete (1 epoch, 100 steps)
- **Evaluation System**: Single + batch evaluation scripts ready
- **Demo Dataset**: 80 samples prepared (40 Stage A + 40 Stage B)
- **Model Checkpoints**: Saved at steps 50 and 100
- **EHR Integration**: 40 EHR JSON files generated for Stage B demo

### 🔄 **In Progress**
- **Metrics Computation**: Minor technical issues being resolved
- **Streamlit Demo**: A/B testing interface development

### ⏳ **Next 24 Hours**
- **Complete Streamlit Demo**: Interactive web interface
- **Model Optimization**: GPU acceleration for smooth demo
- **Final Testing**: Comprehensive testing across all scenarios

### 🎯 **Demo Capabilities**
- **Demo A**: Image-only → Impression + CheXpert labels
- **Demo B**: Image+EHR → Impression + CheXpert + ICD labels
- **A/B Testing**: Same model, toggle EHR ON/OFF at inference

---

## 🎯 Datasets

### Primary Training Data (FINAL CLEAN DATASET)
| File | Size | Records | Purpose | Location |
|------|------|---------|---------|----------|
| `curriculum_train_final_clean.jsonl` | 13.2 MB | 4,360 | **MAIN TRAINING DATA** | `src/data/processed/` |
| `curriculum_val_final_clean.jsonl` | 2.3 MB | 770 | **VALIDATION DATA** | `src/data/processed/` |

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

### Sample Images
- **Location**: `src/data/sample_images/`
- **Count**: 3 sample chest X-ray images
- **Size**: ~5.6 MB total (1.5-2.2 MB each)
- **Purpose**: Sample images for testing and demonstration

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

### Option A: Streamlit Demo (Recommended) ⭐ NEW!

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run the Streamlit Demo
streamlit run app_demo.py

# 3. Open browser to http://localhost:8501
# 4. Select Stage A (image-only) or Stage B (image+EHR)
# 5. Choose a sample and generate reports!
```

**Best for**: Interactive demo, A/B testing, easy visualization

### Option B: Programmatic Inference

```python
# 1. Import the pipeline
from src.inference.pipeline import generate

# 2. Stage A (image-only)
result = generate("path/to/xray.jpg")

# 3. Stage B (image+EHR)
ehr_data = {"Age": 65, "Sex": "M", "Vitals": {...}}
result = generate("path/to/xray.jpg", ehr_data)

# 4. Access results
print(result['impression'])
print(result['chexpert'])
print(result['icd'])
```

**Best for**: Integration into other applications, batch processing

### Option C: Use Ollama (Alternative)

```bash
# 1. Install Ollama (if not installed)
# Visit: https://ollama.ai or run: brew install ollama

# 2. Pull medical LLaVA model (~5 GB, pre-built for Mac)
ollama pull rohithbojja/llava-med-v1.6

# 3. Use for inference (no training needed!)
ollama run rohithbojja/llava-med-v1.6 --image path/to/xray.jpg
```

**Best for**: Quick testing without fine-tuned model

---

## 📁 Project Structure

```
radiology_report/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── app_demo.py                  # 🆕 Streamlit demo app
├── generate_demo_manifest.py    # Demo dataset generator
├── src/                         # Source code
│   ├── inference/               # 🆕 Inference pipeline
│   │   └── pipeline.py          # Main inference pipeline
│   ├── utils/                   # 🆕 Utility functions
│   │   └── load_finetuned_model.py # Model loader
│   ├── evaluation/              # Evaluation scripts
│   │   ├── eval_simple.py       # Single sample evaluation
│   │   └── eval_batch_simple.py # Batch evaluation (fixed)
│   ├── training/                # Training modules
│   │   ├── advanced_trainer.py  # Main trainer
│   │   └── dataset.py           # Dataset handling
│   └── data/processed/          # Processed training data
│       ├── curriculum_train_final_clean.jsonl   # Main training data (4,360 samples)
│       ├── curriculum_val_final_clean.jsonl     # Validation data (770 samples)
│       └── chexpert_dict.json                   # CheXpert labels mapping
├── evaluation/                  # 🆕 Demo evaluation data
│   ├── demo_manifest.csv        # Demo samples manifest
│   └── demo_ehr/                # EHR JSON files for Stage B
├── files/p10/                   # Chest X-ray images (10,003 JPGs)
├── backups/archived_scripts/    # 🆕 Obsolete scripts
│   ├── ehr_cxr_qc.py           # Archived data QC script
│   ├── migrate_data_with_fixes.py # Archived migration script
│   ├── eval_single.py          # Archived single evaluation
│   └── eval_batch.py           # Archived batch evaluation
└── configs/                     # Configuration files
    └── advanced_training_config.yaml
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
- **Training Data**: 4,360 clean training samples with curriculum learning
- **Data Quality**: 41.4% duplicates removed, 42.6% vitals coverage, 94.1% labs coverage
- **Environment**: Python 3.9.6 + PyTorch 2.8.0 + CPU optimization
- **Code Transfer**: All essential files transferred to remote server
- **Dependencies**: All packages installed and configured
- **Model Training**: 1 epoch completed (100 steps) ✅ **COMPLETED**
- **Evaluation System**: Fixed and functional ✅ **COMPLETED**
- **Demo Dataset**: 80 samples prepared (40 Stage A + 40 Stage B) ✅ **COMPLETED**
- **Streamlit Demo**: A/B testing interface ✅ **COMPLETED**
- **Model Loader**: One-liner model loading utility ✅ **COMPLETED**
- **Inference Pipeline**: Minimal I/O pipeline ✅ **COMPLETED**

### 🎉 **DEMO READY - ALL SYSTEMS GO!**
- **Model Training**: 100% complete with checkpoints saved
- **Evaluation System**: Fixed label mismatch and ICD conversion issues
- **Demo Dataset**: 80 samples ready for demonstration
- **Streamlit Demo**: Complete A/B testing interface with image+EHR support
- **Model Infrastructure**: Clean, modular codebase with easy model loading

### 📁 **Repository Cleanup**
- **Obsolete Scripts**: Moved to `backups/archived_scripts/`
- **Active Scripts**: Only essential files remain in main directory
- **Documentation**: Updated to reflect current status

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

**Last Updated**: October 19, 2024  
**Status**: 100% Training Complete - Evaluation & Demo Preparation Phase  
**Next Milestone**: Demo Presentation (October 20, 2024)  
**Repository**: [https://github.com/rahul370139/radiology_report](https://github.com/rahul370139/radiology_report)