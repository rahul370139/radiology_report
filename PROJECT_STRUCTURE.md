# Project Structure

This document outlines the organized structure of the Radiology Report Generation project.

## 📁 Directory Structure

```
radiology_report/
├── README.md                           # Main project documentation
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── configs/                           # Configuration files
│   ├── advanced_training_config.yaml  # Main training configuration
│   └── model_config.yaml             # Model-specific settings
│
├── src/                               # Source code
│   ├── models/                        # Model definitions and architectures
│   │   ├── llava_med.py              # LLaVA-Med model implementation
│   │   └── curriculum_model.py       # Curriculum learning model
│   │
│   ├── data/                          # Data processing modules
│   │   ├── dataset.py                # Dataset classes
│   │   ├── preprocessing.py          # Data preprocessing
│   │   └── curriculum_builder.py     # Curriculum learning builder
│   │
│   ├── training/                      # Training modules
│   │   ├── advanced_trainer.py       # Main training orchestrator
│   │   ├── curriculum_trainer.py     # Curriculum learning trainer
│   │   ├── dataset.py                # Dataset handling
│   │   └── metrics.py                # Evaluation metrics
│   │
│   ├── utils/                         # Utility functions
│   │   ├── accelerator_patch.py      # Compatibility patches
│   │   ├── deploy_to_server.py       # Server deployment
│   │   ├── download_llava_med.py     # Model download utility
│   │   ├── quick_compatibility_check.py # Environment checks
│   │   └── smoke_test.py             # Basic functionality tests
│   │
│   └── evaluation/                    # Evaluation modules
│       ├── validator.py              # Model validation
│       ├── metrics_calculator.py     # Metrics calculation
│       └── performance_analyzer.py   # Performance analysis
│
├── scripts/                           # Executable scripts
│   ├── setup/                         # Setup scripts
│   │   └── setup_environment.sh      # Environment setup
│   │
│   ├── training/                      # Training scripts
│   │   └── start_training.sh         # Training launcher
│   │
│   └── deployment/                    # Deployment scripts
│       └── deploy_to_server.py       # Server deployment
│
└── src/data/                          # Data directory
    ├── processed/                     # Processed datasets (mirror)
    │   ├── curriculum_train_final_clean.jsonl  # Main training data
    │   ├── curriculum_val_final_clean.jsonl    # Validation data
    │   ├── sample_curriculum_train.jsonl       # Sample training data
    │   ├── sample_curriculum_val.jsonl         # Sample validation data
    │   ├── impressions.jsonl                   # Raw impressions
    │   ├── phaseA_manifest.jsonl               # Phase A manifest
    │   └── chexpert_dict.json                  # CheXpert labels
    │
    ├── raw_reports/                   # Sample raw reports
    │   ├── sample_report_1.txt        # Sample radiology report 1
    │   ├── sample_report_2.txt        # Sample radiology report 2
    │   ├── sample_report_3.txt        # Sample radiology report 3
    │   └── sample_report_4.txt        # Sample radiology report 4
    │
    └── sample_images/                 # Sample chest X-ray images
        ├── sample_xray_1.jpg          # Sample chest X-ray 1 (~1.5 MB)
        ├── sample_xray_2.jpg          # Sample chest X-ray 2 (~1.7 MB)
        └── sample_xray_3.jpg          # Sample chest X-ray 3 (~2.2 MB)
│
├── docs/                              # Documentation
│   ├── api/                           # API documentation
│   ├── deployment/                    # Deployment guides
│   └── examples/                      # Usage examples
│
├── updates/                           # Project updates
│   ├── PROJECT_STATUS.md             # Current project status
│   └── TECHNICAL_STATUS_REPORT.md    # Detailed technical report
│
├── checkpoints/                       # Model checkpoints (not in git)
├── logs/                              # Training logs (not in git)
└── venv/                              # Virtual environment (not in git)
```

## 🔧 **Key Files Description**

### Configuration Files (`configs/`)
- **`advanced_training_config.yaml`**: Main training configuration with all hyperparameters
- **`model_config.yaml`**: Model-specific settings and architecture parameters

### Source Code (`src/`)

#### Models (`src/models/`)
- **`llava_med.py`**: LLaVA-Med model implementation and loading
- **`curriculum_model.py`**: Curriculum learning model wrapper

#### Data (`src/data/`)
- **`dataset.py`**: Dataset classes for training and validation
- **`preprocessing.py`**: Data preprocessing and cleaning utilities
- **`curriculum_builder.py`**: Curriculum learning data preparation

#### Training (`src/training/`)
- **`advanced_trainer.py`**: Main training orchestrator with curriculum learning
- **`curriculum_trainer.py`**: Specialized curriculum learning trainer
- **`metrics.py`**: Evaluation metrics and performance calculation

#### Utils (`src/utils/`)
- **`accelerator_patch.py`**: Compatibility patches for different versions
- **`deploy_to_server.py`**: Server deployment utilities
- **`download_llava_med.py`**: Model download and setup utilities

### Scripts (`scripts/`)
- **`setup/setup_environment.sh`**: Environment setup and dependency installation
- **`training/start_training.sh`**: Training launcher with proper environment setup
- **`deployment/deploy_to_server.py`**: Production deployment scripts

### Data (`data/`)
- **`processed/`**: All processed and cleaned datasets
- **`raw/`**: Raw data files (excluded from git)

## 🚀 **Usage Patterns**

### Training
```bash
# Start training
./scripts/training/start_training.sh

# Or directly
python src/training/advanced_trainer.py --config configs/advanced_training_config.yaml
```

### Data Processing
```bash
# Process raw data
python src/data/preprocessing.py --input data/raw/ --output data/processed/

# Build curriculum
python src/data/curriculum_builder.py --config configs/curriculum_config.yaml
```

### Evaluation
```bash
# Run validation
python src/evaluation/validator.py --model checkpoints/best_model --data data/processed/curriculum_val_final_clean.jsonl
```

## 📋 **Development Guidelines**

1. **Code Organization**: Keep related functionality in appropriate modules
2. **Configuration**: Use YAML files for all configuration parameters
3. **Documentation**: Update this file when adding new modules
4. **Testing**: Add tests for new functionality
5. **Git**: Keep data files and checkpoints out of version control

## 🔄 **Migration Notes**

This structure was reorganized from the original flat structure to improve:
- **Maintainability**: Clear separation of concerns
- **Scalability**: Easy to add new modules
- **Documentation**: Self-documenting structure
- **Deployment**: Clear separation of scripts and source code

**Last Updated**: October 10, 2025
