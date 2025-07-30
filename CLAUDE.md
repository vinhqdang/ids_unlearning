# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an intrusion detection system project using machine learning algorithms to classify network traffic as normal or malicious using the NSL-KDD dataset. The project focuses on building multiple classification models for comparative analysis and eventually implementing unlearning algorithms.

## Environment Setup

### Conda Environment
```bash
# Create and activate environment
conda env create -f environment.yml
source ~/anaconda3/bin/activate ids_unlearning

# Alternative activation (if conda init is configured)
conda activate ids_unlearning
```

### GPU Support
The project is configured to use GPU acceleration where available:
- XGBoost: Uses `tree_method='gpu_hist'`
- LightGBM: Uses `device='gpu'` 
- CatBoost: Uses `task_type='GPU'`
- System has NVIDIA A5000 GPU with CUDA support

## Common Commands

### Data Preparation
```bash
# Test basic data loading
python test_basic.py

# Download datasets (already included in data/)
# Training: data/KDDTrain_plus.txt  
# Testing: data/KDDTest_plus.txt
```

### Model Training and Evaluation
```bash
# Run basic scikit-learn models
python run_basic_models.py

# Run full model suite (including GPU-accelerated models)
python main.py
```

### Results
Results are saved to `results/` directory:
- `classification_results.csv` - Performance metrics for all models
- `basic_models_results.csv` - Results from basic scikit-learn models
- `model_comparison.png` - Visualization of model performance
- `roc_curves.png` - ROC curves for all models

### Visualization
```bash
# Generate ROC curves for all models
python plot_roc_curves.py
```

## Code Architecture

### Core Modules
- `src/data_loader.py` - NSL-KDD dataset loading and preprocessing
- `src/classifiers.py` - Multiple ML classification algorithms 
- `main.py` - Main benchmark script
- `run_basic_models.py` - Basic models without GPU dependencies

### Dataset Information
- **NSL-KDD**: Improved version of KDD Cup 99 dataset
- **Features**: 41 numerical features after preprocessing
- **Labels**: Binary classification (normal vs attack) or multi-class
- **Training samples**: 125,973
- **Testing samples**: 22,544

### Model Performance (Binary Classification)
Current best performers on basic models:
1. Gradient Boosting: F1-Score 0.8005, ROC-AUC 0.9625
2. Neural Network: F1-Score 0.7721, ROC-AUC 0.9123
3. Naive Bayes: F1-Score 0.7675, ROC-AUC 0.8298

**Top ROC-AUC Performers:**
1. Gradient Boosting: ROC-AUC 0.9625
2. Random Forest: ROC-AUC 0.9620
3. Neural Network: ROC-AUC 0.9123

## Data Processing Pipeline

1. **Loading**: Read CSV files with proper feature names
2. **Encoding**: Label encoding for categorical features (protocol_type, service, flag)
3. **Scaling**: StandardScaler for numerical features  
4. **Target**: Binary (normal=0, attack=1) or multi-class encoding

## Machine Unlearning Implementation

### Unlearning Algorithms
The project implements 2025 state-of-the-art unlearning algorithms:

1. **SISA (Sharded, Isolated, Sliced, and Aggregated)**
   - Maintains multiple model shards for efficient unlearning
   - Only retrains affected shards containing poisoned data
   - Excellent for large-scale deployments

2. **Gradient Ascent Unlearning**
   - Direct optimization to remove data influence
   - Works with differentiable models (LogisticRegression)
   - Fine-grained control over forgetting process

3. **Ensemble Unlearning**
   - Combines multiple unlearning approaches
   - Provides robustness through diversity

### Data Poisoning Tests
Multiple poisoning attacks implemented:
- **Label Noise**: Flips labels randomly (simulates annotation errors)
- **Feature Noise**: Adds Gaussian noise to features
- **Adversarial Samples**: Creates targeted perturbations
- **Backdoor Attacks**: Inserts hidden triggers

### Unlearning Commands
```bash
# Quick unlearning demonstration (10k samples)
python quick_unlearning_test.py

# Full unlearning experiment (all data, longer runtime)
python unlearning_experiment.py
```

### Unlearning Results
**Recent experiment results (10k subset, 10% noise):**

| Model | Poison Type | Baseline F1 | Poisoned F1 | Best Method | Recovery F1 | Recovery Rate |
|-------|-------------|-------------|-------------|-------------|-------------|---------------|
| Logistic Regression | Label Noise | 0.7363 | 0.7290 | Gradient Ascent | 0.7567 | 376% |
| Random Forest | Label Noise | 0.7723 | 0.7730 | - | 0.7730 | 0% |

**Key Findings:**
- **Gradient Ascent** most effective for logistic regression models
- **SISA** provides good balance of efficiency and effectiveness
- Random Forest naturally robust to label noise
- Feature noise has minimal impact on well-regularized models

## Future Development

- Implement differential privacy-based unlearning
- Add cross-validation and hyperparameter tuning
- Implement advanced ensemble methods
- Add model explainability features
- Performance optimization for large-scale deployment
- Extend to multi-class unlearning scenarios