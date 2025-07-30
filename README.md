# Machine Unlearning for Intrusion Detection Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![NSL-KDD](https://img.shields.io/badge/Dataset-NSL--KDD-green.svg)](https://www.unb.ca/cic/datasets/nsl.html)

A comprehensive implementation of state-of-the-art machine unlearning algorithms for intrusion detection systems, featuring advanced data poisoning attacks and robust defense mechanisms.

## 🚀 Features

### 🛡️ **Intrusion Detection**
- **NSL-KDD Dataset**: Enhanced version of KDD Cup 99 for network intrusion detection
- **10+ ML Algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Neural Networks
- **GPU Acceleration**: CUDA support for XGBoost, LightGBM, and CatBoost
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC with visualizations

### 🧠 **Machine Unlearning (2025 State-of-the-Art)**
- **SISA (Sharded, Isolated, Sliced, and Aggregated)**: Efficient unlearning through model sharding
- **Gradient Ascent Unlearning**: Direct optimization to remove data influence
- **Ensemble Unlearning**: Robust forgetting through algorithmic diversity
- **🆕 AMDU (Adaptive Memory Distillation Unlearning)**: Our novel algorithm combining memory networks with knowledge distillation

### 🎯 **Data Poisoning Attacks**
- **Label Noise**: Random label flipping attacks
- **Feature Noise**: Gaussian noise injection
- **Adversarial Samples**: Targeted perturbations with label flipping
- **Backdoor Attacks**: Hidden trigger insertion

### 📊 **Experimental Framework**
- **Automated Benchmarking**: Complete pipeline for model comparison
- **Performance Recovery Analysis**: Quantifies unlearning effectiveness
- **Visualization Tools**: ROC curves, performance plots, recovery analysis

## 📋 Quick Start

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (optional, for acceleration)
- Conda or pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/vinhqdang/ids_unlearning.git
cd ids_unlearning
```

2. **Set up environment**
```bash
# Using Conda (recommended)
conda env create -f environment.yml
conda activate ids_unlearning

# Or using pip
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python test_basic.py
```

## 🔧 Usage

### Basic Classification Benchmark
```bash
# Run basic scikit-learn models
python run_basic_models.py

# Full model suite with GPU acceleration
python main.py
```

### Machine Unlearning Experiments
```bash
# Quick demonstration (10k samples, ~5 minutes)
python quick_unlearning_test.py

# Full experiment (125k samples, ~30 minutes)
python unlearning_experiment.py

# Aggressive poisoning scenarios
python test_aggressive_poisoning.py
```

### Visualization
```bash
# Generate ROC curves for all models
python plot_roc_curves.py
```

## 📊 Results

### Model Performance (NSL-KDD Binary Classification)

| Model | Accuracy | F1-Score | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| **Gradient Boosting** | 0.8064 | **0.8005** | **0.9625** | 13.0s |
| **Random Forest** | 0.7707 | 0.7544 | 0.9620 | 0.7s |
| **Neural Network** | 0.7849 | 0.7721 | 0.9123 | 39.4s |
| **Logistic Regression** | 0.7539 | 0.7407 | 0.8715 | 1.7s |

### Unlearning Effectiveness

**AMDU vs Existing Methods (NSL-KDD Dataset):**

| Method | Avg F1-Score | Avg Recovery | Time (s) | Status |
|--------|--------------|--------------|----------|---------|
| **🆕 AMDU** | **0.7778** | **+3.96%** | 5.90 | **Best Overall** |
| SISA | 0.7571 | +1.89% | 1.57 | Good |
| Gradient Ascent | 0.7552 | +1.69% | 0.67 | Fast |
| Simple Retraining | 0.7470 | +0.87% | 0.36 | Baseline |

**Performance Recovery under Data Poisoning:**

| Scenario | AMDU F1 | Best Baseline | AMDU Advantage |
|----------|----------|---------------|----------------|
| **10% Label Noise** | 0.7739 | 0.7613 (GA) | **+1.26%** |
| **20% Label Noise** | 0.7920 | 0.7639 (GA) | **+2.81%** |
| **15% Feature Noise** | 0.7677 | 0.7578 (SISA) | **+0.99%** |

*AMDU consistently outperforms all existing methods across multiple attack scenarios*

## 🏗️ Project Structure

```
ids_unlearning/
├── src/
│   ├── data_loader.py          # NSL-KDD dataset processing
│   ├── classifiers.py          # ML classification algorithms
│   ├── data_poisoner.py        # Poisoning attack implementations
│   ├── unlearning_algorithms.py # Existing unlearning methods
│   └── amdu_unlearning.py      # 🆕 AMDU algorithm implementation
├── research/
│   ├── novel_algorithm_design.md # AMDU design document
│   └── amdu_paper_draft.md      # Research paper draft
├── data/
│   ├── KDDTrain_plus.txt      # Training dataset
│   └── KDDTest_plus.txt       # Testing dataset
├── results/                    # Experimental results and plots
├── main.py                    # Full benchmark pipeline
├── quick_unlearning_test.py   # Fast unlearning demo
├── test_amdu_quick.py         # 🆕 AMDU quick test
├── amdu_comparison_test.py    # 🆕 Comprehensive comparison
└── environment.yml            # Conda environment
```

## 🔬 Key Algorithms

### 🆕 AMDU (Our Novel Algorithm)
```python
from src.amdu_unlearning import AMDUUnlearner

# Initialize with memory networks and adversarial validation
amdu = AMDUUnlearner(
    input_dim=X_train.shape[1],
    memory_dim=64,
    learning_rate=0.001,
    device='cuda'  # GPU acceleration
)

# Fit with teacher model
amdu.fit(X_train_poisoned, y_train_poisoned, teacher_model=original_model)

# Perform selective unlearning
amdu.unlearn(X_train_poisoned, y_train_poisoned, poison_indices)

# Evaluate forgetting effectiveness
effectiveness = amdu.evaluate_forgetting_effectiveness(X_forgotten)
```

### SISA Unlearning
```python
from src.unlearning_algorithms import SISAUnlearner

sisa = SISAUnlearner(base_model, n_shards=5)
sisa.fit(X_train, y_train)
sisa.unlearn(X_train, y_train, poison_indices)
```

### Gradient Ascent Unlearning
```python
from src.unlearning_algorithms import GradientAscentUnlearner

ga_unlearner = GradientAscentUnlearner(model, learning_rate=0.01)
ga_unlearner.fit(X_train, y_train)
ga_unlearner.unlearn(X_forget, y_forget, X_retain, y_retain)
```

### Data Poisoning
```python
from src.data_poisoner import DataPoisoner

poisoner = DataPoisoner()
X_poisoned, y_poisoned, poison_idx = poisoner.add_label_noise(
    X_train, y_train, noise_ratio=0.1
)
```

## 📈 Performance Highlights

- **🎯 High Accuracy**: Up to 96.25% ROC-AUC on NSL-KDD dataset
- **⚡ GPU Acceleration**: 5-10x speedup with CUDA-enabled models
- **🛡️ Robust Unlearning**: 81-103% performance recovery even with 40% data poisoning
- **📊 Comprehensive Evaluation**: ROC curves, confusion matrices, detailed metrics

## 🔧 Configuration

### GPU Support
The project automatically detects and uses GPU acceleration when available:
- **XGBoost**: `tree_method='gpu_hist'`
- **LightGBM**: `device='gpu'`
- **CatBoost**: `task_type='GPU'`

### Hyperparameters
Key parameters can be adjusted in each script:
- **Noise ratios**: `[0.05, 0.10, 0.15, 0.20]`
- **SISA shards**: `n_shards=5`
- **Learning rate**: `learning_rate=0.01`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citations

If you use this work in your research, please cite:

```bibtex
@misc{ids_unlearning_2025,
  title={Machine Unlearning for Intrusion Detection Systems},
  author={Vinh, DQ},
  year={2025},
  url={https://github.com/vinhqdang/ids_unlearning}
}
```

### Dataset Citation
```bibtex
@misc{nsl-kdd,
  title={NSL-KDD dataset},
  author={Tavallaee, Mahbod and Bagheri, Ebrahim and Lu, Wei and Ghorbani, Ali A},
  year={2009},
  url={https://www.unb.ca/cic/datasets/nsl.html}
}
```

## 🙏 Acknowledgments

- **NSL-KDD Dataset**: University of New Brunswick
- **SISA Algorithm**: Inspired by "Machine Unlearning" by Bourtoule et al.
- **Gradient-based Unlearning**: Based on recent advances in algorithmic stability
- **Claude AI**: Code generation and optimization assistance

## 📞 Contact

- **Author**: Vinh DQ
- **Email**: dqvinh87@gmail.com
- **GitHub**: [@vinhqdang](https://github.com/vinhqdang)

## 🚨 Disclaimer

This project is for research and educational purposes only. The implemented poisoning attacks are designed to test defense mechanisms and should not be used maliciously. Always follow ethical guidelines and legal requirements when conducting security research.

---

⭐ **Star this repository if you find it useful!** ⭐