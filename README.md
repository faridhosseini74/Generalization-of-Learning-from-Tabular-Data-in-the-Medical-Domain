# 🩺 Glucose Prediction with Transformer Models

A comprehensive deep learning framework for glucose prediction using state-of-the-art transformer architectures. This repository provides a unified interface for training and evaluating multiple transformer models on tabular time-series glucose data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Available Models](#available-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architectures](#model-architectures)
- [Usage Examples](#usage-examples)
- [Model Selection Guide](#model-selection-guide)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a unified framework for glucose prediction using advanced transformer architectures. It supports multiple state-of-the-art models specifically adapted for tabular time-series data, making it ideal for continuous glucose monitoring (CGM) data analysis and prediction.

### Key Highlights

- **7 Transformer Architectures**: SAINT, FT-Transformer, TabPFN, PatchTST, TST, Informer, TabNet
- **Traditional ML Models**: Support for XGBoost, LightGBM, CatBoost, Random Forest, and MLP
- **Unified Interface**: Single API for all models with consistent input/output formats
- **Production Ready**: Comprehensive error handling, logging, and model checkpointing
- **Flexible Configuration**: Easy hyperparameter tuning and model customization

## ✨ Features

🚀 **Universal Dataset Support**
- Automatically detects dataset structure (patient ID, timestamp, glucose columns)
- Works with any CSV or Parquet file containing glucose time series data
- No need for hardcoded dataset configurations

🧠 **Multiple Transformer Models**
- **SAINT**: Self-Attention and Intersample Attention Transformer for tabular data
- **FT-Transformer**: Feature Tokenizer Transformer optimized for mixed data types
- **TabPFN**: Prior-Fitted Networks with meta-learning for small datasets
- **PatchTST**: Patch-based Time Series Transformer for long sequences
- **TST**: Time Series Transformer with learnable positional encoding
- **Informer**: Efficient transformer with ProbSparse attention for scalability
- **TabNet**: Attention-based tabular network with interpretable feature selection

🔍 **Model Explainability**
- Integrated SHAP analysis for all model architectures
- Automatic generation of interpretability plots
- Feature importance rankings and visualizations

⚡ **Intelligent Feature Discovery**
- Automatically identifies categorical and continuous features
- Extracts temporal features (hour, day of week, cyclic features)
- Smart cardinality-based feature classification

## 🤖 Available Models

### Transformer Models

| Model | Description | Best For | Paper |
|-------|-------------|----------|-------|
| **SAINT** | Self-Attention and Intersample Attention Network | Tabular + sequential data | [Link](https://arxiv.org/abs/2106.01342) |
| **FT-Transformer** | Feature Tokenizer Transformer | Mixed tabular features | [Link](https://arxiv.org/abs/2106.11959) |
| **TabPFN** | Prior-Fitting Transformer | Small datasets | [Link](https://arxiv.org/abs/2207.01848) |
| **PatchTST** | Patching Time Series Transformer | Long sequences | [Link](https://arxiv.org/abs/2211.14730) |
| **TST** | Time Series Transformer | Temporal patterns | [Link](https://arxiv.org/abs/2010.02803) |
| **Informer** | Efficient Transformer with ProbSparse | Large sequences | [Link](https://arxiv.org/abs/2012.07436) |
| **TabNet** | Attentive Interpretable Tabular Learning | Feature selection | [Link](https://arxiv.org/abs/1908.07442) |

### Traditional ML Models

- **XGBoost** - Gradient boosting framework
- **LightGBM** - Fast gradient boosting
- **CatBoost** - Categorical boosting
- **Random Forest** - Ensemble learning
- **MLP** - Multi-layer Perceptron

## 🔧 Installation

### Requirements

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/glucose-prediction.git
cd glucose-prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
from model import ModelFactory

# Create a model
model = ModelFactory.create_model(
    model_name='saint',
    categories=[24, 60, 7, 31, 12, 2, 4],  # Categorical feature dimensions
    num_continuous=6,  # Number of continuous features
    dim=128,  # Model dimension
    depth=6,  # Number of layers
    heads=8  # Number of attention heads
)

# Forward pass
predictions = model(x_categorical, x_continuous)
```

### Training with Different Models

```bash
# Train SAINT model
python train.py --mode individual --data_path your_dataset.csv --model saint --epochs 100

# Train FT-Transformer
python train.py --mode individual --data_path your_dataset.csv --model ft-transformer --epochs 100

# Train PatchTST
python train.py --mode individual --data_path your_dataset.csv --model patchtst --epochs 100

# Train with SHAP analysis
python train.py --mode individual --data_path your_dataset.csv --model saint --epochs 100 --shap
```

### Get Model Recommendations

```python
from model import ModelFactory

# Get recommendations based on your data
recommendations = ModelFactory.get_recommendations(
    sequence_length=24,  # Number of timesteps
    num_features=13,  # Total features
    dataset_size=50000,  # Number of samples
    has_missing_values=True
)

for model, reason in recommendations.items():
    print(f"{model}: {reason}")
```

## 📊 Model Selection Guide

### By Dataset Size

| Dataset Size | Recommended Models |
|--------------|-------------------|
| < 10,000 samples | TabPFN, TabNet |
| 10,000 - 100,000 | SAINT, FT-Transformer, TST |
| > 100,000 samples | Informer, PatchTST, SAINT |

### By Sequence Length

| Sequence Length | Recommended Models |
|-----------------|-------------------|
| < 10 timesteps | TabNet, FT-Transformer |
| 10 - 50 timesteps | SAINT, TST |
| > 50 timesteps | Informer, PatchTST |

### By Feature Type

| Feature Type | Recommended Models |
|--------------|-------------------|
| Mostly categorical | FT-Transformer, SAINT |
| Mostly continuous | TST, Informer, PatchTST |
| Mixed features | SAINT, FT-Transformer |
| High-dimensional | TabNet, SAINT |

## Dataset Requirements

Your dataset should contain:
- **Patient/Subject ID column**: Unique identifier for each patient
- **Timestamp column**: Time information (various formats supported)
- **Glucose column**: Glucose measurements (mg/dL or mmol/L)
- **Additional columns**: Any other features (automatically detected and classified)

### Supported Formats
- CSV files (`.csv`)
- Parquet files (`.parquet`, `.pq`)

### Example Dataset Structure

```csv
PatientID,Timestamp,Glucose,Age,Gender,BMI
P001,2023-01-01 08:00:00,120,25,M,22.5
P001,2023-01-01 08:15:00,115,25,M,22.5
P002,2023-01-01 09:00:00,98,30,F,24.1
...
```

## File Organization

```
glucose-prediction/
├── data_utils.py          # Universal data preprocessing
├── model.py               # Unified model factory + QISN architecture  
├── train.py               # Training pipeline with multi-model CLI
├── shap_utils.py          # Universal SHAP analysis for all models
├── transformer_models/    # Transformer implementations
│   ├── model_factory.py   # Model factory and configurations
│   ├── transformer_models.py  # All transformer architectures
│   └── saint_model.py     # SAINT implementation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Command Line Interface

### Model Selection

| Argument | Description | Options |
|----------|-------------|---------|
| `--model` | Model architecture | `qisn`, `saint`, `ft-transformer`, `tabpfn`, `patchtst`, `tst`, `informer`, `tabnet` |

### Model Hyperparameters

| Argument | Description | Default | Applies To |
|----------|-------------|---------|------------|
| `--dim` | Hidden dimension size | 128 | All models |
| `--depth` | Number of layers | 6 | All models |
| `--heads` | Number of attention heads | 8 | Transformer models |
| `--attn_dropout` | Attention dropout rate | 0.1 | Transformer models |
| `--ff_dropout` | Feed-forward dropout rate | 0.1 | All models |
| `--n_states` | Number of quantum states | 4 | QISN only |
| `--patch_len` | Patch length | 3 | PatchTST only |
| `--stride` | Patch stride | 1 | PatchTST only |
| `--factor` | Sparse attention factor | 5 | Informer only |

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Training mode: individual, combined, finetune | Required |
| `--data_path` | Path to dataset file | Required |
| `--patient_col` | Patient ID column name (auto-detect if not provided) | Auto |
| `--time_col` | Timestamp column name (auto-detect if not provided) | Auto |
| `--glucose_col` | Glucose column name (auto-detect if not provided) | Auto |
| `--batch_size` | Training batch size | 64 |
| `--epochs` | Number of training epochs | 100 |
| `--lr` | Learning rate | 1e-4 |
| `--window_size` | Input sequence length | 12 |
| `--prediction_horizon` | Steps ahead to predict | 3 |

### SHAP Analysis Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--shap` | Enable SHAP analysis | False |
| `--shap_samples` | Number of samples for SHAP | 1000 |
| `--shap_evals` | Maximum SHAP evaluations | 2000 |

### System Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--device` | GPU device ID | 0 |
| `--output_dir` | Output directory | outputs |
| `--world_size` | Number of GPUs | Auto-detect |

## Model Architecture Details

### QISN (Quantum-Inspired Superposition Network)
- **Innovation**: Quantum superposition states for feature representation
- **Strengths**: Handles missing values naturally, complex feature interactions
- **Best for**: Datasets with missing values, medium-sized sequences
- **Parameters**: `--n_states` (quantum states), `--dim`, `--depth`

### SAINT (Self-Attention and Intersample Attention Transformer)
- **Innovation**: Intersample attention for tabular data
- **Strengths**: Mixed categorical/continuous features, attention interpretability
- **Best for**: Tabular time series with diverse feature types
- **Parameters**: `--dim`, `--depth`, `--heads`, `--attn_dropout`

### FT-Transformer (Feature Tokenizer Transformer)
- **Innovation**: Feature tokenization for tabular learning
- **Strengths**: Feature-wise attention, handles mixed data types well
- **Best for**: Datasets with many categorical features
- **Parameters**: `--dim`, `--depth`, `--heads`, `--ff_dropout`

### TabPFN (Tabular Prior-Fitted Networks)
- **Innovation**: Meta-learning with learned priors
- **Strengths**: Excellent on small datasets, fast inference
- **Best for**: Limited training data, few-shot learning scenarios
- **Parameters**: `--dim`, `--depth`, `--heads`

### PatchTST
- **Innovation**: Patch-based processing for time series
- **Strengths**: Efficient for long sequences, good temporal modeling
- **Best for**: Long time series, forecasting tasks
- **Parameters**: `--patch_len`, `--stride`, `--dim`, `--depth`

### TST (Time Series Transformer)
- **Innovation**: Learnable positional encoding for time series
- **Strengths**: Temporal pattern recognition, sequence modeling
- **Best for**: Time series with strong temporal dependencies
- **Parameters**: `--dim`, `--depth`, `--heads`, `--window_size`

### Informer
- **Innovation**: ProbSparse attention for long sequences
- **Strengths**: Scales to very long sequences, memory efficient
- **Best for**: Very long time series, large-scale forecasting
- **Parameters**: `--factor`, `--dim`, `--depth`, `--window_size`

### TabNet
- **Innovation**: Attention-based feature selection
- **Strengths**: Interpretable feature selection, handles tabular data well
- **Best for**: Feature selection needs, interpretability requirements
- **Parameters**: `--dim`, `--depth`, `--n_steps` (internal parameter)

## Examples by Use Case

### Example 1: Small Dataset (< 10k samples)
```bash
# TabPFN - excellent for small datasets
python train.py --mode individual --data_path small_dataset.csv \
    --model tabpfn --dim 64 --epochs 50

# QISN - handles missing values well
python train.py --mode individual --data_path small_dataset.csv \
    --model qisn --n_states 6 --dim 128 --epochs 100
```

### Example 2: Long Time Series (> 50 timesteps)
```bash
# Informer - designed for long sequences
python train.py --mode individual --data_path long_series.csv \
    --model informer --window_size 96 --factor 3 --epochs 100

# PatchTST - efficient patching approach
python train.py --mode individual --data_path long_series.csv \
    --model patchtst --window_size 48 --patch_len 6 --epochs 100
```

### Example 3: Many Categorical Features
```bash
# FT-Transformer - excellent feature tokenization
python train.py --mode individual --data_path categorical_data.csv \
    --model ft-transformer --dim 256 --heads 12 --epochs 150

# SAINT - handles mixed data types well
python train.py --mode individual --data_path categorical_data.csv \
    --model saint --dim 192 --depth 8 --epochs 150
```

### Example 4: Interpretability Focus
```bash
# TabNet with SHAP - built-in feature selection + SHAP
python train.py --mode individual --data_path interpretable_data.csv \
    --model tabnet --epochs 100 --shap --shap_samples 2000

# Any model with comprehensive SHAP analysis
python train.py --mode individual --data_path any_data.csv \
    --model saint --epochs 100 --shap --shap_samples 1500 --shap_evals 3000
```

### Example 5: High Performance Setup
```bash
# Informer with optimal settings for large datasets
python train.py --mode individual --data_path large_dataset.csv \
    --model informer --dim 512 --depth 8 --heads 16 \
    --batch_size 256 --lr 0.0001 --window_size 72 \
    --epochs 200 --patience 25
```
| `--output_dir` | Output directory | outputs |
| `--world_size` | Number of GPUs | Auto-detect |

## Model Architecture

The QISN (Quantum-Inspired Superposition Network) uses:

1. **Quantum State Layer**: Represents features in superposition states
2. **Entanglement Layer**: Multi-head attention for feature interactions  
3. **Collapse Mechanism**: Smart feature selection based on thresholds
4. **Temporal Encoding**: Handles sequential glucose patterns

## Automatic Feature Detection

The system automatically:

- **Detects dataset structure**: Finds patient, time, and glucose columns
- **Classifies features**: Separates categorical vs continuous features
- **Extracts time features**: Hour, day of week, seasonality patterns
- **Handles missing data**: Intelligent imputation strategies
- **Scales data**: Automatic normalization for model training

## SHAP Analysis

When using `--shap`, the system generates:

- **Feature Importance Plot**: Top contributing features
- **Summary Plot**: SHAP values distribution
- **Feature Importance CSV**: Detailed rankings and statistics
- **Dependence Plots**: Feature interaction analysis

Output saved to `<output_dir>/shap_analysis/`

## Examples

### Example 1: Simple Training
```bash
python train.py --mode individual --data_path diabetes_data.csv --epochs 50
```

### Example 2: Full Analysis
```bash
python train.py --mode individual --data_path cgm_data.parquet \
    --epochs 100 --batch_size 128 --lr 0.0005 \
    --shap --shap_samples 2000 --output_dir results/
```

### Example 3: Manual Column Specification
```bash
python train.py --mode individual --data_path custom_data.csv \
    --patient_col SubjectID --time_col DateTime --glucose_col BG \
    --window_size 24 --epochs 150
```

## Output Structure

After training, you'll find:

```
outputs/
├── individual/
│   └── <dataset_name>/
│       ├── best_model.pt          # Trained model
│       ├── training_log.txt       # Training logs
│       ├── metrics.png            # Training curves
│       └── shap_analysis/         # SHAP results (if enabled)
│           ├── feature_importance.png
│           ├── shap_summary.png
│           └── feature_importance.csv
```

## Technical Details

### Preprocessing Pipeline
1. Automatic dataset structure detection
2. Data cleaning and validation
3. Time feature extraction
4. Feature type classification
5. Sequence generation with sliding windows
6. Data scaling and normalization

### Model Training
1. Multi-GPU distributed training support
2. Early stopping with validation monitoring
3. Learning rate scheduling
4. Gradient clipping for stability
5. Model checkpointing

### Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)
- R² Score
- Feature importance rankings

## Troubleshooting

### Common Issues

**Dataset not detected properly:**
- Manually specify column names using `--patient_col`, `--time_col`, `--glucose_col`
- Check that your dataset has the required columns

**SHAP analysis fails:**
- Install SHAP: `pip install shap`
- Reduce sample size with `--shap_samples 500`

**Out of memory errors:**
- Reduce `--batch_size`
- Use fewer `--shap_samples`
- Use smaller `--window_size`

**Poor training performance:**
- Increase `--epochs`
- Adjust `--lr` learning rate
- Try different `--window_size`

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- PyTorch ≥ 1.9.0
- pandas ≥ 1.3.0
- numpy ≥ 1.21.0
- scikit-learn ≥ 1.0.0
- SHAP ≥ 0.40.0 (optional, for explainability)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/glucose-prediction.git
cd glucose-prediction

# Install dependencies
pip install -r requirements.txt

# Run verification
python verify_model_update.py
```

### Contribution Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{glucose_prediction_transformers,
  title={Glucose Prediction with Transformer Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/glucose-prediction}
}
```

### Related Papers

Please also consider citing the original papers for the models you use:

**SAINT:**
```bibtex
@article{somepalli2021saint,
  title={SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training},
  author={Somepalli, Gowthami and Goldblum, Micah and Schwarzschild, Avi and Bruss, C Bayan and Goldstein, Tom},
  journal={arXiv preprint arXiv:2106.01342},
  year={2021}
}
```

**FT-Transformer:**
```bibtex
@article{gorishniy2021revisiting,
  title={Revisiting Deep Learning Models for Tabular Data},
  author={Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},
  journal={arXiv preprint arXiv:2106.11959},
  year={2021}
}
```

**TabPFN:**
```bibtex
@article{hollmann2022tabpfn,
  title={TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  journal={arXiv preprint arXiv:2207.01848},
  year={2022}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original transformer implementations from respective papers
- PyTorch team for the excellent deep learning framework
- The open-source community for inspiration and support

## 📞 Contact

For questions and support:
- **Create an issue** on GitHub
- **Check the documentation** in the wiki
- **Review the examples** in the repository

---

**Note**: This is a research project. Please consult with healthcare professionals before using any glucose prediction models in clinical settings.

**⭐ If you find this project helpful, please consider giving it a star!**

**Made with ❤️ for the diabetes research community**
