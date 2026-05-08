# XMalNet Framework

XMalNet is an automated machine learning framework for malware classification that integrates ensemble learning, SHAP-based explainability, feature selection, and data balancing techniques into a unified pipeline.

The framework is designed to simplify experimentation and benchmarking for malware detection tasks while maintaining interpretability and reproducibility.

---

# Features

* Automated training of multiple ensemble models:

  * Random Forest
  * XGBoost
  * LightGBM

* SHAP-based explainability:

  * Uses the trained XGBoost model for feature importance analysis
  * Generates SHAP importance scores for all features

* Quartile-based feature selection:

  * 25th percentile (Q1)
  * 50th percentile / median (Q2)
  * 75th percentile (Q3)

* Imbalance handling with multiple oversampling strategies:

  * SMOTE
  * BorderlineSMOTE
  * SMOTETomek
  * SMOTEENN

* Automated evaluation pipeline:

  * Accuracy
  * Precision
  * Recall
  * F1-score

* CSV result export

* Progress tracking with tqdm

* Fully reproducible experimentation workflow

---

# Framework Pipeline

```text
Input Dataset
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
(Random Forest, XGBoost, LightGBM)
      │
      ▼
Best XGBoost Model Selection
      │
      ▼
SHAP Analysis
      │
      ▼
SHAP-Based Feature Selection
(Q1 / Q2 / Q3)
      │
      ▼
Oversampling
(SMOTE Variants)
      │
      ▼
Model Retraining & Evaluation
      │
      ▼
Best Combination Selection
      │
      ▼
CSV Summary Export
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/muhammadzuama/URXMalNet.git
cd xmalnet
```

## Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

The framework was developed and tested using the following library versions:

| Library          | Version |
| ---------------- | ------- |
| NumPy            | 2.4.4   |
| Pandas           | 3.0.2   |
| Matplotlib       | 3.10.8  |
| Scikit-learn     | 1.8.0   |
| XGBoost          | 3.2.0   |
| LightGBM         | 4.6.0   |
| SHAP             | 0.51.0  |
| Imbalanced-learn | 0.14.1  |
| tqdm             | 4.67.3  |

---

# Usage

```python
from xmalnet import XMalNet

# Initialize framework
xmn = XMalNet(class_labels=[0, 1, 2, 3])

# Train pipeline
xmn.fit(X_train, X_test, y_train, y_test)

# Display experiment summary
xmn.summary()

# Show best result
xmn.best_result()
```

---

# SHAP-Based Feature Selection

XMalNet applies feature selection using SHAP importance values generated from the trained XGBoost model.

The framework evaluates three quartile-based thresholds:

| Quartile             | Description                                                          |
| -------------------- | -------------------------------------------------------------------- |
| Q1 (25th percentile) | Retains features with SHAP values above the lower quartile threshold |
| Q2 (50th percentile) | Retains features above the median SHAP importance                    |
| Q3 (75th percentile) | Retains only highly important features                               |

This strategy enables automatic exploration of different feature subset granularities.

---

# Oversampling Strategies

The framework supports four imbalance handling techniques:

| Method          | Description                                  |
| --------------- | -------------------------------------------- |
| SMOTE           | Synthetic Minority Oversampling Technique    |
| BorderlineSMOTE | Generates samples near decision boundaries   |
| SMOTETomek      | Combines SMOTE with Tomek Links cleaning     |
| SMOTEENN        | Combines SMOTE with Edited Nearest Neighbors |

Each oversampling method is evaluated across all quartile-based feature subsets.

---

# Output Results

The framework automatically stores experiment results into CSV format.

Example output:

| Method          | Percentile | Features | Accuracy | F1 Macro |
| --------------- | ---------- | -------- | -------- | -------- |
| SMOTE           | Q1         | 118      | 0.9556   | 0.9471   |
| SMOTETomek      | Q2         | 87       | 0.9612   | 0.9538   |
| BorderlineSMOTE | Q3         | 42       | 0.9480   | 0.9395   |

---

# Example Workflow

```python
from sklearn.model_selection import train_test_split
from xmalnet import XMalNet

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Initialize framework
xmn = XMalNet(class_labels=[0, 1, 2, 3])

# Execute full pipeline
xmn.fit(X_train, X_test, y_train, y_test)

# Print all results
xmn.summary()

# Print best configuration
xmn.best_result()
```

---

# Evaluation Metrics

XMalNet evaluates classification performance using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report

These metrics help assess both overall performance and class-level behavior.

---

# Reproducibility

To ensure reproducibility:

* Use fixed random seeds
* Maintain consistent train-test splits
* Use the specified dependency versions

---

# Citation

If you use XMalNet in your research, please cite:

```bibtex
comming soon
```

---

# Author

Developed for automated malware classification research using explainable machine learning and imbalance-aware optimization.
