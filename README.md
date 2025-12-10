# 📘 DNA Promoter Classification Using Classical Machine Learning  
*A Bioinformatics Pipeline for Promoter vs Non-Promoter Sequence Prediction*

---

## 🧬 Overview

Promoter regions are essential DNA segments that regulate gene expression. Automatically identifying promoter sequences is a key task in computational genomics.

This repository implements a **full machine learning pipeline** for promoter classification using the **UCI Molecular Biology (Promoter Gene Sequences)** dataset, including preprocessing, one-hot encoding, model training, evaluation, and comparative analysis.

---

## 📂 Repository Structure

📁 promoter-classification-ml/
│
├── data/
│ └── promoter_sequences.csv # (optional local dataset copy)
│
├── notebooks/
│ └── promoter_classification.ipynb # full Jupyter notebook analysis
│
├── src/
│ ├── preprocess.py # sequence cleaning + DataFrame creation
│ ├── encoding.py # one-hot encoding utilities
│ ├── models.py # model training scripts
│ └── evaluation.py # metrics and tables
│
├── results/
│ └── metrics_summary.csv # performance table
│
├── README.md
└── requirements.txt

---

## 📊 Dataset Description

- **Source:** UCI Machine Learning Repository  
- **Samples:** 106 DNA sequences  
- **Labels:**  
  - `+` → Promoter  
  - `-` → Non-promoter  
- **Nucleotides:** A, C, G, T  
- **Structure:** Fixed-length sequences  

Processing steps:

1. Remove irregular characters (tabs)
2. Split sequences into nucleotides
3. Convert into a row-based DataFrame
4. Add labels and index identifiers

---

## 🔧 Feature Engineering

### ✔️ One-Hot Encoding

Each nucleotide position is expanded into four binary indicators:

pos0_A, pos0_C, pos0_G, pos0_T
pos1_A, pos1_C, pos1_G, pos1_T


This produces a **high-dimensional**, **sparse**, **binary** feature matrix.

---

## 🤖 Models Implemented

The following scikit-learn algorithms were evaluated:

- K-Nearest Neighbors (KNN)  
- Multi-Layer Perceptron (MLP)  
- Decision Tree  
- AdaBoost  
- Gaussian Naive Bayes  
- Support Vector Machines (SVM):
  - Linear
  - RBF
  - Sigmoid

Data is split into training and test sets using `train_test_split`.

---

## 🧪 Results

| Model        | Accuracy | Precision (−) | Recall (−) | F1 (−) | Precision (+) | Recall (+) | F1 (+) |
|-------------|----------|--------------|-----------|--------|---------------|-----------|--------|
| KNN         | ~0.68    | ~0.83        | ~0.45     | ~0.59  | ~0.63         | ~0.91     | ~0.74  |
| MLP         | ~0.77    | ~0.88        | ~0.64     | ~0.74  | ~0.71         | ~0.91     | ~0.80  |
| DecisionTree| ~0.68    | ~0.83        | ~0.45     | ~0.59  | ~0.63         | ~0.91     | ~0.74  |
| AdaBoost    | ~0.95    | 1.00         | ~0.91     | ~0.95  | ~0.92         | 1.00      | ~0.96  |
| GaussianNB  | ~0.95    | ~0.92        | 1.00      | ~0.96  | 1.00          | ~0.91     | ~0.95  |
| SVM-Linear  | ~0.86    | 1.00         | ~0.73     | ~0.84  | ~0.79         | 1.00      | ~0.88  |
| SVM-RBF     | ~0.91    | 1.00         | ~0.82     | ~0.90  | ~0.85         | 1.00      | ~0.92  |
| SVM-Sigmoid | ~0.91    | 1.00         | ~0.82     | ~0.90  | ~0.85         | 1.00      | ~0.92  |

### ⭐ Best Performing Models

- **AdaBoost** (~95% accuracy)  
- **Gaussian Naive Bayes** (~95% accuracy)  
- Strong performance also from **SVM-RBF** and **SVM-Sigmoid**

---

## 🧠 Discussion

- **Gaussian NB** performs exceptionally well because one-hot encoded nucleotides satisfy near-independence assumptions.
- **AdaBoost** effectively combines weak learners to model complex feature interactions.
- **SVM (RBF/Sigmoid)** captures non-linear classes efficiently.
- **KNN** and **Decision Trees** struggle with:
  - High dimensionality  
  - Sparse binary features  
  - Small dataset size  

---

## 🚀 Running the Project
2. Run the notebook
3. 

### 1. Install dependencies
pip install -r requirements.txt

📈 Future Work
GridSearchCV hyperparameter optimization
Cross-validation
ROC-AUC and PR-curve evaluation
Ensemble models (Random Forest, XGBoost, LightGBM)
Position-level interpretability
Comparison with deep learning sequence models (CNN/RNN)

⚠️ Disclaimer
This project is intended for research and educational purposes only.
It is not suitable for clinical or diagnostic use without extensive validation.
