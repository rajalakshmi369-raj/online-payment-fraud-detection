# Online Payment Fraud Detection Management Using Machine Learning Algorithms

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced, end-to-end Data Science and Machine Learning management system for detecting and preventing fraudulent financial transactions in real-time. Built with high-performance classifiers, interactive financial visualization dashboards, Explainable AI (XAI) risk drivers, and bulk batch transaction processing.

---

## 📸 Dashboard Interface Overview (FraudGuard)
The interface is designed with a modern financial operations theme:
* **Real-Time KPI Cards:** Total Transactions, Legitimate Volume, Fraudulent Incidents, and Fraud Rate %.
* **Transaction Trend Timeline:** Temporal area chart tracking transactions across 30 days.
* **Class Distribution Donut:** Visualizing the 99.87% legitimate vs. 0.13% fraud class imbalance.
* **Live Feed Table:** Formatted real-time transaction feed with color-coded risk status pills.
* **Model Benchmark & Live Confusion Matrix:** Interactive confusion matrix and multi-metric accuracy cards.
* **Explainable AI Prediction:** Immediate single-transaction scoring with behavioral root cause breakdown.
* **Bulk Batch CSV Processing:** Upload and score multi-thousand row transaction files with one-click CSV export.

---

## 📊 Dataset Description & Access
This project is built using the **PaySim Synthetic Financial Dataset for Fraud Detection**:
* **Source:** [Kaggle - PaySim Mobile Money Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
* **Scale:** 6,362,620 transactions over a 744-hour (30-day) simulation.
* **Ground Truth Fraud Cases:** 8,213 frauds (~0.129%).
* **Total Stolen Attempted Volume:** Exceeds $12.05 Billion.

### Core Features:
| Feature | Type | Description |
|---|---|---|
| `step` | Integer | Simulation time unit in hours (1-744). |
| `type` | Categorical | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER. |
| `amount` | Float | Transaction amount in local currency ($). |
| `nameOrig` | String | Originating sender account identifier. |
| `oldbalanceOrg` | Float | Sender balance prior to transaction. |
| `newbalanceOrig` | Float | Sender balance after transaction. |
| `nameDest` | String | Recipient account identifier (C: Customer, M: Merchant). |
| `oldbalanceDest` | Float | Recipient balance prior to transaction. |
| `newbalanceDest` | Float | Recipient balance after transaction. |
| `isFraud` | Binary | Ground truth label (1 = Fraudulent, 0 = Legitimate). |

---

## 🤖 Machine Learning Model Benchmarks

Evaluation results computed strictly on the held-out test dataset (**57,562 transactions**):

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|---|---|---|---|---|---|---|
| **Random Forest (Champion)** | **99.99%** | **99.94%** | **99.57%** | **99.76%** | **99.97%** | 2.74s |
| **XGBoost Classifier** | **99.98%** | **99.63%** | **99.63%** | **99.63%** | **99.96%** | 1.85s |
| **Decision Tree** | 99.97% | 99.15% | 99.63% | 99.39% | 99.81% | 1.00s |
| **Logistic Regression** | 95.68% | 39.48% | 96.10% | 55.96% | 99.19% | 0.41s |

### 🎯 Key Engineering Takeaways:
1. **The Accuracy Paradox:** Because 99.87% of the dataset is clean, a naive model predicting all transactions as "Legitimate" would achieve 99.87% accuracy while missing 100% of frauds ($0\%$ Recall). Hence, **Recall** and **F1-Score** are our primary optimization targets.
2. **Attack Vectors:** 100% of frauds occur in `TRANSFER` (4,097) and `CASH_OUT` (4,116). Attackers transfer victim funds into temporary mule accounts, then immediately cash out to physical cash.
3. **Engineered Discrepancy Features:**
   * `errorBalanceOrig`: $newbalanceOrig + amount - oldbalanceOrg$
   * `errorBalanceDest`: $oldbalanceDest + amount - newbalanceDest$
   These two features alone account for >70% of feature importance across tree models.

---

## 📁 Project Directory Structure
```
online-payment-fraud-detection/
│
├── data/
│   ├── README.md                           # Dataset documentation & Kaggle instructions
│   ├── PS_20174392719_1491204439457_log.csv# Raw PaySim CSV (Optional local path)
│   ├── paysim_dashboard_sample.parquet    # Optimized stratified sample (287,807 rows)
│   ├── sample_batch_test.csv               # 50-row test dataset for batch prediction
│   └── summary_stats.json                  # Global dataset counts (6.36M records)
│
├── models/
│   ├── best_model.joblib                   # Serialized champion model (Random Forest)
│   ├── random_forest.joblib                # Random Forest classifier
│   ├── xgboost.joblib                      # XGBoost classifier
│   ├── decision_tree.joblib                # Decision Tree classifier
│   ├── logistic_regression.joblib          # Logistic Regression baseline
│   ├── preprocessor.joblib                 # StandardScaler & column mapping pipeline
│   ├── model_metrics.json                  # Held-out test evaluation scores
│   └── feature_importance.json             # Feature weights by model
│
├── outputs/
│   ├── model_comparison.png                # Comparative evaluation bar chart
│   ├── confusion_matrices.png              # 4-model confusion matrix grid
│   ├── roc_curves.png                      # Multi-model ROC curves
│   └── feature_importance.png              # Top feature drivers plot
│
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── data_preprocessing.py               # Feature engineering, scaling & train-test split
│   ├── eda.py                              # Statistical aggregations & radar plot data
│   ├── train_model.py                      # Pipeline to train and benchmark all 4 models
│   ├── evaluate_model.py                   # Plotting script generating outputs/ charts
│   └── prediction.py                       # Single & batch inference engine with XAI
│
├── .streamlit/
│   └── config.toml                         # Dashboard visual styling and theme config
├── app.py                                  # Main Streamlit web application
├── start_dashboard.bat                     # Windows 1-click startup launcher
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart Guide (Local Execution)

### 1. Clone or Open Project Directory
```powershell
cd "C:\Users\LOKESH ELUMALAI\.gemini\antigravity\scratch\online_payment_fraud_detection"
```

### 2. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 3. (Optional) Retrain Machine Learning Models
```powershell
python src/train_model.py
python src/evaluate_model.py
```

### 4. Launch the Interactive Dashboard
Double-click `start_dashboard.bat` or run:
```powershell
streamlit run app.py
```
Open your web browser at: **`http://localhost:8501/`**

---

## 🌐 Public Cloud Deployment Guide

This project is fully structured for free public cloud hosting on **Streamlit Community Cloud** or **Render**:

### Deploying to Streamlit Community Cloud (Recommended):
1. **Push to GitHub:**
   * Create a new GitHub repository named `online-payment-fraud-detection`.
   * Note: The large raw file (`PS_20174392719_1491204439457_log.csv`) is automatically excluded via `.gitignore`.
   * Commit and push the project files (the optimized `paysim_dashboard_sample.parquet` is under 14MB, well below GitHub's 100MB limit).
2. **Connect to Streamlit Cloud:**
   * Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
   * Click **New App**.
   * Select your repository: `online-payment-fraud-detection`.
   * Branch: `main`.
   * Main file path: `app.py`.
   * Click **Deploy!**
3. **Enjoy Live Access:**
   * Streamlit Cloud provisions a secure public URL (e.g., `https://fraudguard-detection.streamlit.app`).

---

## 🎓 Academic Viva Voce Questions & Answers
* **Q: Why does Random Forest outperform Logistic Regression?**  
  *A:* Fraud in financial transactions is inherently non-linear and conditional on interactive rules (e.g. `type == TRANSFER` AND `newbalanceOrig == 0` AND `amount > oldbalanceOrg`). Linear models like Logistic Regression struggle with complex multi-feature interactions, whereas decision tree ensembles naturally isolate these non-linear subspaces.
* **Q: How is class imbalance handled?**  
  *A:* We use stratified sampling to preserve fraud proportions across train and test sets, and apply balanced class weighting (`class_weight='balanced'` in sklearn and `scale_pos_weight` in XGBoost) to heavily penalize errors on fraud transactions.
* **Q: What is the business impact of a False Positive?**  
  *A:* A False Positive triggers temporary two-factor verification or human review for a legitimate customer. While causing slight user friction, its financial risk is negligible compared to a False Negative where stolen funds are unrecoverable.

---

## 📄 License
This academic project is open-source under the [MIT License](LICENSE).
