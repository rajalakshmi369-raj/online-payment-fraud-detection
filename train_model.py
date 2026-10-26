import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from data_preprocessing import create_train_test_split, FEATURE_COLS

def train_and_evaluate_all():
    print("=" * 70)
    print("Online Payment Fraud Detection - 4-Model Training & Evaluation Pipeline")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    sample_path = os.path.join(data_dir, "paysim_dashboard_sample.parquet")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Parquet sample dataset not found at: {sample_path}")
        
    print(f"Loading representative sample from: {sample_path}")
    t0 = time.time()
    df = pd.read_parquet(sample_path)
    print(f"Loaded {len(df):,} records in {time.time() - t0:.2f}s.")
    print(f"Total Fraud Cases: {df['isFraud'].sum():,} | Legit Cases: {(df['isFraud'] == 0).sum():,}")
    
    # Stratified Train-Test Split & Scaler fitting
    print("\nExecuting stratified train-test split (80/20)...")
    X_train_scaled, X_test_scaled, y_train, y_test, preprocessor = create_train_test_split(df, test_size=0.20, random_state=42)
    
    print(f"Training set: {len(X_train_scaled):,} samples (Fraud: {y_train.sum():,})")
    print(f"Testing set:  {len(X_test_scaled):,} samples (Fraud: {y_test.sum():,})")
    
    # Save preprocessor
    prep_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, prep_path)
    print(f"Saved preprocessor pipeline to: {prep_path}")
    
    # Define 4 Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, 
            class_weight='balanced', 
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=12, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
    }
    
    if XGB_AVAILABLE:
        # scale_pos_weight balances positive class in XGBoost
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_weight = float(neg_count / max(pos_count, 1))
        
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=min(scale_weight, 50.0), # robust balanced weighting
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        
    metrics = {}
    conf_matrices = {}
    feature_importances = {}
    saved_model_objs = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        t_start = time.time()
        model.fit(X_train_scaled, y_train)
        fit_time = time.time() - t_start
        
        y_pred = model.predict(X_test_scaled)
        
        # Predicted probabilities for ROC-AUC and PR-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
            pr_auc = float(average_precision_score(y_test, y_prob))
        else:
            y_prob = y_pred
            roc_auc = 0.5
            pr_auc = 0.0
            
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        
        metrics[name] = {
            "Accuracy": round(acc * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1_Score": round(f1 * 100, 2),
            "ROC_AUC": round(roc_auc * 100, 2),
            "PR_AUC": round(pr_auc * 100, 2),
            "Training_Time_s": round(fit_time, 2)
        }
        
        conf_matrices[name] = {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "matrix": [[tn, fp], [fn, tp]]
        }
        
        # Save feature importance if available
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_.tolist()
            feature_importances[name] = dict(zip(FEATURE_COLS, [round(val, 4) for val in fi]))
            
        print(f"{name} Results: Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%, ROC-AUC: {roc_auc*100:.2f}%")
        print(f"Confusion Matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,} (FN represents missed frauds)")
        
        # Save model file
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(models_dir, filename))
        saved_model_objs[name] = model
        
    # Best Model Selection based on combined F1 and Recall
    # In fraud detection, missing a fraud (False Negative) has massive financial penalties,
    # so Recall and F1-Score are the primary selection criteria rather than pure Accuracy.
    best_name = max(metrics, key=lambda k: (metrics[k]["F1_Score"], metrics[k]["Recall"]))
    print(f"\n================================================================")
    print(f"Selected Best Model: {best_name}")
    print(f"Metrics -> F1: {metrics[best_name]['F1_Score']}%, Recall: {metrics[best_name]['Recall']}%, Precision: {metrics[best_name]['Precision']}%")
    print(f"================================================================")
    
    # Save best model copy
    joblib.dump(saved_model_objs[best_name], os.path.join(models_dir, "best_model.joblib"))
    
    eval_results = {
        "best_model_name": best_name,
        "test_size": len(y_test),
        "test_fraud_count": int(y_test.sum()),
        "test_legit_count": int(len(y_test) - y_test.sum()),
        "models": metrics,
        "confusion_matrices": conf_matrices
    }
    
    with open(os.path.join(models_dir, "model_metrics.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
        
    with open(os.path.join(models_dir, "feature_importance.json"), "w") as f:
        json.dump(feature_importances, f, indent=2)
        
    print("\nModel training and metric serialization complete.")
    return metrics, conf_matrices

if __name__ == "__main__":
    train_and_evaluate_all()
