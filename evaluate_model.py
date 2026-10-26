import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from data_preprocessing import create_train_test_split, FEATURE_COLS

def generate_evaluation_visualizations():
    print("Generating comprehensive evaluation visualizations in 'outputs/'...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load dataset sample
    sample_path = os.path.join(data_dir, "paysim_dashboard_sample.parquet")
    df = pd.read_parquet(sample_path)
    
    X_train_scaled, X_test_scaled, y_train, y_test, preprocessor = create_train_test_split(df, test_size=0.20, random_state=42)
    
    # Load models
    model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
    models = {}
    for name in model_names:
        fname = name.lower().replace(" ", "_") + ".joblib"
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            
    with open(os.path.join(models_dir, "model_metrics.json"), "r") as f:
        metrics_data = json.load(f)
        
    metrics = metrics_data["models"]
    conf_matrices = metrics_data["confusion_matrices"]
    
    # Set styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    sns.set_theme(style="whitegrid", palette="muted")
    
    # -------------------------------------------------------------
    # 1. MODEL PERFORMANCE COMPARISON CHART
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    available_models = [m for m in model_names if m in metrics]
    metric_keys = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    
    x = np.arange(len(available_models))
    width = 0.15
    colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed"]
    
    for i, m_key in enumerate(metric_keys):
        vals = [metrics[m].get(m_key, 0) for m in available_models]
        rects = ax.bar(x + i*width, vals, width, label=m_key.replace("_", " "), color=colors[i], alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')
            
    ax.set_ylabel("Score (%)", fontsize=12, fontweight='bold')
    ax.set_title("Machine Learning Algorithm Performance Comparison (Held-out Test Data)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(available_models, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "model_comparison.png"), dpi=300)
    plt.close()
    print("Saved outputs/model_comparison.png")
    
    # -------------------------------------------------------------
    # 2. CONFUSION MATRICES GRID
    # -------------------------------------------------------------
    n_models = len(available_models)
    cols = 2
    rows = (n_models + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = np.array(axes).reshape(-1)
    
    for idx, name in enumerate(available_models):
        ax = axes[idx]
        mat = conf_matrices[name]["matrix"]
        cm_array = np.array(mat)
        
        sns.heatmap(cm_array, annot=True, fmt=',d', cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"fontsize": 13, "fontweight": "bold"})
        ax.set_title(f"{name}\nAcc: {metrics[name]['Accuracy']}% | F1: {metrics[name]['F1_Score']}%", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight='bold')
        ax.set_ylabel("Actual Label", fontsize=10, fontweight='bold')
        ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
        ax.set_yticklabels(["Legitimate (0)", "Fraud (1)"])
        
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
        
    plt.suptitle("Confusion Matrix Heatmaps by Algorithm (Test Set: 57,562 Transactions)", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "confusion_matrices.png"), dpi=300)
    plt.close()
    print("Saved outputs/confusion_matrices.png")
    
    # -------------------------------------------------------------
    # 3. ROC CURVES
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = {"Logistic Regression": "#3b82f6", "Decision Tree": "#10b981", "Random Forest": "#8b5cf6", "XGBoost": "#ef4444"}
    
    for name in available_models:
        model = models.get(name)
        if model and hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.4f})", lw=2.2, color=palette.get(name, "#333333"))
            
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.50)")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Positive Rate / Recall (TPR)", fontsize=11, fontweight='bold')
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.95, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "roc_curves.png"), dpi=300)
    plt.close()
    print("Saved outputs/roc_curves.png")
    
    # -------------------------------------------------------------
    # 4. FEATURE IMPORTANCE (RANDOM FOREST & XGBOOST)
    # -------------------------------------------------------------
    feat_json = os.path.join(models_dir, "feature_importance.json")
    if os.path.exists(feat_json):
        with open(feat_json, "r") as f:
            feat_imp = json.load(f)
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for i, target_model in enumerate(["Random Forest", "XGBoost"]):
            if target_model in feat_imp:
                ax = axes[i]
                series = pd.Series(feat_imp[target_model]).sort_values(ascending=True)
                colors = ["#2563eb" if "error" in k or "TRANSFER" in k or "CASH_OUT" in k else "#94a3b8" for k in series.index]
                series.plot(kind="barh", ax=ax, color=colors)
                ax.set_title(f"{target_model} Feature Importance", fontsize=12, fontweight='bold')
                ax.set_xlabel("Importance Weight", fontsize=10)
            else:
                axes[i].axis('off')
                
        plt.suptitle("Key Feature Drivers in Fraud Classification (Engineered Features Highlighted in Blue)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, "feature_importance.png"), dpi=300)
        plt.close()
        print("Saved outputs/feature_importance.png")
        
    print("All evaluation plots generated successfully.")

if __name__ == "__main__":
    generate_evaluation_visualizations()
