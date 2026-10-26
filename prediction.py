import os
import json
import numpy as np
import pandas as pd
import joblib

from data_preprocessing import transform_new_data, FEATURE_COLS, NUM_COLS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

class FraudPredictor:
    def __init__(self):
        self.preprocessor = None
        self.models = {}
        self.best_model_name = "Random Forest"
        self._load_artifacts()
        
    def _load_artifacts(self):
        prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        if os.path.exists(prep_path):
            self.preprocessor = joblib.load(prep_path)
            
        metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
                self.best_model_name = data.get("best_model_name", "Random Forest")
                
        # Load available models
        for m_name in ["Random Forest", "XGBoost", "Decision Tree", "Logistic Regression"]:
            fname = m_name.lower().replace(" ", "_") + ".joblib"
            p = os.path.join(MODELS_DIR, fname)
            if os.path.exists(p):
                self.models[m_name] = joblib.load(p)
                
    def predict_single(self, transaction: dict, model_name: str = None) -> dict:
        """
        Takes a single transaction dictionary and returns prediction, probability,
        risk level, and rule-based behavioral explanation.
        """
        if not self.preprocessor:
            self._load_artifacts()
            if not self.preprocessor:
                raise RuntimeError("Preprocessor artifact not found. Please run model training first.")
                
        chosen_model_name = model_name or self.best_model_name
        model = self.models.get(chosen_model_name)
        if not model:
            # Fallback to any available model
            if self.models:
                chosen_model_name = list(self.models.keys())[0]
                model = self.models[chosen_model_name]
            else:
                raise RuntimeError(f"Model '{chosen_model_name}' not available.")
                
        df_single = pd.DataFrame([transaction])
        X_scaled = transform_new_data(df_single, self.preprocessor)
        
        pred = int(model.predict(X_scaled)[0])
        
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_scaled)[0, 1])
        else:
            prob = float(pred)
            
        # Determine Risk Level
        if prob >= 0.70:
            risk_level = "High"
        elif prob >= 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        # Explainable AI Insights (Why the model flagged or cleared it)
        reasons = []
        trans_type = transaction.get('type', '')
        amount = float(transaction.get('amount', 0))
        old_orig = float(transaction.get('oldbalanceOrg', 0))
        new_orig = float(transaction.get('newbalanceOrig', 0))
        old_dest = float(transaction.get('oldbalanceDest', 0))
        new_dest = float(transaction.get('newbalanceDest', 0))
        
        # Attack vector check
        if trans_type in ['TRANSFER', 'CASH_OUT']:
            reasons.append(f"Transaction type is high-risk '{trans_type}' (the primary vector for fraudulent cash liquidation).")
        else:
            reasons.append(f"Transaction type '{trans_type}' has historically negligible fraud incidence.")
            
        # Account emptying check
        if old_orig > 0 and new_orig == 0 and abs(amount - old_orig) < 1.0:
            reasons.append("Entire origin account balance was drained to $0.00 in a single transaction (classic account takeover indicator).")
        elif new_orig > old_orig and trans_type not in ['CASH_IN']:
            reasons.append("Origin balance increased despite outflow (balance anomaly).")
            
        # Destination zero anomaly
        if trans_type == 'TRANSFER' and old_dest == 0 and new_dest == 0:
            reasons.append("Destination account had $0 balance before and after the transaction (ghost destination account).")
            
        if amount > 200000:
            reasons.append(f"Large transfer amount (${amount:,.2f}) exceeds regulatory threshold of $200,000.")
            
        return {
            "prediction": "Fraudulent" if pred == 1 else "Legitimate",
            "is_fraud": pred,
            "fraud_probability": round(prob * 100, 2),
            "risk_level": risk_level,
            "model_used": chosen_model_name,
            "risk_factors": reasons
        }
        
    def predict_batch(self, df: pd.DataFrame, model_name: str = None) -> (pd.DataFrame, dict):
        """
        Vectorized batch inference for uploaded CSV files.
        """
        if not self.preprocessor:
            self._load_artifacts()
            
        chosen_model_name = model_name or self.best_model_name
        model = self.models.get(chosen_model_name)
        if not model:
            chosen_model_name = list(self.models.keys())[0]
            model = self.models[chosen_model_name]
            
        # Transform batch
        X_scaled = transform_new_data(df, self.preprocessor)
        preds = model.predict(X_scaled)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[:, 1]
        else:
            probs = preds.astype(float)
            
        df_result = df.copy()
        df_result['Predicted_Status'] = np.where(preds == 1, 'Fraudulent', 'Legitimate')
        df_result['Fraud_Probability_%'] = np.round(probs * 100, 2)
        df_result['Risk_Level'] = np.where(probs >= 0.70, 'HIGH', np.where(probs >= 0.35, 'MEDIUM', 'LOW'))
        
        summary = {
            "total_records": len(df_result),
            "flagged_fraud": int((preds == 1).sum()),
            "cleared_legitimate": int((preds == 0).sum()),
            "fraud_percentage": round(float((preds == 1).sum() / len(df_result) * 100), 2),
            "model_used": chosen_model_name
        }
        
        return df_result, summary
