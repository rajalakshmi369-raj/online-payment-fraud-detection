import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig', 
    'errorBalanceDest', 'hourOfDay',
    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

NUM_COLS = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig', 
    'errorBalanceDest', 'hourOfDay'
]

TRANSACTION_TYPES = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']

def engineer_features(df):
    """
    Applies domain feature engineering to financial transaction records.
    Computes origin balance discrepancies, destination discrepancies, and temporal features.
    """
    df = df.copy()
    
    # Financial discrepancy features:
    # If a transaction was legitimate, newbalanceOrig should equal oldbalanceOrg - amount
    # Discrepancy indicates balance tampering, zero-balance anomalies, or ghost credits
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # Temporal feature: Hour of day (step represents hours in simulation)
    df['hourOfDay'] = df['step'] % 24
    
    # One-hot encoding for transaction type
    for t in TRANSACTION_TYPES:
        df[f'type_{t}'] = (df['type'] == t).astype(int)
        
    return df

def prepare_features_target(df):
    """Extracts features X and binary target y from DataFrame."""
    df_feat = engineer_features(df)
    
    # Ensure all required columns exist
    for col in FEATURE_COLS:
        if col not in df_feat.columns:
            df_feat[col] = 0
            
    X = df_feat[FEATURE_COLS]
    y = df_feat['isFraud'].astype(int) if 'isFraud' in df_feat.columns else None
    return X, y

def create_train_test_split(df, test_size=0.20, random_state=42):
    """
    Performs stratified train-test split and fits StandardScaler strictly on training set.
    Prevents data leakage by fitting scaler only on X_train.
    """
    X, y = prepare_features_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test_scaled[NUM_COLS] = scaler.transform(X_test[NUM_COLS])
    
    preprocessor = {
        'scaler': scaler,
        'feature_cols': FEATURE_COLS,
        'num_cols': NUM_COLS,
        'transaction_types': TRANSACTION_TYPES
    }
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor

def transform_new_data(df, preprocessor):
    """
    Transforms new unlabelled or test transactions using the saved preprocessor pipeline.
    """
    scaler = preprocessor['scaler']
    feature_cols = preprocessor['feature_cols']
    num_cols = preprocessor['num_cols']
    
    df_feat = engineer_features(df)
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0
            
    X = df_feat[feature_cols].copy()
    X[num_cols] = scaler.transform(X[num_cols])
    return X
