import numpy as np
import pandas as pd

def compute_eda_insights(df):
    """
    Computes key exploratory data analysis metrics and aggregations for dashboard visuals.
    """
    total = len(df)
    fraud_df = df[df['isFraud'] == 1]
    legit_df = df[df['isFraud'] == 0]
    
    # 1. Type Breakdown
    type_counts = df['type'].value_counts().to_dict()
    fraud_by_type = fraud_df['type'].value_counts().to_dict()
    
    # 2. Hourly Breakdown (step % 24)
    df_temp = df.copy()
    df_temp['hour'] = df_temp['step'] % 24
    hourly_fraud = df_temp[df_temp['isFraud'] == 1].groupby('hour').size().reindex(range(24), fill_value=0)
    hourly_legit = df_temp[df_temp['isFraud'] == 0].groupby('hour').size().reindex(range(24), fill_value=0)
    
    # Normalize hourly for comparative line graph
    hourly_df = pd.DataFrame({
        'Hour': range(24),
        'Fraud_Count': hourly_fraud.values,
        'Legit_Count': hourly_legit.values,
        'Fraud_Rate_%': (hourly_fraud.values / np.maximum(hourly_fraud.values + hourly_legit.values, 1) * 100).round(2)
    })
    
    # 3. Daily Breakdown (step // 24)
    df_temp['day'] = (df_temp['step'] // 24) + 1
    daily_fraud = df_temp[df_temp['isFraud'] == 1].groupby('day').size()
    daily_legit = df_temp[df_temp['isFraud'] == 0].groupby('day').size()
    all_days = sorted(list(set(df_temp['day'])))
    daily_df = pd.DataFrame({
        'Day': all_days,
        'Fraud_Count': [daily_fraud.get(d, 0) for d in all_days],
        'Legit_Count': [daily_legit.get(d, 0) for d in all_days]
    })
    
    # 4. Amount Statistics
    amount_stats = {
        'overall_mean': float(df['amount'].mean()),
        'overall_median': float(df['amount'].median()),
        'fraud_mean': float(fraud_df['amount'].mean()) if len(fraud_df) > 0 else 0,
        'fraud_median': float(fraud_df['amount'].median()) if len(fraud_df) > 0 else 0,
        'fraud_max': float(fraud_df['amount'].max()) if len(fraud_df) > 0 else 0,
        'legit_mean': float(legit_df['amount'].mean()) if len(legit_df) > 0 else 0,
        'legit_median': float(legit_df['amount'].median()) if len(legit_df) > 0 else 0,
        'total_fraud_loss': float(fraud_df['amount'].sum()) if len(fraud_df) > 0 else 0
    }
    
    # 5. Radar Plot Data (Normalized [0, 100] profile comparing Fraud vs Legit)
    # Dimensions:
    # 1. High Amount ($ > 100k)
    # 2. Account Drained to 0
    # 3. Destination Balance Discrepancy
    # 4. Night-time / Off-hours (Hour 22-06)
    # 5. High-Risk Type (TRANSFER/CASH_OUT)
    
    fraud_radar = [
        round(float((fraud_df['amount'] > 100000).mean() * 100), 1),
        round(float(((fraud_df['oldbalanceOrg'] > 0) & (fraud_df['newbalanceOrig'] == 0)).mean() * 100), 1),
        round(float(((fraud_df['oldbalanceDest'] == 0) & (fraud_df['newbalanceDest'] == 0)).mean() * 100), 1),
        round(float(((df_temp[df_temp['isFraud'] == 1]['hour'] < 7) | (df_temp[df_temp['isFraud'] == 1]['hour'] > 21)).mean() * 100), 1),
        round(float((fraud_df['type'].isin(['TRANSFER', 'CASH_OUT'])).mean() * 100), 1)
    ]
    
    legit_radar = [
        round(float((legit_df['amount'] > 100000).mean() * 100), 1),
        round(float(((legit_df['oldbalanceOrg'] > 0) & (legit_df['newbalanceOrig'] == 0)).mean() * 100), 1),
        round(float(((legit_df['oldbalanceDest'] == 0) & (legit_df['newbalanceDest'] == 0)).mean() * 100), 1),
        round(float(((df_temp[df_temp['isFraud'] == 0]['hour'] < 7) | (df_temp[df_temp['isFraud'] == 0]['hour'] > 21)).mean() * 100), 1),
        round(float((legit_df['type'].isin(['TRANSFER', 'CASH_OUT'])).mean() * 100), 1)
    ]
    
    radar_categories = [
        'Large Amount (> $100K)',
        'Origin Drained to $0',
        'Dest Zero Anomaly',
        'Off-Hour Activity',
        'Transfer/CashOut Type'
    ]
    
    return {
        'type_counts': type_counts,
        'fraud_by_type': fraud_by_type,
        'hourly_df': hourly_df,
        'daily_df': daily_df,
        'amount_stats': amount_stats,
        'radar': {
            'categories': radar_categories,
            'fraud_values': fraud_radar,
            'legit_values': legit_radar
        }
    }
