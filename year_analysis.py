import pandas as pd
import numpy as np

YEAR_STATS = {
    "2023": {
        "year": "2023",
        "total_tx": 1320000,
        "fraud_tx": 1450,
        "legit_tx": 1318550,
        "fraud_rate": 0.110,
        "total_amount": 235000000000,
        "fraud_amount": 1920000000,
        "monthly_frauds": [95, 110, 105, 120, 115, 130, 125, 140, 135, 150, 180, 245],
        "types": {"TRANSFER": 710, "CASH_OUT": 740, "PAYMENT": 0, "CASH_IN": 0, "DEBIT": 0}
    },
    "2024": {
        "year": "2024",
        "total_tx": 1850000,
        "fraud_tx": 2180,
        "legit_tx": 1847820,
        "fraud_rate": 0.118,
        "total_amount": 348000000000,
        "fraud_amount": 3150000000,
        "monthly_frauds": [140, 155, 160, 175, 170, 190, 185, 205, 195, 220, 260, 325],
        "types": {"TRANSFER": 1080, "CASH_OUT": 1100, "PAYMENT": 0, "CASH_IN": 0, "DEBIT": 0}
    },
    "2025": {
        "year": "2025",
        "total_tx": 2100000,
        "fraud_tx": 2860,
        "legit_tx": 2097140,
        "fraud_rate": 0.136,
        "total_amount": 420000000000,
        "fraud_amount": 4280000000,
        "monthly_frauds": [190, 205, 210, 230, 225, 250, 245, 270, 260, 290, 345, 430],
        "types": {"TRANSFER": 1420, "CASH_OUT": 1440, "PAYMENT": 0, "CASH_IN": 0, "DEBIT": 0}
    },
    "2026": {
        "year": "2026 (Current Benchmark)",
        "total_tx": 2450000,
        "fraud_tx": 3723,
        "legit_tx": 2446277,
        "fraud_rate": 0.152,
        "total_amount": 510000000000,
        "fraud_amount": 5650000000,
        "monthly_frauds": [250, 270, 285, 305, 300, 330, 320, 355, 340, 380, 440, 548],
        "types": {"TRANSFER": 1887, "CASH_OUT": 1836, "PAYMENT": 0, "CASH_IN": 0, "DEBIT": 0}
    }
}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def get_yearly_summary_dataframe():
    records = []
    for y_key, info in YEAR_STATS.items():
        records.append({
            "Year": y_key,
            "Total_Transactions": info["total_tx"],
            "Fraud_Count": info["fraud_tx"],
            "Fraud_Rate_%": info["fraud_rate"],
            "Stolen_Amount_$B": round(info["fraud_amount"] / 1e9, 2),
            "Clean_Transactions": info["legit_tx"]
        })
    return pd.DataFrame(records)

def get_monthly_trend_for_year(year_key):
    info = YEAR_STATS.get(year_key, YEAR_STATS["2026"])
    frauds = info["monthly_frauds"]
    
    # Calculate estimated monthly transactions
    monthly_tx = int(info["total_tx"] / 12)
    return pd.DataFrame({
        "Month": MONTH_NAMES,
        "Fraud_Count": frauds,
        "Total_Transactions": [monthly_tx] * 12,
        "Fraud_Rate_%": [round(f / monthly_tx * 100, 3) for f in frauds],
        "Stolen_Est_Millions": [round(f * (info["fraud_amount"] / info["fraud_tx"]) / 1e6, 2) for f in frauds]
    })
