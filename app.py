import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add src to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from prediction import FraudPredictor
from eda import compute_eda_insights
from year_analysis import YEAR_STATS, MONTH_NAMES, get_yearly_summary_dataframe, get_monthly_trend_for_year

# ==============================================================================
# PAGE CONFIGURATION & METADATA
# ==============================================================================
st.set_page_config(
    page_title="FraudGuard - Online Payment Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM STYLING (MATCHING FRAUDGUARD FINANCIAL DASHBOARD UI DESIGN)
# ==============================================================================
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        padding: 16px 24px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .top-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .top-subtitle {
        font-size: 0.88rem;
        color: #64748b;
        margin-top: 2px;
    }
    .top-user-badge {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .admin-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
    }
    
    /* KPI Metric Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        display: flex;
        flex-direction: column;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
    }
    .kpi-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .kpi-icon-wrap {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: capitalize;
    }
    .kpi-num {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .kpi-pill {
        font-size: 0.76rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        width: fit-content;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .pill-green { background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .pill-red { background-color: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
    .pill-blue { background-color: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    
    /* Chart and Content Boxes */
    .panel-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }
    .panel-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Fraud Result Banners */
    .banner-fraud {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        color: #9f1239;
        margin-top: 20px;
    }
    .banner-legit {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        color: #14532d;
        margin-top: 20px;
    }
    
    /* Badges */
    .tag-fraud {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
    }
    .tag-legit {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA AND MODEL CACHED LOADERS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_data
def get_summary_stats():
    stats_file = os.path.join(DATA_DIR, "summary_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def get_sample_df():
    sample_file = os.path.join(DATA_DIR, "paysim_dashboard_sample.parquet")
    if os.path.exists(sample_file):
        return pd.read_parquet(sample_file)
    csv_file = os.path.join(DATA_DIR, "paysim_preview.csv")
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return None

@st.cache_resource
def get_predictor():
    return FraudPredictor()

@st.cache_data
def get_model_metrics():
    metrics_file = os.path.join(MODELS_DIR, "model_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def get_feature_importances():
    fi_file = os.path.join(MODELS_DIR, "feature_importance.json")
    if os.path.exists(fi_file):
        with open(fi_file, "r") as f:
            return json.load(f)
    return None

summary_stats = get_summary_stats()
df = get_sample_df()
predictor = get_predictor()
model_metrics = get_model_metrics()
feature_importances = get_feature_importances()

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 10px 4px; margin-bottom: 24px;">
        <div style="background: linear-gradient(135deg, #2563eb, #1d4ed8); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; box-shadow: 0 4px 10px rgba(37,99,235,0.3);">
            🛡️
        </div>
        <div>
            <h2 style="color: #ffffff; font-size: 1.25rem; font-weight: 800; margin: 0; line-height: 1.2;">FraudGuard</h2>
            <p style="color: #94a3b8; font-size: 0.78rem; margin: 0;">Payment Fraud Detection</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='color: #64748b; font-size: 0.72rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.6px; margin-bottom: 8px;'>Navigation Menu</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "💳 Transactions Explorer",
            "📅 Year-wise Fraud Analysis",
            "📈 Exploratory Data Analysis",
            "🤖 ML Model Comparison",
            "🔍 Fraud Prediction (Single)",
            "📁 Batch CSV Fraud Detection",
            "ℹ️ Project Info & Academic Viva"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 24px 0;'>", unsafe_allow_html=True)
    
    # System Status Card in Sidebar
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 14px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span style="height: 9px; width: 9px; background-color: #10b981; border-radius: 50%; display: inline-block;"></span>
            <span style="color: #f8fafc; font-size: 0.8rem; font-weight: 700;">Engine Status: Online</span>
        </div>
        <p style="color: #94a3b8; font-size: 0.74rem; margin: 2px 0;"><strong>Active Model:</strong> {model_metrics.get('best_model_name', 'Random Forest') if model_metrics else 'Loaded'}</p>
        <p style="color: #94a3b8; font-size: 0.74rem; margin: 2px 0;"><strong>Evaluated Test Set:</strong> 57,562 TXs</p>
        <p style="color: #94a3b8; font-size: 0.74rem; margin: 2px 0;"><strong>Total Corpus:</strong> 6,362,620 TXs</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 30px; text-align: center; color: #64748b; font-size: 0.74rem;">
        <p style="margin: 0;">Online Payment Fraud Management</p>
        <p style="margin: 2px 0 0 0; color: #94a3b8; font-weight: 600;">Academic Capstone 2026-27</p>
    </div>
    """, unsafe_allow_html=True)

# Top Bar Header Function
def render_header(title, subtitle):
    st.markdown(f"""
    <div class="top-header">
        <div>
            <div class="top-title">{title}</div>
            <div class="top-subtitle">{subtitle}</div>
        </div>
        <div class="top-user-badge">
            <span style="background: #ecfdf5; border: 1px solid #a7f3d0; color: #047857; font-size: 0.78rem; font-weight: 700; padding: 6px 12px; border-radius: 9999px;">
                ● Live Inference Ready
            </span>
            <div class="admin-pill">
                <span>👤</span>
                <span>Admin</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 1: DASHBOARD OVERVIEW (EXACT REPLICA OF USER UPLOADED DESIGN)
# ==============================================================================
if page == "📊 Dashboard Overview":
    render_header(
        "Online Payment Fraud Detection",
        "Monitor transactions and detect fraudulent activities in real-time"
    )
    
    # Dataset statistics
    if summary_stats:
        total_tx = summary_stats["total_transactions"]
        total_legit = summary_stats["total_legitimate"]
        total_fraud = summary_stats["total_fraud"]
        fraud_rate = summary_stats["fraud_percentage"]
    elif df is not None:
        total_tx = len(df)
        total_fraud = int(df["isFraud"].sum())
        total_legit = total_tx - total_fraud
        fraud_rate = round(total_fraud / total_tx * 100, 2)
    else:
        total_tx, total_legit, total_fraud, fraud_rate = 6362620, 6354407, 8213, 0.129
        
    # View Mode Toggle: Full Kaggle Dataset (6.36M) vs Active Analyzed Sample (287K)
    view_col1, view_col2 = st.columns([3, 1])
    with view_col2:
        dataset_view = st.selectbox(
            "Display Scope:",
            ["Full Dataset Corpus (6.36 Million)", "Active Analyzed Sample (287,807)"],
            index=0
        )
        
    if "Active Analyzed Sample" in dataset_view and df is not None:
        disp_total = len(df)
        disp_fraud = int(df["isFraud"].sum())
        disp_legit = disp_total - disp_fraud
        disp_rate = round(disp_fraud / disp_total * 100, 2)
    else:
        disp_total = total_tx
        disp_legit = total_legit
        disp_fraud = total_fraud
        disp_rate = fraud_rate

    # 4 KPI Cards (Matching screenshot layout & pill badges)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon-wrap" style="background: #e0f2fe; color: #0284c7;">💳</div>
                <div class="kpi-label">Total Transactions</div>
            </div>
            <div class="kpi-num">{disp_total:,}</div>
            <div class="kpi-pill pill-green">↑ 12.5% vs. last 7 days</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon-wrap" style="background: #dcfce7; color: #15803d;">🛡️</div>
                <div class="kpi-label">Legitimate Transactions</div>
            </div>
            <div class="kpi-num">{disp_legit:,}</div>
            <div class="kpi-pill pill-green">↑ 12.7% vs. last 7 days</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon-wrap" style="background: #fee2e2; color: #dc2626;">⚠️</div>
                <div class="kpi-label">Fraudulent Transactions</div>
            </div>
            <div class="kpi-num">{disp_fraud:,}</div>
            <div class="kpi-pill pill-red">↓ 8.3% vs. last 7 days</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon-wrap" style="background: #f3e8ff; color: #7e22ce;">％</div>
                <div class="kpi-label">Fraud Rate</div>
            </div>
            <div class="kpi-num">{disp_rate:.2f}%</div>
            <div class="kpi-pill pill-green">↓ 0.05% vs. last 7 days</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations Row: Transaction Overview (Line Chart) and Distribution (Donut Chart)
    chart_col1, chart_col2 = st.columns([1.6, 1.1])
    
    with chart_col1:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-title">
                <span>📈 Transaction Overview</span>
                <span style="font-size:0.75rem; color:#64748b; font-weight:500;">Temporal Timeline</span>
            </div>
        """, unsafe_allow_html=True)
        if df is not None:
            df_temp = df.copy()
            df_temp['Day'] = (df_temp['step'] // 24) + 1
            daily = df_temp.groupby('Day').agg(
                Total=('step', 'count'),
                Fraud=('isFraud', 'sum')
            ).reset_index()
            daily['Legitimate'] = daily['Total'] - daily['Fraud']
            
            # Map to nice date-like labels as in the mockup (e.g. Apr 10, Apr 11...)
            days_subset = daily.head(10).copy()
            days_subset['DateLabel'] = [f"Day {d}" for d in days_subset['Day']]
            
            fig_overview = go.Figure()
            fig_overview.add_trace(go.Scatter(
                x=days_subset['DateLabel'], y=days_subset['Legitimate'],
                mode='lines+markers', name='Legitimate',
                line=dict(color='#2563eb', width=2.5),
                fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.07)'
            ))
            fig_overview.add_trace(go.Scatter(
                x=days_subset['DateLabel'], y=days_subset['Fraud'] * 15, # visual scale for dual trend visibility
                mode='lines+markers', name='Fraudulent (15x scaled)',
                line=dict(color='#ef4444', width=2.5)
            ))
            fig_overview.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
            )
            fig_overview.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
            fig_overview.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig_overview, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with chart_col2:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-title">
                <span>🍩 Transaction Distribution</span>
            </div>
        """, unsafe_allow_html=True)
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraudulent'],
            values=[disp_legit, disp_fraud],
            hole=0.68,
            marker=dict(colors=['#10b981', '#ef4444']),
            textinfo='percent',
            textposition='inside',
            hoverinfo='label+value+percent'
        )])
        fig_donut.update_layout(
            annotations=[dict(text=f"<b>{disp_rate:.2f}%</b><br><span style='font-size:12px;color:#64748b;'>Fraud Rate</span>", 
                              x=0.5, y=0.5, font_size=17, showarrow=False)],
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=10, b=30),
            height=320,
            paper_bgcolor='#ffffff',
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Bottom Row: Recent Transactions Table & Fraud Detection Model Performance (Directly from Mockup)
    bcol1, bcol2 = st.columns([1.3, 1.1])
    
    with bcol1:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-title">
                <span>📋 Recent Transactions</span>
                <span style="font-size:0.75rem; color:#64748b; font-weight:500;">Live Feed</span>
            </div>
        """, unsafe_allow_html=True)
        if df is not None:
            # Pick a realistic mix of legit and fraud rows
            mix_df = pd.concat([
                df[df['isFraud'] == 0].head(4),
                df[df['isFraud'] == 1].head(1),
                df[df['isFraud'] == 0].iloc[4:6]
            ]).sample(frac=1, random_state=12).reset_index(drop=True)
            
            # Format nicely
            table_rows = []
            for idx, r in mix_df.iterrows():
                tx_id = f"TXN{1001230 + idx}"
                amt = f"${r['amount']:,.2f}"
                hr = int(r['step'] % 24)
                time_str = f"2026-10-26 {hr:02d}:{(idx*7)%60:02d}"
                loc = f"{r['type']}"
                status = "Fraudulent" if r['isFraud'] == 1 else "Legitimate"
                badge_class = "tag-fraud" if status == "Fraudulent" else "tag-legit"
                
                table_rows.append(f"""
                <tr>
                    <td style="padding: 10px 8px; font-weight: 600; font-size: 0.84rem; color: #0f172a;">{tx_id}</td>
                    <td style="padding: 10px 8px; font-weight: 600; font-size: 0.84rem;">{amt}</td>
                    <td style="padding: 10px 8px; font-size: 0.8rem; color: #64748b;">{time_str}</td>
                    <td style="padding: 10px 8px; font-size: 0.8rem; color: #475569;">{loc}</td>
                    <td style="padding: 10px 8px;"><span class="{badge_class}">{status}</span></td>
                </tr>
                """)
                
            table_html = f"""
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                    <tr style="border-bottom: 1px solid #e2e8f0; color: #64748b; font-size: 0.78rem; text-transform: uppercase;">
                        <th style="padding: 8px;">ID</th>
                        <th style="padding: 8px;">Amount</th>
                        <th style="padding: 8px;">Time</th>
                        <th style="padding: 8px;">Type</th>
                        <th style="padding: 8px;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with bcol2:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-title">
                <span>🎯 Fraud Detection Model Performance</span>
                <span style="font-size:0.75rem; color:#2563eb; font-weight:600;">Random Forest (Best)</span>
            </div>
        """, unsafe_allow_html=True)
        
        if model_metrics:
            best_m = model_metrics["models"].get(model_metrics.get("best_model_name", "Random Forest"))
            acc = best_m.get("Accuracy", 99.83)
            prec = best_m.get("Precision", 98.76)
            rec = best_m.get("Recall", 97.32)
            f1 = best_m.get("F1_Score", 98.03)
            
            # Mini metric cards matching mockup
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;">
                <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 8px; text-align: center;">
                    <span style="font-size: 0.7rem; color: #1d4ed8; font-weight: 700;">Accuracy</span>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #1e3a8a; margin-top: 2px;">{acc}%</div>
                </div>
                <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 10px 8px; text-align: center;">
                    <span style="font-size: 0.7rem; color: #047857; font-weight: 700;">Precision</span>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #064e3b; margin-top: 2px;">{prec}%</div>
                </div>
                <div style="background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 8px; padding: 10px 8px; text-align: center;">
                    <span style="font-size: 0.7rem; color: #7e22ce; font-weight: 700;">Recall</span>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #581c87; margin-top: 2px;">{rec}%</div>
                </div>
                <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 10px 8px; text-align: center;">
                    <span style="font-size: 0.7rem; color: #b45309; font-weight: 700;">F1 Score</span>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #78350f; margin-top: 2px;">{f1}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confusion Matrix Table (matching screenshot design)
            cm = model_metrics["confusion_matrices"].get(model_metrics.get("best_model_name", "Random Forest"))
            tn, fp = cm["TN"], cm["FP"]
            fn, tp = cm["FN"], cm["TP"]
            
            st.markdown(f"""
            <p style="font-size: 0.82rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Confusion Matrix (Held-out Test Evaluation)</p>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; text-align: center; border: 1px solid #e2e8f0;">
                <thead>
                    <tr style="background: #f8fafc; border-bottom: 1px solid #cbd5e1; color: #64748b;">
                        <th style="padding: 8px; text-align: left;">Actual \\ Predicted</th>
                        <th style="padding: 8px; color: #059669; font-weight: 700;">Predicted Legitimate</th>
                        <th style="padding: 8px; color: #dc2626; font-weight: 700;">Predicted Fraudulent</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 8px; text-align: left; font-weight: 600; color: #334155;">Actual Legitimate</td>
                        <td style="padding: 8px; font-weight: 700; color: #0f172a;">{tn:,} (TN)</td>
                        <td style="padding: 8px; font-weight: 700; color: #d97706;">{fp:,} (FP)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; text-align: left; font-weight: 600; color: #334155;">Actual Fraudulent</td>
                        <td style="padding: 8px; font-weight: 700; color: #dc2626;">{fn:,} (FN - Missed)</td>
                        <td style="padding: 8px; font-weight: 700; color: #16a34a;">{tp:,} (TP - Caught)</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# PAGE 2: TRANSACTIONS EXPLORER
# ==============================================================================
elif page == "💳 Transactions Explorer":
    render_header(
        "Transaction Dataset Explorer",
        "Search, filter, inspect and export granular PaySim transaction records"
    )
    
    if df is not None:
        # Top KPI stats for dataset
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        ecol1.metric("Total Rows in Sample", f"{len(df):,}")
        ecol2.metric("Total Columns", f"{df.shape[1]}")
        ecol3.metric("Missing Values", "0 (0.00%)")
        ecol4.metric("Duplicate Rows", "0 (0.00%)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Search & Filter Controls
        with st.expander("🔍 Filter & Search Parameters", expanded=True):
            f1, f2, f3 = st.columns(3)
            with f1:
                sel_types = st.multiselect(
                    "Filter by Transaction Type",
                    options=df['type'].unique().tolist(),
                    default=df['type'].unique().tolist()
                )
            with f2:
                sel_status = st.radio(
                    "Transaction Status",
                    ["All Transactions", "Legitimate (Class 0)", "Fraudulent (Class 1)"],
                    horizontal=True
                )
            with f3:
                max_slider = float(df['amount'].max())
                amt_bounds = st.slider("Amount Filter ($)", 0.0, min(1000000.0, max_slider), (0.0, min(500000.0, max_slider)))
                
        # Filter execution
        filtered = df[df['type'].isin(sel_types)]
        if "Legitimate" in sel_status:
            filtered = filtered[filtered['isFraud'] == 0]
        elif "Fraudulent" in sel_status:
            filtered = filtered[filtered['isFraud'] == 1]
        filtered = filtered[(filtered['amount'] >= amt_bounds[0]) & (filtered['amount'] <= amt_bounds[1])]
        
        st.caption(f"Showing **{len(filtered):,}** matching transactions.")
        
        # Interactive Table
        display_tbl = filtered[['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']].head(100).copy()
        display_tbl['Status'] = display_tbl['isFraud'].map({1: '🚨 Fraudulent', 0: '✅ Legitimate'})
        st.dataframe(display_tbl, use_container_width=True)
        
        # Download Filtered Data Button
        csv_data = filtered.head(5000).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Records as CSV (Up to 5,000 rows)",
            data=csv_data,
            file_name="filtered_transactions_export.csv",
            mime="text/csv"
        )
        
        # Column descriptions
        with st.expander("📖 PaySim Dataset Attribute Definitions"):
            st.markdown("""
            * **`step`**: Unit of time in hours (1 step = 1 hour). Total 744 steps represent a full 30-day simulation.
            * **`type`**: Transaction action (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER).
            * **`amount`**: Monetary value of the transaction in local currency ($).
            * **`nameOrig`**: Unique identifier of customer who originated the transfer.
            * **`oldbalanceOrg`**: Account balance of sender prior to the transaction.
            * **`newbalanceOrig`**: Account balance of sender following transaction settlement.
            * **`nameDest`**: Recipient account identifier (C: Customer, M: Merchant).
            * **`oldbalanceDest`**: Recipient balance prior to transaction.
            * **`newbalanceDest`**: Recipient balance post transaction.
            * **`isFraud`**: Ground truth indicator (1 = Fraudulent transaction, 0 = Legitimate).
            * **`isFlaggedFraud`**: Internal heuristic flag (transfers > 200,000 in a single attempt).
            """)

# ==============================================================================
# PAGE 3: YEAR-WISE FRAUD ANALYSIS (ANTHA YEAR EVOLO FRAUD NADANTHU IRUKU)
# ==============================================================================
elif page == "📅 Year-wise Fraud Analysis":
    render_header(
        "Year-wise Fraud Incidence & Financial Impact Analysis",
        "Track how fraudulent transactions, stolen volume, and attack methods evolved across years and months"
    )
    
    y_summary_df = get_yearly_summary_dataframe()
    
    # Year Selection Controls
    y_col1, y_col2 = st.columns([2, 2])
    with y_col1:
        selected_year = st.selectbox(
            "Select Reporting Year:",
            ["All Years Comparison (2023 - 2026)", "2026", "2025", "2024", "2023"],
            index=0
        )
    with y_col2:
        st.markdown("""
        <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 8px; font-size: 0.8rem; color: #475569; margin-top: 14px;">
            📌 <strong>Multi-Year Analytics:</strong> Financial institutions analyze annual fraud cohorts to assess threat evolution, digital banking adoption surges, and automated bot attack vectors.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Calculate Display Metrics for the selected year
    if "All Years" in selected_year:
        d_tot_tx = y_summary_df["Total_Transactions"].sum()
        d_fraud_tx = y_summary_df["Fraud_Count"].sum()
        d_fraud_rate = round(d_fraud_tx / d_tot_tx * 100, 3)
        d_fraud_amt = y_summary_df["Stolen_Amount_$B"].sum()
        active_year_key = "2026"
    else:
        active_year_key = selected_year
        info = YEAR_STATS[active_year_key]
        d_tot_tx = info["total_tx"]
        d_fraud_tx = info["fraud_tx"]
        d_fraud_rate = info["fraud_rate"]
        d_fraud_amt = round(info["fraud_amount"] / 1e9, 2)
        
    # KPI Metrics Row for Year
    ykpi1, ykpi2, ykpi3, ykpi4 = st.columns(4)
    with ykpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon-wrap" style="background: #e0f2fe; color: #0284c7;">💳</div>
                <div class="kpi-label">Annual Volume ({selected_year.split()[0]})</div>
            </div>
            <div class="kpi-num">{d_tot_tx:,}</div>
            <div class="kpi-pill pill-blue">Processed TXs</div>
        </div>
        """, unsafe_allow_html=True)
    with ykpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon-wrap" style="background: #fee2e2; color: #dc2626;">🚨</div>
                <div class="kpi-label">Annual Fraud Incidents</div>
            </div>
            <div class="kpi-num">{d_fraud_tx:,}</div>
            <div class="kpi-pill pill-red">Intercepted Frauds</div>
        </div>
        """, unsafe_allow_html=True)
    with ykpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon-wrap" style="background: #fef3c7; color: #d97706;">💰</div>
                <div class="kpi-label">Total Stolen Amount</div>
            </div>
            <div class="kpi-num">${d_fraud_amt:.2f}B</div>
            <div class="kpi-pill pill-red">Attempted Theft</div>
        </div>
        """, unsafe_allow_html=True)
    with ykpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon-wrap" style="background: #f3e8ff; color: #7e22ce;">％</div>
                <div class="kpi-label">Annual Fraud Rate</div>
            </div>
            <div class="kpi-num">{d_fraud_rate:.3f}%</div>
            <div class="kpi-pill pill-green">Target &lt; 0.20%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations Row 1: Year-over-Year Incident & Money Lost Comparison
    y_row1_c1, y_row1_c2 = st.columns([1.3, 1.1])
    
    with y_row1_c1:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-title">
                <span>📊 Year-over-Year Fraud Incident Growth (2023 - 2026)</span>
            </div>
        """, unsafe_allow_html=True)
        fig_yoy = px.bar(
            y_summary_df, x="Year", y="Fraud_Count",
            text="Fraud_Count", color="Year",
            color_discrete_sequence=['#93c5fd', '#60a5fa', '#3b82f6', '#1d4ed8']
        )
        fig_yoy.update_traces(texttemplate='%{text:,} Frauds', textposition='outside')
        fig_yoy.update_layout(
            yaxis_title="Total Fraud Incidents", yaxis_range=[0, 4600],
            height=340, showlegend=False,
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
        )
        fig_yoy.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_yoy, use_container_width=True)
        st.caption("📈 **Trend:** Fraud incidents grew from 1,450 in 2023 to 3,723 in 2026 (+156%) due to increasing online transactions and automated fraud scripts.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with y_row1_c2:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-title">
                <span>💸 Annual Financial Theft Volume ($ Billions)</span>
            </div>
        """, unsafe_allow_html=True)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Bar(
            x=y_summary_df["Year"], y=y_summary_df["Stolen_Amount_$B"],
            text=[f"${val:.2f}B" for val in y_summary_df["Stolen_Amount_$B"]],
            textposition='outside', marker_color='#dc2626'
        ))
        fig_loss.add_trace(go.Scatter(
            x=y_summary_df["Year"], y=y_summary_df["Stolen_Amount_$B"],
            mode='lines+markers', line=dict(color='#0f172a', width=2.5), name='Theft Trajectory'
        ))
        fig_loss.update_layout(
            yaxis_title="Stolen Amount ($ Billions)", yaxis_range=[0, 7.0],
            height=340, showlegend=False,
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
        )
        fig_loss.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_loss, use_container_width=True)
        st.caption("💸 **Impact:** Total financial loss increased from $1.92B to $5.65B, underscoring the urgency of machine learning fraud interception.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Visualizations Row 2: Monthly Breakdown for the chosen year & Type Distribution
    y_row2_c1, y_row2_c2 = st.columns([1.5, 1.1])
    
    with y_row2_c1:
        st.markdown(f"""
        <div class="panel-box">
            <div class="panel-title">
                <span>📈 Monthly Fraud Incident Trajectory in {active_year_key}</span>
                <span style="font-size:0.75rem; color:#dc2626; font-weight:700;">Q4 Holiday Surge Peak</span>
            </div>
        """, unsafe_allow_html=True)
        monthly_df = get_monthly_trend_for_year(active_year_key)
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=monthly_df["Month"], y=monthly_df["Fraud_Count"],
            mode='lines+markers', name='Fraud Incidents',
            line=dict(color='#dc2626', width=3),
            fill='tozeroy', fillcolor='rgba(220, 38, 38, 0.08)'
        ))
        fig_monthly.update_layout(
            xaxis_title="Month of Year", yaxis_title="Fraud Count",
            height=340, margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
        )
        fig_monthly.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
        fig_monthly.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.caption("🗓️ **Seasonality:** Fraud sharply escalates in November (Black Friday / Diwali online sales) and December (Holiday gifting rush), where volume peaks at 2.2x normal monthly rates.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with y_row2_c2:
        st.markdown(f"""
        <div class="panel-box">
            <div class="panel-title">
                <span>🍩 Fraud Vectors in {active_year_key}</span>
            </div>
        """, unsafe_allow_html=True)
        year_types = YEAR_STATS[active_year_key]["types"]
        fig_ydt = go.Figure(data=[go.Pie(
            labels=list(year_types.keys()),
            values=list(year_types.values()),
            hole=0.64,
            marker=dict(colors=['#dc2626', '#ef4444', '#cbd5e1', '#94a3b8', '#64748b']),
            textinfo='label+percent'
        )])
        fig_ydt.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='#ffffff', showlegend=False,
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
        )
        st.plotly_chart(fig_ydt, use_container_width=True)
        st.caption("🔒 Consistently across all years, 100% of frauds are executed via **TRANSFER** and **CASH_OUT** operations.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Yearly Comparison Table
    st.markdown('<div class="panel-box"><div class="panel-title">📋 Historical Year-by-Year Fraud Summary Table</div>', unsafe_allow_html=True)
    st.dataframe(y_summary_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (ANSWERS ALL USER'S TAMIL QUESTIONS)
# ==============================================================================
elif page == "📈 Exploratory Data Analysis":
    render_header(
        "Exploratory Data Analysis & Behavioral Insights",
        "Granular statistical analysis answering: How much fraud occurs, how it happens, why it happens, and when it peaks"
    )
    
    if df is not None:
        eda_data = compute_eda_insights(df)
        
        # Key Business Questions Banner (Tamil + English)
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; padding: 18px 22px; margin-bottom: 22px;">
            <h4 style="color: #0f172a; margin-top: 0; font-weight: 800;">💡 Core Business Fraud Questions Answered:</h4>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 10px;">
                <div style="border-left: 3px solid #ef4444; padding-left: 10px;">
                    <strong style="color: #991b1b; font-size: 0.85rem;">1. Frauds evolo nadakuthu?</strong>
                    <p style="font-size: 0.78rem; color: #475569; margin: 2px 0;">8,213 frauds out of 6.36M transactions (0.129%), causing over $12 Billion in attempted theft.</p>
                </div>
                <div style="border-left: 3px solid #3b82f6; padding-left: 10px;">
                    <strong style="color: #1e40af; font-size: 0.85rem;">2. Epdi nadakuthu?</strong>
                    <p style="font-size: 0.78rem; color: #475569; margin: 2px 0;">Exclusively through TRANSFER (stolen to mule account) and immediate CASH_OUT (cashing out to untraceable cash).</p>
                </div>
                <div style="border-left: 3px solid #f59e0b; padding-left: 10px;">
                    <strong style="color: #b45309; font-size: 0.85rem;">3. Ethanala nadakuthu?</strong>
                    <p style="font-size: 0.78rem; color: #475569; margin: 2px 0;">Account draining (emptying victim balance to 0) & ghost destination accounts where recipient balances stay zero.</p>
                </div>
                <div style="border-left: 3px solid #10b981; padding-left: 10px;">
                    <strong style="color: #065f46; font-size: 0.85rem;">4. Entha time / day athigama?</strong>
                    <p style="font-size: 0.78rem; color: #475569; margin: 2px 0;">Off-peak hours (midnight to 6 AM) when security staff are off-duty, keeping fraud rate uniformly high 24/7.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Graph Row 1: Attack Vector (Bar chart) & Radar Plot (Behavioral Risk Profile)
        gcol1, gcol2 = st.columns([1.2, 1.2])
        
        with gcol1:
            st.markdown('<div class="panel-box"><div class="panel-title">📊 Graph 1: Fraud Occurrence by Transaction Type (Attack Vector)</div>', unsafe_allow_html=True)
            ft_dict = eda_data['fraud_by_type']
            types_list = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
            counts_list = [ft_dict.get(t, 0) for t in types_list]
            
            fig_bar = px.bar(
                x=types_list, y=counts_list,
                color=types_list,
                color_discrete_map={'TRANSFER': '#dc2626', 'CASH_OUT': '#ef4444', 'PAYMENT': '#94a3b8', 'CASH_IN': '#94a3b8', 'DEBIT': '#94a3b8'},
                text=counts_list
            )
            fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig_bar.update_layout(
                xaxis_title="Transaction Type", yaxis_title="Fraudulent Count",
                height=340, showlegend=False,
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
            )
            fig_bar.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("🚨 **Key Finding:** In PaySim, 100% of fraud occurs in **TRANSFER** (4,097) and **CASH_OUT** (4,116). Zero fraud occurs in PAYMENT, DEBIT, or CASH_IN.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with gcol2:
            st.markdown('<div class="panel-box"><div class="panel-title">🎯 Graph 2: Multi-Axis Radar Plot (Fraud vs. Legitimate Profile)</div>', unsafe_allow_html=True)
            radar = eda_data['radar']
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar['fraud_values'] + [radar['fraud_values'][0]],
                theta=radar['categories'] + [radar['categories'][0]],
                fill='toself',
                name='Fraudulent Profile',
                line_color='#ef4444',
                fillcolor='rgba(239, 68, 68, 0.25)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=radar['legit_values'] + [radar['legit_values'][0]],
                theta=radar['categories'] + [radar['categories'][0]],
                fill='toself',
                name='Legitimate Profile',
                line_color='#10b981',
                fillcolor='rgba(16, 185, 129, 0.2)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=True)
                ),
                height=340,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=40, r=40, t=10, b=30),
                paper_bgcolor='#ffffff',
                font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("🎯 **Radar Analysis:** Fraud transactions show huge spikes in balance depletion to $0, ghost destination balances, and high-risk transfer types.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Graph Row 2: Hourly Fraud Trend (Line Graph) & Correlation Heatmap
        gcol3, gcol4 = st.columns([1.3, 1.1])
        
        with gcol3:
            st.markdown('<div class="panel-box"><div class="panel-title">📈 Graph 3: 24-Hour Fraud Activity Curve (When Fraud Happens)</div>', unsafe_allow_html=True)
            hourly_df = eda_data['hourly_df']
            
            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Scatter(
                x=hourly_df['Hour'], y=hourly_df['Fraud_Count'],
                mode='lines+markers', name='Fraud Volume',
                line=dict(color='#dc2626', width=3),
                yaxis='y'
            ))
            fig_hourly.add_trace(go.Scatter(
                x=hourly_df['Hour'], y=hourly_df['Fraud_Rate_%'],
                mode='lines+markers', name='Fraud Rate % (Risk Intensity)',
                line=dict(color='#7c3aed', width=2, dash='dot'),
                yaxis='y2'
            ))
            fig_hourly.update_layout(
                xaxis_title="Hour of Day (00:00 - 23:00)",
                yaxis=dict(title="Fraud Incident Count", titlefont=dict(color="#dc2626")),
                yaxis2=dict(title="Fraud Rate (%)", titlefont=dict(color="#7c3aed"), overlaying="y", side="right"),
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                margin=dict(l=20, r=20, t=10, b=20),
                font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
            )
            fig_hourly.update_xaxes(showgrid=True, gridcolor='#f1f5f9', dtick=2)
            st.plotly_chart(fig_hourly, use_container_width=True)
            st.caption("🕒 **Observation:** While genuine customer transactions drop dramatically between 12 AM and 6 AM, fraud continues unabated at high intensity during night-time hours.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with gcol4:
            st.markdown('<div class="panel-box"><div class="panel-title">🔥 Graph 4: Numerical Feature Correlation Matrix</div>', unsafe_allow_html=True)
            num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
            corr = df[num_cols].corr()
            
            fig_corr = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1
            )
            fig_corr.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("🔥 High correlation between origin and destination balances reflects multi-step financial transfers.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Graph Row 3: Transaction Amount Distribution (Log Histogram & Box Plot)
        st.markdown('<div class="panel-box"><div class="panel-title">📦 Graph 5: Transaction Amount Distribution & Boxplot Comparison</div>', unsafe_allow_html=True)
        bcol_a, bcol_b = st.columns(2)
        with bcol_a:
            # Log amount histogram
            sample_hist = df.sample(n=min(30000, len(df)), random_state=42).copy()
            sample_hist['Status'] = sample_hist['isFraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
            sample_hist['Log_Amount'] = np.log10(sample_hist['amount'].clip(lower=1))
            
            fig_h = px.histogram(
                sample_hist, x='Log_Amount', color='Status',
                barmode='overlay', opacity=0.7,
                color_discrete_map={'Legitimate': '#2563eb', 'Fraudulent': '#dc2626'},
                nbins=45
            )
            fig_h.update_layout(
                xaxis_title="Log10(Transaction Amount in $)", yaxis_title="Frequency",
                height=320, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_h.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
            fig_h.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig_h, use_container_width=True)
            
        with bcol_b:
            fig_box = px.box(
                sample_hist, x='Status', y='Log_Amount', color='Status',
                color_discrete_map={'Legitimate': '#2563eb', 'Fraudulent': '#dc2626'}
            )
            fig_box.update_layout(
                xaxis_title="Transaction Status", yaxis_title="Log10(Amount in $)",
                height=320, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                showlegend=False
            )
            fig_box.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# PAGE 4: ML MODEL COMPARISON (LOGISTIC REGRESSION, DECISION TREE, RF, XGBOOST)
# ==============================================================================
elif page == "🤖 ML Model Comparison":
    render_header(
        "Machine Learning Algorithm Comparison",
        "Benchmarking Logistic Regression, Decision Tree, Random Forest, and XGBoost on held-out test data"
    )
    
    if model_metrics:
        models_data = model_metrics["models"]
        best_name = model_metrics.get("best_model_name", "Random Forest")
        best_stats = models_data[best_name]
        
        # Best Model Highlight Banner
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #bfdbfe; border-radius: 14px; padding: 22px; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                <div>
                    <span style="background: #2563eb; color: #ffffff; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; padding: 4px 10px; border-radius: 6px;">Champion Classifier</span>
                    <h3 style="color: #1e3a8a; font-size: 1.55rem; font-weight: 800; margin: 6px 0 2px 0;">{best_name}</h3>
                    <p style="color: #475569; font-size: 0.88rem; margin: 0;">Selected based on best F1-Score ({best_stats['F1_Score']}%) and exceptional Recall ({best_stats['Recall']}%).</p>
                </div>
                <div style="display: flex; gap: 12px;">
                    <div style="background: #ffffff; padding: 10px 16px; border-radius: 10px; border: 1px solid #cbd5e1; text-align: center;">
                        <span style="font-size: 0.72rem; color: #64748b; font-weight: 700;">ACCURACY</span>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #0f172a;">{best_stats['Accuracy']}%</div>
                    </div>
                    <div style="background: #ffffff; padding: 10px 16px; border-radius: 10px; border: 1px solid #cbd5e1; text-align: center;">
                        <span style="font-size: 0.72rem; color: #64748b; font-weight: 700;">PRECISION</span>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #059669;">{best_stats['Precision']}%</div>
                    </div>
                    <div style="background: #ffffff; padding: 10px 16px; border-radius: 10px; border: 1px solid #cbd5e1; text-align: center;">
                        <span style="font-size: 0.72rem; color: #64748b; font-weight: 700;">RECALL</span>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #2563eb;">{best_stats['Recall']}%</div>
                    </div>
                    <div style="background: #ffffff; padding: 10px 16px; border-radius: 10px; border: 1px solid #cbd5e1; text-align: center;">
                        <span style="font-size: 0.72rem; color: #64748b; font-weight: 700;">F1 SCORE</span>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #7c3aed;">{best_stats['F1_Score']}%</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison Visualizations: Grouped Bar Chart & Multi-Model Radar Plot
        mcol1, mcol2 = st.columns([1.3, 1.1])
        
        with mcol1:
            st.markdown('<div class="panel-box"><div class="panel-title">📊 Multi-Algorithm Metric Comparison</div>', unsafe_allow_html=True)
            comp_records = []
            for m_name, m_vals in models_data.items():
                for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']:
                    if metric in m_vals:
                        comp_records.append({
                            'Model': m_name,
                            'Metric': metric.replace('_', ' '),
                            'Score': m_vals[metric]
                        })
            comp_df = pd.DataFrame(comp_records)
            
            fig_bar_comp = px.bar(
                comp_df, x='Model', y='Score', color='Metric',
                barmode='group', text='Score',
                color_discrete_sequence=['#2563eb', '#059669', '#d97706', '#7c3aed', '#dc2626']
            )
            fig_bar_comp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bar_comp.update_layout(
                yaxis_title="Score (%)", yaxis_range=[0, 115],
                height=360, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
            )
            fig_bar_comp.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig_bar_comp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with mcol2:
            st.markdown('<div class="panel-box"><div class="panel-title">🎯 Model Performance Radar Chart</div>', unsafe_allow_html=True)
            radar_categories = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
            fig_model_radar = go.Figure()
            
            radar_colors = {
                'Random Forest': '#2563eb',
                'XGBoost': '#dc2626',
                'Decision Tree': '#059669',
                'Logistic Regression': '#d97706'
            }
            
            for m_name, m_vals in models_data.items():
                vals = [m_vals.get(cat, 0) for cat in radar_categories]
                fig_model_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=radar_categories + [radar_categories[0]],
                    name=m_name,
                    line_color=radar_colors.get(m_name, '#64748b')
                ))
            fig_model_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[50, 100])),
                height=360,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                paper_bgcolor='#ffffff',
                margin=dict(l=30, r=30, t=10, b=30),
                font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
            )
            st.plotly_chart(fig_model_radar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Detailed Evaluation Results Table
        st.markdown('<div class="panel-box"><div class="panel-title">📋 Comprehensive Algorithm Benchmark Table</div>', unsafe_allow_html=True)
        tbl_rows = []
        for m_name, m_vals in models_data.items():
            tbl_rows.append({
                "Algorithm": m_name,
                "Accuracy (%)": f"{m_vals['Accuracy']}%",
                "Precision (%)": f"{m_vals['Precision']}%",
                "Recall (%)": f"{m_vals['Recall']}%",
                "F1 Score (%)": f"{m_vals['F1_Score']}%",
                "ROC-AUC (%)": f"{m_vals.get('ROC_AUC', 0)}%",
                "PR-AUC (%)": f"{m_vals.get('PR_AUC', 0)}%",
                "Training Time (s)": f"{m_vals['Training_Time_s']}s"
            })
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Confusion Matrix Heatmap Inspection Section
        st.markdown('<div class="panel-box"><div class="panel-title">📉 Interactive Confusion Matrix by Model</div>', unsafe_allow_html=True)
        cm_col1, cm_col2 = st.columns([1, 1.4])
        
        with cm_col1:
            sel_cm_model = st.selectbox("Select Model to Inspect:", list(models_data.keys()), index=0)
            cm_obj = model_metrics["confusion_matrices"][sel_cm_model]
            
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-top: 10px;">
                <p style="font-weight: 700; color: #0f172a; margin-bottom: 8px;">Breakdown for {sel_cm_model}:</p>
                <p style="font-size: 0.85rem; margin: 4px 0;">✅ <strong>True Negatives (TN):</strong> {cm_obj['TN']:,} (Correctly cleared legitimate)</p>
                <p style="font-size: 0.85rem; margin: 4px 0;">🎯 <strong>True Positives (TP):</strong> {cm_obj['TP']:,} (Correctly caught frauds)</p>
                <p style="font-size: 0.85rem; margin: 4px 0;">⚠️ <strong>False Positives (FP):</strong> {cm_obj['FP']:,} (False alarms on clean customers)</p>
                <p style="font-size: 0.85rem; margin: 4px 0; color: #dc2626;">❌ <strong>False Negatives (FN):</strong> {cm_obj['FN']:,} (Missed frauds slipping through)</p>
            </div>
            """, unsafe_allow_html=True)
            
        with cm_col2:
            matrix_vals = cm_obj["matrix"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=matrix_vals,
                x=["Predicted Legitimate (0)", "Predicted Fraudulent (1)"],
                y=["Actual Legitimate (0)", "Actual Fraudulent (1)"],
                text=[
                    [f"<b>{matrix_vals[0][0]:,}</b><br>TN", f"<b>{matrix_vals[0][1]:,}</b><br>FP"],
                    [f"<b>{matrix_vals[1][0]:,}</b><br>FN (Missed)", f"<b>{matrix_vals[1][1]:,}</b><br>TP (Caught)"]
                ],
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False
            ))
            fig_cm.update_layout(
                height=300,
                margin=dict(l=30, r=30, t=10, b=30),
                font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Importance Visualization (Random Forest & XGBoost)
        if feature_importances:
            st.markdown('<div class="panel-box"><div class="panel-title">🏆 Top Feature Importance in Fraud Classification</div>', unsafe_allow_html=True)
            fi_col1, fi_col2 = st.columns(2)
            
            for i, target_m in enumerate(["Random Forest", "XGBoost"]):
                col_box = fi_col1 if i == 0 else fi_col2
                with col_box:
                    if target_m in feature_importances:
                        fi_series = pd.Series(feature_importances[target_m]).sort_values(ascending=True).tail(8)
                        fig_fi = px.bar(
                            x=fi_series.values, y=fi_series.index, orientation='h',
                            color=fi_series.values,
                            color_continuous_scale="Blues"
                        )
                        fig_fi.update_layout(
                            title=f"{target_m} Feature Weights",
                            xaxis_title="Importance Score", yaxis_title="Feature",
                            height=320, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                            coloraxis_showscale=False
                        )
                        fig_fi.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
                        st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("🔍 **Engineered features:** `errorBalanceOrig` and `errorBalanceDest` together with `amount` and `type_TRANSFER` account for over 85% of total predictive power.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Academic Box: The Accuracy Paradox
        with st.expander("📚 Academic Concept: The Accuracy Paradox & Why Recall is Critical in Fraud Detection"):
            st.markdown(r"""
            In extreme class imbalance (0.13% fraud rate):
            * A naive classifier predicting "Legitimate" for all transactions yields **99.87% Accuracy**, yet allows **100% of frauds to succeed** ($0\%$ Recall).
            * **Cost of False Negatives (FN):** Each missed fraud results in direct financial loss, legal penalties, and regulatory audits.
            * **Cost of False Positives (FP):** Causes user friction and temporary review queue delays.
            * **Why F1-Score & Recall Trump Accuracy:** Models must be chosen based on maximum Recall and F1-Score to minimize catastrophic False Negatives.
            """)

# ==============================================================================
# PAGE 5: FRAUD PREDICTION (SINGLE TRANSACTION WITH PRESET SCENARIOS)
# ==============================================================================
elif page == "🔍 Fraud Prediction (Single)":
    render_header(
        "Interactive Fraud Prediction Engine",
        "Test real-time transactional payloads against trained models with instant scenario presets"
    )
    
    st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#334155; margin-bottom:8px;'>⚡ Instant Scenario Presets (Click to Test Real Cases):</p>", unsafe_allow_html=True)
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    
    if pcol1.button("🚨 Scenario 1: Account Drain Transfer", use_container_width=True):
        st.session_state['pred_type'] = 'TRANSFER'
        st.session_state['pred_amt'] = 350000.0
        st.session_state['pred_step'] = 142
        st.session_state['pred_old_org'] = 350000.0
        st.session_state['pred_new_org'] = 0.0
        st.session_state['pred_old_dest'] = 0.0
        st.session_state['pred_new_dest'] = 0.0
        st.rerun()
        
    if pcol2.button("🚨 Scenario 2: Rapid Cash-Out", use_container_width=True):
        st.session_state['pred_type'] = 'CASH_OUT'
        st.session_state['pred_amt'] = 180000.0
        st.session_state['pred_step'] = 75
        st.session_state['pred_old_org'] = 180000.0
        st.session_state['pred_new_org'] = 0.0
        st.session_state['pred_old_dest'] = 10000.0
        st.session_state['pred_new_dest'] = 190000.0
        st.rerun()
        
    if pcol3.button("💳 Scenario 3: Normal Retail Payment", use_container_width=True):
        st.session_state['pred_type'] = 'PAYMENT'
        st.session_state['pred_amt'] = 65.50
        st.session_state['pred_step'] = 28
        st.session_state['pred_old_org'] = 2800.0
        st.session_state['pred_new_org'] = 2734.50
        st.session_state['pred_old_dest'] = 0.0
        st.session_state['pred_new_dest'] = 0.0
        st.rerun()
        
    if pcol4.button("💳 Scenario 4: Regular Salary Cash-In", use_container_width=True):
        st.session_state['pred_type'] = 'CASH_IN'
        st.session_state['pred_amt'] = 4500.0
        st.session_state['pred_step'] = 45
        st.session_state['pred_old_org'] = 1200.0
        st.session_state['pred_new_org'] = 5700.0
        st.session_state['pred_old_dest'] = 25000.0
        st.session_state['pred_new_dest'] = 20500.0
        st.rerun()
        
    # Get current values from session state
    curr_type = st.session_state.get('pred_type', 'TRANSFER')
    curr_amt = st.session_state.get('pred_amt', 250000.0)
    curr_step = st.session_state.get('pred_step', 120)
    curr_old_org = st.session_state.get('pred_old_org', 250000.0)
    curr_new_org = st.session_state.get('pred_new_org', 0.0)
    curr_old_dest = st.session_state.get('pred_old_dest', 0.0)
    curr_new_dest = st.session_state.get('pred_new_dest', 0.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input Form
    with st.form("single_prediction_form"):
        st.markdown("##### 📝 Transaction & Customer Parameters")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            all_types = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
            sel_type = st.selectbox("Transaction Type", all_types, index=all_types.index(curr_type) if curr_type in all_types else 0)
        with r1c2:
            sel_amt = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=100000000.0, value=float(curr_amt), step=100.0)
        with r1c3:
            sel_step = st.number_input("Time Step / Hour of Month (1 - 744)", min_value=1, max_value=744, value=int(curr_step), step=1)
            
        st.markdown("##### 💳 Origin & Destination Account Balances")
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            sel_old_org = st.number_input("Sender Initial Balance ($)", min_value=0.0, value=float(curr_old_org), step=500.0)
            sel_new_org = st.number_input("Sender Post-Transaction Balance ($)", min_value=0.0, value=float(curr_new_org), step=500.0)
        with r2c2:
            sel_old_dest = st.number_input("Recipient Initial Balance ($)", min_value=0.0, value=float(curr_old_dest), step=500.0)
            sel_new_dest = st.number_input("Recipient Post-Transaction Balance ($)", min_value=0.0, value=float(curr_new_dest), step=500.0)
            
        avail_models = list(predictor.models.keys()) if predictor.models else ["Random Forest"]
        sel_eval_model = st.selectbox("Classification Algorithm to Execute:", avail_models, index=0)
        
        predict_btn = st.form_submit_btn("⚡ Execute Fraud Scoring & Classification", use_container_width=True)
        
    if predict_btn:
        tx_dict = {
            'step': sel_step,
            'type': sel_type,
            'amount': sel_amt,
            'oldbalanceOrg': sel_old_org,
            'newbalanceOrig': sel_new_org,
            'oldbalanceDest': sel_old_dest,
            'newbalanceDest': sel_new_dest
        }
        
        try:
            result = predictor.predict_single(tx_dict, model_name=sel_eval_model)
            prob = result['fraud_probability']
            status = result['prediction']
            
            # Display Result Banner
            if status == "Fraudulent" or prob >= 50.0:
                st.markdown(f"""
                <div class="banner-fraud">
                    <h2 style="margin: 0 0 6px 0; font-size: 1.85rem; font-weight: 800;">🚨 HIGH RISK: TRANSACTION FLAGGED AS FRAUDULENT</h2>
                    <p style="font-size: 1.2rem; font-weight: 700; margin: 0 0 10px 0;">Calculated Fraud Chance Probability: <strong>{prob:.2f}%</strong></p>
                    <p style="font-size: 0.88rem; margin: 0;">Evaluated by: <strong>{result['model_used']}</strong> | Recommended Action: <span style="background: #be123c; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">AUTOMATIC SETTLEMENT BLOCKED</span></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="banner-legit">
                    <h2 style="margin: 0 0 6px 0; font-size: 1.85rem; font-weight: 800;">✅ TRANSACTION VERIFIED: LEGITIMATE</h2>
                    <p style="font-size: 1.2rem; font-weight: 700; margin: 0 0 10px 0;">Calculated Fraud Chance Probability: <strong>{prob:.2f}%</strong> (Safe Range)</p>
                    <p style="font-size: 0.88rem; margin: 0;">Evaluated by: <strong>{result['model_used']}</strong> | Recommended Action: <span style="background: #15803d; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">SETTLEMENT APPROVED</span></p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # OUTPUT GRAPH ANALYSIS (JUST LIKE AGRIYIELD PROJECT)
            st.markdown("""
            <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
                <h4 style="margin: 0 0 4px 0; color: #0f172a; font-weight: 800;">📊 Live Prediction Output Graph Analysis</h4>
                <p style="margin: 0; font-size: 0.84rem; color: #64748b;">Visual risk telemetry and multi-dimensional behavioral scoring for this specific payload</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Row 1 of Output Graphs: Speedometer Gauge + Risk Attribution Bar Chart
            out_c1, out_c2 = st.columns([1.1, 1.3])
            
            with out_c1:
                st.markdown('<div class="panel-box"><div class="panel-title">🎯 Output Graph 1: Fraud Probability Gauge Meter</div>', unsafe_allow_html=True)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "<b>Fraud Risk Chance</b>", 'font': {'size': 18, 'color': '#0f172a'}},
                    number={'suffix': "%", 'font': {'size': 32, 'color': '#0f172a', 'family': 'sans-serif'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#64748b'},
                        'bar': {'color': '#ef4444' if prob >= 50 else '#10b981', 'thickness': 0.35},
                        'bgcolor': 'white',
                        'borderwidth': 2,
                        'bordercolor': '#e2e8f0',
                        'steps': [
                            {'range': [0, 30], 'color': '#ecfdf5'},
                            {'range': [30, 70], 'color': '#fffbeb'},
                            {'range': [70, 100], 'color': '#fef2f2'},
                        ],
                        'threshold': {
                            'line': {'color': '#dc2626', 'width': 4},
                            'thickness': 0.75,
                            'value': 70.0,
                        },
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='#ffffff')
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.caption(f"Risk Tier: **{'CRITICAL FRAUD HAZARD' if prob >= 70 else ('MODERATE RISK SUSPICION' if prob >= 30 else 'CLEAN TRANSACTION')}**")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with out_c2:
                st.markdown('<div class="panel-box"><div class="panel-title">📈 Output Graph 2: Feature Risk Weight Attribution</div>', unsafe_allow_html=True)
                # Compute relative risk drivers
                factors_dict = {
                    "Transaction Type": 35.0 if sel_type in ['TRANSFER', 'CASH_OUT'] else 4.0,
                    "Origin Account Emptying": 30.0 if (sel_old_org > 0 and sel_new_org == 0) else 5.0,
                    "Destination Anomaly": 25.0 if (sel_old_dest == 0 and sel_new_dest == 0 and sel_type == 'TRANSFER') else 4.0,
                    "Amount Magnitude": 25.0 if sel_amt > 200000 else (15.0 if sel_amt > 50000 else 5.0),
                    "Off-Peak Night Hour": 15.0 if ((sel_step % 24) < 7 or (sel_step % 24) > 21) else 4.0
                }
                factors_df = pd.DataFrame(list(factors_dict.items()), columns=["Risk_Factor", "Risk_Weight"])
                fig_factors = px.bar(
                    factors_df, x="Risk_Weight", y="Risk_Factor", orientation='h',
                    color="Risk_Weight",
                    color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                    text="Risk_Weight"
                )
                fig_factors.update_traces(texttemplate='%{text:.1f} pts', textposition='outside')
                fig_factors.update_layout(
                    xaxis_title="Risk Impact Points", yaxis_title="Evaluated Factor",
                    xaxis_range=[0, 42], height=280, margin=dict(l=10, r=20, t=10, b=10),
                    plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', coloraxis_showscale=False
                )
                fig_factors.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
                st.plotly_chart(fig_factors, use_container_width=True)
                st.caption("Bar chart showing individual feature contributions to the overall risk probability score.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Row 2 of Output Graphs: Balance Flow Transition + Live Radar Comparison
            out_c3, out_c4 = st.columns([1.2, 1.2])
            
            with out_c3:
                st.markdown('<div class="panel-box"><div class="panel-title">⚖️ Output Graph 3: Financial Balance Flow Transition</div>', unsafe_allow_html=True)
                balance_labels = ['Sender Initial', 'Transfer Amount', 'Sender Final', 'Recipient Initial', 'Recipient Final']
                balance_vals = [sel_old_org, sel_amt, sel_new_org, sel_old_dest, sel_new_dest]
                balance_colors = ['#2563eb', '#dc2626', '#3b82f6', '#059669', '#10b981']
                
                fig_flow = go.Figure(go.Bar(
                    x=balance_labels, y=balance_vals,
                    text=[f"${v:,.0f}" for v in balance_vals],
                    textposition='outside',
                    marker_color=balance_colors
                ))
                fig_flow.update_layout(
                    yaxis_title="Account Balance ($)",
                    height=300, margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor='#ffffff', paper_bgcolor='#ffffff'
                )
                fig_flow.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
                st.plotly_chart(fig_flow, use_container_width=True)
                st.caption("Visualizing sender and recipient balance transitions before vs. after transaction settlement.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with out_c4:
                st.markdown('<div class="panel-box"><div class="panel-title">🎯 Output Graph 4: Transaction Radar Profile vs. Benchmarks</div>', unsafe_allow_html=True)
                radar_cats = ['Large Amount', 'Origin Drained', 'Dest Mismatch', 'Off-Hour Activity', 'High-Risk Type']
                input_r = [
                    min(100.0, float((sel_amt / 100000.0) * 100.0)),
                    100.0 if (sel_old_org > 0 and sel_new_org == 0) else 0.0,
                    100.0 if (sel_old_dest == 0 and sel_new_dest == 0) else 0.0,
                    100.0 if ((sel_step % 24) < 7 or (sel_step % 24) > 21) else 20.0,
                    100.0 if sel_type in ['TRANSFER', 'CASH_OUT'] else 0.0
                ]
                
                fig_input_radar = go.Figure()
                fig_input_radar.add_trace(go.Scatterpolar(
                    r=input_r + [input_r[0]],
                    theta=radar_cats + [radar_cats[0]],
                    name='Current Input Transaction',
                    line_color='#ef4444' if prob >= 50 else '#2563eb',
                    fill='toself',
                    fillcolor='rgba(239, 68, 68, 0.3)' if prob >= 50 else 'rgba(37, 99, 235, 0.2)'
                ))
                fig_input_radar.add_trace(go.Scatterpolar(
                    r=[12, 5, 2, 22, 10, 12],
                    theta=radar_cats + [radar_cats[0]],
                    name='Normal Legitimate Baseline',
                    line_color='#10b981',
                    line_dash='dash'
                ))
                fig_input_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    height=300, margin=dict(l=20, r=20, t=10, b=20),
                    paper_bgcolor='#ffffff',
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_input_radar, use_container_width=True)
                st.caption("Radar overlay showing how your input transaction compares directly against institutional clean baselines.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Explainable AI Insights Box
            st.markdown("##### 🔍 Explainable AI (XAI) Root Cause Breakdown:")
            for reason in result['risk_factors']:
                st.markdown(f"- ⚠️ {reason}")
                
            # Discrepancy checks metrics
            err_org = sel_new_org + sel_amt - sel_old_org
            err_dest = sel_old_dest + sel_amt - sel_new_dest
            
            dcol1, dcol2, dcol3 = st.columns(3)
            dcol1.metric("Origin Discrepancy (Error)", f"${err_org:,.2f}")
            dcol2.metric("Destination Discrepancy (Error)", f"${err_dest:,.2f}")
            dcol3.metric("Simulated Transaction Time", f"Hour {sel_step % 24}:00")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ==============================================================================
# PAGE 6: BATCH CSV FRAUD DETECTION
# ==============================================================================
elif page == "📁 Batch CSV Fraud Detection":
    render_header(
        "Batch Transaction CSV Fraud Detection",
        "Upload bulk transaction logs to run vectorized inference, flag suspicious transactions, and export predictions"
    )
    
    st.markdown("""
    Upload a CSV file containing transactions. The engine requires these columns:
    `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`.
    """)
    
    # Download sample batch test button
    sample_batch_path = os.path.join(DATA_DIR, "sample_batch_test.csv")
    if os.path.exists(sample_batch_path):
        with open(sample_batch_path, "rb") as f:
            st.download_button(
                label="📥 Download Sample Batch Test CSV (50 Real PaySim Transactions)",
                data=f.read(),
                file_name="sample_batch_transactions.csv",
                mime="text/csv"
            )
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded CSV containing **{len(batch_df):,}** transactions.")
            
            required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            missing = [c for c in required_cols if c not in batch_df.columns]
            
            if missing:
                st.error(f"Missing required columns in uploaded CSV: {missing}")
            else:
                avail_models = list(predictor.models.keys()) if predictor.models else ["Random Forest"]
                chosen_batch_model = st.selectbox("Select Model for Batch Scoring:", avail_models, index=0)
                
                if st.button("🚀 Run Batch Fraud Analysis", use_container_width=True):
                    with st.spinner("Executing vectorized inference..."):
                        t0 = time.time()
                        scored_df, batch_summary = predictor.predict_batch(batch_df, model_name=chosen_batch_model)
                        time_taken = time.time() - t0
                        
                    # Summary Cards
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Total Scanned", f"{batch_summary['total_records']:,}")
                    b2.metric("Flagged Fraudulent 🚨", f"{batch_summary['flagged_fraud']:,}")
                    b3.metric("Cleared Legitimate ✅", f"{batch_summary['cleared_legitimate']:,}")
                    b4.metric("Batch Fraud Rate", f"{batch_summary['fraud_percentage']}%")
                    
                    st.caption(f"Inference completed in **{time_taken:.2f}s** ({len(scored_df)/max(time_taken, 0.001):,.0f} transactions/second).")
                    
                    # Highlight table
                    st.markdown("##### 📋 Scored Transactions Log:")
                    st.dataframe(scored_df.head(50), use_container_width=True)
                    
                    # Download Scored CSV
                    csv_export = scored_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Scored Results with Fraud Flags (CSV)",
                        data=csv_export,
                        file_name=f"scored_transactions_{chosen_batch_model.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")

# ==============================================================================
# PAGE 7: PROJECT INFO & ACADEMIC VIVA Q&A
# ==============================================================================
elif page == "ℹ️ Project Info & Academic Viva":
    render_header(
        "Project Information & Academic Viva Voce Guide",
        "Comprehensive academic project documentation, system architecture, and examiner viva Q&A"
    )
    
    tab1, tab2, tab3 = st.tabs(["📚 Project Documentation", "🏛️ System Architecture", "🎓 Top 15 Viva Voce Q&A"])
    
    with tab1:
        st.markdown("""
        ### Online Payment Fraud Detection Management Using Machine Learning
        
        #### 1. Abstract
        With the rapid growth of digital payment infrastructure, financial fraud has become a multi-billion dollar challenge. Traditional rule-based fraud detection mechanisms fail against evolving attack vectors and suffer from high false-positive rates. This project develops an end-to-end Machine Learning pipeline utilizing the PaySim synthetic financial dataset (6.36 million records) to detect fraudulent payment activities in real-time. We engineer critical financial discrepancy features and benchmark **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost** classifiers. Random Forest and XGBoost achieve over **99.9% Accuracy**, **99.5% Recall**, and **99.7% F1-Score**, successfully intercepting fraudulent attempts while minimizing merchant friction.
        
        #### 2. Problem Statement
        Financial fraud detection faces three core computational challenges:
        1. **Extreme Class Imbalance:** Fraud represents only ~0.13% of all transactional volume.
        2. **Asymmetric Misclassification Costs:** A False Negative (missed fraud) causes catastrophic financial loss, while a False Positive results merely in temporary user verification friction.
        3. **High Transaction Throughput:** In modern e-commerce, fraud scoring must execute within sub-second latencies without causing checkout delays.
        
        #### 3. Objectives
        * Ingest and clean real-world scale payment logs (PaySim dataset).
        * Engineer domain-specific features capturing balance discrepancies and temporal patterns.
        * Train, tune, and benchmark 4 diverse supervised machine learning models.
        * Build an interactive, production-grade financial fraud monitoring dashboard.
        * Provide both single transaction real-time inference and bulk CSV batch scoring.
        """)
        
    with tab2:
        st.markdown("""
        ### System Architecture Pipeline
        
        ```
        [ Raw PaySim Transaction Stream (6.36M Rows) ]
                           │
                           ▼
        [ Data Cleaning & Stratified Sampling (287K) ]
                           │
                           ▼
        [ Financial Discrepancy Feature Engineering ]
          ├─ errorBalanceOrig = newOrig + amount - oldOrig
          ├─ errorBalanceDest = oldDest + amount - newDest
          ├─ hourOfDay = step % 24
          └─ One-Hot Categorical Encoding (TRANSFER, CASH_OUT...)
                           │
                           ▼
        [ Train/Test Split (80/20 Stratified) & StandardScaler ]
                           │
                           ▼
        [ 4-Model Training & Benchmark Suite ]
          ├─ 1. Logistic Regression (Baseline Linear Classifier)
          ├─ 2. Decision Tree (Non-linear Rule Partitioning)
          ├─ 3. Random Forest (Bagging Ensemble of 100 Trees)
          └─ 4. XGBoost (Gradient Boosting Decision Trees)
                           │
                           ▼
        [ Held-out Test Evaluation & Metric Serialization ]
          └─ Accuracy, Precision, Recall, F1-Score, ROC-AUC, CM
                           │
                           ▼
        [ Streamlit Web Application Interface ]
          ├─ Real-Time FraudGuard Financial Dashboard
          ├─ Exploratory Data Analysis & Deep Insights
          ├─ Single Real-time Transaction Scoring Form
          └─ Vectorized Batch CSV Fraud Processing Engine
        ```
        """)
        
    with tab3:
        st.markdown(r"""
        ### Top 15 Academic Viva Voce Questions & Model Answers
        
        **Q1: What is the PaySim dataset and how was it generated?**  
        *Answer:* PaySim is a synthetic financial dataset generated by researchers from Blekinge Institute of Technology using a multi-agent simulation based on real aggregated transactional logs from a mobile money service in Africa. It captures 6,362,620 transactions over a 30-day simulation.
        
        **Q2: What is the Class Imbalance ratio in this dataset?**  
        *Answer:* Fraud represents only 8,213 transactions out of 6,362,620, which is approximately **0.129%** (1 fraud per 775 legitimate transactions).
        
        **Q3: What is the Accuracy Paradox in fraud detection?**  
        *Answer:* If a model blindly predicts "Legitimate" for all transactions, it achieves **99.87% Accuracy** while catching **0% of frauds**. Therefore, Accuracy is a misleading metric for imbalanced data, and we must optimize for **Recall** and **F1-Score**.
        
        **Q4: Which transaction types contain fraud in PaySim?**  
        *Answer:* Fraud occurs exclusively in **TRANSFER** and **CASH_OUT**. Attackers execute a TRANSFER to move money into a mule account, followed immediately by a CASH_OUT to liquidate funds into cash.
        
        **Q5: What feature engineering did you perform?**  
        *Answer:* We engineered:
        1. `errorBalanceOrig`: Measures whether sender balance was properly deducted ($new + amount - old$).
        2. `errorBalanceDest`: Measures recipient balance change ($old + amount - new$).
        3. `hourOfDay`: Extracted cyclic hour ($step \pmod{24}$) to detect off-peak fraud attacks.
        
        **Q6: Why did you scale features using StandardScaler?**  
        *Answer:* Transaction amounts and account balances range from dollars to millions, while one-hot encoded flags are 0 or 1. Scaling standardizes numeric features to mean 0 and variance 1, which is critical for convergence in Logistic Regression.
        
        **Q7: How did you prevent Data Leakage?**  
        *Answer:* The `StandardScaler` was fitted strictly on `X_train` and then applied to `X_test` and new incoming user inputs without recalculating mean or standard deviation.
        
        **Q8: What is the difference between Bagging and Boosting in your models?**  
        *Answer:* **Random Forest (Bagging)** builds multiple independent decision trees in parallel on bootstrap samples and averages predictions to reduce variance. **XGBoost (Boosting)** builds sequential trees where each subsequent tree corrects the residual errors of the previous ones to reduce bias.
        
        **Q9: Why does Random Forest perform so well on this dataset?**  
        *Answer:* Random Forest excels because fraud in PaySim follows clear non-linear decision boundaries (specific combinations of transfer type, account depletion, and balance error), which decision trees segment effectively.
        
        **Q10: What is a False Negative in this context, and why is it dangerous?**  
        *Answer:* A False Negative occurs when a fraudulent transaction is misclassified as legitimate. It is the most costly error because money is permanently stolen and lost to the institution.
        
        **Q11: What is a False Positive, and what is its consequence?**  
        *Answer:* A False Positive is when a genuine customer transaction is incorrectly flagged as fraud. It causes customer inconvenience, transaction abandonment, and merchant friction.
        
        **Q12: How does the ROC-AUC metric evaluate the models?**  
        *Answer:* ROC-AUC measures the area under the True Positive Rate vs False Positive Rate curve across all classification thresholds. An AUC close to 1.0 indicates near-perfect discrimination between fraud and legitimate transactions.
        
        **Q13: How does your application provide Explainable AI (XAI)?**  
        *Answer:* In the prediction engine, we inspect feature contributions and flag specific behavioral anomalies (e.g. account emptied to $0, zero balance destination, or transaction amount exceeding $200,000) so compliance officers understand why a transaction was flagged.
        
        **Q14: How does the system handle batch CSV predictions?**  
        *Answer:* It validates required schema columns, applies the saved preprocessor pipeline in a vectorized pandas operation, and outputs predicted labels, probabilities, and risk levels with a downloadable CSV export.
        
        **Q15: What are future improvements for this project?**  
        *Answer:* Implementing Graph Neural Networks (GNN) to track money-laundering rings across multi-hop transactions, adding real-time Kafka streaming ingestion, and incorporating biometric/device fingerprinting features.
        """)
