@echo off
title FraudGuard - Online Payment Fraud Detection Dashboard
echo ===================================================================
echo       Online Payment Fraud Detection Management System
echo                  Powered by Machine Learning
echo ===================================================================
echo.
echo Launching Streamlit Dashboard on http://localhost:8501/ ...
echo.
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501
pause
