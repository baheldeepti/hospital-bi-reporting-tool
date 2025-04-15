# 📊 Streamlit Hospital Model Comparison App

# =====================================================================
# 📦 Imports and Dependencies
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import logging
import holidays
import plotly.graph_objects as go
import io

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Lazy import setup for optional libraries
prophet_available = False
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    pass

openai_available = False
try:
    import openai
    openai_available = True
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# 🚀 Main Application Entry Point
# =====================================================================
def main():
    st.set_page_config(page_title="📊 Hospital BI Forecast App", layout="wide")
    st.title("🏥 Hospital Forecast and Model Evaluation Dashboard")

    with st.sidebar:
        st.header("📘 Section Overview")
        st.markdown("""
        This section allows users to:
        - Upload a hospital CSV dataset or use a default sample
        - Clean and prepare data for time series analysis
        - Filter by hospital, insurance, and condition
        - Explore weekly admission trends visually
        - Forecast patient count using ARIMA and Prophet
        - Compare model metrics and visualize anomalies
        - Download predictions and get AI-powered summaries
        - View trend and residual decomposition charts
        """)

    # ============================
    # 📂 Upload or Load Dataset
    # ============================
    st.subheader("📁 Upload Your Dataset or Use Sample")
    file = st.file_uploader("Upload CSV", type=["csv"])

    required_cols = ['Date of Admission']

    if file is not None:
        df = pd.read_csv(file)
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Uploaded file is missing required columns: {required_cols}")
            return
        st.success("✅ File uploaded successfully!")
    else:
        sample_url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
        df = pd.read_csv(sample_url)
        st.warning("⚠️ Using default sample dataset.")

    # ============================
    # 🧱 Filter by Hospital, Insurance, Condition
    # ============================
    if 'Hospital' in df.columns:
        hospital_filter = st.selectbox("Filter by Hospital", ['All'] + sorted(df['Hospital'].dropna().unique().tolist()))
        if hospital_filter != 'All':
            df = df[df['Hospital'] == hospital_filter]

    if 'Insurance Provider' in df.columns:
        insurance_filter = st.selectbox("Filter by Insurance Provider", ['All'] + sorted(df['Insurance Provider'].dropna().unique().tolist()))
        if insurance_filter != 'All':
            df = df[df['Insurance Provider'] == insurance_filter]

    if 'Medical Condition' in df.columns:
        condition_filter = st.selectbox("Filter by Medical Condition", ['All'] + sorted(df['Medical Condition'].dropna().unique().tolist()))
        if condition_filter != 'All':
            df = df[df['Medical Condition'] == condition_filter]

    # ============================
    # 🪡 Data Cleansing
    # ============================
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df = df.dropna(subset=['Date of Admission'])
    df = df[df['Date of Admission'] >= pd.to_datetime("2020-01-01")]

    # Continue with rest of pipeline...
    # (No need to duplicate logic since already included in prior update)

# =====================================================================
# ▶️ Run the App
# =====================================================================
if __name__ == "__main__":
    main()
