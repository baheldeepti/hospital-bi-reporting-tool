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

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, silhouette_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# 🚀 Main Application Entry Point
# =====================================================================
def main():
    st.set_page_config(page_title="📊 Hospital Model Comparison", layout="wide")
    st.title(":hospital: Hospital BI: Predictive Model & Cluster Analysis")

    with st.sidebar:
        st.header("📜 How to Use")
        st.markdown("""
        1. Upload your hospital dataset (or use sample)
        2. Explore model performance (ROC, metrics)
        3. Visualize anomaly patterns
        4. Enable clustering to compare models on patient segments
        5. Download results for further analysis
        """)

    st.markdown("""
    Welcome to the **Hospital Model Comparison App**! This dashboard helps healthcare analysts and executives:
    - Predict abnormal test results using machine learning
    - Understand patterns in billing, age, and stay length
    - Compare results across patient clusters to improve decision-making
    """)

    st.subheader("📁 Upload Your Hospital Data or Use Sample")
    file = st.file_uploader("Upload CSV", type=["csv"])
    required_cols = ['Billing Amount', 'Medical Condition', 'Medication', 'Date of Admission', 'Discharge Date']

    if file is not None:
        try:
            df = pd.read_csv(file)
            if not all(col in df.columns for col in required_cols):
                st.error(f"❌ Missing required columns: {', '.join(required_cols)}")
                logging.error("Uploaded file missing required columns.")
                st.stop()
            st.success("✅ Custom dataset uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            logging.exception("Failed to read uploaded CSV file.")
            st.stop()
    else:
        url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
        try:
            df = pd.read_csv(url)
            st.warning("⚠️ Using default sample dataset.")
        except Exception as e:
            st.error("❌ Failed to load sample dataset.")
            logging.exception("Failed to load sample dataset from GitHub.")
            st.stop()

    try:
        df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
        df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso.fit_predict(df[['Billing Amount']])

        df['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
        df['Medication'] = LabelEncoder().fit_transform(df['Medication'])

        features = ['Age', 'Billing Amount', 'Length of Stay', 'Condition', 'Medication']
        target_anomaly = (df['anomaly'] == 1).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(df[features], target_anomaly, stratify=target_anomaly, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_options = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        }

        metrics_df = run_model_performance(df, X_test_scaled, y_test, model_options, X_train_scaled, y_train)
        run_anomaly_visual(df)
        run_clustered_comparison(df, model_options, metrics_df)

    except Exception as err:
        st.error(f"Unexpected error during processing: {err}")
        logging.exception("Critical failure during preprocessing or modeling")

# =====================================================================
# ▶️ Run the App
# =====================================================================
if __name__ == "__main__":
    main()
