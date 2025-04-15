# 📊 Streamlit Hospital Model Comparison App

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import logging
import zipfile
from io import BytesIO
from fpdf import FPDF
import time
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 🔍 Evaluate unclustered models on the dataset

def run_model_performance(X_test, y_test, models, X_train, y_train):
    import plotly.graph_objects as go
    roc_fig = go.Figure()
    st.subheader("📈 Model Performance Summary")
    metrics = []
    best_auc = 0
    best_model = ''

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc_score = auc(fpr, tpr)
        metrics.append({"Model": name, "Accuracy": acc, "F1 Score": f1, "Precision": prec, "Recall": rec, "AUC": auc_score})
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = name
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))

    roc_fig.update_layout(title="Combined ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(roc_fig, use_container_width=True)

    roc_path = "/tmp/roc_combined.png"
    roc_fig.write_image(roc_path)
    with open(roc_path, "rb") as f:
        st.download_button("📈 Download ROC Curve", f.read(), file_name="roc_combined.png", mime="image/png")

    st.markdown("### 📊 Model Comparison Table")
    sorted_df = pd.DataFrame(metrics).sort_values(by="AUC", ascending=False)
    styled_table = sorted_df.style.format("{:.2f}")\
        .background_gradient(subset=["Accuracy", "F1 Score", "Precision", "Recall", "AUC"], cmap="Greens")\
        .apply(lambda x: ['background-color: lightgreen' if v == best_auc else '' for v in x] if x.name == 'AUC' else ['' for _ in x], axis=1)
    with st.expander("📊 View All Model Metrics", expanded=True):
        st.dataframe(styled_table)

    st.download_button("📅 Download Model Metrics CSV", sorted_df.to_csv(index=False).encode(), file_name="model_metrics.csv", mime="text/csv")

    return pd.DataFrame(metrics), best_model, best_auc

# 🚨 Visualize anomalies detected using Isolation Forest

def run_anomaly_visual(df):
    st.subheader("🚨 Anomaly Detection Overview")
    fig = px.scatter(df, x="Billing Amount", y="Length of Stay", color=df['anomaly'].map({1: "Anomaly", -1: "Normal"}),
                     title="Anomaly Detection Results")
    st.plotly_chart(fig, use_container_width=True)

# 📦 ZIP and PDF Export Utilities

def generate_zip_export(files):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        index_lines = ["Dashboard Visual Index:"]
        for path, label in files:
            if os.path.exists(path):
                filename = os.path.basename(path)
                zf.write(path, arcname=filename)
                index_lines.append(f"- {label}: {filename}")
        zf.writestr("index.txt", "\n".join(index_lines))
    zip_buffer.seek(0)
    return zip_buffer

def generate_pdf_report(title, content_blocks):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    for block in content_blocks:
        pdf.multi_cell(0, 10, block)
        pdf.ln(5)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Appendix: Dashboard Index", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, "This section lists all exported visuals available in the ZIP file.")
    path = "/tmp/hospital_summary.pdf"
    pdf.output(path)
    return path

# 🚀 Main App Logic

def main():
    st.set_page_config(page_title="📊 Hospital Model Comparison", layout="wide")
    st.title(":hospital: Hospital BI: Predictive Model & Cluster Analysis")

    st.sidebar.header("📜 How to Use")
    st.sidebar.markdown("""
    1. Upload your hospital dataset (or use sample)
    2. Explore model performance (ROC, metrics)
    3. Visualize anomaly patterns
    4. Run clustering and compare segmented models
    """)

    st.subheader("📁 Upload Your Hospital Data or Use Sample")
    file = st.file_uploader("Upload CSV", type=["csv"])
    required_cols = ['Billing Amount', 'Medical Condition', 'Medication', 'Date of Admission',
                     'Discharge Date', 'Age', 'Gender', 'Insurance Provider']

    if file:
        df = pd.read_csv(file)
        st.success("✅ File uploaded successfully.")
    else:
        df = pd.read_csv("https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv")
        st.warning("⚠️ Using sample dataset")

    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns in dataset.")
        st.stop()

    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    df['anomaly'] = IsolationForest(contamination=0.05, random_state=42).fit_predict(df[['Billing Amount']])
    df['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
    df['Medication'] = LabelEncoder().fit_transform(df['Medication'])
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'].fillna("Unknown"))
    df['Insurance'] = pd.factorize(df['Insurance Provider'])[0]

    features = ['Age', 'Billing Amount', 'Length of Stay', 'Condition', 'Medication', 'Gender', 'Insurance']
    target = (df['anomaly'] == 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(df[features], target, stratify=target, test_size=0.3, random_state=42)
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

    metrics_df, best_model, best_auc = run_model_performance(X_test_scaled, y_test, model_options, X_train_scaled, y_train)
    run_anomaly_visual(df)
    run_clustered_comparison(df, metrics_df)

    # 📆 PDF Export Section
    summary_blocks = [
        f"Best unclustered model: {best_model} (AUC: {best_auc:.2f})",
        "Unclustered model performance table:\n" + metrics_df.to_string(index=False)
    ]
    pdf_path = generate_pdf_report("Hospital BI Report", summary_blocks)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Summary PDF", f.read(), file_name="hospital_summary.pdf", mime="application/pdf")

    # 📦 ZIP Export Section
    chart_paths = [
        ("/tmp/roc_combined.png", "ROC Curve")
    ]
    zip_buffer = generate_zip_export(chart_paths)
    st.download_button("📦 Download All Visuals (ZIP)", zip_buffer.read(), file_name="charts_bundle.zip", mime="application/zip")

if __name__ == "__main__":
    main()
