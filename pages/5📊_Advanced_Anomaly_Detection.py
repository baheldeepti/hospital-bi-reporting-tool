# 📊 Streamlit Hospital Model Comparison App

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
import zipfile
from io import BytesIO
from fpdf import FPDF
import time
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.base import clone
from kneed import KneeLocator

try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 🔍 Evaluate unclustered models on the dataset

def run_model_performance(X_test, y_test, models, X_train, y_train):
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

# 🔬 Clustering and model comparison with elbow, visualizations and line charts

def run_clustered_comparison(df):
    st.subheader("🧪 Cluster-Based Model Comparison")
    features = ['Age', 'Billing Amount', 'Length of Stay', 'Gender', 'Insurance']
    distortions = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df[features])
        distortions.append(kmeans.inertia_)
    kl = KneeLocator(range(1, 10), distortions, curve='convex', direction='decreasing')
    best_k = kl.elbow if kl.elbow else 3
    st.plotly_chart(px.line(x=range(1, 10), y=distortions, markers=True, title="Elbow Plot to Find Optimal K"))
    st.success(f"Best K determined: {best_k}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }
    results = []
    for k in range(2, best_k + 1):
        df['cluster'] = KMeans(n_clusters=k, random_state=42).fit_predict(df[features])
        for model_name, model_def in models.items():
            for cluster in sorted(df['cluster'].unique()):
                cdf = df[df['cluster'] == cluster]
                if len(cdf) < 10: continue
                X = cdf[features]
                y = (cdf['anomaly'] == 1).astype(int)
                if len(np.unique(y)) < 2: continue
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model = clone(model_def)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                results.append({
                    "K": k, "Cluster": cluster, "Model": model_name,
                    "F1 Score": f1_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred),
                    "Recall": recall_score(y_test, y_pred),
                    "AUC": auc(*roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])[:2])
                })
    clustered_df = pd.DataFrame(results)
    st.dataframe(clustered_df)
    st.plotly_chart(px.line(clustered_df, x='K', y='AUC', color='Model', markers=True, title="AUC by Model & K"))
    return clustered_df

# 📦 PDF and ZIP export utilities

def generate_pdf_report(title, insights, metrics_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    for block in insights:
        pdf.multi_cell(0, 10, block)
        pdf.ln(3)
    pdf.ln(5)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 10, metrics_df.to_string(index=False))
    path = "/tmp/report.pdf"
    pdf.output(path)
    return path

def generate_zip_export(paths):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipf:
        for p in paths:
            if os.path.exists(p):
                zipf.write(p, os.path.basename(p))
    buffer.seek(0)
    return buffer

# 🚀 Main app logic

def main():
    st.set_page_config(page_title="📊 Hospital Model Comparison", layout="wide")
    st.title("🏥 Hospital BI: Predictive & Cluster Analysis")

    file = st.file_uploader("Upload hospital dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.success("✅ File uploaded successfully.")
    else:
        df = pd.read_csv("https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv")
        st.warning("⚠️ Using sample dataset")

    required = ['Billing Amount', 'Medical Condition', 'Medication', 'Date of Admission', 'Discharge Date', 'Age', 'Gender', 'Insurance Provider']
    if not all(col in df.columns for col in required):
        st.error("❌ Missing required columns.")
        return

    # 🧼 Data cleansing
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
    clustered_df = run_clustered_comparison(df)

    # 🤖 GPT narrative insight with retry logic
    if openai_available:
        api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
            prompt = f"Best model: {best_model} (AUC: {best_auc:.2f}). Compare unclustered and clustered models."
            try:
                for attempt in range(3):
                    try:
                        response = openai.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a healthcare data analyst."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        st.markdown(response.choices[0].message.content)
                        break
                    except Exception as e:
                        if 'rate limit' in str(e).lower() or 'timeout' in str(e).lower():
                            st.warning("Retrying due to timeout...")
                            time.sleep(2 ** attempt)
                        else:
                            raise
            except Exception as err:
                st.error("Failed to get GPT response")
                st.exception(err)

    # 📦 Export all
    pdf_path = generate_pdf_report("Hospital BI Report", [f"Best Model: {best_model}"], metrics_df)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download PDF Report", f.read(), file_name="hospital_report.pdf")

    zip_buf = generate_zip_export(["/tmp/roc_combined.png", pdf_path])
    st.download_button("📦 Download All Charts as ZIP", zip_buf.read(), file_name="charts_bundle.zip", mime="application/zip")

if __name__ == "__main__":
    main()
