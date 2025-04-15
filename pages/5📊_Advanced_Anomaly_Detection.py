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

# 🔬 Clustering and model comparison with export and GPT integration

def run_clustered_comparison(df, metrics_df):
    from sklearn.base import clone
    from kneed import KneeLocator
    import plotly.graph_objects as go
    import itertools

    st.subheader("🧪 Cluster-Based Comparison")

    features = ["Age", "Billing Amount", "Length of Stay", "Gender", "Insurance"]

    distortions = []
    K_range = range(1, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df[features])
        distortions.append(kmeans.inertia_)

    kl = KneeLocator(K_range, distortions, curve="convex", direction="decreasing")
    best_k = kl.elbow if kl.elbow else 3

    st.markdown("### 🔍 Elbow Method for Optimal Clusters")
    fig_elbow = px.line(x=list(K_range), y=distortions, markers=True, labels={'x': 'K', 'y': 'Inertia'}, title="Elbow Curve")
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.success(f"Best K determined from elbow method: {best_k}")

    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    cluster_results = []
    for k in range(2, best_k + 1):
        df['cluster'] = KMeans(n_clusters=k, random_state=42).fit_predict(df[features])
        for model_name, model_def in model_defs.items():
            for cluster in sorted(df['cluster'].unique()):
                cluster_df = df[df['cluster'] == cluster]
                if len(cluster_df) < 10:
                    continue
                X = cluster_df[features]
                y = (cluster_df['anomaly'] == 1).astype(int)
                if len(np.unique(y)) < 2:
                    continue
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model = clone(model_def)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                f1 = f1_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                auc_score = auc(*roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])[:2])
                cluster_results.append({"K": k, "Cluster": cluster, "Model": model_name, "F1 Score": f1, "Precision": prec, "Recall": rec, "AUC": auc_score})

    clustered_df = pd.DataFrame(cluster_results)
    st.markdown("### 📊 Clustered Model Metrics")
    st.dataframe(clustered_df.round(2))

    # 📦 Export to CSV
    st.download_button("📥 Download Clustered Metrics CSV", clustered_df.to_csv(index=False).encode(), file_name="clustered_model_metrics.csv", mime="text/csv")

    # 🤖 GPT Narrative Summary with retry
    if openai_available:
        api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
            with st.expander("🧠 GPT-Based Executive Summary", expanded=True):
                prompt = f"""
Summarize model comparisons from unclustered and clustered analysis.
Best model from unclustered: {metrics_df.sort_values('AUC', ascending=False).iloc[0]['Model']}
Clustered metrics summary:
{clustered_df.head(10).to_markdown()}
"""
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = openai.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a healthcare analyst generating cluster model summaries."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        st.markdown(response.choices[0].message.content)
                        break
                    except Exception as e:
                        error_str = str(e).lower()
                        if 'rate limit' in error_str or 'timeout' in error_str:
                            st.warning(f"Attempt {attempt + 1} failed due to rate limit or timeout. Retrying...")
                            time.sleep(2 ** attempt)
                        elif 'invalid api key' in error_str:
                            st.error("Invalid OpenAI API key. Please check your credentials.")
                            break
                        elif 'quota' in error_str:
                            st.error("You have exceeded your OpenAI API quota.")
                            break
                        else:
                            st.error("An unexpected error occurred while calling OpenAI API.")
                            st.exception(e)
                            break

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

if __name__ == "__main__":
    main()
