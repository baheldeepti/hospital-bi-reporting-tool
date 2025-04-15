# 📊 Streamlit Hospital Model Comparison App

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import logging

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_model_performance(df, X_test, y_test, models, X_train, y_train):
    st.subheader("📈 Model Performance Summary")
    metrics = []
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

        fig = px.area(x=fpr, y=tpr, title=f"ROC Curve - {name}", labels=dict(x="False Positive Rate", y="True Positive Rate"))
        st.plotly_chart(fig, use_container_width=True)

    return pd.DataFrame(metrics)


def run_anomaly_visual(df):
    st.subheader("🚨 Anomaly Detection Overview")
    fig = px.scatter(df, x="Billing Amount", y="Length of Stay", color=df['anomaly'].map({1: "Anomaly", -1: "Normal"}),
                     title="Anomaly Detection Results")
    st.plotly_chart(fig, use_container_width=True)


def run_clustered_comparison(df, model_options, metrics_df):
    st.subheader("🧪 Cluster-Based Comparison")
    cluster_model = KMeans(n_clusters=3, random_state=42)
    features = ["Age", "Billing Amount", "Length of Stay"]
    df['cluster'] = cluster_model.fit_predict(df[features])
    fig = px.scatter_3d(df, x="Age", y="Billing Amount", z="Length of Stay", color="cluster", symbol="anomaly",
                        title="3D View of Clusters with Anomalies")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(metrics_df)

    if openai_available:
        api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
            with st.expander("🧠 Narrative Summary from GPT", expanded=True):
                try:
                    prompt = f"""
                    Based on the following model metrics:
                    {metrics_df.to_markdown(index=False)}
                    Provide a brief executive summary comparing the models and noting which performed best.
                    """
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a healthcare analyst generating executive summaries."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error("Failed to retrieve GPT summary.")
                    st.exception(e)


def main():
    st.set_page_config(page_title="📊 Hospital Model Comparison", layout="wide")
    st.title(":hospital: Hospital BI: Predictive Model & Cluster Analysis")

    st.sidebar.header("📜 How to Use")
    st.sidebar.markdown("""
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

    if file:
        try:
            df = pd.read_csv(file)
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing required columns: {', '.join(required_cols)}")
                return
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return
    else:
        url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
        try:
            df = pd.read_csv(url)
            st.warning("Using default sample dataset")
        except Exception:
            st.error("Unable to load sample dataset")
            return

    try:
        df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
        df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
        df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso.fit_predict(df[['Billing Amount']])

        df['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
        df['Medication'] = LabelEncoder().fit_transform(df['Medication'])

        features = ['Age', 'Billing Amount', 'Length of Stay', 'Condition', 'Medication']
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

        metrics_df = run_model_performance(df, X_test_scaled, y_test, model_options, X_train_scaled, y_train)
        run_anomaly_visual(df)
        run_clustered_comparison(df, model_options, metrics_df)

    except Exception as err:
        st.error(f"Unexpected error during processing: {err}")
        logging.exception("Critical failure during preprocessing or modeling")


if __name__ == "__main__":
    main()
