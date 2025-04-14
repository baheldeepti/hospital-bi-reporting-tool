# 📊 Clustered Model Comparison Page
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io

# -------------------------------------
# Title & Intro
# -------------------------------------
st.set_page_config(page_title="📈 Which Model Predicts Best for Each Patient Type?", layout="wide")
st.title("📈 Which Model Predicts Best for Each Patient Type?")

st.markdown("""
Welcome to the **Try Different Models and See What Works Best** page! 🎯

Here, you can test different AI models to see how well they predict **abnormal test results** for patients.  
First, you'll try the models on the whole hospital dataset.  
Then, you'll see if grouping similar patients together (using clustering) helps the models do even better!

This page helps you answer: **Which model works best — and for whom?**
""")


# -------------------------------------
# Upload or Load Sample Dataset
# -------------------------------------
st.subheader("📁 Upload Your Hospital Data or Use Sample")
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.success("✅ Custom dataset uploaded successfully!")
else:
    url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
    df = pd.read_csv(url)
    st.warning("⚠️ Using default sample dataset.")

# -------------------------------------
# Preprocessing
# -------------------------------------
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Test Results Encoded'] = df['Test Results'].map({'Normal': 0, 'Abnormal': 1, 'Inconclusive': 2})

X = df[['Age', 'Billing Amount', 'Length of Stay']].copy()
X['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
X['Medication'] = LabelEncoder().fit_transform(df['Medication'])
y = (df['Test Results Encoded'] == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------
# Model Selection and Training (Non-Clustered)
# -------------------------------------
st.subheader("📈 Model Performance Without Clustering")
model_options = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

selected_models = st.multiselect("Choose Models", list(model_options.keys()), default=["Logistic Regression", "Random Forest"])

df_metrics = []
report_text = ""

for name in selected_models:
    clf = model_options[name]
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"])
    report_text += f"=== {name} ===\n{report}\n\n"

    df_metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

st.dataframe(pd.DataFrame(df_metrics).sort_values(by="Accuracy", ascending=False))

# 📥 Download Buttons
st.download_button("📥 Download Metrics CSV", pd.DataFrame(df_metrics).to_csv(index=False), file_name="model_performance_metrics.csv")
st.download_button("📥 Download Classification Reports (.txt)", report_text, file_name="classification_reports.txt")

# -------------------------------------
# Enable Clustering
# -------------------------------------
if st.checkbox("🔘 Enable Clustered Model Comparison"):
    st.header("🔍 K-Means Clustering")
    cluster_features = ['Age', 'Billing Amount', 'Length of Stay']
    cluster_df = df[cluster_features].copy()
    cluster_df['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
    scaled_cluster = StandardScaler().fit_transform(cluster_df)

    sse, silhouettes = [], []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42).fit(scaled_cluster)
        sse.append(km.inertia_)
        silhouettes.append(silhouette_score(scaled_cluster, km.labels_))

    fig_elbow = px.line(x=list(range(2, 10)), y=sse, labels={'x': 'k', 'y': 'SSE'}, title="📉 Elbow Method")
    st.plotly_chart(fig_elbow)

    best_k = np.argmax(silhouettes) + 2
    st.success(f"✅ Optimal number of clusters: {best_k}")

    df['Cluster'] = KMeans(n_clusters=best_k, random_state=42).fit_predict(scaled_cluster)

    st.subheader("📊 Clustered Multi-Model Evaluation")
    clustered_metrics = []
    cluster_model_choices = st.multiselect("Select models for clustered evaluation", list(model_options.keys()), default=selected_models)

    for cluster_id in range(best_k):
        cluster_data = df[df['Cluster'] == cluster_id]
        X_c = cluster_data[['Age', 'Billing Amount', 'Length of Stay']].copy()
        X_c['Condition'] = LabelEncoder().fit_transform(cluster_data['Medical Condition'])
        X_c['Medication'] = LabelEncoder().fit_transform(cluster_data['Medication'])
        y_c = (cluster_data['Test Results'].map({'Normal': 0, 'Abnormal': 1, 'Inconclusive': 2}) == 1).astype(int)

        if len(y_c.unique()) < 2 or len(cluster_data) < 20:
            continue

        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, stratify=y_c, random_state=42)
        scaler_c = StandardScaler()
        X_train_scaled_c = scaler_c.fit_transform(X_train_c)
        X_test_scaled_c = scaler_c.transform(X_test_c)

        for model_name in cluster_model_choices:
            model = model_options[model_name]
            model.fit(X_train_scaled_c, y_train_c)
            y_pred_c = model.predict(X_test_scaled_c)
            clustered_metrics.append({
                "Cluster": f"Cluster {cluster_id}",
                "Model": model_name,
                "Accuracy": accuracy_score(y_test_c, y_pred_c),
                "Precision": precision_score(y_test_c, y_pred_c),
                "Recall": recall_score(y_test_c, y_pred_c),
                "F1 Score": f1_score(y_test_c, y_pred_c)
            })

    if clustered_metrics:
        df_clustered = pd.DataFrame(clustered_metrics)
        st.subheader("📋 Per-Cluster Performance")
        st.dataframe(df_clustered)

        for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            fig = px.bar(df_clustered, x="Cluster", y=metric, color="Model", barmode="group", title=f"{metric} by Cluster")
            st.plotly_chart(fig)

        st.subheader("🔳 Aggregated Performance (Avg. across Clusters)")
        aggregated = df_clustered.groupby("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].mean().reset_index()
        st.dataframe(aggregated.style.highlight_max(axis=0, color="lightgreen"))

        with st.expander("📘 What do these metrics mean?"):
            st.markdown("""
            - **Accuracy**: % of all predictions that were correct
            - **Precision**: % of predicted 'Abnormal' that were truly 'Abnormal'
            - **Recall**: % of true 'Abnormal' cases that were correctly found
            - **F1 Score**: A balance of precision and recall
            """)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(aggregated.set_index("Model"), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        st.pyplot(fig)

        # 📥 Download
        st.download_button("📥 Download Clustered Metrics CSV", df_clustered.to_csv(index=False), file_name="clustered_model_metrics.csv")
