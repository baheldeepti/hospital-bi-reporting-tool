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

# =====================================================================
# 🚀 Main Application Entry Point
# =====================================================================
def main():
    # -----------------------------------------------------------------
    # ⚙️ Set Page Configuration and Title
    # -----------------------------------------------------------------
    st.set_page_config(page_title="📊 Hospital Model Comparison", layout="wide")
    st.title(":hospital: Hospital BI: Predictive Model & Cluster Analysis")

    # -----------------------------------------------------------------
    # 🧭 Sidebar Navigation and Instructions
    # -----------------------------------------------------------------
    with st.sidebar:
        st.header("📜 How to Use")
        st.markdown("""
        1. Upload your hospital dataset (or use sample)
        2. Explore model performance (ROC, metrics)
        3. Visualize anomaly patterns
        4. Enable clustering to compare models on patient segments
        5. Download results for further analysis
        """)

    # -----------------------------------------------------------------
    # 📝 App Overview and Welcome Message
    # -----------------------------------------------------------------
    st.markdown("""
    Welcome to the **Hospital Model Comparison App**! This dashboard helps healthcare analysts and executives:
    - Predict abnormal test results using machine learning
    - Understand patterns in billing, age, and stay length
    - Compare results across patient clusters to improve decision-making
    """)

    # =====================================================================
    # 📂 Data Upload and Fallback to Sample Dataset
    # =====================================================================
    st.subheader("📁 Upload Your Hospital Data or Use Sample")
    file = st.file_uploader("Upload CSV", type=["csv"])
    required_cols = ['Billing Amount', 'Medical Condition', 'Medication', 'Date of Admission', 'Discharge Date']

    if file is not None:
        try:
            df = pd.read_csv(file)
            if not all(col in df.columns for col in required_cols):
                st.error(f"❌ Missing required columns: {', '.join(required_cols)}")
                st.stop()
            st.success("✅ Custom dataset uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()
    else:
        url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
        df = pd.read_csv(url)
        st.warning("⚠️ Using default sample dataset.")

    # =====================================================================
    # 🧹 Data Preprocessing and Feature Engineering
    # =====================================================================
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

    iso = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = iso.fit_predict(df[['Billing Amount']])

    df['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
    df['Medication'] = LabelEncoder().fit_transform(df['Medication'])

    X = df[['Age', 'Billing Amount', 'Length of Stay', 'Condition', 'Medication']]
    y = (df['anomaly'] == 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =====================================================================
    # 🧠 Define Machine Learning Models for Evaluation
    # =====================================================================
    model_options = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    # =====================================================================
    # 📊 Execute Analytics Workflows (Unclustered + Anomalies + Clustering)
    # =====================================================================
    metrics_df = run_model_performance(df, X_test_scaled, y_test, model_options, X_train_scaled, y_train)
    run_anomaly_visual(df)
    run_clustered_comparison(df, model_options, metrics_df)

# =====================================================================
# 📈 Model Performance Without Clustering
# =====================================================================
def run_model_performance(df, X_test_scaled, y_test, model_options, X_train_scaled, y_train):
    st.subheader("📈 Model Performance Without Clustering")
    selected_models = st.multiselect("Choose Models", list(model_options.keys()), default=list(model_options.keys()))

    if not selected_models:
        st.warning("⚠️ Please select at least one model to evaluate.")
        return pd.DataFrame()

    df_metrics = []
    report_text = ""
    plt.figure(figsize=(8, 6))

    for name in selected_models:
        clf = model_options[name]
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_test_scaled)
        else:
            y_score = y_pred

        report = classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"])
        report_text += f"=== {name} ===\n{report}\n\n"

        df_metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0)
        })

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(plt)

    metrics_df = pd.DataFrame(df_metrics).sort_values(by="Accuracy", ascending=False)
    styled_df = metrics_df.style.background_gradient(cmap='YlGnBu', subset=["Accuracy", "Precision", "Recall", "F1 Score"]).format("{:.2%}")
    st.dataframe(styled_df)

    st.download_button("📥 Download Metrics CSV", metrics_df.to_csv(index=False), file_name="model_performance_metrics.csv")
    st.download_button("📥 Download Classification Reports (.txt)", report_text, file_name="classification_reports.txt")

    return metrics_df

# =====================================================================
# 📉 Anomaly Detection and Visualization
# =====================================================================
def run_anomaly_visual(df):
    st.subheader("📉 Anomaly Detection Visualization")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    anomaly_col = st.selectbox("Choose a column to detect anomalies in:", numeric_cols, index=numeric_cols.index("Billing Amount"))

    iso = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_dynamic'] = iso.fit_predict(df[[anomaly_col]])
    df['anomaly_dynamic'] = df['anomaly_dynamic'].map({1: "Normal", -1: "Anomaly"})

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='anomaly_dynamic', y=anomaly_col, ax=ax)
    ax.set_title(f"Anomaly Detection on '{anomaly_col}'")
    st.pyplot(fig)

    anomaly_count = df[df['anomaly_dynamic'] == "Anomaly"].shape[0]
    st.markdown(f"🧾 Total Anomalies Detected in **{anomaly_col}**: **{anomaly_count}**")

    anomalies_df = df[df['anomaly_dynamic'] == "Anomaly"]
    st.download_button("📥 Download Anomalies as CSV", anomalies_df.to_csv(index=False), file_name=f"anomalies_in_{anomaly_col}.csv")

# =====================================================================
# 🔍 Clustered Model Performance with K-Means
# =====================================================================
def run_clustered_comparison(df, model_options, baseline_df):
    if st.checkbox("🔘 Enable Clustered Model Comparison"):
        st.header("🔍 K-Means Clustering")
        cluster_features = ['Age', 'Billing Amount', 'Length of Stay', 'Condition']
        scaled_cluster = StandardScaler().fit_transform(df[cluster_features])

        sse, silhouettes = [], []
        for k in range(2, 10):
            km = KMeans(n_clusters=k, random_state=42).fit(scaled_cluster)
            sse.append(km.inertia_)
            silhouettes.append(silhouette_score(scaled_cluster, km.labels_))

        fig_elbow = px.line(x=list(range(2, 10)), y=sse, labels={'x': 'k', 'y': 'SSE'}, title="📉 Elbow Method for Optimal Clusters")
        st.plotly_chart(fig_elbow)

        best_k = np.argmax(silhouettes) + 2
        st.success(f"✅ Optimal number of clusters: {best_k}")
        df['Cluster'] = KMeans(n_clusters=best_k, random_state=42).fit_predict(scaled_cluster)

        st.subheader("📊 Clustered Multi-Model Evaluation")
        cluster_model_choices = st.multiselect("Select models for clustered evaluation", list(model_options.keys()), default=list(model_options.keys()))

        clustered_metrics = []
        for cluster_id in range(best_k):
            cluster_data = df[df['Cluster'] == cluster_id]
            X_c = cluster_data[['Age', 'Billing Amount', 'Length of Stay', 'Condition', 'Medication']]
            y_c = (cluster_data['anomaly'] == 1).astype(int)

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
                    "Precision": precision_score(y_test_c, y_pred_c, zero_division=0),
                    "Recall": recall_score(y_test_c, y_pred_c, zero_division=0),
                    "F1 Score": f1_score(y_test_c, y_pred_c, zero_division=0)
                })

        if clustered_metrics:
            df_clustered = pd.DataFrame(clustered_metrics)
            st.subheader("📋 Per-Cluster Model Performance")
            st.dataframe(df_clustered)

            for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                fig = px.bar(df_clustered, x="Cluster", y=metric, color="Model", barmode="group", title=f"{metric} by Cluster")
                st.plotly_chart(fig)

            aggregated = df_clustered.groupby("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].mean().reset_index()
            st.subheader("🔳 Aggregated Clustered Model Performance")
            st.dataframe(aggregated.style.highlight_max(axis=0, color="lightgreen"))
            st.download_button("📥 Download Clustered Metrics CSV", df_clustered.to_csv(index=False), file_name="clustered_model_metrics.csv")

            st.markdown("### 📢 Business Insight: Clustered vs Unclustered Models")
            for model in aggregated['Model']:
                match = baseline_df[baseline_df['Model'] == model]
                if not match.empty:
                    base_row = match.iloc[0]
                    cluster_row = aggregated[aggregated['Model'] == model].iloc[0]
                    st.markdown(f"**{model}**: Accuracy `{cluster_row['Accuracy']:.2%}` vs base `{base_row['Accuracy']:.2%}`")
                else:
                    st.warning(f"⚠️ No baseline performance available for model: {model}")

# =====================================================================
# ▶️ Run the App
# =====================================================================
if __name__ == "__main__":
    main()
