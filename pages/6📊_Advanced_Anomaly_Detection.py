# 📘 Streamlit App with Chat-Based Analytics for Anomaly Detection
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import shap
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import (
    classification_report, roc_curve, roc_auc_score, f1_score, accuracy_score, precision_score
)

st.set_page_config(page_title="Hospital Anomaly Detection", layout="wide")

# 📊 App Overview
st.markdown("""
## 👋 Welcome to the Hospital Anomaly Finder!

This app helps you find unusual or surprising data in hospital records. Think of it like a smart detective that looks at patient info, figures out what's normal, and points out what’s not!

### What can you do here?
- 📂 Load hospital data (or use the sample)
- 🧮 Let the app clean and prepare it for you
- 🚨 Find out which cases look strange or risky
- 🧠 See which data points are most important
- 🤖 Compare smart models to see which one works best
- 💬 Get simple explanations of how everything works

It's like having a data scientist in your pocket! 😄
""")

# -- Sample Dataset URL
sample_url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"

# -- App Configuration
# (Moved set_page_config to the top as required)
st.title("🏥 Hospital Analytics Chat & Model Assistant")

# -- File Upload or Sample
st.sidebar.header("1️⃣ Upload Data or Use Sample")
file = st.file_uploader("📁 Upload Your Hospital Dataset", type=["csv"])
required_cols = ['Billing Amount', 'Medical Condition', 'Medication', 'Date of Admission',
                 'Discharge Date', 'Age', 'Gender', 'Insurance Provider']

if file:
    df = pd.read_csv(file)
else:
    df = pd.read_csv("https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv")
    st.warning("⚠️ Using sample dataset")

if not all(col in df.columns for col in required_cols):
    st.error("Dataset missing required columns.")
    st.stop()

# -- Check Required Columns
required_cols = ['Date of Admission', 'Discharge Date', 'Billing Amount', 'Medical Condition', 'Insurance Provider', 'Gender', 'Medication', 'Age']
if not all(col in df.columns for col in required_cols):
    st.error(f"Missing required columns: {set(required_cols) - set(df.columns)}")
    st.stop()

# -- Data Cleansing & Feature Engineering
with st.expander("🔧 Data Cleansing & Feature Engineering", expanded=True):
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
    df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    df = df.dropna(subset=['Billing Amount', 'Length of Stay'])

    df['Gender'] = df['Gender'].fillna("Unknown")
    df['Insurance'] = pd.factorize(df['Insurance Provider'])[0]
    df['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
    df['Medication'] = LabelEncoder().fit_transform(df['Medication'])

    df = df.dropna(subset=['Age', 'Billing Amount', 'Length of Stay', 'Condition', 'Medication', 'Insurance'])

    df['Is Weekend'] = (df['Date of Admission'].dt.weekday >= 5).astype(int)
    df['Admission Month'] = df['Date of Admission'].dt.month
    df['Day of Week'] = df['Date of Admission'].dt.dayofweek
    df['Log Billing Amount'] = np.log1p(df['Billing Amount'])

    df['Is Long Stay'] = (df['Length of Stay'] > df['Length of Stay'].quantile(0.95)).astype(int)
    df['Billing per Day'] = df['Billing Amount'] / df['Length of Stay']
    df['Age x Billing'] = df['Age'] * df['Billing Amount']
    df['Age x Stay'] = df['Age'] * df['Length of Stay']
    df['Weekend x Billing'] = df['Is Weekend'] * df['Billing Amount']
    df['Insurance x Condition'] = df['Insurance'] * df['Condition']

    all_features = [
        'Age','Length of Stay', 'Condition', 'Medication', 'Insurance',
        'Is Weekend', 'Admission Month', 'Day of Week', 'Is Long Stay',
        'Billing per Day', 'Age x Billing', 'Age x Stay', 'Weekend x Billing', 'Insurance x Condition'
    ]

# -- Anomaly Detection
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = iso.fit_predict(df[['Billing Amount']])
df['anomaly_label'] = (df['anomaly_score'] == -1).astype(int)

# -- Visualize Anomalies
import plotly.express as px
st.subheader("📉 Anomaly Score Visualization")
fig_score = px.histogram(df, x='anomaly_score', nbins=50, title="Distribution of Anomaly Scores")
st.plotly_chart(fig_score, use_container_width=True)

fig_scatter = px.scatter(
    df,
    x="Billing Amount",
    y="Length of Stay",
    color=df['anomaly_score'].map({1: "Normal", -1: "Anomaly"}),
    title="Billing Amount vs Length of Stay with Anomalies"
)
st.plotly_chart(fig_scatter, use_container_width=True)

anomalies = df[df['anomaly_score'] == -1]
st.download_button("📁 Download Detected Anomalies", anomalies.to_csv(index=False), file_name="anomalies.csv", mime="text/csv")

st.markdown("""
### 💡 Insights from Anomaly Detection

The histogram of anomaly scores shows a clear distribution tail, indicating potential outliers in billing behavior.
Use these visual cues to:
- Flag and investigate unusual billing instances
- Audit patient records with short stays but high expenses
- Guide resource allocation or fraud detection strategies
""")
df['anomaly_score'] = iso.fit_predict(df[['Billing Amount']])
df['anomaly_label'] = (df['anomaly_score'] == -1).astype(int)

# -- Feature Importance (Top 10)
X_all = df[all_features]
y = df['anomaly_label']
X_train_all, X_test_all, y_train, y_test = train_test_split(X_all, y, stratify=y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled_all = scaler.fit_transform(X_train_all)
X_test_scaled_all = scaler.transform(X_test_all)

model = XGBClassifier(eval_metric='logloss')
model.fit(X_train_scaled_all, y_train)
importances = model.feature_importances_
feature_scores = pd.Series(importances, index=all_features).sort_values(ascending=False)
top_features = feature_scores.head(10).index.tolist()

# -- Feature Importance Chart
st.subheader("🧠 Top 10 Predictive Features")
fig, ax = plt.subplots()
top_10 = feature_scores.head(10).sort_values()
top_10.plot(kind='barh', ax=ax)
for i, (feature, value) in enumerate(top_10.items()):
    ax.text(value, i, f'{value:.3f}', va='center', ha='left')
ax.set_title("Feature Importances (XGBoost)")
st.pyplot(fig)

# -- SHAP Explainability
st.subheader("🔍 SHAP Explanation (Top 10 Features)")
explainer = shap.Explainer(model)
shap_values = explainer(X_test_scaled_all)
shap.summary_plot(shap_values, X_test_all, plot_type="bar", show=False)
fig = plt.gcf()
st.pyplot(fig)




# -- Model Comparison
st.sidebar.header("📊 Model Evaluation (Top 3 features selected by default)")
selected = st.sidebar.multiselect("Select Features for Modeling", top_features, default=top_features[:3])
if selected:
    X = df[selected]
    y = df['anomaly_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "SVM (RBF)": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "AdaBoost": AdaBoostClassifier(random_state=42)
    }
    st.subheader("📈 ROC Curves & Metrics")
    fig, ax = plt.subplots(figsize=(10, 6))
    results = []
    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        y_prob = m.predict_proba(X_test_scaled)[:, 1]
        y_pred = m.predict(X_test_scaled)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)

        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
        results.append([name, auc, f1, acc, prec])

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    results_df = pd.DataFrame(results, columns=["Model", "AUC", "F1", "Accuracy", "Precision"])
    st.dataframe(results_df.sort_values(by="AUC", ascending=False))

    # Show SHAP summary plot for top model
    best_model_name = results_df.sort_values(by="AUC", ascending=False).iloc[0]['Model']
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    explainer_top = shap.Explainer(best_model)
    shap_values_top = explainer_top(X_test_scaled)

    st.subheader(f"🔎 SHAP Summary for Best Model: {best_model_name}")
    shap.summary_plot(shap_values_top, X_test, plot_type="dot", show=False)
    st.pyplot(bbox_inches='tight')
    st.markdown("---")
 
    # --Narrative Summary Section 
    st.header("📘 Narrative Insights")
    st.markdown(f"""
    - A total of **{df.shape[0]}** records were processed. Anomaly detection flagged **{df['anomaly_label'].sum()}** high-risk cases using Isolation Forest.
    - Top predictive features include: **{', '.join(top_features[:5])}**, with **{top_features[0]}** contributing most to the model's decisions.
    - Among all tested models, **{results_df.sort_values(by='AUC', ascending=False).iloc[0]['Model']}** showed the highest performance with AUC of **{results_df.sort_values(by='AUC', ascending=False).iloc[0]['AUC']:.2f}**.
    - SHAP analysis confirms the impact of these features, providing interpretable justifications for predictions.
    """)

    st.subheader("🧠 Per-Model Evaluation Summary")
    
    st.markdown("### 💡 Recommendation: Use the best-performing model ({}) for critical alerts due to its AUC of {:.2f} and precision of {:.2f}.".format(
        best_model_name, results_df.sort_values(by='AUC', ascending=False).iloc[0]['AUC'], results_df.sort_values(by='AUC', ascending=False).iloc[0]['Precision']
    ))
    # -- GPT Summary
import openai
import time

def generate_gpt_summary(metrics_df, best_model, best_auc):
    api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI API key is missing. Please provide it in your app secrets or session.")
        return

    openai.api_key = api_key
    prompt = f"""
    Based on the model evaluation:
    Best model: {best_model} with AUC {best_auc:.2f}
    Metrics table:
    {metrics_df.to_markdown(index=False)}
    Provide 2-3 executive summary insights.
    """

    with st.spinner("Generating GPT summary..."):
        for attempt in range(3):
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a healthcare analyst generating executive summaries."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.subheader("🧠 AI-Generated Insight Summary")
                st.markdown(response.choices[0].message.content)
                return response.choices[0].message.content
            except Exception as e:
                if 'rate limit' in str(e).lower() or 'timeout' in str(e).lower():
                    st.warning("Retrying due to temporary OpenAI API issue...")
                    time.sleep(2 ** attempt)
                else:
                    st.error(f"GPT summary failed: {e}")
                    return ""


    # -- AI-Generated Summary Section
st.markdown("---")
st.subheader("🧠 AI-Generated Insight Summary")
generate_gpt_summary(results_df, best_model_name, results_df.sort_values(by='AUC', ascending=False).iloc[0]['AUC'])
