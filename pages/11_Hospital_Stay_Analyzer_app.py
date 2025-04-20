# Streamlit Hospital Stay Analyzer App

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for torch

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import openai

from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- App Layout ---
st.set_page_config(page_title="Hospital Stay Analyzer", layout="wide")
st.title("🏥 Hospital Stay Analyzer")
st.markdown("""
Analyze hospital stays, detect billing anomalies, and uncover drivers of prolonged length of stay.

**Features:**
- Upload or use default dataset
- Custom categorization of stay duration
- Anomaly detection on billing
- Feature importance insights with XGBoost + SHAP
- Model evaluation with AI-generated summaries
""")

# --- Data Upload or Default ---
st.sidebar.header("Data Options")
upload = st.sidebar.file_uploader("Upload Your CSV", type=["csv"])

if upload:
    df = pd.read_csv(upload)
else:
    df = pd.read_csv("https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv")

# --- Preprocessing ---
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
df.dropna(subset=['Date of Admission', 'Discharge Date'], inplace=True)
df['Length_of_Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df = df[df['Length_of_Stay'] > 0]

def custom_stay_category(days):
    if days <= 5:
        return 'Short'
    elif 5 < days <= 15:
        return 'Medium'
    elif 15 < days <= 45:
        return 'Long'
    else:
        return 'Very Long'

# Apply logic
df['Stay_Category_Custom'] = df['Length_of_Stay'].apply(custom_stay_category)

# ICD Mapping
icd_mapping = {
    'Infections': 'A49.9', 'Flu': 'J10.1', 'Cancer': 'C80.1', 'Asthma': 'J45.909',
    'Heart Disease': 'I51.9', 'Alzheimer’s': 'G30.9', 'Diabetes': 'E11.9', 'Obesity': 'E66.9'
}
df['ICD_Code'] = df['Medical Condition'].map(icd_mapping)
df['ICD_Chapter'] = df['ICD_Code'].str[0]
df['Is_Chronic'] = df['ICD_Code'].isin(['C80.1','J45.909','I51.9','G30.9','E11.9','E66.9']).astype(int)

# Feature Engineering
df['Is_Emergency'] = df['Admission Type'].str.lower().str.contains('emergency').astype(int)
df['Quarter'] = df['Date of Admission'].dt.quarter
df['Is_Flu_Season'] = df['Date of Admission'].dt.month.isin([10, 11, 12, 1, 2]).astype(int)
df['Age_Group'] = df['Age'].apply(lambda x: 'Child' if x <= 18 else ('Adult' if x <= 60 else 'Senior'))

for col in ['Gender', 'Hospital', 'Admission Type', 'Medical Condition', 'Blood Type', 'ICD_Chapter', 'Age_Group']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Billing Features & Anomaly Detection
df['Billing_per_day'] = df['Billing Amount'] / df['Length_of_Stay']
df['Log_Billing'] = np.log1p(df['Billing Amount'])
df['Log_Billing_per_day'] = np.log1p(df['Billing_per_day'])

iso = IsolationForest(contamination=0.05, random_state=42)
df['Billing_Anomaly'] = iso.fit_predict(df[['Log_Billing', 'Log_Billing_per_day']].fillna(0))
df['Billing_Anomaly'] = df['Billing_Anomaly'].map({1: 0, -1: 1})

# Interaction Features
df['Billing_x_Chronic'] = df['Billing Amount'] * df['Is_Chronic']
df['Billing_x_Condition'] = df['Billing Amount'] * df['Medical Condition']
df['ICD_x_Emergency'] = df['ICD_Chapter'] * df['Is_Emergency']
df['Anomaly_x_Condition'] = df['Billing_Anomaly'] * df['Medical Condition']

# --- Filter Sidebar ---
st.sidebar.header("Filters")
year_filter = st.sidebar.multiselect("Year", options=sorted(df['Date of Admission'].dt.year.unique()), default=None)
month_filter = st.sidebar.multiselect("Month", options=sorted(df['Date of Admission'].dt.month.unique()), default=None)

if year_filter:
    df = df[df['Date of Admission'].dt.year.isin(year_filter)]
if month_filter:
    df = df[df['Date of Admission'].dt.month.isin(month_filter)]

# --- Visualizations ---
df['Month'] = df['Date of Admission'].dt.to_period('M')
los_by_month = df.groupby('Month')['Length_of_Stay'].mean()
st.subheader("📊 Average Length of Stay by Month")
fig, ax = plt.subplots()
los_by_month.plot(kind='line', marker='o', ax=ax)
ax.set_ylabel("Days")
ax.set_xlabel("Month")
st.pyplot(fig)

stay_pct = df.groupby(['Month', 'Stay_Category_Custom']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack().fillna(0)
st.subheader("📊 Stay Category % Distribution Over Time")
fig2, ax2 = plt.subplots(figsize=(10, 5))
stay_pct.plot(kind='bar', stacked=True, ax=ax2)
ax2.set_ylabel("Percentage")
ax2.set_xlabel("Month")
st.pyplot(fig2)

# --- Model Training ---
stay_mapping = {'Short': 0, 'Medium': 1, 'Long': 2, 'Very Long': 3}
df['Stay_Class'] = df['Stay_Category_Custom'].map(stay_mapping)

features = [
    'Medical Condition', 'Billing Amount', 'ICD_Chapter', 'Is_Chronic', 'Billing_Anomaly',
    'Billing_x_Chronic', 'Billing_x_Condition', 'ICD_x_Emergency', 'Anomaly_x_Condition'
]

X = df[features]
y = df['Stay_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test_scaled)
# st.subheader("📋 Classification Report")
# st.text(classification_report(y_test, y_pred, target_names=stay_mapping.keys()))

cm = confusion_matrix(y_test, y_pred)
# fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(stay_mapping.keys()))
# disp.plot(ax=ax_cm, cmap='Blues', values_format='d')
# st.pyplot(fig_cm)

# --- SHAP Feature Importance ---
st.subheader("🧠 SHAP Feature Importances")
explainer = shap.Explainer(model)
shap_values = explainer(X_train_scaled)
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
fig_shap = plt.gcf()
st.pyplot(fig_shap)

# --- GPT Summary Insights ---
api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
if api_key:
    openai.api_key = api_key
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prompt = f"""
    Based on the classification results:
    - Accuracy: {acc:.2f}
    - F1 Score: {f1:.2f}
    - Top 3 features: {feat_imp.sort_values(ascending=False).head(3).to_dict()}

    Provide a short executive summary with 2-3 key insights to help hospital operations.
    """
    with st.spinner("Generating AI Insights..."):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a healthcare analyst generating executive summaries."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.subheader("💡 AI-Generated Insights")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"OpenAI API call failed: {e}")
else:
    st.warning("🔐 OpenAI API Key not found. Please add it in Streamlit secrets to enable GPT insights.")

# --- Model Comparison ---
st.subheader("\U0001F916 Model Comparison")

# --- Compare Multiple Models ---
st.markdown("### \U0001F5A5️ Evaluating Multiple Models")
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

features = ['Medical Condition', 'Billing Amount', 'ICD_Chapter', 'Is_Chronic', 'Billing_Anomaly']
X = df[features]
y = df['Stay_Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "XGBoost": XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(multi_class='multinomial', max_iter=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({"Model": name, "Accuracy": acc, "F1-Score": f1})

results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
st.dataframe(results_df)

# --- GPT Summary ---
st.markdown("---")
st.subheader("\U0001F4AC GPT Summary of Insights")
api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
if api_key:
    import openai
    openai.api_key = api_key
    best_model_name = results_df.iloc[0]['Model']
    best_f1 = results_df.iloc[0]['F1-Score']
    prompt = f"""
    Based on the model evaluation:
    Best model: {best_model_name} with F1-score {best_f1:.2f}
    Metrics table:
    {results_df.to_markdown(index=False)}
    Provide 2-3 executive summary insights in simple business language.
    """

    with st.spinner("Generating GPT insights..."):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a healthcare analyst generating executive summaries."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"GPT summary failed: {e}")
else:
    st.warning("Please add your OpenAI API key to Streamlit secrets or session state.")
