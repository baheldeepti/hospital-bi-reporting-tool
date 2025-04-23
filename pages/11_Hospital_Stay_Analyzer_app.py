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


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_score, recall_score

# --- App Layout ---
st.set_page_config(page_title="Hospital Stay Analyzer", layout="wide")
st.title("🏥 Hospital Stay Analyzer")
st.markdown("""
Analyze hospital stays, detect billing anomalies, and uncover drivers of prolonged length of stay.

**Features:**
- Upload or use default dataset
- Custom classification of hospital stay length: Short, Medium, Long, Very Long
  - **Short**: ≤ 5 days
  - **Medium**: 6 to 15 days
  - **Long**: 16 to 45 days
  - **Very Long**: > 45 days
- Anomaly detection on billing
- Feature importance insights with XGBoost 
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

st.subheader("\U0001F4CA Stay Category Distribution Over Time")

# Ensure 'Year' column exists
df['Year'] = df['Date of Admission'].dt.year

# Compute % distribution of stay categories by Year
stay_pct = (
    df.groupby(['Year', 'Stay_Category_Custom'])
    .size()
    .groupby(level=0)
    .apply(lambda x: 100 * x / x.sum())
    .unstack()
    .fillna(0)
)

# Plot stacked bar chart
fig2, ax2 = plt.subplots(figsize=(12, 6))
stay_pct.plot(kind='bar', stacked=True, ax=ax2)
ax2.set_title("Stay Categories % Distribution Over Time")
ax2.set_ylabel("% Share")
ax2.set_xlabel("Year")
ax2.set_xticks(range(len(stay_pct)))
ax2.set_xticklabels([str(year) for year in stay_pct.index], rotation=45, ha='right')
ax2.legend(title="Stay Category", bbox_to_anchor=(1.05, 1), loc='upper left')
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
report = classification_report(y_test, y_pred, target_names=['Short', 'Medium', 'Long', 'Very Long'])
report_dict = classification_report(y_test, y_pred, target_names=['Short', 'Medium', 'Long', 'Very Long'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)

# --- Feature Importance ---
st.subheader("🔍 Feature Importance with SHAP & XGBoost")

st.subheader("📊 Classification Report Table")
st.dataframe(report_df.style.background_gradient(cmap='Oranges'))

# Display Confusion Matrix as Table
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Short', 'Medium', 'Long', 'Very Long'], columns=['Pred: Short', 'Pred: Medium', 'Pred: Long', 'Pred: Very Long'])
st.subheader("\U0001F4CB Confusion Matrix Table")
st.dataframe(cm_df.style.background_gradient(cmap='Blues').format("{:.0f}"))



# --- Add Feature Importance Plot ---
st.subheader("\U0001F9E0 XGBoost Feature Importances")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values()
fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
feat_imp.plot(kind='barh', ax=ax_imp)
ax_imp.set_title("XGBoost Feature Importances")
ax_imp.set_xlabel("Importance Score")
st.pyplot(fig_imp)


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
st.subheader("\U0001F52C Model Performance & ROC Comparison")
st.subheader("🧠 Approach & Methodology for Model Evaluation")

st.markdown("""
To evaluate how well our models predict the **Length of Stay** categories, we followed a structured approach:

---

### 📌 1. Data Preparation
- Selected clinical and administrative features such as:
  - **Medical Condition**, **Billing Amount**, **ICD Codes**, **Chronic Status**, and **Anomaly Detection**.
- Created a target variable with four classes: **Short**, **Medium**, **Long**, and **Very Long** stays.

---

### 📈 3. Feature Importance via XGBoost
- XGBoost was used to rank features based on their predictive contribution.
- The **top 5 features** impacting hospital stay prediction were used.

### 🤖 3. Model Training (One-vs-Rest)
- Trained five models using a One-vs-Rest (OvR) strategy:
  - ✅ XGBoost  
  - ✅ Random Forest  
  - ✅ Logistic Regression  
  - ✅ Gradient Boosting  
  - ✅ AdaBoost  
- All models were trained on scaled features and evaluated on a 20% test split.

---



These features provide strong signals about cost, severity, and urgency of cases—key drivers of prolonged hospital stays.

---

### 🧪 4. ROC Curve Analysis
- Used one-hot encoding to compute ROC curves for each class.
- Averaged ROC curves across 4 classes to visualize trade-off between **True Positive Rate (TPR)** and **False Positive Rate (FPR)**.

---

### 🧾 5. Performance Metrics
- Computed:
  - **Accuracy**
  - **F1 Score**
  - **Precision**
  - **Recall**
- Best model was selected based on highest **F1 Score** and ROC AUC.

---

This robust modeling pipeline enables hospitals to better forecast resource needs and improve operational planning.
""")

# --- Model Comparison (Corrected) ---
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Prepare Data
df['Stay_Class'] = df['Stay_Category_Custom'].map({'Short': 0, 'Medium': 1, 'Long': 2, 'Very Long': 3})
features = ['Medical Condition', 'Billing Amount', 'ICD_Chapter', 'Is_Chronic', 'Billing_Anomaly']
X = df[features]
y = df['Stay_Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Binarize for ROC
y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "XGBoost": XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(multi_class='multinomial', max_iter=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

results = []
plt.figure(figsize=(10, 7))

for name, base_model in models.items():
    clf = OneVsRestClassifier(base_model)
    clf.fit(X_train_scaled, y_train_bin)
    
    y_score = clf.predict_proba(X_test_scaled)  # shape: (n_samples, n_classes)
    y_pred_bin = clf.predict(X_test_scaled)     # shape: (n_samples, n_classes)
    y_pred = np.argmax(y_score, axis=1)         # convert to single-label prediction
    
    # ROC Curves
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr[i], tpr[i]) for i in range(n_classes)], axis=0)
    avg_auc = np.mean(list(roc_auc.values()))
    
    plt.plot(mean_fpr, mean_tpr, label=f'{name} (Avg AUC = {avg_auc:.2f})')

    # Metrics
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'ROC AUC': avg_auc
    })

# Plot settings
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve Comparison (Multi-Class OvR)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
st.pyplot(plt.gcf())

# Results table
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
st.subheader("📋 Model Performance Table")

# Apply background gradient to key metric columns
styled_results_df = results_df.style.background_gradient(
    subset=["F1-Score", "Accuracy", "Precision", "Recall", "ROC AUC"],
    cmap='YlGn'  # Yellow-Green color map (good for performance heatmaps)
).format({
    "F1-Score": "{:.2f}",
    "Accuracy": "{:.2f}",
    "Precision": "{:.2f}",
    "Recall": "{:.2f}",
    "ROC AUC": "{:.2f}"
})

st.dataframe(styled_results_df)



# --- GPT Summary ---
st.subheader("\U0001F4AC GPT Summary of Insights")
api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
if api_key:
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
