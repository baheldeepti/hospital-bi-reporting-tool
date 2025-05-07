import streamlit as st

st.set_page_config(page_title="🏥 Stay Analyzer Overview", layout="wide")

st.markdown("""
# 🏥 Hospital Stay Analyzer

An intelligent **Streamlit dashboard** for analyzing hospital stays, classifying patients by length of stay, detecting anomalies in billing, and evaluating predictive models with SHAP and AI-powered GPT summaries.

---

## 📊 Key Features

### 🩺 Length of Stay Classification
- Categorizes patient stays into:
  - **Short** (≤ 5 days)
  - **Medium** (6–15 days)
  - **Long** (16–45 days)
  - **Very Long** (> 45 days)
- Visualizes trends in stay duration by **month** and **year**
- Helps assess shifts in hospital capacity and burden

### ⚠️ Billing Anomaly Detection
- Uses **Isolation Forest** to flag anomalous billing cases
- Features include interactions with:
  - **Chronic illness**
  - **Admission type**
  - **Medical condition**

### 🧠 Feature Engineering & Enrichment
- ICD code mapping for medical condition analysis
- Chronic flags and emergency admission markers
- Seasonal variables (e.g., flu season), age groups, and socio-demographics

### 🔍 SHAP & Feature Importance
- Uses **XGBoost** for feature importance ranking
- **SHAP plots** provide global and local explanations for length of stay predictions

### 🧪 Model Training & Evaluation
- Trains and evaluates classifiers:
  - XGBoost, Random Forest, Logistic Regression, Gradient Boosting, AdaBoost
- Evaluates multi-class predictions across 4 stay categories
- Key metrics:
  - **Accuracy**
  - **F1 Score**
  - **Precision**
  - **Recall**
  - **ROC AUC** (multi-class)

### 📈 ROC Curve Comparison
- Implements One-vs-Rest (OvR) evaluation
- Aggregated ROC curves show model effectiveness across all classes

---

## 🧠 AI-Powered Insights (GPT)
- Summarizes key findings using **OpenAI GPT-3.5**
- Highlights model strengths, key features, and potential actions
- Requires an `OPENAI_API_KEY` in `.streamlit/secrets.toml` or session state

---

## 📁 Required Dataset Columns

Ensure your dataset includes:
- `Date of Admission`, `Discharge Date`, `Billing Amount`, `Medical Condition`
- `Admission Type`, `Hospital`, `Gender`, `Age`, `Blood Type`

Optional enrichments:
- `ICD Code`, `Is_Chronic`

---
""", unsafe_allow_html=True)
