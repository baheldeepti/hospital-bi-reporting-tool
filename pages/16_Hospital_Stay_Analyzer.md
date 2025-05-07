# 🏥 Hospital Stay Analyzer

An intelligent Streamlit dashboard for analyzing hospital stays, classifying patients by length of stay, detecting anomalies in billing, and evaluating predictive models with SHAP and AI-powered GPT summaries.

---

## 📊 Key Features

### 🩺 Length of Stay Classification
- Custom classification into:
  - **Short** (≤5 days)
  - **Medium** (6–15 days)
  - **Long** (16–45 days)
  - **Very Long** (>45 days)
- Visualize trends in stay duration over months and years
- Understand shifts in hospital burden

### ⚠️ Billing Anomaly Detection
- Detects high-risk or irregular billing cases using **Isolation Forest**
- Adds features for interaction between billing and chronic conditions, admission type, and medical conditions

### 🧠 Feature Engineering & Enrichment
- Maps **ICD codes** and chronic flags to medical conditions
- Adds seasonal markers (flu season, emergency visits)
- Includes socio-demographic features (age group, admission type, etc.)

### 🔍 SHAP & Feature Importance
- Uses **XGBoost** to calculate feature importance
- Interactive SHAP summary plots reveal what drives longer stays

### 🧪 Model Training & Evaluation
- Trains classifiers using:
  - XGBoost, Random Forest, Logistic Regression, Gradient Boosting, AdaBoost
- Multi-class classification with 4 target classes
- Metrics reported:
  - **Accuracy**
  - **F1 Score**
  - **Precision**
  - **Recall**
  - **Multi-class ROC AUC**

### 📈 ROC Curve Comparison
- One-vs-Rest (OvR) strategy for all models
- Plots aggregated ROC curves across all four stay categories

---

## 🧠 AI-Powered Insights (GPT)
- Generates executive summaries using **OpenAI GPT-3.5**
- Explains top features, model metrics, and takeaways for hospital administrators
- Requires `OPENAI_API_KEY` (add via `.streamlit/secrets.toml`)

---

## 📁 Required Dataset Columns

Make sure the dataset includes at least:
- `Date of Admission`, `Discharge Date`, `Billing Amount`, `Medical Condition`
- `Admission Type`, `Hospital`, `Gender`, `Age`, `Blood Type`

Optional enrichments:
- `ICD codes`, `Chronic condition flag`

---
