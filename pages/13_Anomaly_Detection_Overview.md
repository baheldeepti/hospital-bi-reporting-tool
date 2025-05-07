# 🏥 Streamlit App with Chat-Based Analytics for Anomaly Detection

An interactive analytics dashboard to explore, detect, and explain **anomalies in hospital data** — powered by machine learning, explainable AI (SHAP), and natural language summaries using OpenAI GPT.

---

## 🔍 Key Features

### 📂 Load and Clean Hospital Data
- Upload your own hospital dataset or use a preloaded sample
- Auto-cleansing and feature engineering on:
  - Date columns
  - Missing values
  - Derived fields like `Length of Stay`, `Billing per Day`, and interaction terms

### 🚨 Detect Anomalies
- Uses **Isolation Forest** to identify unusual billing or patient stay patterns
- Visualize anomaly scores and label distributions
- Download outlier records for audit or investigation

### 🧠 Feature Importance and Explainability
- **Top 10 features** influencing anomaly detection using XGBoost
- **SHAP explanations** for both global and per-model insights

### ⚖️ Model Comparison and Evaluation
- Train and evaluate multiple models:
  - Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, Naive Bayes, KNN, AdaBoost
- Metrics provided:
  - AUC, F1 Score, Accuracy, Precision
- Visualize ROC curves for all models

### 💬 AI-Powered Narrative Insights
- Summarizes key findings using **GPT-3.5**:
  - Executive-ready insights on anomalies and top features
  - Comparison of models using easy-to-understand language
- Requires OpenAI API key (via Streamlit secrets or session state)

---

## 📊 Outputs

- Anomaly detection charts (Histogram, Scatter)
- Feature importances (Bar chart)
- SHAP summary plots (Bar and dot plots)
- Model performance summary table
- Narrative summaries for business teams

---

## 📁 Required Data Columns

Ensure the dataset includes:

- `Date of Admission`, `Discharge Date`, `Billing Amount`, `Medical Condition`,  
- `Insurance Provider`, `Gender`, `Medication`, `Age`

---

