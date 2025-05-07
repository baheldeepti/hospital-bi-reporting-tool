import streamlit as st

st.set_page_config(page_title="🏥 Strategy Optimization Overview", layout="wide")

st.markdown("""
# 🏥 Hospital Strategy Optimization Dashboard

Welcome! This dashboard helps hospitals save money and improve patient care using data, math, and a bit of AI. You don't need to be a data scientist to understand this guide — it's written for anyone, even a 15-year-old!

---

## 🚀 What This App Does
This app takes hospital patient data (like costs, stay duration, and weekend admissions) and helps you:

- Compare **three strategies** for cost-saving:
  - 🧮 **IP (Integer Programming)** – Smart and precise, like solving a puzzle.
  - 📏 **LP (Linear Programming)** – A relaxed version, more flexible but less exact.
  - ⚡ **Greedy** – Quick & easy, just picks the cheapest patients first.

- See which strategy is **best** using charts and scorecards.
- Find out which patients might be **cost anomalies** (too expensive).
- Get **AI suggestions** from ChatGPT based on your results.

---

## 📂 How to Use It (Step by Step)
1. **Upload Data** (CSV format) OR let the app load the default sample data.
2. **Adjust Settings**:
   - Use sliders to control things like how many patients to select, or how strict to be with costs.
3. **Compare Strategies**:
   - View metrics like total cost, average cost per patient, and anomalies.
4. **Dive into Patient Details**:
   - Look at real patient-level data by strategy.
5. **View Trend Charts**:
   - Explore monthly cost trends and compare across strategies.
6. **Download Results**:
   - Export strategy tables and patient data to CSV.

---

## 🧠 Features (In Simple Terms)
- **Strategy Comparison**: Helps you choose the best way to reduce billing costs.
- **KPI Cards**: Tiny dashboards that show you top metrics.
- **Line Charts & Pie Charts**: Make it easy to spot patterns.
- **Heatmap**: Shows how things like long stays or weekends relate to high billing.
- **Business Tips**: Gives you advice if too many patients are anomalies or long stay.
- **AI Assistant**: ChatGPT gives smart suggestions just for your data.

---

## 🧾 Requirements
This runs on **Streamlit Cloud** and uses the following Python packages:

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
pulp
openai
