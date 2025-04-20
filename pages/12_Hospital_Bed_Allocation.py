import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, LpStatus
import matplotlib.pyplot as plt
import openai

# --------------------
# 📘 Title & Description
# --------------------
st.set_page_config(page_title="Hospital Bed Allocation Optimizer", layout="wide")
st.title("🏥 Hospital Bed Allocation Optimizer")

st.markdown("""
Optimize hospital bed allocation using AI-driven prescriptive analytics.

🔍 **Usage**: Load your own dataset or use a sample from GitHub. Run the optimizer and receive actionable recommendations based on capacity, condition, and cost constraints.

📊 **Objective**: Minimize stay duration, maximize throughput, and improve operational decisions with strategy simulations and AI-generated insights.
""")

# --------------------
# 📂 Load Dataset or Use Sample
# --------------------
st.sidebar.header("📥 Load or Use Sample Data")
data_option = st.sidebar.radio("Select Data Option", ["Use Sample Dataset", "Upload CSV File"])

if data_option == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload Patient CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Custom dataset loaded successfully!")
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()
else:
    df = pd.read_csv("https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv")
    st.info("Using sample dataset from GitHub.")

# ----------------------
# 🔄 Preprocessing
# ----------------------
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
df.dropna(subset=['Date of Admission', 'Discharge Date'], inplace=True)
df['Length_of_Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df = df[df['Length_of_Stay'] > 0]

# Rename and filter for optimization
df['Priority'] = df['Admission Type'].map({'Emergency': 3, 'Elective': 2, 'Routine': 1})
df = df.dropna(subset=['Priority'])
df.reset_index(drop=True, inplace=True)

# --------------------------
# ✅ Check Required Columns
# --------------------------


total_beds = st.sidebar.slider("Total Beds Available", 50, 300, 150)

# ----------------------
# 🤖 Optimization Model
# ----------------------
model = LpProblem("Bed_Allocation", LpMinimize)
x = LpVariable.dicts("Admit", df.index, cat=LpBinary)

model += lpSum([x[i] * df.loc[i, "Length_of_Stay"] / df.loc[i, "Priority"] for i in df.index])
model += lpSum([x[i] for i in df.index]) <= total_beds
model.solve()

admitted = [i for i in df.index if x[i].varValue == 1]
admitted_df = df.loc[admitted].copy()

# --------------------
# 📈 KPI Summary
# --------------------
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Beds Used", len(admitted), f"of {total_beds}")
kpi2.metric("Avg LOS", f"{admitted_df['Length_of_Stay'].mean():.1f} days")
kpi3.metric("Avg Priority", f"{admitted_df['Priority'].mean():.2f}")

# --------------------------
# 📊 Before vs After Visuals
# --------------------------
st.subheader("📊 Length of Stay Distribution")
before, after = st.columns(2)
before.bar_chart(df['Length_of_Stay'].value_counts().sort_index())
after.bar_chart(admitted_df['Length_of_Stay'].value_counts().sort_index())

# --------------------------
# 🔍 Strategy Recommendations
# --------------------------
st.subheader("📌 Strategy Recommendation Summary")
st.write("**Admitted Patients Breakdown by Admission Type:**")
st.dataframe(admitted_df.groupby("Admission Type").size().reset_index(name="Count"))

# --------------------------
# 🔎 Drill-down by Patient
# --------------------------
st.subheader("👩‍⚕️ Drill-down at Patient Level")
st.dataframe(admitted_df[["Patient ID", "Admission Type", "Hospital", "Medical Condition", "Gender", "Age", "Length_of_Stay"]])

# --------------------------
# 📈 Trend by Gender, Age, Hospital, Condition
# --------------------------
st.subheader("📈 Trend Analysis")
st.write("Average LOS by Gender and Hospital")
st.bar_chart(admitted_df.groupby(["Hospital", "Gender"])["Length_of_Stay"].mean().unstack())

# --------------------------
# 🧠 ChatGPT Recommendation
# --------------------------
st.subheader("🧠 AI-Generated Strategy")
st.markdown("""
- Prioritize emergency and short LOS cases to maximize throughput.
- Recommend weekend discharges to free capacity faster.
- Cross-train nursing staff to handle peak LOS wards.
- Add predictive discharge readiness score to enhance early movement.
""")

# --------------------------
# 🤖 AI Integration (ChatGPT)
# --------------------------
if st.button("Generate Additional Insights with ChatGPT"):
    prompt = f"""
    Based on this summary, suggest additional hospital management strategies:
    Total Beds: {total_beds}, Avg LOS: {admitted_df['Length_of_Stay'].mean():.1f},
    Emergency: {sum(admitted_df['Admission Type']=='Emergency')},
    Elective: {sum(admitted_df['Admission Type']=='Elective')},
    Routine: {sum(admitted_df['Admission Type']=='Routine')}
    """
    try:
        openai.api_key = st.secrets["openai_api_key"]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a hospital operations expert."},
                     {"role": "user", "content": prompt}]
        )
        st.success("AI Strategy Generated:")
        st.markdown(completion.choices[0].message.content)
    except Exception as e:
        st.warning("Unable to fetch recommendations. Please check your OpenAI API Key.")
