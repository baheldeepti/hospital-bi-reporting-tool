# ----------------------
# 📘 Title & Introduction
# ----------------------
import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary
import plotly.express as px
import openai

st.set_page_config(page_title="Hospital Bed Allocation Optimizer", layout="wide")
st.title("🏥 Hospital Bed Allocation Optimizer")

st.markdown("""
Optimize hospital bed allocation using AI-driven prescriptive analytics.

🔍 **Usage**: Load your own dataset or use a sample. Simulate scenarios, visualize KPI trends, and receive AI-generated business recommendations.
""")

# ----------------------
# 📥 Load Dataset
# ----------------------
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
# 📂 Preprocessing
# ----------------------
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
df.dropna(subset=['Date of Admission', 'Discharge Date'], inplace=True)
df['Length_of_Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df = df[df['Length_of_Stay'] > 0]
df['Priority'] = df['Admission Type'].map({'Emergency': 3, 'Elective': 2, 'Routine': 1})
df = df.dropna(subset=['Priority'])
df.reset_index(drop=True, inplace=True)

# ----------------------
# # Sidebar Constraints
# ----------------------
with st.sidebar.expander("🔍 Filter Data"):
    hospital_filter = st.multiselect("Hospital", sorted(df['Hospital'].dropna().unique()), default=list(df['Hospital'].dropna().unique()))
    gender_filter = st.radio("Gender", ["All"] + sorted(df['Gender'].dropna().unique().tolist()))
    age_range = st.slider("Age Range", 0, 100, (0, 100))
    date_range = st.date_input("Admission Date Range", [df['Date of Admission'].min().date(), df['Date of Admission'].max().date()])

    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
    else:
        st.stop()

filtered_df = df[df['Hospital'].isin(hospital_filter) &
                 df['Age'].between(age_range[0], age_range[1]) &
                 df['Date of Admission'].between(start_date, end_date)]
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]

# ----------------------
# 🌪 Surge Scenario Simulation
# ----------------------
st.subheader("⚙️ Optimization Simulation Controls")
total_beds = st.slider("Total Beds Available", 50, 500, 150)
staffing_capacity = st.slider("Available Staff Count", 10, 100, 50, step=1)
icu_beds = st.slider("ICU Beds Reserved", 0, 100, 10, step=1)
weekend_discharges = st.slider("Weekend Discharge Boost (%)", 0, 50, 10, step=1)

surge_scenario = st.toggle("Activate Surge Scenario (e.g., flu season, pandemic surge)")

if surge_scenario:
    st.warning("🚨 Surge scenario is active. Emergency cases and average LOS will increase.")
    filtered_df = filtered_df.copy()
    filtered_df.loc[filtered_df['Admission Type'] == 'Emergency', 'Length_of_Stay'] *= 1.2
    filtered_df = pd.concat([filtered_df, filtered_df.sample(frac=0.2, replace=True)], ignore_index=True)

scenario = st.radio("Scenario Strategy", ["Prioritize Emergency", "Minimize LOS", "Maximize Elective Intake"])

# ----------------------
# 🤖 Strategy Simulation
# ----------------------
model = LpProblem("Bed_Allocation", LpMinimize)
x = LpVariable.dicts("Admit", filtered_df.index, cat=LpBinary)

if scenario == "Prioritize Emergency":
    model += -lpSum([x[i] * (1 if filtered_df.loc[i, "Admission Type"] == "Emergency" else 0) for i in filtered_df.index])
elif scenario == "Minimize LOS":
    model += lpSum([x[i] * filtered_df.loc[i, "Length_of_Stay"] for i in filtered_df.index])
else:  # Maximize Elective
    model += -lpSum([x[i] * (1 if filtered_df.loc[i, "Admission Type"] == "Elective" else 0) for i in filtered_df.index])


model += lpSum([x[i] for i in filtered_df.index]) <= total_beds
model.solve()

admitted = [i for i in filtered_df.index if x[i].varValue == 1]
admitted_df = filtered_df.loc[admitted].copy()

# ----------------------
# 📋 Strategy Recommendation Engine
# ----------------------
st.header("📋 Strategy Recommendation Engine")
avg_los = admitted_df['Length_of_Stay'].mean() if not admitted_df.empty else 0
avg_priority = admitted_df['Priority'].mean() if not admitted_df.empty else 0

st.markdown(f"""
- **Scenario Applied**: `{scenario}`  
- **Beds Used**: {len(admitted)} / {total_beds}  
- **Avg LOS**: {avg_los:.2f} days
""")

# ----------------------
# 📊 KPI Summary
# ----------------------
st.subheader("📊 KPI Summary")
if admitted_df.empty:
    st.error("No patients were admitted based on the selected constraints. Try adjusting filters or sliders.")
else:
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Beds Used", len(admitted), f"of {total_beds}")
    kpi2.metric("Avg LOS", f"{avg_los:.1f} days")
    kpi3.metric("Avg Priority", f"{avg_priority:.2f}")

# ----------------------
# 💡 Business Recommendations based on KPI
# ----------------------
st.subheader("💡 Business Recommendations")
st.markdown("""
- Prioritize short LOS + high priority admissions during peak days.
- Improve ICU rotation efficiency to reduce congestion.
- Deploy weekend discharge planning teams to optimize bed turnover.
""")

# ----------------------
# 🤖 AI-Powered Suggestions (Dynamic with OpenAI)
# ----------------------
if st.button("🧠 Ask ChatGPT for Suggestions"):
    prompt = f"""
    Based on a hospital scenario where beds used = {len(admitted)},
    average LOS = {avg_los:.1f} days,
    and scenario selected is '{scenario}', provide 3 strategic actions for improvement.
    """
    try:
        openai.api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
        with st.spinner("Asking ChatGPT for strategy suggestions..."):
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a hospital strategy advisor."},
                    {"role": "user", "content": prompt}
                ]
            )
        st.success("AI Recommendations:")
        st.markdown(response.choices[0].message.content)
    except Exception as e:
        st.warning("Could not fetch OpenAI suggestions. Check your API key or try again later.")
