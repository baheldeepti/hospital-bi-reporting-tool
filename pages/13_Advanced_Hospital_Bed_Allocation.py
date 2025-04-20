# ----------------------
# 📘 Title & Introduction
# ----------------------
import streamlit as st
import pandas as pd
import numpy as np
import time
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary
from ortools.linear_solver import pywraplp
from mip import Model, xsum, BINARY
import plotly.express as px

st.set_page_config(page_title="Hospital Bed Allocation Optimizer", layout="wide")
st.title("🏥 Hospital Bed Allocation Optimizer")

st.markdown("""
Optimize hospital bed allocation using AI-driven prescriptive analytics.

🔍 **Usage**: Load your own dataset or use a sample. Simulate scenarios, visualize KPI trends, and benchmark optimization models.
""")

# ----------------------
# 📥 Load Dataset
# ----------------------
st.sidebar.header("📥 Load or Use Sample Data")
data_option = st.sidebar.radio("Select Data Option", ["Use Sample Dataset", "Upload CSV File"])

if data_option == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload Patient CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv("https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv")

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
# 🔍 Filter Data
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
# 🌪 Optimization Simulation Controls
# ----------------------
st.subheader("⚙️ Optimization Simulation Controls")
total_beds = st.slider("Total Beds Available", 50, 500, 150)
staffing_capacity = st.slider("Available Staff Count", 10, 100, 50, step=1)
icu_beds = st.slider("ICU Beds Reserved", 0, 100, 10, step=1)
weekend_discharges = st.slider("Weekend Discharge Boost (%)", 0, 50, 10, step=1)
weekend_fraction = st.slider("% of Patients Discharged Early", 0.0, 1.0, 0.25, step=0.05)

surge_scenario = st.toggle("Activate Surge Scenario (e.g., flu season, pandemic surge)")

if surge_scenario:
    st.warning("🚨 Surge scenario is active. Emergency cases and average LOS will increase.")
    filtered_df = filtered_df.copy()
    filtered_df.loc[filtered_df['Admission Type'] == 'Emergency', 'Length_of_Stay'] *= 1.2
    filtered_df = pd.concat([filtered_df, filtered_df.sample(frac=0.2, replace=True)], ignore_index=True)

# 💡 Apply Weekend Discharge Impact
if weekend_discharges > 0:
    st.info("📅 Weekend Discharge Boost active. Shortening LOS for a portion of patients.")
    weekend_boost_factor = 1 - (weekend_discharges / 100)
    idx_sample = filtered_df.sample(frac=weekend_fraction, random_state=42).index
    filtered_df.loc[idx_sample, 'Length_of_Stay'] = filtered_df.loc[idx_sample, 'Length_of_Stay'] * weekend_boost_factor

scenario = st.radio("Scenario Strategy", ["Prioritize Emergency", "Minimize LOS", "Maximize Elective Intake"])

if surge_scenario:
    st.warning("🚨 Surge scenario is active. Emergency cases and average LOS will increase.")
    filtered_df = filtered_df.copy()
    filtered_df.loc[filtered_df['Admission Type'] == 'Emergency', 'Length_of_Stay'] *= 1.2
    filtered_df = pd.concat([filtered_df, filtered_df.sample(frac=0.2, replace=True)], ignore_index=True)

# ----------------------
# 🤖 Optimization Functions
# ----------------------
def optimize_with_pulp(df, scenario, beds, staff_cap, icu_beds):
    start = time.time()
        model = LpProblem("PULP_Optimizer", LpMinimize)
    x = LpVariable.dicts("Admit", df.index, cat=LpBinary)
    if scenario == "Prioritize Emergency":
        model += -lpSum([x[i] * (1 if df.loc[i, "Admission Type"] == "Emergency" else 0) for i in df.index])
    elif scenario == "Minimize LOS":
        model += lpSum([x[i] * df.loc[i, "Length_of_Stay"] for i in df.index])
    else:
        model += -lpSum([x[i] * (1 if df.loc[i, "Admission Type"] == "Elective" else 0) for i in df.index])
        model += lpSum([x[i] for i in df.index]) <= beds
    model += lpSum([x[i] * (1 if df.loc[i, 'ICU Required'] else 0) for i in df.index]) <= icu_beds
    model += lpSum([x[i] * df.loc[i, 'Priority'] for i in df.index]) <= staff_cap * 3
    model.solve()
    result = [i for i in df.index if x[i].varValue == 1]
    return result, time.time() - start

def optimize_with_ortools(df, scenario, beds, staff_cap, icu_beds):
    start = time.time()
        solver = pywraplp.Solver.CreateSolver('SCIP')
    x = [solver.BoolVar(f'x_{i}') for i in df.index]
    if scenario == "Prioritize Emergency":
        solver.Maximize(solver.Sum([x[i] * (1 if df.loc[i, "Admission Type"] == "Emergency" else 0) for i in df.index]))
    elif scenario == "Minimize LOS":
        solver.Minimize(solver.Sum([x[i] * df.loc[i, "Length_of_Stay"] for i in df.index]))
    else:
        solver.Maximize(solver.Sum([x[i] * (1 if df.loc[i, "Admission Type"] == "Elective" else 0) for i in df.index]))
        solver.Add(solver.Sum([x[i] for i in df.index]) <= beds)
    solver.Add(solver.Sum([x[i] * (1 if df.loc[i, 'ICU Required'] else 0) for i in df.index]) <= icu_beds)
    solver.Add(solver.Sum([x[i] * df.loc[i, 'Priority'] for i in df.index]) <= staff_cap * 3)
    solver.Solve()
    result = [i for i in df.index if x[i].solution_value() == 1]
    return result, time.time() - start

def optimize_with_mip(df, scenario, beds, staff_cap, icu_beds):
    start = time.time()
    model = Model()
    x = [model.add_var(var_type=BINARY) for _ in df.index]
    if scenario == "Prioritize Emergency":
        model.objective = model.maximize(xsum(x[i] * (1 if df.loc[i, "Admission Type"] == "Emergency" else 0) for i in df.index))
    elif scenario == "Minimize LOS":
        model.objective = xsum(x[i] * df.loc[i, "Length_of_Stay"] for i in df.index)
    else:
        model.objective = model.maximize(xsum(x[i] * (1 if df.loc[i, "Admission Type"] == "Elective" else 0) for i in df.index))
        model += xsum(x[i] for i in df.index) <= beds
    model += xsum(x[i] * (1 if df.loc[i, 'ICU Required'] else 0) for i in df.index) <= icu_beds
    model += xsum(x[i] * df.loc[i, 'Priority'] for i in df.index) <= staff_cap * 3
    model.optimize()
    result = [i for i in df.index if x[i].x >= 0.99]
    return result, time.time() - start

# ----------------------
# 🧮 Model Comparison
# ----------------------
relax_constraints = st.toggle("🪄 Relax Constraints (ignore ICU/staff limits)")
if relax_constraints:
    st.warning("Constraints relaxed. Optimization will ignore ICU and staffing limits.")
    icu_beds = filtered_df['ICU Required'].sum()
    staffing_capacity = int(filtered_df['Priority'].sum() / 3)

model_choice = st.selectbox("🧠 Select Optimization Solver", ["PuLP", "OR-Tools", "MIP"])

if model_choice == "PuLP":
    admitted, elapsed = optimize_with_pulp(filtered_df, scenario, total_beds, staffing_capacity, icu_beds)
elif model_choice == "OR-Tools":
    admitted, elapsed = optimize_with_ortools(filtered_df, scenario, total_beds, staffing_capacity, icu_beds)
else:
    admitted, elapsed = optimize_with_mip(filtered_df, scenario, total_beds, staffing_capacity, icu_beds)

admitted_df = filtered_df.loc[admitted].copy()
avg_los = admitted_df['Length_of_Stay'].mean() if not admitted_df.empty else 0
avg_priority = admitted_df['Priority'].mean() if not admitted_df.empty else 0

# ----------------------
# 🧠 AI Insights and Constraint Toggles
# ----------------------
st.header("📋 Strategy Recommendation Engine")
st.markdown(f"""
- **Scenario Applied**: `{scenario}`  
- **Solver Used**: `{model_choice}`  
- **Beds Used**: {len(admitted)} / {total_beds}  
- **Avg LOS**: {avg_los:.2f} days  
- **Solver Time**: {elapsed:.3f} seconds
""")

st.subheader("📊 KPI Summary")
if admitted_df.empty:
    st.error("No patients were admitted based on the selected constraints. Try adjusting filters or sliders.")
else:
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Beds Used", len(admitted), f"of {total_beds}")
    kpi2.metric("Avg LOS", f"{avg_los:.1f} days")
    kpi3.metric("Avg Priority", f"{avg_priority:.2f}")

        # 📍 Visual: ICU vs Non-ICU
    st.subheader("🛏️ ICU vs Non-ICU Patients")
    if 'ICU Required' in admitted_df.columns:
        st.bar_chart(admitted_df['ICU Required'].value_counts().rename({0: 'Non-ICU', 1: 'ICU'}))

    # 📈 Chart Comparison View
    st.subheader("📈 Admitted Patient Profile")
    st.bar_chart(admitted_df['Admission Type'].value_counts())

    st.subheader("💵 Estimated Resource Utilization (Cost Proxy)")
    admitted_df['Cost_Estimate'] = admitted_df['Length_of_Stay'] * admitted_df['Priority'] * 100
    cost_fig = px.box(admitted_df, x='Admission Type', y='Cost_Estimate', title='Estimated Cost Distribution by Admission Type')
        st.plotly_chart(cost_fig)

    # 🤖 AI-Generated Insights
    st.subheader("🤖 AI-Generated Strategic Suggestions")
    import openai
    openai.api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")

    prompt = f"""
    You are a hospital operations advisor. Analyze this hospital optimization scenario:
    - Scenario: {scenario}
    - Solver: {model_choice}
    - Total beds: {total_beds}, ICU beds reserved: {icu_beds}, Staff capacity: {staffing_capacity}
    - Weekend discharge boost: {weekend_discharges}% applied to {int(weekend_fraction*100)}% of patients
    - Average Length of Stay (LOS): {avg_los:.1f} days
    - Average priority of admitted patients: {avg_priority:.2f}
    
    Provide three clear, strategic recommendations to hospital leadership to improve operational efficiency and patient flow.
    """

    try:
        with st.spinner("Asking ChatGPT for strategic suggestions..."):
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a hospital operations advisor."},
                    {"role": "user", "content": prompt}
                ]
            )
        st.success("AI Recommendations:")
        st.markdown(response.choices[0].message.content)
    except Exception as e:
        st.warning("Could not fetch AI suggestions. Please check your OpenAI key or try again later.")
