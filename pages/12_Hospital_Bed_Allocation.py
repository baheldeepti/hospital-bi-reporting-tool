import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, LpStatus
import matplotlib.pyplot as plt
import plotly.express as px
import openai
import datetime

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
df['Priority'] = df['Admission Type'].map({'Emergency': 3, 'Elective': 2, 'Routine': 1})
df = df.dropna(subset=['Priority'])
df.reset_index(drop=True, inplace=True)

# --------------------------
# 🎛️ Interactive Filters
# --------------------------
with st.sidebar.expander("🔍 Filter Data"):
    hospital_filter = st.multiselect("Hospital", options=sorted(df['Hospital'].unique()), default=list(df['Hospital'].unique()))
    gender_filter = st.radio("Gender", options=["All"] + sorted(df['Gender'].unique().tolist()))
    age_range = st.slider("Age Range", min_value=0, max_value=100, value=(0, 100))
    default_start = df['Date of Admission'].min().date()
    default_end = df['Date of Admission'].max().date()
    date_range = st.date_input("Admission Date Range", [default_start, default_end])

    if len(date_range) != 2:
        st.warning("Please select a valid start and end date.")
        st.stop()

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

filtered_df = df[(df['Hospital'].isin(hospital_filter)) &
                 (df['Age'].between(age_range[0], age_range[1])) &
                 (df['Date of Admission'].between(start_date, end_date))]
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]

# --------------------------
# 🔁 Scenario Selector
# --------------------------
scenario = st.sidebar.selectbox("Scenario Strategy", ["Prioritize Emergency", "Minimize LOS", "Maximize Elective Intake"])
total_beds = st.sidebar.slider("Total Beds Available", 50, 300, 150)

# ----------------------
# 🤖 Optimization Model
# ----------------------
model = LpProblem("Bed_Allocation", LpMinimize)
x = LpVariable.dicts("Admit", filtered_df.index, cat=LpBinary)

if scenario == "Prioritize Emergency":
    model += lpSum([x[i] * (1 if filtered_df.loc[i, "Admission Type"] == "Emergency" else 10) for i in filtered_df.index])
elif scenario == "Minimize LOS":
    model += lpSum([x[i] * filtered_df.loc[i, "Length_of_Stay"] for i in filtered_df.index])
else:  # Maximize Elective Intake
    model += lpSum([x[i] * (1 if filtered_df.loc[i, "Admission Type"] == "Elective" else 5) for i in filtered_df.index])

model += lpSum([x[i] for i in filtered_df.index]) <= total_beds
model.solve()

admitted = [i for i in filtered_df.index if x[i].varValue == 1]
admitted_df = filtered_df.loc[admitted].copy()

# --------------------------
# 🧭 Dashboard Tabs
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 KPI Summary", "📊 Trend Analysis", "👩‍⚕️ Patient Drill-down", "💡 AI Strategy"])

with tab1:
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Beds Used", len(admitted), f"of {total_beds}")
    kpi2.metric("Avg LOS", f"{admitted_df['Length_of_Stay'].mean():.1f} days")
    kpi3.metric("Avg Priority", f"{admitted_df['Priority'].mean():.2f}")

    st.subheader("📊 Length of Stay Distribution")
    before, after = st.columns(2)
    fig_before = px.histogram(filtered_df, x="Length_of_Stay", nbins=10, title="Before Optimization", labels={"Length_of_Stay": "LOS (Days)"})
    fig_after = px.histogram(admitted_df, x="Length_of_Stay", nbins=10, title="After Optimization", labels={"Length_of_Stay": "LOS (Days)"})
    before.plotly_chart(fig_before, use_container_width=True)
    after.plotly_chart(fig_after, use_container_width=True)

with tab2:
    st.subheader("Average LOS by Gender and Hospital")
    grouped_data = admitted_df.groupby(["Hospital", "Gender"]).agg({"Length_of_Stay": "mean"}).reset_index()
    fig = px.bar(grouped_data, x="Hospital", y="Length_of_Stay", color="Gender", barmode="group",
                 labels={"Length_of_Stay": "Average LOS"}, title="Avg LOS by Gender & Hospital")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Admitted Patients Breakdown")
    st.dataframe(admitted_df[["Patient ID", "Admission Type", "Hospital", "Medical Condition", "Gender", "Age", "Length_of_Stay"]])

with tab4:
    st.markdown("""
    - Prioritize emergency and short LOS cases to maximize throughput.
    - Recommend weekend discharges to free capacity faster.
    - Cross-train nursing staff to handle peak LOS wards.
    - Add predictive discharge readiness score to enhance early movement.
    """)
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
