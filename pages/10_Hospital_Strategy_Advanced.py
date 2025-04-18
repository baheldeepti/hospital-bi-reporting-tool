import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, LpStatus, value
import openai
openai.api_key = st.secrets.get("openai_api_key")


# ----------------------
# 📘 Title & Introduction
# ----------------------
st.set_page_config(page_title="Hospital Strategy Recommender", layout="wide")
st.title("🏥 Hospital Strategy Optimization Dashboard")

# ----------------------
# 📥 Load Dataset
# ----------------------
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
            df = pd.read_csv(url)
        df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
        df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
        df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
        df = df.dropna(subset=['Billing Amount', 'Length of Stay', 'Age'])
        df['Is Weekend'] = (df['Date of Admission'].dt.weekday >= 5).astype(int)
        df['Is Long Stay'] = (df['Length of Stay'] > df['Length of Stay'].quantile(0.95)).astype(int)
        df['Gender'] = df['Gender'].fillna("Unknown")
        df['Insurance Provider'] = df['Insurance Provider'].fillna("Unknown")
        df['Insurance'] = pd.factorize(df['Insurance Provider'])[0]
        df['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
        df['Medication'] = LabelEncoder().fit_transform(df['Medication'])
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly_score'] = iso.fit_predict(df[['Billing Amount']])
        df['anomaly_prob'] = (df['anomaly_score'] == -1).astype(int)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

uploaded_file = st.sidebar.file_uploader("📂 Upload your hospital data (CSV)", type=["csv"])
df = load_data(uploaded_file)

if df.empty:
    st.stop()

# ----------------------
# ⚙️ Simulation Sliders
# ----------------------
with st.expander("⚙️ Simulate Optimization Thresholds"):
    sim_weekend = st.slider("Max % Weekend Admissions", 0.0, 1.0, 0.2, step=0.05)
    sim_longstay = st.slider("Max % Long Stays", 0.0, 1.0, 0.1, step=0.05)
    sim_anomaly = st.slider("Max % Anomaly", 0.0, 1.0, 0.05, step=0.01)
    sim_min_patients = st.slider("Min Patients Selected", 0, 100, 70, step=5)

# ----------------------
# 🧠 Optimization Comparison
# ----------------------
st.subheader("🧠 Optimization Comparison: LP vs IP")
model_sample = df[['Billing Amount', 'Is Weekend', 'Is Long Stay', 'anomaly_prob']].dropna().sample(n=100, random_state=42)
c = model_sample['Billing Amount'].values
A_ub = [
    model_sample['Is Weekend'].values,
    model_sample['Is Long Stay'].values,
    model_sample['anomaly_prob'].values
]

# IP Model using PuLP
ip_model = LpProblem("Hospital_Optimization_IP", LpMinimize)
x_vars_ip = [LpVariable(f"x_ip_{i}", cat=LpBinary) for i in range(len(c))]
ip_model += lpSum([c[i] * x_vars_ip[i] for i in range(len(c))])
ip_model += lpSum([x_vars_ip[i] for i in range(len(c))]) == sim_min_patients
ip_model += lpSum([A_ub[0][i] * x_vars_ip[i] for i in range(len(c))]) <= sim_weekend * len(model_sample)
ip_model += lpSum([A_ub[1][i] * x_vars_ip[i] for i in range(len(c))]) <= sim_longstay * len(model_sample)
ip_model += lpSum([A_ub[2][i] * x_vars_ip[i] for i in range(len(c))]) <= sim_anomaly * len(model_sample)
status_ip = ip_model.solve()
model_sample['IP_Selected'] = [int(x_vars_ip[i].varValue) for i in range(len(c))] if LpStatus[status_ip] == "Optimal" else 0

# LP model (relaxed version)
x_lp = sim_min_patients / len(model_sample)
model_sample['LP_Selected'] = 0
model_sample.loc[model_sample.sort_values(by='Billing Amount').head(sim_min_patients).index, 'LP_Selected'] = 1

# Greedy baseline
model_sample['Greedy_Selected'] = 0
model_sample.loc[model_sample.sort_values(by='Billing Amount').head(sim_min_patients).index, 'Greedy_Selected'] = 1

# Comparison
strategies = ['IP_Selected', 'LP_Selected', 'Greedy_Selected']
results = []
for strat in strategies:
    subset = model_sample[model_sample[strat] == 1]
    results.append({
        'Strategy': strat.replace('_Selected', ''),
        'Total Cost': subset['Billing Amount'].sum(),
        '% Weekend': 100 * subset['Is Weekend'].mean(),
        '% Long Stay': 100 * subset['Is Long Stay'].mean(),
        '% Anomaly': 100 * subset['anomaly_prob'].mean()
    })

comparison_df = pd.DataFrame(results)
st.dataframe(comparison_df, use_container_width=True)

# ----------------------
# 🧮 Model Performance Comparison
# ----------------------
st.subheader("🧮 Model Performance Metrics")
perf_metrics = []
for strat in strategies:
    subset = model_sample[model_sample[strat] == 1]
    cost = subset['Billing Amount'].sum()
    avg_cost = subset['Billing Amount'].mean()
    patient_count = len(subset)
    anomaly_count = subset['anomaly_prob'].sum()
    perf_metrics.append({
        'Strategy': strat.replace('_Selected', ''),
        'Total Patients': patient_count,
        'Total Cost': f"${cost:,.0f}",
        'Avg Cost per Patient': f"${avg_cost:,.0f}",
        'Anomaly Cases': anomaly_count
    })

perf_df = pd.DataFrame(perf_metrics)
st.dataframe(perf_df, use_container_width=True)
# Visualization of metrics
st.subheader("📊 Performance Metrics Visualization")
fig_perf, ax_perf = plt.subplots(figsize=(10, 5))
for column in ['Total Cost', 'Avg Cost per Patient', 'Anomaly Cases']:
    y_values = perf_df[column]
    if perf_df[column].dtype == 'O':
        y_values = perf_df[column].replace({r'[$,]': ''}, regex=True).astype(float)
    ax_perf.plot(perf_df['Strategy'], y_values, marker='o', label=column)
ax_perf.set_title("Strategy-wise Performance Comparison")
ax_perf.set_ylabel("Metric Value")
ax_perf.set_xlabel("Strategy")
ax_perf.legend()
ax_perf.grid(True)
st.pyplot(fig_perf)
# ----------------------
# 📊 KPI Summary
# ----------------------
top = comparison_df.sort_values(by='Total Cost').iloc[0]
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Top Strategy", top['Strategy'])
kpi2.metric("Total Cost", f"${top['Total Cost']:,.2f}")
kpi3.metric("% Long Stay", f"{top['% Long Stay']:.2f}%")
kpi4.metric("% Anomalies", f"{top['% Anomaly']:.2f}%")

# ----------------------
# 🎯 Patient Drilldown
# ----------------------
st.subheader("🎯 Patient-Level Drilldown")
strategy_choice = st.selectbox("Choose Strategy", ['IP', 'LP', 'Greedy'])
selected_col = f"{strategy_choice}_Selected"
filtered_patients = model_sample[model_sample[selected_col] == 1]
st.dataframe(filtered_patients, use_container_width=True)

# ----------------------
# 📅 Monthly Billing Trends Over Time
# ----------------------
st.subheader("📅 Monthly Billing Trends Over Time")
model_sample['Date'] = df['Date of Admission'].sample(n=100, random_state=42).dt.to_period("M").dt.to_timestamp()
trend_df = model_sample.groupby('Date')['Billing Amount'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(trend_df['Date'], trend_df['Billing Amount'], marker='o')
ax.set_title("Monthly Billing Trends")
ax.set_xlabel("Month")
ax.set_ylabel("Billing Amount")
ax.grid(True)
st.pyplot(fig)

# ----------------------
# 📈 Multi-Strategy Overlay Line Chart
# ----------------------
st.subheader("📈 Strategy-Wise Monthly Billing Trends")
fig3, ax3 = plt.subplots(figsize=(10, 5))
for strat in ['IP_Selected', 'LP_Selected', 'Greedy_Selected']:
    strat_df = model_sample[model_sample[strat] == 1].groupby('Date')['Billing Amount'].sum().reset_index()
    ax3.plot(strat_df['Date'], strat_df['Billing Amount'], label=strat.replace('_Selected', ''))
ax3.set_title("Billing Trends by Strategy")
ax3.set_xlabel("Month")
ax3.set_ylabel("Total Billing Amount")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

# ----------------------
# 📊 Strategy Distribution
# ----------------------
st.subheader("📊 Strategy Selection Distribution")
strategy_counts = {
    'IP': model_sample['IP_Selected'].sum(),
    'LP': model_sample['LP_Selected'].sum(),
    'Greedy': model_sample['Greedy_Selected'].sum()
}
pie_data = pd.DataFrame.from_dict(strategy_counts, orient='index', columns=['Count'])
fig2, ax2 = plt.subplots()
ax2.pie(pie_data['Count'], labels=pie_data.index, autopct='%1.1f%%', startangle=90)
ax2.axis('equal')
st.pyplot(fig2)

# ----------------------
# 🧮 Correlation Heatmap
# ----------------------
st.subheader("🧮 Correlation Heatmap of Key Metrics")
corr_data = model_sample[['Billing Amount', 'Is Weekend', 'Is Long Stay', 'anomaly_prob']]
corr = corr_data.corr()
fig4, ax4 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
ax4.set_title("Correlation Matrix")
st.pyplot(fig4)

# ----------------------
# 💡 Business Recommendations
# ----------------------
st.subheader("💡 Business Recommendations")

# Generate dynamic recommendations with icons
business_recs = []
if top['% Weekend'] > 10:
    business_recs.append("🔄 Consider reducing weekend admissions to control cost surges.")
if top['% Long Stay'] > 10:
    business_recs.append("🛌 Evaluate long-stay cases for opportunities to improve discharge planning.")
if top['% Anomaly'] > 5:
    business_recs.append("🚨 Enhance anomaly detection and investigate high-billing outliers.")
if top['Total Cost'] > 500000:
    business_recs.append("💰 Review billing practices to identify potential cost-saving areas.")
business_recs.append("📊 Reassess constraints regularly to align with operational goals.")

with st.container():
    for rec in business_recs:
        st.markdown(f"- {rec}")

# ----------------------
# 🤖 AI-Powered Suggestions
# ----------------------
st.subheader("🤖 AI Recommendations from ChatGPT")
openai.api_key = st.secrets.get("openai_api_key")
prompt = f"""
You are a healthcare optimization expert. Based on the following results:
- Strategy: {top['Strategy']}
- Total Cost: ${top['Total Cost']:.2f}
- % Weekend: {top['% Weekend']:.2f}%
- % Long Stay: {top['% Long Stay']:.2f}%
- % Anomaly: {top['% Anomaly']:.2f}%

Provide data-driven recommendations to reduce hospital billing while maintaining care quality.
"""

with st.spinner("Generating recommendations..."):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        ai_suggestions = response.choices[0].message.content.strip()
        st.info(ai_suggestions)
    except Exception as e:
        st.error(f"❌ Failed to generate recommendations: {e}")

# ----------------------
# 📥 Download Options
# ----------------------
st.subheader("📥 Download Optimization Data")
st.download_button("Download Comparison Table", data=comparison_df.to_csv(index=False), file_name="strategy_comparison.csv", mime="text/csv")
st.download_button("Download Sample Patient Data", data=model_sample.to_csv(index=False), file_name="patient_sample_data.csv", mime="text/csv")

# ----------------------
# 🛠️ App Help & Guidance
# ----------------------
with st.expander("🛠️ App Help & Guidance"):
    st.markdown("""
    - **Top Strategy**: Selected based on minimum total cost.
    - **Constraints Sliders**: Simulate various admission policy thresholds.
    - **Billing Trends**: Helps visualize seasonal cost behavior.
    - **AI Recommendations**: GPT-4 generated strategy advice based on your dataset.
    - **Drilldowns**: Patient-level data for each strategy.
    - **Charts**: Strategy comparison, pie distribution, and metric correlation.
    """)    
