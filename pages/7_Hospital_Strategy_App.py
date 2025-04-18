import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linprog
import openai
# ----------------------
# 📘 Title & Introduction
# ----------------------
st.set_page_config(page_title="Hospital Strategy Recommender", layout="wide")
st.title("🏥 Hospital Strategy Optimization Dashboard")

with st.expander("❓ Frequently Asked Questions"):
    st.markdown("""
    **Q1: What are the three strategies used?**  
    LP = Linear Programming based on cost optimization with constraints.  
    Greedy = Selects the lowest-cost patients.  
    Heuristic = Avoids weekend & anomaly cases.

    **Q2: How is anomaly detected?**  
    Using Isolation Forest on Billing Amount to flag outliers.

    **Q3: What does 'Constraints Met' mean?**  
    It's the count of how many of your constraints were satisfied by a strategy (out of 4).

    **Q4: Can I download the filtered data?**  
    Yes! You’ll see download buttons below the patient table and chart summary.
    """)

st.markdown("""
### Welcome to the Hospital Strategy Optimization Tool
This dashboard helps hospital administrators evaluate cost-saving strategies while preserving care quality.

---
#### 🔍 What You Can Do:
- Set business constraints
- Compare strategies (LP, Greedy, Heuristic)
- Drill down into patients
- Visualize billing trends
- Download data summaries

---
#### 🛠 How to Use:
1. Use the sidebar to set constraints
2. Review strategy recommendations
3. Explore patient-level data
4. View trends and spikes
5. Read business suggestions
""")

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
# ⚙️ Optimization Simulation Sliders
# ----------------------
with st.expander("⚙️ Simulate Optimization Thresholds"):
    sim_weekend = st.slider("Max % Weekend Admissions", 0.0, 1.0, 0.2, step=0.05)
    sim_longstay = st.slider("Max % Long Stays", 0.0, 1.0, 0.1, step=0.05)
    sim_anomaly = st.slider("Max % Anomaly", 0.0, 1.0, 0.05, step=0.01)
    sim_min_patients = st.slider("Min Patients Selected", 0, 100, 70, step=5)

# ----------------------
# 🧠 Optimization Model Preview
# ----------------------
st.subheader("🧠 Optimization Model: Cost Minimization")
model_sample = df[['Billing Amount', 'Is Weekend', 'Is Long Stay', 'anomaly_prob']].dropna().sample(n=100, random_state=42)
c = model_sample['Billing Amount'].values
A_ub = [
    model_sample['Is Weekend'].values,
    model_sample['Is Long Stay'].values,
    model_sample['anomaly_prob'].values
]
b_ub = [
    sim_weekend * len(model_sample),
    sim_longstay * len(model_sample),
    sim_anomaly * len(model_sample)
]
A_eq = [np.ones(len(model_sample))]
b_eq = [sim_min_patients]
x_bounds = [(0, 1) for _ in range(len(model_sample))]

try:
    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')
    if result.success:
        model_sample['Selected'] = result.x.round()
        model_sample['Index'] = range(len(model_sample))

        fig_model, ax_model = plt.subplots(figsize=(10, 4))
        ax_model.bar(model_sample['Index'], model_sample['Billing Amount'], alpha=0.3, label='Original Billing')
        ax_model.bar(model_sample['Index'], model_sample['Billing Amount'] * model_sample['Selected'], alpha=0.8, label='Optimized Billing')
        ax_model.set_title("📊 Optimization Result: Before vs After Billing")
        ax_model.set_xlabel("Patient Index")
        ax_model.set_ylabel("Billing Amount")
        ax_model.legend()
        ax_model.grid(True)
        st.pyplot(fig_model)

        before_total = model_sample['Billing Amount'].sum()
        after_total = (model_sample['Billing Amount'] * model_sample['Selected']).sum()
        cost_saved = before_total - after_total
        st.markdown(f"**Original Total Billing:** ${before_total:,.2f}")
        st.markdown(f"**Optimized Total Billing:** ${after_total:,.2f}")
        st.markdown(f"**Cost Reduction:** ${cost_saved:,.2f}")
    else:
        failed_constraints = []
        if sim_weekend * len(model_sample) < sum(model_sample['Is Weekend']):
            failed_constraints.append("Weekend %")
        if sim_longstay * len(model_sample) < sum(model_sample['Is Long Stay']):
            failed_constraints.append("Long Stay %")
        if sim_anomaly * len(model_sample) < sum(model_sample['anomaly_prob']):
            failed_constraints.append("Anomaly %")
        if sim_min_patients > len(model_sample):
            failed_constraints.append("Minimum Patient Count")

        msg = "🚫 Optimization model failed to find a feasible solution. Try loosening constraints."
        if failed_constraints:
            msg += "\n\n❗ Constraints potentially too strict: " + ", ".join(failed_constraints)
        st.error(msg)
except Exception as e:
    st.error(f"⚠️ Optimization Error: {e}")
# ----------------------
# 🤖 Strategy Simulation
# ----------------------
def apply_strategies(df):
    opt_df = df[['Billing Amount', 'Is Weekend', 'Is Long Stay', 'anomaly_prob', 'Date of Admission', 'Gender', 'Insurance Provider']].dropna().sample(n=100, random_state=42)
    opt_df['LP_Selected'] = 0
    opt_df.loc[opt_df.sort_values(by='Billing Amount').head(sim_min_patients).index, 'LP_Selected'] = 1
    opt_df['Greedy_Selected'] = 0
    greedy_sorted = opt_df.sort_values(by='Billing Amount').head(sim_min_patients).index
    opt_df.loc[greedy_sorted, 'Greedy_Selected'] = 1
    opt_df['Heuristic_Selected'] = 0
    heuristic_filtered = opt_df[(opt_df['anomaly_prob'] == 0) & (opt_df['Is Weekend'] == 0)]
    heuristic_selected = heuristic_filtered.sort_values(by='Billing Amount').head(sim_min_patients).index
    opt_df.loc[heuristic_selected, 'Heuristic_Selected'] = 1
    return opt_df

opt_df = apply_strategies(df)

# ----------------------
# 📋 Strategy Recommendation Engine
# ----------------------
def recommend_strategy(df, budget=800000, max_anomaly=5, max_weekend=20, max_longstay=10):
    strategies = ['LP_Selected', 'Greedy_Selected', 'Heuristic_Selected']
    recommendation = []
    for strat in strategies:
        sel = df[df[strat] == 1]
        total_cost = sel['Billing Amount'].sum()
        anomaly_pct = sel['anomaly_prob'].mean() * 100
        weekend_pct = sel['Is Weekend'].mean() * 100
        longstay_pct = sel['Is Long Stay'].mean() * 100
        score = sum([
            total_cost <= budget,
            anomaly_pct <= max_anomaly,
            weekend_pct <= max_weekend,
            longstay_pct <= max_longstay
        ])
        recommendation.append({
            "Strategy": strat.replace("_Selected", ""),
            "Total Cost": f"${total_cost:,.2f}",
            "% Anomalies": f"{anomaly_pct:.2f}%",
            "% Weekend": f"{weekend_pct:.2f}%",
            "% Long Stay": f"{longstay_pct:.2f}%",
            "Constraints Met": score
        })
    return pd.DataFrame(recommendation).sort_values(by="Constraints Met", ascending=False)

# Sidebar Constraints
st.sidebar.header("Set Recommendation Constraints")
budget = st.sidebar.slider("💵 Budget Limit ($)", 500000, 2000000, 800000, step=50000)
max_anomaly = st.sidebar.slider("% Anomalies Allowed", 0, 20, 5)
max_weekend = st.sidebar.slider("% Weekend Admissions Allowed", 0, 50, 20)
max_longstay = st.sidebar.slider("% Long Stays Allowed", 0, 30, 10)

recommend_df = recommend_strategy(opt_df, budget, max_anomaly, max_weekend, max_longstay)
#top strategy
top = recommend_df.iloc[0]
# 📊 KPI Summary
st.subheader("📊 Strategy KPIs")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Top Strategy", top["Strategy"])
kpi2.metric("Total Cost", top["Total Cost"])
kpi3.metric("% Anomalies", top["% Anomalies"])
kpi4.metric("Constraints Met", int(top["Constraints Met"]))
# 💡 Business Recommendations
st.subheader("💡 Strategic Recommendations for Leadership")
st.markdown(f"""
**Top Strategy Chosen:** `{top['Strategy']}`

**Key Recommendations:**
- Implement `{top['Strategy']}` to potentially reduce hospital billing costs.
- Review high anomaly or weekend cases to reduce unnecessary expenses.
- Monitor patient length of stay and improve discharge planning.
- Consider enhancing anomaly detection systems to prevent financial leakage.
- Periodically re-evaluate strategies with updated patient and billing data.
""")

st.subheader("📋 Strategy Recommendation Summary")
st.dataframe(recommend_df, use_container_width=True)


# ----------------------
# 🤖 AI-Powered Suggestions (Dynamic with OpenAI)
# ----------------------


st.subheader("🤖 AI Recommendations from ChatGPT")

openai.api_key = st.secrets.get("openai_api_key")  # Secure key storage in .streamlit/secrets.toml

prompt = f"""
You are an AI healthcare strategy advisor. Based on the following strategy data:

- Strategy Chosen: {top['Strategy']}
- Total Cost: {top['Total Cost']}
- % Anomalies: {top['% Anomalies']}
- % Weekend: {top['% Weekend']}
- % Long Stay: {top['% Long Stay']}

Suggest specific, actionable recommendations for hospital leadership to reduce costs while improving patient care. Mention strategies for discharge planning, anomaly management, and weekend admission control.
"""

with st.spinner("Generating AI-powered recommendations..."):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        ai_recommendation = response.choices[0].message.content.strip()
        st.info(ai_recommendation)
    except Exception as e:
        st.error(f"❌ Failed to fetch AI recommendations: {e}")


# ----------------------
# 🎯 Strategy-Specific Patient Drilldown
# ----------------------
st.subheader("🎯 Strategy-Specific Patient Drilldown")
strategy_choice = st.selectbox("Select Strategy for Patient View", ["LP", "Greedy", "Heuristic"])
selected_col = strategy_choice + "_Selected"
patient_view = opt_df[opt_df[selected_col] == 1].copy()

with st.expander("🔍 Filter Patients"):
    genders = st.multiselect("Select Gender", options=sorted(patient_view['Gender'].unique()), default=sorted(patient_view['Gender'].unique()))
    insurers = st.multiselect("Select Insurance Provider", options=sorted(patient_view['Insurance Provider'].unique()), default=sorted(patient_view['Insurance Provider'].unique()))
    patient_view = patient_view[(patient_view['Gender'].isin(genders)) & (patient_view['Insurance Provider'].isin(insurers))]

st.dataframe(patient_view[['Billing Amount', 'Is Weekend', 'Is Long Stay', 'anomaly_prob', 'Date of Admission', 'Gender', 'Insurance Provider']], use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Reset Filters"):
        st.rerun()
with col2:
    st.download_button("📥 Download Patient Data as CSV", data=patient_view.to_csv(index=False), file_name="filtered_patient_data.csv", mime="text/csv")

# ----------------------
# 📅 Billing Trend Over Time with Anomaly Alerts
# ----------------------
st.subheader("📅 Billing Trend Over Time with Anomaly Alerts")
group_by = st.radio("Group Trend By", options=["None", "Gender", "Insurance Provider"])

fig, ax = plt.subplots(figsize=(10, 5))

if group_by == "None":
    billing_over_time = patient_view.groupby(patient_view['Date of Admission'].dt.to_period("M")).agg({'Billing Amount': 'sum', 'anomaly_prob': 'mean'}).reset_index()
    billing_over_time['Date'] = billing_over_time['Date of Admission'].dt.to_timestamp()
    ax.plot(billing_over_time['Date'], billing_over_time['Billing Amount'], marker='o', label='Total Billing')
    anomaly_spikes = billing_over_time[billing_over_time['anomaly_prob'] > 0.05]
    for _, row in anomaly_spikes.iterrows():
        ax.axvline(row['Date'], color='red', linestyle='--', alpha=0.5)
        ax.text(row['Date'], row['Billing Amount'], 'Anomaly Spike', rotation=90, color='red', fontsize=8)
else:
    grouped = patient_view.groupby([patient_view['Date of Admission'].dt.to_period("M"), group_by])['Billing Amount'].sum().unstack().fillna(0)
    grouped.index = grouped.index.to_timestamp()
    for col in grouped.columns:
        ax.plot(grouped.index, grouped[col], marker='o', label=col)

ax.set_title(f"Monthly Billing Trend - {strategy_choice} Strategy")
ax.set_xlabel("Month")
ax.set_ylabel("Total Billing Amount")
ax.grid(True)
ax.legend()
st.pyplot(fig)

csv_data = grouped.reset_index() if group_by != "None" else billing_over_time[['Date', 'Billing Amount']]
st.download_button("📥 Download Billing Trend Data", data=csv_data.to_csv(index=False), file_name="billing_trend_data.csv", mime="text/csv")



kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Top Strategy", top["Strategy"])
kpi2.metric("Total Cost", top["Total Cost"])
kpi3.metric("% Anomalies", top["% Anomalies"])
kpi4.metric("Constraints Met", int(top["Constraints Met"]))
