import streamlit as st

st.set_page_config(page_title="🏥 Strategy Optimization Overview", layout="wide")

st.markdown("""
# 🏥 Hospital Strategy Optimization Dashboard

An interactive **Streamlit** app that simulates and recommends hospital patient management strategies using **Linear/Integer Programming**, **Anomaly Detection**, and **AI-powered decision support**.

---

## 📌 Key Features

### 🎯 Strategic Optimization Using LP/IP
- Compare **Linear Programming (LP)**, **Integer Programming (IP)**, and **Greedy Baseline** strategies
- Constraints include:
  - Maximum % of weekend admissions
  - Maximum % of long stays
  - Maximum % of anomaly-prone patients
  - Minimum required patient count
- Objective: **Minimize total hospital billing cost** under the given constraints

### 🧠 Anomaly Detection
- Detects high-billing or unusual stay patterns using **Isolation Forest**
- Flags records with high anomaly probability
- Uses billing, length of stay, and encoded patient attributes

### 📊 Interactive Visualization & KPIs
- Strategy-wise cost and patient profile comparison
- Monthly billing trends (overall + per-strategy)
- Pie charts showing distribution of strategies
- Correlation heatmap of cost and stay patterns
- Dynamic KPIs for cost, long stays, weekend cases, and anomalies

### 💬 GPT-4 Recommendations *(Optional)*
- Analyzes optimal strategy and constraints
- Delivers **executive-level AI recommendations** for cost reduction and care improvement
- Requires OpenAI API key (`.streamlit/secrets.toml` or session input)

---

## 📁 Required Dataset Format

Ensure your uploaded `.csv` file includes:
- `Date of Admission`, `Discharge Date`, `Billing Amount`
- `Medical Condition`, `Insurance Provider`, `Gender`, `Medication`, `Age`

---

## ⚙️ Optimization Strategies

| Strategy    | Approach             | Optimized For                        |
|-------------|----------------------|--------------------------------------|
| **IP**      | Integer Programming  | Exact patient selection under constraints |
| **LP**      | Linear Programming   | Fast estimation with relaxed constraints |
| **Greedy**  | Cost-based sorting   | Quick and simple baseline comparison |

---
""", unsafe_allow_html=True)
