import streamlit as st

# ✅ Page Setup
st.set_page_config(
    page_title="Hospital BI Reporting Tool",
    layout="wide",
    page_icon="🏥"
)

# ✅ App Title
st.title("🏥 Hospital BI Reporting Tool")

# ℹ️ Welcome Message
st.markdown("""
Welcome to the **Hospital Business Intelligence Reporting Tool**!

This tool allows users to analyze hospital data using AI, explore visual dashboards, and gain operational insights with ease.
""")

# 📘 How It Works Section
with st.expander("ℹ️ How It Works", expanded=False):
    st.markdown("""
1. 📁 Navigate to the **Dashboard** to view key performance metrics.
2. 🤖 Use the **Chat Assistant** to ask natural language questions about the data.
3. 📊 Explore **Feature Overview** sections to understand the tool's capabilities.
4. 🧠 Download chat history, query logs, and charts as needed.

> You can either upload your own hospital dataset or load the sample dataset from within those pages.
""")

# 🔗 Navigation Links
import streamlit as st

# ✅ Page Setup
st.set_page_config(
    page_title="Hospital BI Reporting Tool",
    layout="wide",
    page_icon="🏥"
)

# ✅ App Title
st.title("🏥 Hospital BI Reporting Tool")

# ℹ️ Welcome Message
st.markdown("""
Welcome to the **Hospital Business Intelligence Reporting Tool**!

This tool allows users to analyze hospital data using AI, explore visual dashboards, and gain operational insights with ease.
""")

# 📘 How It Works Section
with st.expander("ℹ️ How It Works", expanded=False):
    st.markdown("""
1. 📁 Navigate to the **Dashboard** to view key performance metrics.
2. 🤖 Use the **Chat Assistant** to ask natural language questions about the data.
3. 📊 Explore **Feature Overview** sections to understand the tool's capabilities.
4. 🧠 Download chat history, query logs, and charts as needed.

> You can either upload your own hospital dataset or load the sample dataset from within those pages.
""")

# 🔗 Navigation Links (Using Markdown for HTML-based links)
st.markdown("### 🔗 Navigate to:")

st.markdown("""
- [📊 Dashboard](pages/1_Dashboard.py)
- [🗨️ Chat Assistant](pages/2_Chat_Assistant.py)
- [🤖 Chat Assistant Features](pages/3_Chat_Assistant_Feature_Overview.py)
- [🧭 Dashboard Features](pages/4_Dashboard_Feature_Overview.py)
- [📈 Time Series Forecasting](pages/5_Time_Series_Forecasting.py)
- [🧠 Advanced Anomaly Detection](pages/6_Advanced_Anomaly_Detection.py)
""")

# 👩‍💻 About the Developer
st.markdown("### 👩‍💻 About the Developer")
st.markdown("""
Built by **Deepti Bahel**, this app combines data engineering, AI, and intuitive dashboards to help hospitals turn raw data into actionable insights.

[Connect on LinkedIn](https://www.linkedin.com/in/deepti-bahel/)
""")
st.markdown("""
<style>
footer {
    visibility: hidden;
}
footer:after {
    content:'Powered by Streamlit, OpenAI & LangChain';
    visibility: visible;
    display: block;
    text-align: center;
    padding: 10px;
    color: gray;
    font-size: 0.85em;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
---
<div style='text-align: center; color: grey; font-size: 0.9em;'>
  🔍 Visit the app anytime at <a href='https://hospital-bi-tool.streamlit.app/' target='_blank'>hospital-bi-tool.streamlit.app</a><br>
  Powered by <strong>OpenAI</strong> and <strong>LangChain</strong>
</div>
""", unsafe_allow_html=True)

