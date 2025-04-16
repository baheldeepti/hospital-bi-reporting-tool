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

