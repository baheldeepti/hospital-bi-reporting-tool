import streamlit as st

# ========================
# 📌 PAGE CONFIGURATION
# ========================
st.set_page_config(
    page_title="Hospital BI Reporting Tool",
    page_icon="🏥",
    layout="wide"
)

# ========================
# 🏥 APP TITLE & WELCOME
# ========================
st.title("🏥 Hospital BI Reporting Tool")

st.markdown("""
Welcome to the **Hospital Business Intelligence (BI) Reporting Tool**!

This interactive application helps hospital administrators, analysts, and decision-makers explore operational insights using dashboards and AI-powered analytics.
""")

# ========================
# ℹ️ HOW IT WORKS SECTION
# ========================
with st.expander("ℹ️ How It Works", expanded=False):
    st.markdown("""
1. **📊 Dashboard** – Analyze hospital KPIs, patient trends, and billing metrics.
2. **🤖 Chat Assistant** – Ask natural language questions and get data-driven responses.
3. **📘 Feature Overview** – Learn about the tools available in the app.
4. **📁 Data Upload** – Use the sample dataset or upload your own.

---

💡 Tip: Export your charts, insights, and logs for external reporting!
""")



# ========================
# 👩‍💻 ABOUT THE DEVELOPER
# ========================
st.markdown("---")
st.markdown("### 👩‍💻 About the Developer")
st.markdown("""
Built by **Deepti Bahel**, a Senior Business Intelligence Engineer blending data, AI, and visualization to drive actionable insights in healthcare.

🔗 [Connect on LinkedIn](https://www.linkedin.com/in/deepti-bahel/)
""")

# ========================
# 🦶 FOOTER
# ========================
st.markdown("---")
st.markdown("#### 🔻 Quick Access & Credits")

st.markdown("Made with ❤️ by Deepti Bahel | Powered by Streamlit + LangChain + Altair")

st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard", icon="📊")
st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Features", icon="📄")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Features", icon="📘")

st.caption("© 2025 Hospital BI Tool. All rights reserved.")
