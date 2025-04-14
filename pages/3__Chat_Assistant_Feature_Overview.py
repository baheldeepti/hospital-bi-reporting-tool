import streamlit as st

st.set_page_config(page_title="Chat Assistant Feature Overview", layout="wide")
st.title("📄 Hospital Chat Assistant - Feature Overview")

with open("docs/feature_overview.md", "r") as f:
    md_content = f.read()

st.markdown(md_content, unsafe_allow_html=True)

st.page_link("pages/1_📊_Dashboard.py", label="📊Dashboard", icon="📊")
st.page_link("pages/2_🤖_Chat_Assistant.py", label="🤖Chat Assistant")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="Dashboard Feature Overview")
# 🔗 Page Navigation
st.page_link("pages/2_🤖_Chat_Assistant.py", label="🤖Chat Assistant")
st.page_link("pages/1_📊_Dashboard.py", label="📊 Dashboard")
st.page_link("pages/4_Dashboard_Feature_Overview.py", label="📘 Dashboard Feature Overview")
# st.page_link("pages/3__Chat_Assistant_Feature_Overview.py", label="📄 Chat Assistant Feature Overview")
