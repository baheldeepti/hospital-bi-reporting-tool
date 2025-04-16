import streamlit as st

st.set_page_config(page_title="Chat Assistant Feature Overview", layout="wide")
st.title("📄 Hospital Chat Assistant - Feature Overview")

with open("docs/feature_overview.md", "r") as f:
    md_content = f.read()

st.markdown(md_content, unsafe_allow_html=True)


