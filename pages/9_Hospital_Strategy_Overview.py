import streamlit as st

st.set_page_config(page_title="Hospital Strategy App Feature Overview", layout="wide")


with open("docs/Hospital_Strategy_App_Overview.md", "r") as f:
    md_content = f.read()

st.markdown(md_content, unsafe_allow_html=True)

