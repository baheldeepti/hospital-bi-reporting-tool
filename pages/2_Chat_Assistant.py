# 📘 Hospital Chat Assistant - v1.4.3 STREAMLIT DEPLOYMENT VERSION

import streamlit as st
import pandas as pd
import os
import altair as alt
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from datetime import datetime
import openai
from openai import OpenAI

# 🌐 ENV + CONFIG
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

st.set_page_config(page_title="🤖 Hospital Chat Assistant", layout="wide")
st.title("🏥 Hospital Chat Assistant")

DEBUG_MODE = st.sidebar.checkbox("🐞 Debug Mode")
def debug_log(msg):
    if DEBUG_MODE:
        st.sidebar.markdown(f"🔍 **Debug**: {msg}")


# 🔁 Session Init
for key in ["main_df", "chat_history", "query_log", "fallback_log", "usage_log"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "log" in key or "history" in key else None

FALLBACK_RESPONSE = """🤖 I’m not able to understand that question right now.

**Try asking something like:**
- *Total billing by hospital*
- *Average stay per condition*
- *Top conditions by test result*"""


# 📁 Load Data UI
def load_data_ui():
    with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
        st.markdown("Upload your CSV or use our sample dataset.")
        if st.button("📥 Load Sample Data"):
            df = pd.read_csv("https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv")
            st.session_state["main_df"] = df
            st.success("✅ Sample dataset loaded.")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            with st.spinner("Loading your file..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state["main_df"] = df
                    st.success("✅ Uploaded data loaded successfully.")
                    st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"Error loading file: {e}")


# 🔍 Filters
def apply_filters(df):
    st.sidebar.markdown("### 🔎 Filter Data")
    hospitals = st.sidebar.multiselect("Hospital", df["Hospital"].dropna().unique())
    insurance = st.sidebar.multiselect("Insurance Provider", df["Insurance Provider"].dropna().unique())
    conditions = st.sidebar.multiselect("Medical Condition", df["Medical Condition"].dropna().unique())
    if hospitals: df = df[df["Hospital"].isin(hospitals)]
    if insurance: df = df[df["Insurance Provider"].isin(insurance)]
    if conditions: df = df[df["Medical Condition"].isin(conditions)]
    return df


# 📈 KPI Cards
def render_kpis(df):
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Total Billing", f"${df['Billing Amount'].sum():,.2f}")
    c2.metric("🛏️ Avg Stay", f"{df['Length of Stay'].mean():.1f} days")
    c3.metric("👥 Total Patients", f"{df['Name'].nunique()}")


# 📉 Billing Trend
def render_billing_trend(df):
    st.subheader("📉 Billing Trend Over Time")
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    trend = df.groupby("Date of Admission")["Billing Amount"].sum().reset_index()
    chart = alt.Chart(trend).mark_line(point=True).encode(
        x="Date of Admission:T", y="Billing Amount:Q"
    ).properties(title="Total Billing Over Time")
    st.altair_chart(chart, use_container_width=True)


# 📊 Modular Chart Functions
def render_bar_chart_with_labels(data, x_col, y_col, title):
    base = alt.Chart(data).mark_bar().encode(x=alt.X(x_col, sort="-y"), y=y_col, tooltip=[x_col, y_col])
    labels = alt.Chart(data).mark_text(dy=-5).encode(x=x_col, y=y_col, text=y_col)
    st.altair_chart(base + labels, use_container_width=True)

def render_line_chart_with_labels(data, x_col, y_col, title):
    chart = alt.Chart(data).mark_line(point=True).encode(x=x_col, y=y_col, tooltip=[x_col, y_col]).properties(title=title)
    st.altair_chart(chart, use_container_width=True)

def render_pie_chart_with_tooltips(data, category_col, value_col):
    chart = alt.Chart(data).mark_arc(innerRadius=50).encode(theta=value_col, color=category_col, tooltip=[category_col, value_col])
    st.altair_chart(chart, use_container_width=True)


# 🔁 Chart Mapping
keyword_chart_map = {
    "total billing by hospital": {"chart_type": "bar", "dimension": "Hospital", "metric": "Billing Amount", "agg": "sum"},
    "total billing by insurance provider": {"chart_type": "bar", "dimension": "Insurance Provider", "metric": "Billing Amount", "agg": "sum"},
    "average stay over time": {"chart_type": "line", "dimension": "Date of Admission", "metric": "Length of Stay", "agg": "mean"},
    "patient count by gender": {"chart_type": "bar", "dimension": "Gender", "agg": "count"}
}

def match_chart_mapping(query):
    query = query.lower()
    for key in keyword_chart_map:
        if key in query:
            return keyword_chart_map[key]
    return None

def respond_to_query(query, df):
    chart_config = match_chart_mapping(query)
    if chart_config:
        dim, metric, agg = chart_config.get("dimension"), chart_config.get("metric"), chart_config.get("agg")
        try:
            if chart_config["chart_type"] == "bar":
                if agg == "count":
                    data = df[dim].value_counts().reset_index()
                    data.columns = [dim, "Count"]
                else:
                    data = df.groupby(dim)[metric].agg(agg).reset_index()
                render_bar_chart_with_labels(data, dim, data.columns[1], query)
                return f"📊 Chart rendered for: {query}"
            elif chart_config["chart_type"] == "line":
                data = df.groupby(dim)[metric].agg(agg).reset_index()
                render_line_chart_with_labels(data, dim, metric, query)
                return f"📈 Line chart shown for: {query}"
        except Exception as e:
            return f"⚠️ Could not generate chart: {e}"
    try:
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0), df=df, verbose=False)
        return agent.run(query)
    except Exception:
        st.session_state["fallback_log"].append(query)
        return FALLBACK_RESPONSE

# 💬 Chat Assistant UI
def render_chat(df):
    st.subheader("💬 Chat Assistant")
    suggestions = [
        "Total billing by hospital",
        "Patient count by gender",
        "Total billing by insurance provider",
        "Average stay over time"
    ]
    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        if cols[i].button(s):
            reply = respond_to_query(s, df)
            st.session_state["chat_history"].append((s, reply))
            st.session_state["query_log"].append(s)
    for i, (q, a) in enumerate(st.session_state["chat_history"]):
        message(q, is_user=True, key=f"u_{i}")
        message(a, key=f"a_{i}")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask anything about the data")
        if st.form_submit_button("Send") and user_input:
            reply = respond_to_query(user_input, df)
            st.session_state["chat_history"].append((user_input, reply))
            st.session_state["query_log"].append(user_input)


# 📖 Narrative Insights
def render_narrative_summary(df):
    st.subheader("📖 Narrative Insights")
    if st.button("Generate Summary"):
        with st.spinner("Analyzing data..."):
            try:
                from langchain.prompts import PromptTemplate
                prompt = PromptTemplate.from_template("Give 3 insights from:\n{summary}")
                summary_text = df.describe(include='all').to_string()
                summary = OpenAI(temperature=0)(prompt.format(summary=summary_text))
                st.markdown(summary)
            except Exception as e:
                st.error(f"Error: {e}")


# 📊 Advanced Insights
def render_advanced_charts(df):
    st.subheader("📊 Explore More Insights")
    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Pie"])
    column = st.selectbox("Analyze by", ["Gender", "Hospital", "Insurance Provider", "Date of Admission"])
    if chart_type == "Bar":
        data = df[column].value_counts().reset_index()
        data.columns = [column, "Count"]
        render_bar_chart_with_labels(data, column, "Count", "Bar Chart")
    elif chart_type == "Line":
        if column == "Date of Admission":
            df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
            data = df.groupby("Date of Admission")["Billing Amount"].mean().reset_index()
            render_line_chart_with_labels(data, "Date of Admission", "Billing Amount", "Line Chart")
    elif chart_type == "Pie":
        data = df[column].value_counts().reset_index()
        data.columns = [column, "Count"]
        render_pie_chart_with_tooltips(data, column, "Count")


# 🏆 Leaderboard & Logs
def render_logs():
    st.subheader("🏆 Leaderboard & Logs")
    if st.session_state["query_log"]:
        log_df = pd.DataFrame(st.session_state["query_log"], columns=["Query"])
        log_df = log_df["Query"].value_counts().reset_index()
        log_df.columns = ["Query", "Count"]
        st.dataframe(log_df)
    if st.session_state["fallback_log"]:
        st.markdown("### Fallback Queries")
        st.dataframe(pd.DataFrame(st.session_state["fallback_log"], columns=["Query"]))


# 📘 Glossary
def render_glossary():
    with st.sidebar.expander("🔍 Glossary"):
        st.text_input("Search glossary", key="glossary_search")
        glossary = {
            "Hospital": "Name of facility",
            "Billing Amount": "Total billed amount",
            "Length of Stay": "Avg. duration of patient stay",
            "Medical Condition": "Primary diagnosis",
            "Insurance Provider": "Insurance company"
        }
        search = st.session_state.get("glossary_search", "").lower()
        matches = [f"- **{k}**: {v}" for k, v in glossary.items() if search in k.lower()]
        st.markdown("\n".join(matches) if matches else "❌ No match found.")




# 🚀 MAIN FLOW
load_data_ui()
if st.session_state["main_df"] is not None:
    df = st.session_state["main_df"]
    df["Billing Amount"] = pd.to_numeric(df["Billing Amount"].replace('[\$,]', '', regex=True), errors="coerce")
    df["Length of Stay"] = pd.to_numeric(df.get("Length of Stay", pd.Series(dtype=float)), errors="coerce")
    filtered_df = apply_filters(df)
    render_kpis(filtered_df)
    render_billing_trend(filtered_df)
    render_chat(filtered_df)
    render_narrative_summary(filtered_df)
    render_advanced_charts(filtered_df)
    render_logs()
    render_glossary()
    #render_footer()
