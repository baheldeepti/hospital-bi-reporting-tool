import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import traceback
import re

# 🔐 Set API key securely
openai.api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")

# 🧠 Initialize session state
if "main_df" not in st.session_state:
    st.session_state["main_df"] = None
if "history" not in st.session_state:
    st.session_state.history = []
if "query_log" not in st.session_state:
    st.session_state["query_log"] = []
if "fallback_log" not in st.session_state:
    st.session_state["fallback_log"] = []

# 📁 File upload UI
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

load_data_ui()

# 📊 Chart display
import altair as alt

# 📊 Improved visualization with tooltips and labels
def try_visualize(result):
    try:
        if isinstance(result, pd.Series):
            df_plot = result.reset_index()
            df_plot.columns = ["Category", "Value"]

            chart = (
                alt.Chart(df_plot)
                .mark_bar()
                .encode(
                    x=alt.X("Category:N", title=None, sort="-y"),
                    y=alt.Y("Value:Q", title="Value"),
                    tooltip=["Category", "Value"]
                )
                .properties(width=600, height=400)
            )

            text = (
                alt.Chart(df_plot)
                .mark_text(align='center', dy=-5, fontSize=12)
                .encode(x="Category:N", y="Value:Q", text="Value")
            )

            st.altair_chart(chart + text, use_container_width=True)

        elif isinstance(result, pd.DataFrame) and result.shape[1] >= 2:
            df_plot = result.reset_index()
            df_plot.columns = ["Category", "Value"]

            chart = (
                alt.Chart(df_plot)
                .mark_bar()
                .encode(
                    x=alt.X("Category:N", title=None, sort="-y"),
                    y=alt.Y("Value:Q", title="Value"),
                    tooltip=["Category", "Value"]
                )
                .properties(width=600, height=400)
            )

            text = (
                alt.Chart(df_plot)
                .mark_text(align='center', dy=-5, fontSize=12)
                .encode(x="Category:N", y="Value:Q", text="Value")
            )

            st.altair_chart(chart + text, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not render interactive chart: {e}")


# 📝 Summary formatter
def format_summary(summary_text: str) -> str:
    summary = re.sub(r"\s+", " ", summary_text).strip()
    summary = re.sub(r"(\d),\s(\d)", r"\1\2", summary)
    hospitals = re.split(r",\s*|\band\b", summary)
    formatted_lines = []
    for hospital in hospitals:
        match = re.search(r"([A-Za-z0-9() \-]+?)\s+billed\s+approximately\s+\$?([\d,]+)", hospital)
        if match:
            name = match.group(1).strip()
            amount = match.group(2).replace(",", "")
            formatted_lines.append(f"- **{name}**: ${int(amount):,}")
    if not formatted_lines:
        return f"📝 **Summary:** {summary}"
    return "📝 **Summary**  \n" + "\n".join(formatted_lines)

# 🧠 GPT-based summary
def get_summary(question, result_str):
    summary_prompt = f"""You are a helpful assistant.
The user asked: {question}
The result of the query was: {result_str}
Summarize the insight clearly."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        raw_summary = response.choices[0].message.content.strip()
        return format_summary(raw_summary)
    except:
        return ""

# 🧠 Main chat handler
def handle_chat(question):
    df = st.session_state["main_df"]
    if df is None:
        st.warning("Please upload a dataset or load the sample.")
        return

    columns = ", ".join(df.columns)

    st.chat_message("user").write(question)
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.query_log.append(question)

    prompt = f"""You are a senior data analyst working with this DataFrame: df
Available columns: {columns}
Conversation so far:
{st.session_state.history}

Write executable pandas code to answer the **last user question only**.
- Assign output to a variable named `result`
- Use only valid column names from the DataFrame
- Do not include explanations or print statements
- Only output valid Python code"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        code = response.choices[0].message.content.strip()
        st.code(code, language="python")

        local_vars = {"df": df}
        exec(code, {}, local_vars)
        result = local_vars.get("result", "No result")

        with st.chat_message("assistant"):
            if isinstance(result, (pd.Series, pd.DataFrame)):
                st.dataframe(result)
                try_visualize(result)
            else:
                st.write(str(result))

            summary = get_summary(question, str(result))
            if summary:
                st.markdown(summary)
                st.session_state.history.append({"role": "assistant", "content": summary})

    except Exception as e:
        st.session_state.fallback_log.append(question)
        st.chat_message("assistant").error(f"Error:\n{traceback.format_exc()}")

# 🪵 Logs only
def render_logs():
    st.subheader("🪵 Conversation Logs")
    query_log = st.session_state.get("query_log", [])
    if query_log:
        st.markdown("### 🔁 Most Asked Questions")
        log_df = pd.DataFrame(query_log, columns=["Query"])
        value_counts = log_df["Query"].value_counts().reset_index()
        value_counts.columns = ["Query", "Count"]
        st.dataframe(value_counts)
    else:
        st.info("No queries logged yet.")
    fallback_log = st.session_state.get("fallback_log", [])
    if fallback_log:
        st.markdown("### ⚠️ Fallback Queries (Unanswered)")
        fallback_df = pd.DataFrame(fallback_log, columns=["Query"])
        st.dataframe(fallback_df)

# 🧪 App UI
st.title("🏥 Hospital Chat Assistant")
st.markdown("Ask questions about hospital data. Get real answers with charts and code-backed insights!")

if prompt := st.chat_input("Ask a question about the hospital dataset..."):
    handle_chat(prompt)

# 🪵 Logs display
st.markdown("---")
render_logs()
