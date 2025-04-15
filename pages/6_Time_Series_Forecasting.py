import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import logging
import holidays

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Lazy import setup for optional libraries
prophet_available = False
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    pass

openai_available = False
try:
    import openai
    from openai import ChatCompletion
    openai_available = True
except ImportError:
    pass

st.set_page_config(page_title="📊 Hospital BI Forecast App", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data
def load_sample_data():
    url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
    return pd.read_csv(url)

@st.cache_data(show_spinner="Training ARIMA model...")
def run_arima(ts):
    model = ARIMA(ts['Patient Count'], order=(2, 1, 2))
    return model.fit()

@st.cache_data(show_spinner="Training Prophet model...")
def run_prophet(prophet_df, horizon, granularity):
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=horizon, freq=granularity)
    return m.predict(future), m

def main():
    st.title("🏥 Hospital Forecast and Model Evaluation Dashboard")

    with st.sidebar:
        st.header("📘 Section Overview")
        st.markdown("""
        - Upload a CSV or use sample data
        - Filter by hospital, insurance, and condition
        - Visualize trends
        - Forecast using ARIMA and Prophet
        - Download forecasts and view AI-powered summaries
        """)

    st.subheader("📁 Upload Your Dataset or Use Sample")
    file = st.file_uploader("Upload CSV", type=["csv"])
    required_cols = ['Date of Admission']

    if file:
        df = pd.read_csv(file)
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns: {required_cols}")
            return
        st.success("File uploaded successfully!")
    else:
        try:
            df = load_sample_data()
            st.warning("Using default sample dataset.")
        except Exception:
            st.error("Failed to load sample dataset.")
            return

    if df.shape[0] < 10:
        st.warning("Few rows in dataset. Forecasting may be unreliable.")

    col1, col2, col3 = st.columns(3)
    if 'Hospital' in df.columns:
        with col1:
            hospital_filter = st.selectbox("Hospital", ['All'] + sorted(df['Hospital'].dropna().unique()))
            if hospital_filter != 'All':
                df = df[df['Hospital'] == hospital_filter]
    if 'Insurance Provider' in df.columns:
        with col2:
            insurance_filter = st.selectbox("Insurance Provider", ['All'] + sorted(df['Insurance Provider'].dropna().unique()))
            if insurance_filter != 'All':
                df = df[df['Insurance Provider'] == insurance_filter]
    if 'Medical Condition' in df.columns:
        with col3:
            condition_filter = st.selectbox("Medical Condition", ['All'] + sorted(df['Medical Condition'].dropna().unique()))
            if condition_filter != 'All':
                df = df[df['Medical Condition'] == condition_filter]

    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df = df.dropna(subset=['Date of Admission'])
    df = df[df['Date of Admission'] >= pd.to_datetime("2020-01-01")]

    st.subheader("📈 Weekly Admissions Trend")
    weekly_df = df.groupby(pd.Grouper(key="Date of Admission", freq="W")).size().reset_index(name="Patients")
    fig = px.line(weekly_df, x="Date of Admission", y="Patients", markers=True)
    fig.update_layout(xaxis_title="Week", yaxis_title="Number of Patients")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗕 Forecast Patient Volume")
    granularity = st.selectbox("Time Granularity", ["D", "W", "M"], format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x])
    forecast_horizon = st.slider("Forecast Horizon", 7, 60, 14, step=7)
    horizon_label = {"D": "days", "W": "weeks", "M": "months"}[granularity]

    ts = df.groupby("Date of Admission").size().rename("Patient Count").to_frame()
    ts = ts.resample(granularity).sum()
    ts['Spike'] = ((ts - ts.mean()) / ts.std())['Patient Count'].abs() > 2

    st.line_chart(ts['Patient Count'])
    st.markdown(f"**Spikes/Dips Detected:** {ts['Spike'].sum()} {horizon_label}")

    try:
        arima_fit = run_arima(ts)
        arima_forecast = arima_fit.forecast(steps=forecast_horizon)
        st.line_chart(arima_forecast.rename("ARIMA Forecast"))
    except Exception as e:
        st.error(f"ARIMA failed: {e}")
        arima_forecast = None

    prophet_forecast_df = None
    if prophet_available:
        try:
            prophet_df = ts[['Patient Count']].reset_index().rename(columns={"Date of Admission": "ds", "Patient Count": "y"})
            prophet_forecast_df, m = run_prophet(prophet_df, forecast_horizon, granularity)
            fig2 = m.plot(prophet_forecast_df)
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Prophet failed: {e}")

    if arima_forecast is not None and prophet_forecast_df is not None:
        actuals = ts['Patient Count'].dropna()[-forecast_horizon:]
        prophet_preds = prophet_forecast_df.set_index('ds').reindex(actuals.index)['yhat']
        arima_preds = arima_forecast[:len(actuals)]

        min_len = min(len(actuals), len(arima_preds), len(prophet_preds))
        actuals = actuals[-min_len:]
        arima_preds = arima_preds[-min_len:]
        prophet_preds = prophet_preds[-min_len:]

        arima_rmse = np.sqrt(mean_squared_error(actuals, arima_preds))
        prophet_rmse = np.sqrt(mean_squared_error(actuals, prophet_preds))
        arima_mae = mean_absolute_error(actuals, arima_preds)
        prophet_mae = mean_absolute_error(actuals, prophet_preds)
        arima_mape = mean_absolute_percentage_error(actuals, arima_preds)
        prophet_mape = mean_absolute_percentage_error(actuals, prophet_preds)

        st.markdown("### 📊 Model Performance Comparison")
        st.write(f"ARIMA - RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}, MAPE: {arima_mape:.2%}")
        st.write(f"Prophet - RMSE: {prophet_rmse:.2f}, MAE: {prophet_mae:.2f}, MAPE: {prophet_mape:.2%}")

        combined_df = pd.DataFrame({
            'Actual': actuals,
            'ARIMA Forecast': arima_preds,
            'Prophet Forecast': prophet_preds
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Actual'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['ARIMA Forecast'], mode='lines+markers', name='ARIMA Forecast'))
        fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Prophet Forecast'], mode='lines+markers', name='Prophet Forecast'))

        spike_dates = ts[ts['Spike']].index.intersection(combined_df.index)
        fig.add_trace(go.Scatter(x=spike_dates, y=combined_df.loc[spike_dates, 'Actual'], mode='markers', marker=dict(color='red', size=10, symbol='x'), name='Spike/Dip'))

        holiday_dates = [d for d in combined_df.index if d in holidays.US()]
        fig.add_trace(go.Scatter(x=holiday_dates, y=combined_df.loc[holiday_dates, 'Actual'], mode='markers', marker=dict(color='orange', size=12, symbol='star'), name='Public Holiday'))

        fig.update_layout(title='Actual vs Forecasted Admissions with Annotations', xaxis_title='Date', yaxis_title='Patient Count', legend_title='Series', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        csv = combined_df.to_csv(index=True).encode()
        st.download_button("Download Forecast CSV", csv, file_name="forecast_results.csv", mime="text/csv")

        st.markdown("### 🧠 AI-Powered Narrative")


        if openai_available:
            api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
                
                import time
                try:
                    user_prompt = st.text_area("✍️ Customize AI Insight Prompt", value=f"""
                    Analyze patient trends and forecasts for {forecast_horizon} {horizon_label}.
                    Include model performance (ARIMA RMSE: {arima_rmse:.2f}, Prophet RMSE: {prophet_rmse:.2f}),
                    spikes detected: {ts['Spike'].sum()}, holidays: {len(holiday_dates)}.
                    Return 2-3 executive summary bullet points.
                    """.strip(), height=160)

                    if st.button("🔍 Generate Narrative Insights"):
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:

                                response = openai.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are a healthcare analyst writing business summaries."},
                                        {"role": "user", "content": user_prompt.strip()}
                                    ]
                                )
                                st.markdown(response.choices[0].message.content)
                                break
                            except Exception as e:
                                error_str = str(e).lower()
                                if 'rate limit' in error_str or 'timeout' in error_str:
                                    st.warning(f"Attempt {attempt + 1} failed due to rate limit or timeout. Retrying...")
                                    time.sleep(2 ** attempt)
                                elif 'invalid api key' in error_str:
                                    st.error("Invalid OpenAI API key. Please check your credentials.")
                                    break
                                elif 'quota' in error_str:
                                    st.error("You have exceeded your OpenAI API quota.")
                                    break
                                else:
                                    st.error("An unexpected error occurred while calling OpenAI API.")
                                    st.exception(e)
                                    break
                except Exception as outer_e:
                    st.error("An error occurred while preparing or submitting the GPT request.")
                    st.exception(outer_e)
               
                               


    

if __name__ == "__main__":
    main()
