# 📊 Streamlit Hospital Model Comparison App

# =====================================================================
# 📦 Imports and Dependencies
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import logging
import holidays
import plotly.graph_objects as go
import io

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# 🚀 Main Application Entry Point
# =====================================================================
def main():
    st.set_page_config(page_title="📊 Hospital BI Forecast App", layout="wide")
    st.title("🏥 Hospital Forecast and Model Evaluation Dashboard")

    with st.sidebar:
        st.header("📘 Section Overview")
        st.markdown("""
        This section allows users to:
        - Upload a hospital CSV dataset or use a default sample
        - Clean and prepare data for time series analysis
        - Filter by hospital, insurance, and condition
        - Explore weekly admission trends visually
        - Forecast patient count using ARIMA and Prophet
        - Compare model metrics and visualize anomalies
        - Download predictions and get AI-powered summaries
        - View trend and residual decomposition charts
        """)

    # ============================
    # 📂 Upload or Load Dataset
    # ============================
    st.subheader("📁 Upload Your Dataset or Use Sample")
    file = st.file_uploader("Upload CSV", type=["csv"])
    required_cols = ['Date of Admission']

    if file is not None:
        df = pd.read_csv(file)
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Uploaded file is missing required columns: {required_cols}")
            return
        st.success("✅ File uploaded successfully!")
    else:
        sample_url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
        try:
            df = pd.read_csv(sample_url)
            st.warning("⚠️ Using default sample dataset.")
        except Exception as e:
            st.error("❌ Failed to load sample dataset.")
            logging.exception("Failed to load sample dataset from GitHub.")
            return

    # ============================
    # 🩡 Filter by Hospital, Insurance, Condition
    # ============================
    if 'Hospital' in df.columns:
        hospital_filter = st.selectbox("Filter by Hospital", ['All'] + sorted(df['Hospital'].dropna().unique().tolist()))
        if hospital_filter != 'All':
            df = df[df['Hospital'] == hospital_filter]

    if 'Insurance Provider' in df.columns:
        insurance_filter = st.selectbox("Filter by Insurance Provider", ['All'] + sorted(df['Insurance Provider'].dropna().unique().tolist()))
        if insurance_filter != 'All':
            df = df[df['Insurance Provider'] == insurance_filter]

    if 'Medical Condition' in df.columns:
        condition_filter = st.selectbox("Filter by Medical Condition", ['All'] + sorted(df['Medical Condition'].dropna().unique().tolist()))
        if condition_filter != 'All':
            df = df[df['Medical Condition'] == condition_filter]

    # ============================
    # 🩡 Data Cleansing
    # ============================
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df = df.dropna(subset=['Date of Admission'])
    df = df[df['Date of Admission'] >= pd.to_datetime("2020-01-01")]

    # ============================
    # 📊 Weekly Trend Visualization
    # ============================
    st.subheader("📈 Weekly Admissions Trend")
    weekly_df = df.groupby(pd.Grouper(key="Date of Admission", freq="W")).size().reset_index(name="Patients")
    fig = px.line(weekly_df, x="Date of Admission", y="Patients", markers=True, title="Weekly Patient Admissions")
    fig.update_layout(xaxis_title="Week", yaxis_title="Number of Patients")
    st.plotly_chart(fig, use_container_width=True)

    # ============================
    # 🗓 Time Series Forecasting
    # ============================
    st.subheader("📅 Forecast Patient Volume")
    granularity = st.selectbox("Choose time granularity", ["D", "W", "M"], format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x])
    forecast_horizon = st.slider("Select forecast horizon", 7, 60, 14, step=7)
    horizon_label = {"D": "days", "W": "weeks", "M": "months"}[granularity]

    ts = df.groupby("Date of Admission").size().rename("Patient Count").to_frame()
    ts = ts.resample(granularity).sum()
    ts['Spike'] = ((ts - ts.mean()) / ts.std())['Patient Count'].abs() > 2

    st.line_chart(ts['Patient Count'])
    st.markdown(f"**🚨 Detected spikes/dips:** {ts['Spike'].sum()} {horizon_label}")

    # ========== Forecasting with ARIMA ==========
    arima_forecast = None
    try:
        arima_model = ARIMA(ts['Patient Count'], order=(2, 1, 2))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=forecast_horizon)
        st.line_chart(arima_forecast.rename("ARIMA Forecast"))
    except Exception as e:
        st.error(f"ARIMA forecast failed: {e}")

    # ========== Forecasting with Prophet ==========
    prophet_forecast_df = None
    if prophet_available:
        try:
            prophet_df = ts[['Patient Count']].reset_index().rename(columns={"Date of Admission": "ds", "Patient Count": "y"})
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=forecast_horizon, freq=granularity)
            prophet_forecast_df = m.predict(future)
            fig2 = m.plot(prophet_forecast_df)
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Prophet forecast failed: {e}")

    # ========== Evaluation Metrics ========== 
    if arima_forecast is not None and prophet_forecast_df is not None:
        actuals = ts['Patient Count'].dropna()[-forecast_horizon:]
        prophet_preds = prophet_forecast_df.set_index('ds').reindex(actuals.index)['yhat']
        arima_preds = arima_forecast[:len(actuals)]

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

        fig.update_layout(title='📈 Actual vs Forecasted Admissions with Annotations', xaxis_title='Date', yaxis_title='Patient Count', legend_title='Series', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        csv = combined_df.to_csv(index=True).encode()
        st.download_button("📥 Download Forecast CSV", csv, file_name="forecast_results.csv", mime="text/csv")

        st.markdown("### 🧠 AI-Powered Narrative")
        if openai_available:
            api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
                try:
                    insights_prompt = f"""
                    Analyze patient admission trends and forecasts for the next {forecast_horizon} {horizon_label}. 
                    Include which model performed better (ARIMA RMSE: {arima_rmse:.2f}, Prophet RMSE: {prophet_rmse:.2f}), 
                    spikes detected: {ts['Spike'].sum()}, holiday overlap: {len(holiday_dates)}. Return executive-style insights in 2–3 bullet points.
                    """
                    response = ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a healthcare analyst writing business summaries."},
                            {"role": "user", "content": insights_prompt}
                        ]
            )
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ GPT narrative failed: {e}")
    else:
        st.info("🧠 AI insights available with OpenAI key configured in secrets or session.")


# =====================================================================
# ▶️ Run the App
# =====================================================================
if __name__ == "__main__":
    main()
