# 🏥Time Series Forecasting Overview

This interactive Streamlit app helps healthcare analysts and hospital executives **forecast patient volume**, **evaluate model performance**, and **generate narrative insights** using **AI**.

---

## 📦 Key Features

- **Data Upload or Sample Dataset**  
  Upload your own hospital admission CSV file or use a built-in sample dataset with patient records.

- **Filter by Hospital Parameters**  
  Narrow down insights by hospital, insurance provider, or medical condition for customized analysis.

- **Visualize Weekly Trends**  
  Analyze weekly patient admission trends via interactive line charts.

- **Forecasting with ARIMA and Prophet**  
  Forecast patient volumes over daily, weekly, or monthly horizons using two robust time-series models:
  - **ARIMA**: AutoRegressive Integrated Moving Average
  - **Prophet** (if available): Facebook’s forecasting model

- **Model Performance Comparison**  
  Evaluate model accuracy using:
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)

- **Spike/Dip and Holiday Annotations**  
  Identify unusual patient volume spikes/dips and overlay U.S. holidays directly on the forecast chart.

- **📥 Forecast Results Download**  
  Export the final forecasts (actual vs. predicted) as a downloadable CSV file.

- **🧠 AI-Powered Narrative Generation** *(Optional with OpenAI)*  
  Use GPT-3.5 to generate summary bullet points explaining trends, anomalies, and model insights.  
  *(Requires OpenAI API key)*

---

## 🔧 Tech Stack

- **Streamlit** for the web app interface
- **Pandas**, **NumPy**, **Plotly**, **Matplotlib**, **Seaborn** for data handling & visualization
- **ARIMA** from `statsmodels` and **Prophet** for forecasting
- **OpenAI GPT-3.5** for AI-powered summaries *(optional)*
- **US Holidays API** for public holiday annotations

---

## 📂 Required CSV Columns

Make sure your uploaded dataset includes at least the following column:

- `Date of Admission`

Optional (but helpful) columns:
- `Hospital`
- `Insurance Provider`
- `Medical Condition`

---

