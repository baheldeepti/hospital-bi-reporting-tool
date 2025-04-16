# 📊 Hospital Dashboard - Feature Overview

This document outlines the current capabilities of the **Hospital Dashboard** page within the Hospital Streamlit App.

---

## 📁 Data Management
- **Data Source**: Uses uploaded or sample data loaded into `st.session_state.main_df`.
- **Data Preparation**:
  - Handles missing values for critical columns.
  - Converts billing and date fields into appropriate formats.

## 📊 Visual Dashboard Features
- **Key Metrics**:
  - Total Patients
  - Total Billing Amount
  - Average Billing per Patient
  - Average Length of Stay
- **Breakdown Charts**:
  - Billing by Hospital
  - Patient Count by Gender
  - Admissions over Time (Line Chart)
  - Billing by Insurance Provider
- **Interactive Filters**:
  - Slicers based on date ranges, gender, hospital, and admission type (if implemented).

## 🧮 Statistical Summary
- **Descriptive Statistics**:
  - Uses `df.describe()` to summarize numerical columns.
  - Showcased in expandable sections for interpretability.

## 📤 Export & Reporting
- **CSV Export**: Download transformed/filtered dataset.
- **Image Export**: PNG export for visualizations (if implemented).

## 🧠 Built-in Logic
- **Dynamic Layout**: Adapts content based on data availability and column presence.
- **Cached Calculations**: Optimized for performance using `@st.cache` or `st.cache_data`.
- **Session Safety**: Prevents crashes if dataset is not yet loaded.

## 🔮 Future Expansion Ideas
- Cross-tab summaries by multiple filters.
- Dashboard snapshots or PDF export for offline reports.
- Custom filter memory across sessions.
- Forecasting widget integration (Prophet/ARIMA).

