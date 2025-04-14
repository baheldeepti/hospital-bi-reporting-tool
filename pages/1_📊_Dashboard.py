# hospital_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

st.title("🏥 Healthcare Data Product with Python & Streamlit")

file = st.file_uploader("Upload your healthcare dataset (.csv)", type=["csv"])

if file:
    df = pd.read_csv(file)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

    # Forecasting: Billing Amount
    st.subheader("📈 Forecast: Monthly Billing")
    billing_df = df[['Date of Admission', 'Billing Amount']].copy()
    billing_df = billing_df.rename(columns={'Date of Admission': 'ds', 'Billing Amount': 'y'})
    billing_df = billing_df.groupby(pd.Grouper(key='ds', freq='M')).sum().reset_index()
    prophet_model = Prophet()
    prophet_model.fit(billing_df)
    future = prophet_model.make_future_dataframe(periods=6, freq='M')
    forecast = prophet_model.predict(future)
    fig1 = prophet_model.plot(forecast)
    st.pyplot(fig1)

    # LOS Forecasting
    st.subheader("📉 Forecast: Length of Stay")
    los_df = df[['Date of Admission', 'Length of Stay']].copy()
    los_df = los_df.rename(columns={'Date of Admission': 'ds', 'Length of Stay': 'y'})
    los_df = los_df.groupby(pd.Grouper(key='ds', freq='M')).mean().reset_index()
    los_model = Prophet()
    los_model.fit(los_df)
    future_los = los_model.make_future_dataframe(periods=6, freq='M')
    los_forecast = los_model.predict(future_los)
    fig2 = los_model.plot(los_forecast)
    st.pyplot(fig2)

    # Anomaly Detection in Billing
    st.subheader("🚨 Anomaly Detection in Billing")
    iso = IsolationForest(contamination=0.02, random_state=42)
    df['anomaly'] = iso.fit_predict(df[['Billing Amount']])
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='anomaly', y='Billing Amount', data=df, ax=ax3)
    st.pyplot(fig3)

    # Patient Segmentation
    st.subheader("🧪 Patient Segmentation with KMeans")
    clustering_data = df[['Age', 'Billing Amount', 'Length of Stay']].copy()
    clustering_data['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
    scaled = StandardScaler().fit_transform(clustering_data)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x='Billing Amount', y='Length of Stay', hue='Cluster', data=df, palette='Set2', ax=ax4)
    st.pyplot(fig4)

    # Logistic Regression for Predicting Abnormal Test Results
    st.subheader("🧠 Predicting Abnormal Test Results (Logistic Regression)")
    df['Test Results Encoded'] = df['Test Results'].map({'Normal': 0, 'Abnormal': 1, 'Inconclusive': 2})
    X = df[['Age', 'Billing Amount', 'Length of Stay']].copy()
    X['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
    X['Medication'] = LabelEncoder().fit_transform(df['Medication'])
    y = (df['Test Results Encoded'] == 1).astype(int)  # binary classification: Abnormal vs not

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred))
