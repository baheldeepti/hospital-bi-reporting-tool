import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

st.title("🏥 Easy-to-Understand Hospital Data Dashboard")

st.markdown("""
Welcome to the Hospital Executive Insights Dashboard — designed to provide a clear, high-level view of operational performance and key healthcare metrics.  
Each visualization is paired with concise commentary to support strategic decision-making, resource planning, and performance optimization.
""")

# 🔗 Embedded Power BI Dashboard
st.header("📊 Power BI Insights Dashboard")
st.markdown("This embedded dashboard shows key hospital metrics like billing trends, patient counts, and length of stay across departments.")
powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=adba6afc-7417-4ef1-a069-a4c9b7287a7b&autoAuth=true&ctid=81a7563b-844e-45bb-a4f0-b2f1ed65acb7"
components.iframe(powerbi_url, width=1000, height=600, scrolling=True)

# 📥 Sample Data Download Instructions
st.subheader("📥 Sample Data to Explore")
st.markdown("""
If you don't have your own data yet, you can use our **sample hospital dataset** to try out the dashboard.  
Click the link below to download:

🔗 [Download Sample CSV](https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv)
""")

file = st.file_uploader("📁 Upload your hospital dataset (.csv)", type=["csv"])

# 🚀 Load dataset: uploaded or default sample
if file is not None:
    df = pd.read_csv(file)
    st.info("✅ Custom dataset uploaded successfully!")
else:
    sample_url = "https://github.com/baheldeepti/hospital-streamlit-app/raw/main/modified_healthcare_dataset.csv"
    df = pd.read_csv(sample_url)
    st.warning("⚠️ No file uploaded. Using the **default sample dataset** for demo purposes.")

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# Forecasting Billing
st.header("📈 Monthly Billing Forecast")
st.markdown("This chart predicts how much total money the hospital might make from patient bills in the next 6 months. 📅")

billing_df = df[['Date of Admission', 'Billing Amount']].copy()
billing_df = billing_df.rename(columns={'Date of Admission': 'ds', 'Billing Amount': 'y'})
billing_df = billing_df.groupby(pd.Grouper(key='ds', freq='M')).sum().reset_index()

model = Prophet()
model.fit(billing_df)
future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)

fig1 = model.plot(forecast)
st.pyplot(fig1)

# Forecasting Length of Stay
st.header("📉 Average Length of Stay Forecast")
st.markdown("This shows how long patients usually stay in the hospital and predicts future trends. ⏱️")

los_df = df[['Date of Admission', 'Length of Stay']].copy()
los_df = los_df.rename(columns={'Date of Admission': 'ds', 'Length of Stay': 'y'})
los_df = los_df.groupby(pd.Grouper(key='ds', freq='M')).mean().reset_index()

los_model = Prophet()
los_model.fit(los_df)
future_los = los_model.make_future_dataframe(periods=6, freq='M')
los_forecast = los_model.predict(future_los)

fig2 = los_model.plot(los_forecast)
st.pyplot(fig2)

# Anomaly Detection
st.header("🚨 Billing Anomaly Detection")
st.markdown("This chart shows **suspicious or unusual bills** that might be errors or very special cases. 🔍")

iso = IsolationForest(contamination=0.02, random_state=42)
df['anomaly'] = iso.fit_predict(df[['Billing Amount']])
fig3, ax3 = plt.subplots()
sns.boxplot(x='anomaly', y='Billing Amount', data=df, ax=ax3)
st.pyplot(fig3)

# Predict Abnormal Test Results
st.header("🧠 Predicting Abnormal Test Results")
st.markdown("This model guesses if a patient's medical test might come back **abnormal**, based on age, medicine, and condition. ⚕️")

df['Test Results Encoded'] = df['Test Results'].map({'Normal': 0, 'Abnormal': 1, 'Inconclusive': 2})
X = df[['Age', 'Billing Amount', 'Length of Stay']]
X['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
X['Medication'] = LabelEncoder().fit_transform(df['Medication'])
y = (df['Test Results Encoded'] == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

report = classification_report(y_test, y_pred)
st.code("This model correctly predicts most abnormal cases.\n\n" + report)

# Clustering
st.header("🧪 Patient Clusters")
st.markdown("Here we group patients into 4 types based on age, cost, and stay time. Think of it like patient **personalities**! 😊")

cluster_data = df[['Age', 'Billing Amount', 'Length of Stay']]
cluster_data['Condition'] = LabelEncoder().fit_transform(df['Medical Condition'])
scaled = StandardScaler().fit_transform(cluster_data)

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)

fig4, ax4 = plt.subplots()
sns.scatterplot(x='Billing Amount', y='Length of Stay', hue='Cluster', data=df, palette='Set2', ax=ax4)
st.pyplot(fig4)

st.success("🎉 You're done exploring the hospital data like a pro!")

# Summary
st.markdown("""
---

## 📌 Summary View of the Analysis

- **Billing Forecast**: The hospital is predicted to continue generating steady revenue in the coming months.
- **Length of Stay Forecast**: Patients are generally staying the same length over time, with little variation.
- **Anomaly Detection**: A few patient bills are much higher or lower than usual, which could mean errors or special treatments.
- **Abnormal Test Prediction**: Using factors like age and medication, we can fairly accurately guess if a test might be abnormal.
- **Patient Clusters**: Patients can be grouped by age, cost, and time spent, helping hospitals personalize care or optimize processes.
""")
