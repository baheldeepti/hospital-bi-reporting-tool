# 📄 Overview: Hospital Strategy Optimization App (Explained Simply)

This Streamlit app is like a smart assistant for hospital administrators. It helps them understand which strategies can save the hospital money **without hurting patient care**.

Let me explain what each part of the code does — like you're 15 years old and just learning data science 👩‍🔬👨‍🔬:

---

## 🧱 1. What’s the App About?
It’s a dashboard that lets you:
- Upload hospital data 📂
- Find expensive or strange cases 😵
- Simulate cost-saving strategies 💰
- Get smart recommendations 📊
- Even see what ChatGPT suggests 🤖

---

## 🛠 2. How It Works

### 🧩 Libraries Used
- `streamlit`: for building the web app UI
- `pandas`, `numpy`: for handling and transforming data
- `matplotlib`: for creating charts
- `sklearn`: for anomaly detection (Isolation Forest)
- `scipy`: for optimization (figuring out the best patients to select)

### 📥 Data Upload
Users can upload a CSV of hospital data, or the app uses a default one from GitHub.

### 🧼 Data Cleaning
The app:
- Converts dates
- Calculates how long each patient stayed
- Checks if they were admitted on a weekend
- Uses machine learning to detect “anomalies” (strange billing cases)

---

## 🎛 3. Simulating Cost-Saving

The user sets sliders like:
- Max % of weekend cases
- Max % of long stays
- Max % anomalies allowed
- How many patients to include at minimum

Then the app tries to **find the best combination of patients** that fits the rules and minimizes total cost.

It shows you:
- A before/after billing chart 📉
- How much money you saved 💸

---

## 🧠 4. Strategy Options

The app compares 3 ways to pick patients:
- **LP (Linear Programming)**: Smart math to minimize cost with constraints
- **Greedy**: Just pick the cheapest patients
- **Heuristic**: Avoid weekend and anomaly cases

Then it recommends the best strategy based on your goals.

---

## 📊 5. Summary & Drilldown
You can:
- See strategy summary with metrics
- View patient-level details (filter by gender, insurance)
- Download selected patient data

---

## 📅 6. Billing Trends Over Time
This part shows:
- How billing changes month-by-month
- Anomaly spikes with red warning lines
- Breakdown by gender or insurance

---

## 💡 7. Recommendations
There are two kinds:
- **Static tips** from business rules
- **Live AI advice** from OpenAI’s ChatGPT API (e.g. "Reduce weekend admissions")

The AI sees the strategy and data, then suggests actions a hospital could take.

---

## 🔐 8. Security Tip
For AI to work, users need to add their OpenAI key in a secret file.

```toml
# In .streamlit/secrets.toml
openai_api_key = "your-key-here"
```

---

## ✅ Conclusion
This app helps hospital teams make **data-driven decisions** and plan better strategies using:
- Math 🧮
- AI 🤖
- Charts 📈


