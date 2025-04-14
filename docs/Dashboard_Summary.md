# 🤖 Hospital Chat Assistant (v1.4.3)

The Hospital Chat Assistant is an AI-powered Streamlit app designed to analyze hospital patient datasets using natural language queries and interactive visualizations.

---

## 🔧 Features

- **Data Upload or Sample Dataset Support**
- **Dynamic Filtering by Hospital, Insurance Provider, and Medical Condition**
- **KPI Dashboard for Billing, Stay Length, and Patient Count**
- **Billing Trend Line Chart**
- **AI Chat Assistant (GPT + Altair Charting)**
- **Chart Suggestions + Free-form Query Matching**
- **Narrative Summary Generator**
- **Advanced Chart Explorer (Bar, Pie, Line)**
- **Data Glossary with Search**
- **Leaderboards and Usage Logs**
- **Export CSV for all charts and logs**
- **Debug Mode + Session Snapshot**

---

## 📁 File Structure

```bash
📦 hospital-chat-assistant/
├── Hospital_Chat_Assistant_v1_4_3_streamlit_cloud.py
├── requirements.txt
└── README.md
```

---

## 🚀 Deployment (Streamlit Cloud)

1. Fork or clone this repo
2. Add your OpenAI key as a secret:
   - In Streamlit Cloud → `Settings` → `Secrets`:
     ```
     OPENAI_API_KEY=your-key-here
     ```
3. Deploy using Streamlit Cloud
4. Use the UI to:
   - Upload your own `.csv`
   - Or load the sample dataset

---

## 📊 Sample Query Prompts

- `"Total billing by hospital"`
- `"Patient count by gender"`
- `"Average stay over time"`
- `"Billing trend"`

These will automatically render visual charts and support CSV download.

---

## ❤️ Credits

Built with Streamlit, Altair, LangChain, and OpenAI  
Crafted by Deepti Bahel  
Date: 2025-04-14
