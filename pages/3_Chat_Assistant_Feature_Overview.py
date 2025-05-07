# 🏥 Hospital Chat Assistant

A **Streamlit-powered AI chatbot** for exploring and analyzing hospital datasets using natural language. Users can upload a dataset or use the built-in sample, ask questions about the data, and receive chart-backed, code-generated answers—powered by GPT-4 and pandas.

---

## 📌 Features

- 🧠 **GPT-4 powered query interpretation** to convert natural language into executable Python (pandas) code.
- 📊 **Automatic visualizations** (Altair bar charts) for numerical insights.
- 📁 **Custom or sample dataset support** via file upload or built-in link.
- 📝 **AI-generated summaries** of insights for executive readability.
- 🪵 **Conversation logs and fallback tracker** for debugging or improving model interactions.

---

## ⚙️ How It Works

1. **User uploads a dataset** (CSV) or loads a default hospital dataset.
2. The user **asks a natural language question** in the chat input.
3. The app sends context (column names, history) to GPT-4 via OpenAI API.
4. GPT-4 responds with **executable pandas code**, which is:
   - Displayed in the app
   - Executed on the dataset
   - Visualized if applicable (via Altair bar chart)
   - Summarized using a follow-up GPT call
5. All interactions are **logged** for review and debugging.

---

## 🧪 Example Questions

- *"Show me average billing amount by hospital."*
- *"How many patients were admitted in 2021?"*
- *"Which medical condition has the longest length of stay?"*

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- An OpenAI API Key
- Streamlit


