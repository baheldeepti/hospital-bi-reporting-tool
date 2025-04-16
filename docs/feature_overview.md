🧠 **Hospital Chat Assistant - Feature Overview**

This document outlines all current capabilities of the **Hospital Chat Assistant** Streamlit application.

---

📁 **Data Management**
- **Default Dataset**: Loads a sample CSV file directly from GitHub on button click.
- **User Upload**: Allows users to upload their own hospital CSV files.
- **Source Attribution**: Sidebar banner with reference to the original Kaggle dataset.

📚 **Sidebar Features**
- **Glossary**: Clear description of all dataset columns (e.g., Age, Gender, Medical Condition, etc.).
- **About This App**: Overview of features, stack, and usage instructions.
- **Dataset Loader**: Upload CSV or load sample dataset.

💬 **Chat Interface**
- **Chat History**: Maintains chronological conversation between user and assistant.
- **Dynamic Quick Suggestions**: Suggested prompts based on available columns in the dataset.
- **Natural Language Querying**:
  - Attempts retrieval-augmented generation (RAG) via LangChain.
  - Falls back to LangChain Pandas DataFrame agent if RAG is not configured.
- **Medical Term Tooltips**: Auto-appends relevant explanations (e.g., "billing", "condition").

📊 **Visual Analytics**
- **Auto Chart Previews**: Generates Altair bar charts in response to relevant queries.
  - Billing trend by hospital
  - Patient count by gender
  - Top medical conditions by frequency
- **Export Options**:
  - Download PNG for each chart
  - Download PDF for each chart using Matplotlib
- **Seasonal Decomposition (Planned)**:
  - Intended to show trend, seasonal, and residual components using statsmodels.

📤 **Export & Downloads**
- **Chat History**: Download full chat as CSV.
- **Query Leaderboard**: Track and download most clicked quick suggestions.
- **Fallback Query Log**: View/download queries that triggered the fallback agent.

📈 **Tracking & Logs**
- **Query Tracking**: Clicks on suggestion buttons are counted.
- **Fallback Log**: Logs inputs answered by the fallback DataFrame agent.
- *(Future)* Summary tagging and filtering.

✅ **Built-in Intelligence**
- **Token Overload Prevention**: Avoids full DataFrame context passing by loading a smaller sample by default.
- **Dynamic UX**: Suggestion logic adjusts to visible columns in the dataset.

🔮 **Future Expansion Ideas**
- Add seasonal decomposition chart using monthly billing.
- Add user profile for login-based dashboards.
- Memory summarization across sessions.
- Schedule automatic reporting.
- Search/filter saved summaries.

