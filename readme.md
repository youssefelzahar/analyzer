# 🧠 SmartFin – AI-Powered Personal Finance Advisor

SmartFin is an AI-powered personal finance assistant built with Python. It analyzes your transactions, forecasts future spending, categorizes expenses using NLP, and provides budget alerts through a chatbot — all via a clean Streamlit dashboard.

---

## 🚀 Features

* 📊 **Transaction Analysis**: Visualize spending patterns, category breakdowns, and monthly trends
* 🧠 **NLP Category Prediction**: Predicts transaction categories from text descriptions using TF-IDF + Random Forest
* 📈 **Spending Forecasting**: Forecasts future expenses using SARIMA time-series modeling
* 🤖 **Chatbot Alerts**: Smart chatbot alerts users on overspending and responds to budget-related questions
* 🌐 **Streamlit Dashboard**: Interactive UI for uploading data and getting insights

---

## 📂 Project Structure

```
smartfin/
│
├── config.py                 # File paths and configuration
├── main.py                   # Entry point to run everything
├── app/
│   └── streamlit_app.py      # Streamlit dashboard app
├── src/
│   ├── preprocessing.py      # Data cleaning and encoding
│   ├── predict.py            # Category prediction model
│   ├── forecast.py           # Forecasting logic using SARIMA
│   ├── analysis.py           # Visual + statistical analysis
│   └── chatbot.py            # Rule-based alert chatbot
├── models/                   # Saved ML models
├── data/
│   └── personal_transactions.csv  # Original dataset
│   └── personal_transactions_cleaned.csv  # Processed dataset
└── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone the repo:**

```bash
git clone https://github.com/youssefelzahar/smartfin.git
cd smartfin
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app:**

```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Example Questions for Chatbot

* “Did I overspend?”
* “What’s my highest spending category?”
* “Give me a spending summary”

---

## 📈 Model & Tools Used

* `Pandas`, `NumPy` – Data manipulation
* `Scikit-learn` – TF-IDF, classification, preprocessing
* `Statsmodels` – Time-series forecasting with SARIMA
* `Streamlit` – Dashboard interface
* `Joblib` – Model saving/loading
* `Matplotlib` / `Seaborn` – Visualizations

---

## 📌 Dataset

Used [Personal Finance Dataset](https://www.kaggle.com/datasets/bukolafatunde/personal-finance) from Kaggle, and added additional structure for forecasting and classification tasks.

---

## ✨ Future Improvements

* Add user authentication and custom budgets per user
* Integrate real-time bank data using Plaid or Salt Edge
* Make chatbot more intelligent using LLMs (e.g., GPT API)

---

## 📬 Contact

For questions or collaborations, feel free to reach out on [LinkedIn](https://www.linkedin.com/in/youssef-elzahar/).
