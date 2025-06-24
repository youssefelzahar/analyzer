import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from src.analysis import Analysis

# === Load Model and Tools ===
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/model_vectorizer.pkl")
#label_encoder = joblib.load("models/model_label_encoder.pkl")

# === Streamlit Setup ===
st.set_page_config(page_title="SmartFin", layout="wide")
st.title("💰 SmartFin: AI-Powered Personal Finance Advisor")

# === Section 1: Category Prediction ===
st.header("🔮 Predict Transaction Category")
desc_input = st.text_input("Enter a transaction description:")

if st.button("Predict Category"):
    if not desc_input.strip():
        st.warning("Please enter a valid description.")
    else:
        X = vectorizer.transform([desc_input])
        pred = model.predict(X)[0]
        #label = label_encoder.inverse_transform([pred])[0]
        #st.success(f"Predicted Category: **{label}**")

# === Section 2: Data Analysis ===
st.header("📊 Transaction Data Analysis")

uploaded_file = st.file_uploader("Upload cleaned transaction CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    analyzer = Analysis(df)

    st.subheader("📌 Summary Statistics")
    st.dataframe(analyzer.summary_statistics())

    st.subheader("📈 Monthly Spending Trend")
    monthly = analyzer.monthly_spending()
    st.line_chart(monthly['Amount'])

    st.subheader("📂 Category Counts")
    st.bar_chart(analyzer.category_count())

    st.subheader("💸 Spending per Category")
    st.bar_chart(analyzer.category_spending())

else:
    st.info("Upload your cleaned CSV to begin analysis.")
