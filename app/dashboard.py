import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from src.analysis import Analysis



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
    st.subheader("🔍 Transaction Type Counts")
    st.bar_chart(analyzer.transaction_type_count())
    st.subheader("📊 Transaction Type by Category")
    st.dataframe(analyzer.transaction_type_category())

else:
    st.info("Upload your cleaned CSV to begin analysis.")
