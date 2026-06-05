import streamlit as st
import joblib
import pandas as pd

from src.forecasting import forecast_country
from src.ai_helper import chat_with_ollama

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Economic Recovery Predictor",
    layout="wide"
)

st.title("🌍 AI Economic Recovery Prediction (2026 Forecast)")

# =========================
# LOAD MODEL + DATA
# =========================
model = joblib.load("models/economic_recovery_model.pkl")
df = pd.read_csv("Data/world_bank_data_2025.csv")

# =========================
# FEATURE COLUMNS
# =========================
feature_cols = [
    "Inflation (CPI %)",
    "GDP (Current USD)",
    "GDP per Capita (Current USD)",
    "Unemployment Rate (%)",
    "Inflation (GDP Deflator, %)",
    "Current Account Balance (% GDP)",
    "Gross National Income (USD)",
    "gdp_lag_1",
    "gdp_lag_2",
    "gdp_lag_3",
    "gdp_roll3",
    "gdp_trend",
    "inflation_unemployment_gap",
    "gni_to_gdp_ratio",
    "gdp_inflation_ratio"
]

# =========================
# UI - COUNTRY ONLY
# =========================
countries = sorted(df["country_name"].unique())
country = st.selectbox("Select Country", countries)

# =========================
# BUTTON
# =========================
if st.button("Predict 2026 Recovery"):

    # =========================
    # FORECAST
    # =========================
    pred, prob = forecast_country(
        model=model,
        df=df,
        feature_cols=feature_cols,
        country=country
    )

    prediction_text = "Recovery Expected" if pred == 1 else "Recovery Not Expected"

    # =========================
    # RESULT UI
    # =========================
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.success(prediction_text)
    else:
        st.error(prediction_text)

    st.metric("Confidence Score", f"{prob:.2%}")

    # =========================
    # GET LATEST DATA
    # =========================
    country_df = df[df["country_name"].str.lower() == country.lower()].sort_values("year")
    latest = country_df.iloc[-1]

    # =========================
    # AI EXPLANATION
    # =========================
    with st.spinner("Generating AI Explanation..."):

        prompt = f"""
Country: {country}
Forecast Year: 2026

Prediction: {prediction_text}
Confidence: {prob:.2%}

GDP: {latest['GDP (Current USD)']}
GDP Per Capita: {latest['GDP per Capita (Current USD)']}
Inflation: {latest['Inflation (CPI %)']}
Unemployment: {latest['Unemployment Rate (%)']}
GNI: {latest['Gross National Income (USD)']}

Explain:
1. Meaning
2. Key reasons
3. Opportunities
4. Risks
5. Investment outlook
6. Final recommendation
"""

        explanation = chat_with_ollama(prompt)

    st.subheader("🧠 AI Explanation")
    st.write(explanation)

# =========================
# 📊 SIMPLE CLEAN GRAPHS
# =========================
st.subheader("📊 Economic Trends (Historical)")

country_df = df[df["country_name"].str.lower() == country.lower()].sort_values("year")

col1, col2, col3 = st.columns(3)

with col1:
    st.line_chart(country_df.set_index("year")["GDP (Current USD)"])

with col2:
    st.line_chart(country_df.set_index("year")["Inflation (CPI %)"])

with col3:
    st.line_chart(country_df.set_index("year")["Unemployment Rate (%)"])