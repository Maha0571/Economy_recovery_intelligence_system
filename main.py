from src.data_loader import load_data
from src.preprocessing import *
from src.feature_engineering import *
from src.train_test_split import split_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.shap_analysis import run_shap_analysis
from src.forecasting import forecast_country
from src.model_saver import save_model
from src.ai_helper import chat_with_ollama

# =========================
# LOAD DATA
# =========================
df = load_data("Data/world_bank_data_2025.csv")

# =========================
# PREPROCESSING
# =========================
df = clean_missing_values(df)
df = sort_data(df)

# =========================
# FEATURE ENGINEERING
# =========================
df = create_target(df)
df = create_lags(df)
df = create_ratio_features(df)
df = final_cleanup(df)

# =========================
# SPLIT DATA
# =========================
train_df, test_df = split_data(df)

# =========================
# FEATURES
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
# TRAIN MODEL
# =========================
model = train_model(
    train_df[feature_cols],
    train_df["Recovery"]
)

# =========================
# EVALUATION
# =========================
evaluate_model(
    model,
    test_df[feature_cols],
    test_df["Recovery"],
    feature_cols
)

# =========================
# SHAP ANALYSIS
# =========================
run_shap_analysis(
    model,
    test_df[feature_cols]
)

# =========================
# SAVE MODEL
# =========================
save_model(model, feature_cols)

# =========================
# USER INPUT
# =========================
print("\n🌍 Top Countries:")
print(df["country_name"].value_counts().head(20))

country = input("\nEnter Country Name: ")

# =========================
# YEAR INPUT (IMPORTANT FIX)
# =========================
years_available = sorted(df[df["country_name"].str.lower() == country.lower()]["year"].unique())

print("\nAvailable Years:", years_available)

year = int(input("Enter Year: "))

# filter country + year
country_df = df[
    (df["country_name"].str.lower() == country.lower()) &
    (df["year"] == year)
]

if country_df.empty:
    raise ValueError("No data for selected country/year")

# =========================
# FORECAST
# =========================
pred, prob = forecast_country(
    model=model,
    df=df,
    feature_cols=feature_cols,
    country=country
)

pred = int(pred)

prediction_text = (
    "Recovery Expected"
    if pred == 1
    else "Recovery Not Expected"
)

print("\n====================")
print("Country:", country)
print("Year:", year)
print("Prediction:", prediction_text)
print("Confidence:", f"{prob:.2%}")
print("====================")

# =========================
# AI EXPLANATION
# =========================
latest = country_df.iloc[-1]

prompt = f"""
You are a professional Economic Analyst.

Country: {country}


Prediction: {prediction_text}
Confidence: {prob:.2%}

Economic Indicators:
GDP: {latest['GDP (Current USD)']}
GDP Per Capita: {latest['GDP per Capita (Current USD)']}
Inflation: {latest['Inflation (CPI %)']}
Unemployment: {latest['Unemployment Rate (%)']}
GNI: {latest['Gross National Income (USD)']}
Current Account Balance: {latest['Current Account Balance (% GDP)']}

Explain:
1. Meaning of prediction
2. Why model predicted this
3. Key factors
4. Opportunities
5. Risks
6. Country type
7. Final recommendation
"""

explanation = chat_with_ollama(prompt)

print("\n🧠 AI Explanation:\n")
print(explanation)