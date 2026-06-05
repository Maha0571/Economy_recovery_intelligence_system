import pandas as pd

def forecast_country(model, df, feature_cols, country):

    # =========================
    # FILTER COUNTRY
    # =========================
    country_df = df[df["country_name"].str.lower() == country.lower()].copy()
    country_df = country_df.sort_values("year")

    if len(country_df) < 4:
        raise ValueError("Not enough data (need at least 4 years)")

    # =========================
    # TAKE LAST 3 YEARS
    # =========================
    latest = country_df.iloc[-1]   # most recent year (2024/2025)
    prev1 = country_df.iloc[-2]
    prev2 = country_df.iloc[-3]

    # =========================
    # OPTIONAL: simple future trend (2026 estimation)
    # =========================
    gdp_trend_future = latest["GDP Growth (% Annual)"] - prev1["GDP Growth (% Annual)"]

    # =========================
    # BUILD 2026 FEATURE VECTOR
    # =========================
    future_input = pd.DataFrame([{
        # raw economic indicators (latest known values)
        "Inflation (CPI %)": latest["Inflation (CPI %)"],
        "GDP (Current USD)": latest["GDP (Current USD)"],
        "GDP per Capita (Current USD)": latest["GDP per Capita (Current USD)"],
        "Unemployment Rate (%)": latest["Unemployment Rate (%)"],
        "Inflation (GDP Deflator, %)": latest["Inflation (GDP Deflator, %)"],
        "Current Account Balance (% GDP)": latest["Current Account Balance (% GDP)"],
        "Gross National Income (USD)": latest["Gross National Income (USD)"],

        # =========================
        # LAG FEATURES (LAST 3 YEARS)
        # =========================
        "gdp_lag_1": latest["GDP Growth (% Annual)"],
        "gdp_lag_2": prev1["GDP Growth (% Annual)"],
        "gdp_lag_3": prev2["GDP Growth (% Annual)"],

        # =========================
        # ROLLING FEATURES
        # =========================
        "gdp_roll3": country_df["GDP Growth (% Annual)"].tail(3).mean(),

        # =========================
        # TREND (FOR 2026 FORECAST)
        # =========================
        "gdp_trend": gdp_trend_future,

        # =========================
        # RATIOS
        # =========================
        "inflation_unemployment_gap":
            latest["Inflation (CPI %)"] - latest["Unemployment Rate (%)"],

        "gni_to_gdp_ratio":
            latest["Gross National Income (USD)"] / (latest["GDP (Current USD)"] + 1),

        "gdp_inflation_ratio":
            latest["GDP (Current USD)"] / (latest["Inflation (CPI %)"] + 1)
    }])

    # =========================
    # PREDICT 2026
    # =========================
    pred = model.predict(future_input)[0]
    prob = model.predict_proba(future_input)[0][1]

    return pred, prob