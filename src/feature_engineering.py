import numpy as np
import pandas as pd


# =========================
# TARGET (SAFE + NO LEAKAGE)
# =========================
def create_target(df):

    df = df.copy()
    df = df.sort_values(["country_name", "year"])

    df["future_gdp_growth"] = (
        df.groupby("country_name")["GDP Growth (% Annual)"].shift(-1)
    )

    # binary target
    df["Recovery"] = (df["future_gdp_growth"] > 3).astype(int)

    df = df.dropna(subset=["future_gdp_growth"])

    return df


# =========================
# LAG FEATURES (STABLE VERSION)
# =========================
def create_lags(df):

    df = df.copy()
    df = df.sort_values(["country_name", "year"])

    group = df.groupby("country_name")["GDP Growth (% Annual)"]

    df["gdp_lag_1"] = group.shift(1)
    df["gdp_lag_2"] = group.shift(2)
    df["gdp_lag_3"] = group.shift(3)

    # rolling mean (stable trend signal)
    df["gdp_roll3"] = group.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # trend (normalized version FIX)
    df["gdp_trend"] = (
        df["gdp_lag_1"] - df["gdp_lag_3"]
    ) / (df["gdp_lag_3"].abs() + 1)

    # volatility (important FIX for confidence stability)
    df["gdp_volatility"] = group.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).std()
    ).fillna(0)

    # momentum feature (NEW STRONG SIGNAL)
    df["gdp_momentum"] = (
        df["gdp_lag_1"] - df["gdp_lag_2"]
    )

    return df


# =========================
# RATIO FEATURES (CLEAN + STABLE)
# =========================
def create_ratio_features(df):

    df = df.copy()

    # inflation vs unemployment pressure
    df["inflation_unemployment_gap"] = (
        df["Inflation (CPI %)"] - df["Unemployment Rate (%)"]
    )

    # stability ratio
    df["gni_to_gdp_ratio"] = (
        df["Gross National Income (USD)"] /
        (df["GDP (Current USD)"] + 1)
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # inflation sensitivity ratio
    df["gdp_inflation_ratio"] = (
        df["GDP (Current USD)"] /
        (df["Inflation (CPI %)"] + 1)
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # 🔥 NEW IMPORTANT FEATURE (fix prediction bias)
    df["economic_pressure_index"] = (
        df["Unemployment Rate (%)"] +
        df["Inflation (CPI %)"] -
        df["GDP Growth (% Annual)"]
    )

    # normalize extreme values (IMPORTANT FIX)
    df["economic_pressure_index"] = np.clip(
        df["economic_pressure_index"],
        -100,
        100
    )

    return df


# =========================
# FINAL CLEANUP (SAFE + ROBUST)
# =========================
def final_cleanup(df):

    df = df.copy()

    essential_cols = [
        "gdp_lag_1",
        "gdp_lag_2",
        "gdp_lag_3",
        "gdp_roll3",
        "gdp_trend",
        "gdp_volatility",
        "gdp_momentum"
    ]

    df = df.dropna(subset=essential_cols)

    df = df.reset_index(drop=True)

    return df