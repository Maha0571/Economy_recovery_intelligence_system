

import numpy as np
import pandas as pd

def clean_missing_values(df):
    """
    Fill missing values
    Numeric -> Median
    Categorical -> Mode
    """

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Missing values cleaned")

    return df


def sort_data(df):
    """
    Sort country and year
    """

    df = df.sort_values(
        ["country_name", "year"]
    ).reset_index(drop=True)

    print("Data sorted")

    return df

