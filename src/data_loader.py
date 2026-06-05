import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    print("Initial Shape:", df.shape)
    print("Year Range:", df["year"].min(), "-", df["year"].max())

    return df
