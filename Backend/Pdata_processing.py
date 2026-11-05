import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMNS = (
    ["unit_id", "cycle"] +
    [f"op{i}" for i in range(1, 4)] +
    [f"s{i}" for i in range(1, 22)]
)

def load_cmapss(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.iloc[:, :-2]  # remove empty cols
    df.columns = COLUMNS
    return df

def compute_rul(df):
    max_cycle = df.groupby("unit_id")["cycle"].max()
    df["RUL"] = df.apply(lambda r: max_cycle[r.unit_id] - r.cycle, axis=1)
    return df

def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler
