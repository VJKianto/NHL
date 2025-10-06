import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# === Consensus importance weights ===
consensus_scores = {
    "PDO": 0.246,
    "HDGF": 0.088,
    "SH%": 0.083,
    "HDGA": 0.081,
    "TOI": 0.072,
    "SV%": 0.069,
    "MDGF": 0.064,
    "MDGA": 0.063,
    "LDGF": 0.040,
    "LDGA": 0.039
}

rolling_window = 5  # Number of past games to average

# === LOAD TRAIN DATA (2021–2023) ===
df_train = pd.read_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\output data\games_21_23_all_v2.csv")
df_train.replace('-', np.nan, inplace=True)
df_train["Date"] = pd.to_datetime(df_train["Date"])
df_train = df_train.sort_values(["Team", "Date"])

# Select variables to use
use_cols = [col for col in consensus_scores.keys() if col in df_train.columns]

# Compute rolling averages per team (using only past games)
for col in use_cols:
    df_train[col + "_roll"] = (
        df_train.groupby("Team")[col]
        .transform(lambda x: x.shift(1).rolling(rolling_window, min_periods=1).mean())
    )

# Drop rows with NaNs (teams' first few games may not have enough history)
train_features = df_train[[c + "_roll" for c in use_cols]].fillna(0)
train_target = df_train["Win"]

# Apply consensus weights
for col, weight in consensus_scores.items():
    col_roll = col + "_roll"
    if col_roll in train_features.columns:
        train_features[col_roll] *= weight

# === LOAD TEST DATA (2024 season) ===
df_test = pd.read_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\input data\games_2024_all.csv")
df_test["Win"] = (df_test["GF"] > df_test["GA"]).astype(int)
df_test.replace('-', np.nan, inplace=True)
import re
import pandas as pd

# Extract first valid date (YYYY-MM-DD) from the 'Game' column if it exists
df_test["Date"] = (
    df_test["Game"]
    .astype(str)
    .str.extract(r"(\d{4}-\d{2}-\d{2})")[0]  # get substrings like 2024-09-30
)

# Convert to datetime, coercing invalid ones to NaT (not-a-time)
df_test["Date"] = pd.to_datetime(df_test["Date"], errors="coerce")

# Optional: Drop rows with invalid/missing dates
df_test = df_test.dropna(subset=["Date"])


# Compute 5-game rolling averages (use previous games only)
for col in use_cols:
    df_test[col + "_roll"] = (
        df_test.groupby("Team")[col]
        .transform(lambda x: x.shift(1).rolling(rolling_window, min_periods=1).mean())
    )

test_features = df_test[[c + "_roll" for c in use_cols]].fillna(0)
test_target = df_test["Win"]

# Apply same weights
for col, weight in consensus_scores.items():
    col_roll = col + "_roll"
    if col_roll in test_features.columns:
        test_features[col_roll] *= weight

# === TRAIN & TEST MODEL ===
model = RandomForestClassifier(random_state=42, n_estimators=500, max_depth=6)
model.fit(train_features, train_target)

y_pred = model.predict(test_features)
print("Predictive accuracy (based on past games):", round(accuracy_score(test_target, y_pred), 3))
