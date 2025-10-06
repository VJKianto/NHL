import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


df = pd.read_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\Input data\games_21_23_all.csv")

# Create target variable
df["Win"] = (df["GF"] > df["GA"]).astype(int)
df=df.drop(columns=["TOI", "Unnamed: 2", "Attendance"])
df["Date"]=df["Game"].str[:11]
df["Trend"] = 0
for team, group in df.groupby("Team"):
    streak = []
    counter = 0
    # Sort by date to ensure correct order
    group_sorted = group.sort_values("Date")
    for idx, row in group_sorted.iterrows():
        month = str(row["Date"])[5:7]  # Assumes 'YYYY-MM-DD' format
        if month == "10":  # October
            counter = 0    # Reset streak for new season
        if row["Win"] == 1:
            counter += 1
        else:
            counter = 0
        streak.append(counter)
    df.loc[group_sorted.index, "Trend"] = streak

print(df)
print(df.head(10))
df.to_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\Output data\games_21_23_all_v2.csv")

from sklearn.model_selection import train_test_split

features = df.drop(columns=["Win", "GF", "GA", "Date", "Game", "Team"])
target = df["Win"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))