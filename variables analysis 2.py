import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE, mutual_info_classif, f_classif
import numpy as np


# ======================
# Load data
# ======================
df = pd.read_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\Input data\games_2022_2024_all.csv")


# Create target variable
df["Win"] = (df["GF"] > df["GA"]).astype(int)


# Select numeric features only
features = df.drop(columns=["Win", "GF", "GA"]).select_dtypes(include="number")
target = df["Win"]


# Split train/test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# ======================
# 3. Logistic Regression (scaled)
# ======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_model = LogisticRegression(max_iter=1000, random_state=42, penalty="l2")
log_model.fit(X_train_scaled, y_train)
log_importances = pd.Series(abs(log_model.coef_[0]), index=features.columns).sort_values(ascending=False)
log_importances.to_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\Output data\NHL_logistic.csv")


# ======================
# 4. Permutation Importance
# ======================
perm_result = permutation_importance(log_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
perm_importances = pd.Series(perm_result.importances_mean, index=features.columns).sort_values(ascending=False)
perm_importances.to_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\Output data\NHL_perm.csv")




# ======================
# 6. LassoCV (L1 Logistic Regression)
# ======================
lasso = LassoCV(cv=5, random_state=42).fit(X_train_scaled, y_train)
lasso_importances = pd.Series(abs(lasso.coef_), index=features.columns).sort_values(ascending=False)
lasso_importances.to_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\Output data\NHL_lasso.csv")


# ======================
# 7. Recursive Feature Elimination (RFE with Logistic Regression)
# ======================
rfe = RFE(estimator=LogisticRegression(max_iter=1000, solver="liblinear"), n_features_to_select=10)
rfe.fit(X_train_scaled, y_train)
rfe_importances = pd.Series(rfe.ranking_, index=features.columns).sort_values()
rfe_importances.to_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\Output data\NHL_rfe.csv")


# ======================
# 8. Mutual Information
# ======================
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_importances = pd.Series(mi_scores, index=features.columns).sort_values(ascending=False)
mi_importances.to_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\Output data\NHL_mutual_info.csv")


# ======================
# 9. ANOVA F-test
# ======================
f_scores, p_values = f_classif(X_train, y_train)
anova_importances = pd.Series(f_scores, index=features.columns).sort_values(ascending=False)
anova_importances.to_csv(r"C:\Users\vonak\OneDrive - LUT University\Documents\Data science\Output data\NHL_anova.csv")



