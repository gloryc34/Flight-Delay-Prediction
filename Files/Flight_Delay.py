# Flight Delay Prediction Project
# Author: Colin Glory
# -----------------------------------------
# End-to-end machine learning pipeline:
# 1. Load and clean data
# 2. Feature engineering
# 3. Handle class imbalance
# 4. Train/test split
# 5. Model training (Random Forest, XGBoost)
# 6. Evaluation
# 7. Feature importance visualization
# -----------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------------------
# 1. Load Data
# -----------------------------------------
df = pd.read_csv("T_ONTIME_REPORTING.csv")  

print("Initial shape:", df.shape)

# -----------------------------------------
# 2. Data Cleaning
# -----------------------------------------
# Keep essential columns
cols = [
    "FL_DATE", "DAY_OF_WEEK", "OP_UNIQUE_CARRIER",
    "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "ARR_DELAY",
    "CANCELLED", "DIVERTED", "DISTANCE", "CRS_ELAPSED_TIME"
]
df = df[[c for c in cols if c in df.columns]]

# Remove cancelled or diverted flights
df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]

# Drop rows with missing ARR_DELAY
df = df.dropna(subset=["ARR_DELAY"])

print("After cleaning:", df.shape)

# -----------------------------------------
# 3. Feature Engineering
# -----------------------------------------
# Target: binary classification (delayed if ARR_DELAY > 15 mins)
df["ARR_DELAY_BIN"] = (df["ARR_DELAY"] > 15).astype(int)

# Departure time bins
def dep_time_bin(x):
    if pd.isnull(x): return "Unknown"
    hour = int(str(int(x)).zfill(4)[:2])  # extract hour
    if 5 <= hour < 12: return "Morning"
    elif 12 <= hour < 17: return "Afternoon"
    elif 17 <= hour < 21: return "Evening"
    else: return "Night"

df["DEP_TIME_BIN"] = df["CRS_DEP_TIME"].apply(dep_time_bin)

# Reduce airport categories to top 10
top_airports = df["ORIGIN_AIRPORT_ID"].value_counts().nlargest(10).index
df["ORIGIN_AIRPORT_ID"] = df["ORIGIN_AIRPORT_ID"].apply(
    lambda x: x if x in top_airports else "Other"
)
top_airports_dest = df["DEST_AIRPORT_ID"].value_counts().nlargest(10).index
df["DEST_AIRPORT_ID"] = df["DEST_AIRPORT_ID"].apply(
    lambda x: x if x in top_airports_dest else "Other"
)

# Ensure correct data types
df["OP_UNIQUE_CARRIER"] = df["OP_UNIQUE_CARRIER"].astype(str)
df["ORIGIN_AIRPORT_ID"] = df["ORIGIN_AIRPORT_ID"].astype(str)
df["DEST_AIRPORT_ID"] = df["DEST_AIRPORT_ID"].astype(str)
df["DEP_TIME_BIN"] = df["DEP_TIME_BIN"].astype(str)

# Features and target
features = [
    "DAY_OF_WEEK", "OP_UNIQUE_CARRIER", "ORIGIN_AIRPORT_ID",
    "DEST_AIRPORT_ID", "DEP_TIME_BIN", "DISTANCE", "CRS_ELAPSED_TIME"
]
X = df[features]
y = df["ARR_DELAY_BIN"]

# -----------------------------------------
# 4. Train/Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------
# 5. Handle Class Imbalance (SMOTE)
# -----------------------------------------
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# -----------------------------------------
# 6. Preprocessing Pipeline
# -----------------------------------------
categorical_features = ["OP_UNIQUE_CARRIER", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "DEP_TIME_BIN"]
numeric_features = [f for f in features if f not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# -----------------------------------------
# 7. Modeling
# -----------------------------------------

# Random Forest
rf_clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])
rf_clf.fit(X_res, y_res)

# XGBoost
xgb_clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
])
xgb_clf.fit(X_res, y_res)

# -----------------------------------------
# 8. Evaluation
# -----------------------------------------
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Results")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

evaluate_model(rf_clf, X_test, y_test, "Random Forest")
evaluate_model(xgb_clf, X_test, y_test, "XGBoost")

# -----------------------------------------
# 9. Feature Importance (Random Forest)
# -----------------------------------------
# Extract encoded feature names
ohe = rf_clf.named_steps['preprocessor'].named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(categorical_features)
all_features = np.concatenate([ohe_features, numeric_features])

# Extract importances
importances = rf_clf.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)

# Plot top 20
plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp[:20], y=feat_imp.index[:20], palette="viridis")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
