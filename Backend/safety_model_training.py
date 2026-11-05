# ===============================
# SAFETY / AIR QUALITY MODEL TRAINING
# ===============================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score

from sklearn.ensemble import ExtraTreesRegressor

# -------------------------------
# PATHS
# -------------------------------
DATA_PATH = r"E:\Plant Nexus\Datasets\Air Quality.csv"

SAVE_DIR = "Backend/Safety"

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_PATH)

print("Original Shape:", df.shape)

# -------------------------------
# CLEANING
# -------------------------------
df = df.dropna(axis=1, how='all')
df.replace(-200, np.nan, inplace=True)

# -------------------------------
# DATETIME HANDLING
# -------------------------------
if 'Date' in df.columns and 'Time' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    df['Hour'] = df['Datetime'].dt.hour
    df.drop(columns=['Date','Time','Datetime'], inplace=True)

# -------------------------------
# MISSING VALUES
# -------------------------------
imputer = SimpleImputer(strategy="mean")
df[df.columns] = imputer.fit_transform(df)

# -------------------------------
# TARGET & FEATURES
# -------------------------------
TARGET = "CO(GT)"   # Change if needed

X = df.drop(columns=[TARGET])
y = df[TARGET]

# -------------------------------
# SCALING
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# FEATURE SELECTION
# -------------------------------
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X_scaled, y)

selected_features = X.columns[selector.get_support()]
print("Selected Features:", list(selected_features))

# -------------------------------
# SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL + TUNING
# -------------------------------
param_grid = {
    "n_estimators": [200,300],
    "max_depth": [None,10,20],
    "min_samples_split": [2,5]
}

grid = GridSearchCV(
    ExtraTreesRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

# -------------------------------
# EVALUATION
# -------------------------------
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)

print(f"Safety Model R2 Score: {r2*100:.2f}%")

# -------------------------------
# SAVE
# -------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(model, f"{SAVE_DIR}/safety_model.pkl")
joblib.dump(scaler, f"{SAVE_DIR}/safety_scaler.pkl")
joblib.dump(selector, f"{SAVE_DIR}/safety_selector.pkl")

print("Safety Model Saved Successfully")