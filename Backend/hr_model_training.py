# ===============================
# HR ATTRITION MODEL TRAINING
# ===============================

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE

# -------------------------------
# PATH
# -------------------------------
DATA_PATH = r"E:\Plant Nexus\Datasets\HR_Analytics.csv"
SAVE_DIR = "Backend/HR"

# -------------------------------
# LOAD DATA
# -------------------------------
try:
    df = pd.read_csv(DATA_PATH)
except:
    df = pd.read_excel(DATA_PATH)

print("Dataset Loaded:", df.shape)

# -------------------------------
# BASIC CLEANING
# -------------------------------
df.drop_duplicates(inplace=True)

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# TARGET
# -------------------------------
TARGET = "Attrition"
df[TARGET] = df[TARGET].map({"Yes":1, "No":0})

X = df.drop(columns=[TARGET])
y = df[TARGET]

# -------------------------------
# PREPROCESSOR
# -------------------------------
numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include="object").columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -------------------------------
# SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# PROCESS + SMOTE
# -------------------------------
X_train_p = preprocessor.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_p, y_train)

# -------------------------------
# MODEL + TUNING
# -------------------------------
param_grid = {
    "n_estimators":[200,300],
    "max_depth":[None,10,20],
    "min_samples_split":[2,5]
}

grid = GridSearchCV(
    ExtraTreesClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_res, y_res)

model = grid.best_estimator_

# -------------------------------
# EVALUATION
# -------------------------------
X_test_p = preprocessor.transform(X_test)
pred = model.predict(X_test_p)

acc = accuracy_score(y_test, pred)
print(f"HR Model Accuracy: {acc*100:.2f}%")

# -------------------------------
# SAVE
# -------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(model, f"{SAVE_DIR}/hr_model.pkl")
joblib.dump(preprocessor, f"{SAVE_DIR}/hr_preprocessor.pkl")

print("HR Model Saved Successfully")