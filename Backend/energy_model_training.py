import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

DATA_PATH = r"E:\Plant Nexus\Datasets\Energy_Consumption_Efficiency.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# Keep numeric only
df = df.select_dtypes(include=["int64", "float64"])

# Assume last column is energy consumption target
target_col = df.columns[-1]

X = df.drop(target_col, axis=1)
y = df[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
score = r2_score(y_test, y_pred)

print(f"Energy Model R² Score: {score*100:.2f}%")

# Save
os.makedirs("Backend/Energy", exist_ok=True)
joblib.dump(model, "Backend/Energy/energy_model.pkl")
joblib.dump(scaler, "Backend/Energy/energy_scaler.pkl")

print("Energy Model Saved Successfully")