import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from Pdata_processing import load_cmapss, compute_rul, scale_features

DATA_PATH = "data_production_KLIN/train_FD001.txt"
# Load & process data
df = load_cmapss(DATA_PATH)
df = compute_rul(df)

FEATURES = [f"s{i}" for i in range(1, 22)]

df, scaler = scale_features(df, FEATURES)

X = df[FEATURES]
y = df["RUL"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")
# Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully")
