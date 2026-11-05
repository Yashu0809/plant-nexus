import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

DATA_PATH = r"E:\Plant Nexus\Datasets\smart_logistics_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Keep numeric
df = df.select_dtypes(include=["int64", "float64"])

# Save true labels
true_labels = None
if "Logistics_Delay" in df.columns:
    true_labels = df["Logistics_Delay"]
    df_features = df.drop("Logistics_Delay", axis=1)
else:
    df_features = df

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# KMeans Clustering
model = KMeans(n_clusters=2, random_state=42)
clusters = model.fit_predict(X_scaled)

# ============================
# CALCULATE ACCURACY
# ============================
if true_labels is not None:
    # Since cluster labels (0/1) may be reversed,
    # compute both possibilities and take best accuracy
    acc1 = accuracy_score(true_labels, clusters)
    acc2 = accuracy_score(true_labels, 1 - clusters)
    accuracy = max(acc1, acc2)

    print(f"Supply Chain Clustering Accuracy: {accuracy*100:.2f}%")
else:
    print("No true labels available for accuracy calculation.")

# ============================
# SAVE MODEL
# ============================
os.makedirs("Backend/SupplyChain", exist_ok=True)

joblib.dump(model, "Backend/SupplyChain/sc_model.pkl")
joblib.dump(scaler, "Backend/SupplyChain/sc_scaler.pkl")

print("Supply Chain Model Saved Successfully")