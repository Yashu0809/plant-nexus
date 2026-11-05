import pandas as pd

df = pd.read_csv("E:\\Plant Nexus\\Datasets\\vehicle path (Day).csv")

# Normalize columns
df.columns = df.columns.str.strip().str.lower()

df.columns = df.columns.str.strip().str.lower()

ID_COL = "id"
SPEED_COL = "vehicle_speed"

def classify(speed):
    if speed > 80:
        return "OVER SPEED"
    elif speed < 5:
        return "HIGH IDLE"
    else:
        return "NORMAL"

df["status"] = df[SPEED_COL].apply(classify)

# Save to CSV
output_path = "vehicle_status_output.csv"
df.to_csv(output_path, index=False)

print(f"✅ File saved as: {output_path}")