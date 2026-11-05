from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import joblib
import os

from streamlit import cursor
from db_postgres import get_connection

app = FastAPI(title="Plant Nexus Backend")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD MODELS ----------------
prod_model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
prod_scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

sc_model = joblib.load(os.path.join(BASE_DIR, "SupplyChain", "sc_model.pkl"))
sc_scaler = joblib.load(os.path.join(BASE_DIR, "SupplyChain", "sc_scaler.pkl"))

energy_model = joblib.load(os.path.join(BASE_DIR, "Energy", "energy_model.pkl"))
energy_scaler = joblib.load(os.path.join(BASE_DIR, "Energy", "energy_scaler.pkl"))

safety_model = joblib.load(os.path.join(BASE_DIR, "Safety", "safety_model.pkl"))
safety_scaler = joblib.load(os.path.join(BASE_DIR, "Safety", "safety_scaler.pkl"))
safety_selector = joblib.load(os.path.join(BASE_DIR, "Safety", "safety_selector.pkl"))

hr_model = joblib.load(os.path.join(BASE_DIR, "HR", "hr_model.pkl"))
hr_preprocessor = joblib.load(os.path.join(BASE_DIR, "HR", "hr_preprocessor.pkl"))

# ---------------- SCHEMAS ----------------
class ProductionInput(BaseModel):
    sensor_values: List[float]

class SupplyChainInput(BaseModel):
    values: List[float]

class EnergyInput(BaseModel):
    values: List[float]

class SafetyInput(BaseModel):
    values: List[float]

class HRInput(BaseModel):
    data: dict


@app.get("/")
def home():
    return {"status": "PlantNexus backend running successfully "}


# ---------------- PRODUCTION ----------------
@app.post("/production/monitoring")
def production_monitor(data: ProductionInput):

    if len(data.sensor_values) != 21:
        return {"error": "Exactly 21 sensor values required"}

    values = np.array(data.sensor_values)

    avg = np.mean(values)
    variance = np.var(values)
    max_val = np.max(values)

    score = (avg * 10 + variance * 5 + max_val * 2)
    health_score = 100 - score
    health_score = max(0, min(100, health_score))

    if health_score < 30:
        status = "CRITICAL"
    elif health_score < 60:
        status = "WARNING"
    else:
        status = "NORMAL"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO maintenance_logs (sensor_values, predicted_rul, status)
        VALUES (%s, %s, %s)
    """, (str(data.sensor_values), float(health_score), status))
    conn.commit()
    conn.close()

    return {
        "department": "Production",
        "health_score": round(float(health_score), 2),
        "status": status
    }


# ---------------- SUPPLY CHAIN ----------------
@app.post("/supplychain/predict")
def supplychain_predict(data: SupplyChainInput):

    if len(data.values) != 10:
        return {"error": "Exactly 10 values required"}

    x = np.array(data.values).reshape(1, -1)
    x_scaled = sc_scaler.transform(x)
    cluster = sc_model.predict(x_scaled)[0]

    status = "HIGH LOGISTICS RISK" if np.mean(data.values) > 1000 else "LOW LOGISTICS RISK"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO supply_chain_logs (input_values, risk)
        VALUES (%s, %s)
    """, (str(data.values), status))
    conn.commit()
    conn.close()

    return {"department": "Supply Chain", "cluster": int(cluster), "status": status}


# ---------------- ENERGY ----------------
@app.post("/energy/predict")
def energy_predict(data: EnergyInput):

    if len(data.values) != 5:
        return {"error": "Exactly 5 values required"}

    x = np.array(data.values).reshape(1, -1)
    x_scaled = energy_scaler.transform(x)
    prediction = energy_model.predict(x_scaled)[0]

    status = "HIGH ENERGY" if np.mean(data.values) > 200 else "NORMAL"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO energy_logs (input_values, consumption, status)
        VALUES (%s, %s, %s)
    """, (str(data.values), float(prediction), status))
    conn.commit()
    conn.close()

    return {"department": "Energy", "predicted_consumption": float(prediction), "status": status}


# ---------------- SAFETY ----------------
@app.post("/safety/predict")
def safety_predict(data: SafetyInput):

    if len(data.values) != 16:
        return {"error": "Exactly 16 values required"}

    x = np.array(data.values).reshape(1, -1)
    x_scaled = safety_scaler.transform(x)
    x_selected = safety_selector.transform(x_scaled)

    predicted_co = safety_model.predict(x_selected)[0]

    avg = np.mean(data.values)
    max_val = np.max(data.values)

    if predicted_co >= 2 or avg > 2 or max_val > 3:
        status = "UNSAFE AIR QUALITY"
    elif predicted_co >= 1 or avg > 1:
        status = "MODERATE AIR QUALITY"
    else:
        status = "SAFE AIR QUALITY"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO safety_logs (air_values, result)
        VALUES (%s, %s)
    """, (str(data.values), status))
    conn.commit()
    conn.close()

    return {
        "department": "Safety",
        "predicted_CO_level": round(float(predicted_co), 2),
        "status": status
    }


# ---------------- HR ----------------
@app.post("/hr/predict")
def hr_predict(input_data: HRInput):

    df_input = pd.DataFrame([input_data.data])
    expected_cols = hr_preprocessor.feature_names_in_

    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[expected_cols]
    processed = hr_preprocessor.transform(df_input)

    prediction = hr_model.predict(processed)[0]
    probability = hr_model.predict_proba(processed)[0][1]
    status = "HIGH ATTRITION" if prediction == 1 else "LOW ATTRITION"

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO hr_logs (employee_data, prediction, attrition_probability, status)
    VALUES (%s, %s, %s, %s)
    """, (
    str(input_data.data),
    int(prediction),
    float(probability),
    status))

    conn.commit()
    conn.close()
    return {
        "department": "HR",
        "attrition_prediction": int(prediction),
        "attrition_probability": round(float(probability), 2),
        "status": status
    }


# ---------------- VEHICLE ----------------
import math

@app.get("/vehicle/monitor/{vehicle_id}")
def monitor_vehicle(vehicle_id: int):

    df = pd.read_csv(r"E:\Plant Nexus\Datasets\vehicle path (Day).csv")

    record = df[df["id"] == vehicle_id]

    if record.empty:
        return {"error": "Vehicle not found"}

    speed = float(record["vehicle_speed"].values[0])
    x = float(record["vehicle_x"].values[0])
    y = float(record["vehicle_y"].values[0])

    # Simple distance from origin
    distance = math.sqrt(x**2 + y**2)

    if speed > 80:
        status = "OVER SPEED"
    elif speed < 5:
        status = "HIGH IDLE"
    else:
        status = "NORMAL"
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO vehicle_logs (vehicle_id, vehicle_type, speed, distance, status, remark)
    VALUES (%s, %s, %s, %s, %s, %s)
""", (
    vehicle_id,
    "Vehicle",
    float(speed),
    float(distance),
    status,
    "Auto log"))

    conn.commit()
    conn.close()
    
    return {
        "vehicle_id": vehicle_id,
        "speed": speed,
        "distance": round(distance, 2),   # ✅ FIX
        "status": status
    }

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO vehicle_logs (vehicle_id, vehicle_type, speed, distance, status, remark)
    VALUES (%s, %s, %s, %s, %s, %s)
""", (
    vehicle_id,
    "Vehicle",
    float(speed),
    float(distance),
    status,
    "Auto log"
))

    conn.commit()
    conn.close()

    
@app.get("/logs/production")
def get_production_logs():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM maintenance_logs ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data


@app.get("/logs/supplychain")
def get_supplychain_logs():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM supply_chain_logs ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data


@app.get("/logs/energy")
def get_energy_logs():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM energy_logs ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data


@app.get("/logs/safety")
def get_safety_logs():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM safety_logs ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data


@app.get("/logs/hr")
def get_hr_logs():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM hr_logs ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data


@app.get("/logs/vehicle")
def get_vehicle_logs():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicle_logs ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data