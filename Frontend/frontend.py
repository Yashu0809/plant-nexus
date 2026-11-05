import streamlit as st
import requests
import plotly.express as px
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Plant Nexus Dashboard", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.metric-card {
    padding: 20px;
    border-radius: 10px;
    background-color: #111827;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🏭 Plant Nexus Smart Monitoring Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    ["Production", "Supply Chain", "Energy", "Safety", "HR", "Vehicle"]
)

# ---------------- PRODUCTION ----------------
if menu == "Production":
    st.header("⚙️ Production Monitoring")

    values = st.text_input("Enter 21 sensor values (comma separated)")

    if st.button("Analyze"):
        data = [float(x) for x in values.split(",")]

        res = requests.post(f"{BASE_URL}/production/monitoring",
                            json={"sensor_values": data}).json()

        col1, col2 = st.columns(2)
        st.write(res)

        col1.metric("Health Score", res["health_score"])

        status_color = "green" if res["status"] == "NORMAL" else "red"
        col2.markdown(f"<h3 style='color:{status_color}'>{res['status']}</h3>",
                    unsafe_allow_html=True)

        # Chart
        df = pd.DataFrame({"Sensor": range(len(data)), "Value": data})
        fig = px.line(df, x="Sensor", y="Value", title="Sensor Trend")
        st.plotly_chart(fig, use_container_width=True)


# ---------------- SUPPLY CHAIN ----------------
elif menu == "Supply Chain":
    st.header("🚚 Supply Chain Risk")

    values = st.text_input("Enter 10 values")

    if st.button("Predict"):
        data = [float(x) for x in values.split(",")]

        res = requests.post(f"{BASE_URL}/supplychain/predict",
                            json={"values": data}).json()

        st.metric("Cluster", res["cluster"])

        color = "red" if "HIGH" in res["status"] else "green"
        st.markdown(f"<h3 style='color:{color}'>{res['status']}</h3>",
                    unsafe_allow_html=True)


# ---------------- ENERGY ----------------
elif menu == "Energy":
    st.header("⚡ Energy Monitoring")

    values = st.text_input("Enter 5 values")

    if st.button("Predict Energy"):
        data = [float(x) for x in values.split(",")]

        res = requests.post(f"{BASE_URL}/energy/predict",
                            json={"values": data}).json()

        st.metric("Predicted Consumption", res["predicted_consumption"])

        color = "red" if "HIGH" in res["status"] else "green"
        st.markdown(f"<h3 style='color:{color}'>{res['status']}</h3>",
                    unsafe_allow_html=True)

        df = pd.DataFrame({"Index": range(len(data)), "Energy": data})
        fig = px.bar(df, x="Index", y="Energy", title="Energy Input Distribution")
        st.plotly_chart(fig, use_container_width=True)


# ---------------- SAFETY ----------------
elif menu == "Safety":
    st.header("🌫️ Air Quality")

    values = st.text_input("Enter 16 values")

    if st.button("Check"):
        data = [float(x) for x in values.split(",")]

        res = requests.post(f"{BASE_URL}/safety/predict",
                            json={"values": data}).json()

        st.metric("CO Level", res["predicted_CO_level"])

        color = "red" if "UNSAFE" in res["status"] else "green"
        st.markdown(f"<h3 style='color:{color}'>{res['status']}</h3>",
                    unsafe_allow_html=True)


# ---------------- HR ----------------
elif menu == "HR":
    st.header("👨‍💼 HR Attrition")

    age = st.slider("Age", 18, 60)
    income = st.number_input("Income")
    satisfaction = st.slider("Satisfaction", 1, 4)
    years = st.number_input("Years at Company")
    overtime = st.selectbox("Overtime", ["Yes", "No"])

    if st.button("Predict"):
        data = {
            "Age": age,
            "MonthlyIncome": income,
            "JobSatisfaction": satisfaction,
            "YearsAtCompany": years,
            "OverTime": overtime
        }

        res = requests.post(f"{BASE_URL}/hr/predict",
                            json={"data": data}).json()

        st.metric("Attrition Probability", res["attrition_probability"])

        color = "red" if "HIGH" in res["status"] else "green"
        st.markdown(f"<h3 style='color:{color}'>{res['status']}</h3>",
                    unsafe_allow_html=True)


# ---------------- VEHICLE ----------------
elif menu == "Vehicle":
    st.header("🚗 Vehicle Monitoring")

    vid = st.number_input("Vehicle ID", min_value=1)

    if st.button("Track"):
        res = requests.get(f"{BASE_URL}/vehicle/monitor/{int(vid)}").json()

        st.metric("Speed", res["speed"])
        st.metric("Distance", res["distance"])

        color = "red" if res["status"] != "NORMAL" else "green"
        st.markdown(f"<h3 style='color:{color}'>{res['status']}</h3>",
                    unsafe_allow_html=True)