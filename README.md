# 🏭 Plant NEXUS: AI-Powered Industrial Monitoring Platform

An enterprise-grade, end-to-end industrial intelligence platform designed for **Sponge Iron Manufacturing Plants**. Integrates machine learning, real-time IoT/SCADA sensor telemetry, **FastAPI**, **PostgreSQL**, and an interactive **Streamlit** dashboard across **6 critical operational domains**.

---

## 📌 Executive Summary
Sponge iron manufacturing involves high-temperature rotary kilns, extensive material handling, and intensive energy consumption. Operational blindspots often cause unplanned downtime, excessive emissions, and supply disruptions.

**Plant NEXUS** eliminates silos by combining predictive maintenance, environmental air quality monitoring, supply chain risk analytics, energy optimization, workforce dynamics, and yard logistics into a unified real-time dashboard.

---

## 🌟 Six Core Operational Modules

| Module | Purpose & Technique | Machine Learning Model | Key Metric / Outcome |
| :--- | :--- | :--- | :--- |
| **⚙️ Predictive Maintenance (Kiln RUL)** | Predicts Remaining Useful Life (RUL) of rotary kilns from 21 multi-channel sensor feeds. | **Random Forest Regressor** | **Health Score (0–100%)**, Status alerts (NORMAL / CRITICAL) |
| **🛡️ Safety & Air Quality** | Real-time monitoring and forecasting of hazardous emissions ($CO$, toxic pollutants). | **Extra Trees Regressor** | **$R^2$ Score > 94%**, Automated hazard threshold alarms |
| **⚡ Energy Efficiency** | Tracks specific energy usage per ton and identifies abnormal power surges. | **Regression Model** | Identifies peak tariff waste and equipment inefficiency |
| **🔗 Supply Chain Risk** | Analyzes supplier lead times, raw material stocks (iron ore/coal), and transit delay risk. | **Polynomial Regression** | Proactive logistics buffer recommendations |
| **👥 HR & Workforce Analytics** | Evaluates shift schedules, overtime, and predicts attrition risk. | **Classification Model** | Workforce safety and turnover mitigation |
| **🚛 Vehicle Telematics** | Monitors GPS coordinates and yard vehicle speeds in real-time. | **Rule-Based & Clustering** | Flags **HIGH IDLE** and **OVER SPEED** violations |

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                   Streamlit Web Dashboard (UI)                   │
│         (Plotly Charts, Status Badges, Sensor Inputs)            │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ HTTP / JSON
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FastAPI High-Performance Backend                │
│    (Pydantic Validation, Scikit-Learn Inference Pipelines)       │
└──────────────────┬───────────────────────────────┬───────────────┘
                   │                               │
                   ▼                               ▼
       ┌────────────────────────┐      ┌────────────────────────┐
       │ PostgreSQL Database    │      │ Pre-Trained ML Weights │
       │ (Audit & Logs Storage) │      │ (Joblib Serialized)    │
       └────────────────────────┘      └────────────────────────┘
```

---

## 🗂️ Project Structure

```text
├── Backend/
│   ├── Backend_1.py               # Main FastAPI application server & REST endpoints
│   ├── db_postgres.py             # PostgreSQL connection and database helper routines
│   ├── init_db_postgres.py        # Database schema initialization script
│   ├── Pdata_processing.py        # C-MAPSS sensor data cleaning, RUL calculation & scaling
│   ├── Pmodel_training.py         # Production kiln RUL training script (Random Forest)
│   ├── model.pkl                  # Serialized production kiln RUL predictive model
│   ├── scaler.pkl                 # Production sensor feature scaler
│   ├── Energy/                    # Energy model training & serialized weights
│   ├── HR/                        # HR analytics model training & preprocessor
│   ├── Safety/                    # Air quality Extra Trees model & feature selector
│   └── SupplyChain/               # Supply chain risk model & scaler
├── Frontend/
│   └── frontend.py                # Multi-page interactive Streamlit dashboard
├── Datasets/                      # Raw and preprocessed industrial datasets
│   ├── Air Quality.csv            # Environmental sensor records
│   ├── Energy_Consumption_Efficiency.csv # Power and fuel consumption logs
│   ├── HR_Analytics.csv           # Workforce parameters
│   ├── smart_logistics_dataset.csv# Supply chain delivery history
│   └── vehicle path (Day).csv     # Telematics GPS trajectory data
├── data_production_KLIN/          # C-MAPSS turbofan / rotary kiln degradation dataset
├── vehicle check.py               # Telematics speed classifier script
├── requirements.txt               # Project Python dependencies
├── Plant_Nexus_Final_Report.pdf   # Complete project documentation & system report
└── README.md                      # Documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Yashu0809/plant-nexus.git
cd plant-nexus
```

### 2. Set Up a Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure the Database (PostgreSQL)
Ensure PostgreSQL is running locally and update connection settings in `Backend/db_postgres.py`:
```bash
python Backend/init_db_postgres.py
```

### 4. Start the FastAPI Backend
```bash
uvicorn Backend.Backend_1:app --reload --port 8000
```
FastAPI interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 5. Launch the Streamlit Frontend Dashboard
In a separate terminal window:
```bash
streamlit run Frontend/frontend.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Technology Stack
- **Backend & APIs**: FastAPI, Uvicorn, Pydantic
- **Machine Learning**: Scikit-Learn, Joblib, NumPy, Pandas
- **Interactive Frontend**: Streamlit, Plotly Express
- **Database**: PostgreSQL (`psycopg2`)
- **Simulation & Telematics**: SUMO (Simulation of Urban MObility) GPS outputs

---

## 👤 Author
**Yashwanth Jillella**
- GitHub: [@Yashu0809](https://github.com/Yashu0809)
