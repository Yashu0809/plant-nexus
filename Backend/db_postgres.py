import psycopg2

def get_connection():
    conn = psycopg2.connect(
        dbname="plant_nexus",
        user="postgres",
        password="1504",
        host="localhost",
        port="5432"
    )
    return conn


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_logs (
        id SERIAL PRIMARY KEY,
        sensor_values TEXT,
        predicted_rul FLOAT,
        status VARCHAR(20)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hr_logs (
        id SERIAL PRIMARY KEY,
        employee_data TEXT,
        prediction VARCHAR(20)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS safety_logs (
        id SERIAL PRIMARY KEY,
        air_values TEXT,
        result VARCHAR(20)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS energy_logs (
        id SERIAL PRIMARY KEY,
        input_values TEXT,
        consumption FLOAT,
        status VARCHAR(20)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS supply_chain_logs (
        id SERIAL PRIMARY KEY,
        input_values TEXT,
        risk VARCHAR(20)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_logs (
        id SERIAL PRIMARY KEY,
        vehicle_id INT,
        vehicle_type TEXT,
        status VARCHAR(20),
        remark TEXT
    );
    """)

    conn.commit()
    conn.close()