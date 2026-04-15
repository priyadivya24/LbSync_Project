import pandas as pd
import os
import io
from sqlalchemy import create_engine, text

# 🔧 CONFIG
base_path = r"C:\Users\Priyadarshini Maha\OneDrive\Documents\Projects_git\Grafana_XFEL\XFEL.SYNC\LASER.LOCK.XLO\XHEXP1.SLO1\CTRL0.OUT.MEAN.RD"
DB_URL = "postgresql://postgres:Priya%4099@localhost:5432/XFEL.SYNC"

engine = create_engine(DB_URL)

# 🔥 STEP 1: Create table (if not exists)
create_table_query = """
CREATE TABLE IF NOT EXISTS xfel_data (
    timestamp TIMESTAMP,
    value DOUBLE PRECISION,
    system TEXT,
    device TEXT,
    sensor TEXT
);
"""

with engine.connect() as conn:
    conn.execute(text(create_table_query))
    conn.commit()

print("✅ Table ready")

# 🔥 STEP 2: Process files
for root, dirs, files in os.walk(base_path):

    for file in files:

        # ⚡ ONLY October 2023
        if not file.endswith("2023-10.parquet"):
            continue

        file_path = os.path.join(root, file)
        print(f"🔥 Processing: {file_path}")

        try:
            # ⚡ Fast parquet read
            df = pd.read_parquet(file_path, engine="pyarrow")

            # ⚡ Keep only needed columns
            df = df[['timestamp', 'data']].copy()

            df.rename(columns={
                'timestamp': 'timestamp',
                'data': 'value'
            }, inplace=True)

            # ⚡ Fast datetime conversion
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            # Extract metadata
            parts = root.split(os.sep)

            df['system'] = "LASER"
            df['device'] = parts[-2] if len(parts) >= 2 else "NA"
            df['sensor'] = parts[-1]

            # 🚀 FAST INSERT using COPY
            buffer = io.StringIO()
            df.to_csv(buffer, index=False, header=False)
            buffer.seek(0)

            conn = engine.raw_connection()
            cursor = conn.cursor()

            cursor.copy_expert(
                """
                COPY xfel_data (timestamp, value, system, device, sensor)
                FROM STDIN WITH CSV
                """,
                buffer
            )

            conn.commit()
            cursor.close()
            conn.close()

            print(f"✅ Inserted {len(df)} rows")

            # 🧠 Free memory
            del df

        except Exception as e:
            print(f"❌ Error: {e}")

# 🔥 STEP 3: Create indexes (VERY IMPORTANT)
with engine.connect() as conn:
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_time ON xfel_data (timestamp);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sensor ON xfel_data (sensor);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_system ON xfel_data (system);"))
    conn.commit()

print("🚀 Indexes created successfully")