
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "sensor_coverage.csv"

df = pd.read_csv(INPUT)

df["sensor_first_utc"] = pd.to_datetime(
    df["sensor_first_utc"], utc=True, errors="coerce"
)

df["sensor_last_utc"] = pd.to_datetime(
    df["sensor_last_utc"], utc=True, errors="coerce"
)

print("\nSENSOR QUALITY INSPECTION")
print("=" * 65)

# 1. Unit distribution
print("\n1. UNIT DISTRIBUTION")
print(df.groupby(["parameter", "unit"]).size().to_string())

# 2. Sensor date availability
print("\n2. DATE AVAILABILITY")
print(
    df.groupby("parameter").agg(
        total_sensors=("sensor_id", "nunique"),
        missing_first=("sensor_first_utc", lambda x: x.isna().sum()),
        missing_last=("sensor_last_utc", lambda x: x.isna().sum()),
        earliest=("sensor_first_utc", "min"),
        latest=("sensor_last_utc", "max")
    ).to_string()
)

# 3. Sensor historical duration
df["history_days"] = (
    df["sensor_last_utc"] - df["sensor_first_utc"]
).dt.total_seconds() / 86400

print("\n3. HISTORICAL DURATION IN DAYS")
print(
    df.groupby("parameter")["history_days"]
    .describe()
    .round(1)
    .to_string()
)

# 4. Stations with all six pollutants and valid dates
valid = df.dropna(
    subset=["sensor_first_utc", "sensor_last_utc"]
)

station_pollutants = (
    valid.groupby("location_id")["parameter"]
    .nunique()
)

print("\n4. STATIONS WITH VALID SENSOR DATE RANGES")
print("At least 3 pollutants:", (station_pollutants >= 3).sum())
print("All 6 pollutants:", (station_pollutants == 6).sum())

# 5. Candidate sensors with at least one year of history
eligible = df[
    (df["history_days"] >= 365)
    & df["sensor_first_utc"].notna()
    & df["sensor_last_utc"].notna()
]

print("\n5. SENSORS WITH AT LEAST ONE YEAR OF HISTORY")
print("Total:", len(eligible))
print(eligible["parameter"].value_counts().to_string())

# 6. Export detailed inspection
output = ROOT / "data" / "processed" / "sensor_quality_inspection.csv"
df.to_csv(output, index=False)

print("\nSaved:", output)
print("Inspection completed.")
