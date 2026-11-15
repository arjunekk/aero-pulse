
import os
import time
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------
# 1. Configuration
# -----------------------------------------

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENAQ_API_KEY")

BASE_URL = "https://api.openaq.org/v3/locations"

INPUT = ROOT / "data" / "processed" / "indian_sensor_catalog.csv"
OUTPUT = ROOT / "data" / "processed"

CHECKPOINT = OUTPUT / "sensor_coverage_checkpoint.csv"
FINAL_FILE = OUTPUT / "sensor_coverage.csv"

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

if not API_KEY:
    raise ValueError("OPENAQ_API_KEY missing from .env")

HEADERS = {"X-API-Key": API_KEY}

session = requests.Session()
session.headers.update(HEADERS)


# -----------------------------------------
# 2. API request with retry handling
# -----------------------------------------

def fetch_location_sensors(location_id):

    url = f"{BASE_URL}/{location_id}/sensors"

    for attempt in range(5):

        try:
            response = session.get(url, timeout=60)

            if response.status_code == 429:
                wait = int(
                    response.headers.get("Retry-After", 10)
                )
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            return response.json().get("results", [])

        except requests.RequestException as error:

            if attempt == 4:
                print(f"Failed location {location_id}: {error}")
                return None

            time.sleep(2 ** attempt)

    return None


# -----------------------------------------
# 3. Extract sensor-level coverage
# -----------------------------------------

def extract_sensor_data(location, sensors):

    rows = []

    for sensor in sensors:

        parameter = sensor.get("parameter", {})
        name = parameter.get("name")

        if name not in POLLUTANTS:
            continue

        coverage = sensor.get("coverage") or {}
        summary = sensor.get("summary") or {}

        first = sensor.get("datetimeFirst") or {}
        last = sensor.get("datetimeLast") or {}

        rows.append({

            "location_id": location["location_id"],
            "location_name": location["location_name"],

            "latitude": location["latitude"],
            "longitude": location["longitude"],

            "sensor_id": sensor.get("id"),
            "parameter": name,
            "unit": parameter.get("units"),

            "sensor_first_utc": first.get("utc"),
            "sensor_last_utc": last.get("utc"),

            "expected_count": coverage.get("expectedCount"),
            "observed_count": coverage.get("observedCount"),

            "percent_complete": coverage.get("percentComplete"),
            "percent_coverage": coverage.get("percentCoverage"),

            "coverage_from": (
                coverage.get("datetimeFrom", {}).get("utc")
                if coverage.get("datetimeFrom") else None
            ),

            "coverage_to": (
                coverage.get("datetimeTo", {}).get("utc")
                if coverage.get("datetimeTo") else None
            ),

            "average_value": summary.get("avg"),
            "minimum_value": summary.get("min"),
            "maximum_value": summary.get("max"),

        })

    return rows


# -----------------------------------------
# 4. Main audit
# -----------------------------------------

def main():

    catalog = pd.read_csv(INPUT)

    # Select locations with at least 3 target pollutants
    matrix = pd.crosstab(
        catalog[catalog["parameter"].isin(POLLUTANTS)][
            "location_id"
        ],
        catalog[catalog["parameter"].isin(POLLUTANTS)][
            "parameter"
        ]
    )

    matrix = matrix.reindex(
        columns=POLLUTANTS,
        fill_value=0
    )

    candidate_ids = matrix[
        matrix.gt(0).sum(axis=1) >= 3
    ].index.astype(str).tolist()

    location_info = (
        catalog.groupby("location_id")
        .agg(
            location_name=("location_name", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first")
        )
    )

    print("\nOPENAQ SENSOR COVERAGE AUDIT")
    print("=" * 50)

    print("Candidate locations:", len(candidate_ids))

    # Resume from checkpoint if available
    if CHECKPOINT.exists():

        existing = pd.read_csv(CHECKPOINT)

        completed_ids = set(
            existing["location_id"].astype(str)
        )

        rows = existing.to_dict("records")

        print("Previously completed locations:", len(completed_ids))

    else:

        completed_ids = set()
        rows = []

    errors = []

    for index, location_id in enumerate(candidate_ids, 1):

        if location_id in completed_ids:
            continue

        info = location_info.loc[int(location_id)]

        location = {
            "location_id": location_id,
            "location_name": info["location_name"],
            "latitude": info["latitude"],
            "longitude": info["longitude"]
        }

        sensors = fetch_location_sensors(location_id)

        if sensors is None:
            errors.append(location_id)
            continue

        sensor_rows = extract_sensor_data(location, sensors)

        rows.extend(sensor_rows)
        completed_ids.add(location_id)

        if index % 25 == 0:

            pd.DataFrame(rows).to_csv(
                CHECKPOINT, index=False
            )

            print(
                f"Progress: {index}/{len(candidate_ids)} "
                f"| Sensors collected: {len(rows)}"
            )

        time.sleep(0.15)

    # Save final results
    coverage_df = pd.DataFrame(rows)

    coverage_df.to_csv(FINAL_FILE, index=False)

    print("\nAUDIT SUMMARY")
    print("=" * 50)

    print("Locations completed:", len(completed_ids))
    print("Pollutant sensors audited:", len(coverage_df))

    if not coverage_df.empty:

        print("\nSensors by pollutant:")
        print(
            coverage_df["parameter"]
            .value_counts()
            .to_string()
        )

        print("\nMissing sensor date ranges:")
        print(
            coverage_df[
                ["sensor_first_utc", "sensor_last_utc"]
            ].isna().sum().to_string()
        )

        print("\nObserved measurements by pollutant:")
        print(
            coverage_df.groupby("parameter")[
                "observed_count"
            ].agg(["count", "min", "median", "max"])
            .to_string()
        )

        print("\nCoverage percentage distribution:")
        print(
            coverage_df.groupby("parameter")[
                "percent_coverage"
            ].describe().round(2).to_string()
        )

    print("\nLocations with API errors:", len(errors))

    if errors:
        print("Failed location IDs:", errors)

    print("\nSaved:", FINAL_FILE)
    print("Sensor audit completed.")


if __name__ == "__main__":
    main()
