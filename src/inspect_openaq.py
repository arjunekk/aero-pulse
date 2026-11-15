
import os
import json
import time
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------
# 1. Configuration
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENAQ_API_KEY")

BASE_URL = "https://api.openaq.org/v3/locations"

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

if not API_KEY:
    raise ValueError(
        "OPENAQ_API_KEY not found. Check your .env file."
    )

HEADERS = {
    "X-API-Key": API_KEY
}

# -----------------------------------
# 2. Fetch Indian monitoring stations
# -----------------------------------

def fetch_indian_locations():

    all_locations = []
    page = 1
    limit = 1000

    while True:

        params = {
            "iso": "IN",
            "limit": limit,
            "page": page
        }

        print(f"Fetching page {page}...")

        response = requests.get(
            BASE_URL,
            headers=HEADERS,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        payload = response.json()

        results = payload.get("results", [])

        if not results:
            break

        all_locations.extend(results)

        meta = payload.get("meta", {})
        total_found = meta.get("found")

        print(
            f"Retrieved {len(results)} locations. "
            f"Total collected: {len(all_locations)}"
        )

        if len(results) < limit:
            break

        if total_found is not None:
            if len(all_locations) >= int(total_found):
                break

        page += 1
        time.sleep(0.2)

    return all_locations


# -----------------------------------
# 3. Extract station and sensor data
# -----------------------------------

def create_sensor_catalog(locations):

    rows = []

    for location in locations:

        country = location.get("country", {})
        coordinates = location.get("coordinates", {})

        for sensor in location.get("sensors", []):

            parameter = sensor.get("parameter", {})

            rows.append({

                "location_id": location.get("id"),
                "location_name": location.get("name"),
                "city": location.get("locality"),

                "latitude": coordinates.get("latitude"),
                "longitude": coordinates.get("longitude"),

                "country": country.get("name"),
                "timezone": location.get("timezone"),

                "is_monitor": location.get("isMonitor"),

                "sensor_id": sensor.get("id"),
                "parameter": parameter.get("name"),
                "unit": parameter.get("units"),

                "first_measurement": (
                    (location.get("datetimeFirst") or {}).get("utc")
                ),

                "last_measurement": (
                    (location.get("datetimeLast") or {}).get("utc")
                )

            })

    return pd.DataFrame(rows)


# -----------------------------------
# 4. Main execution
# -----------------------------------

def main():

    print("\nOPENAQ DATA EXPLORATION")
    print("=" * 45)

    locations = fetch_indian_locations()

    print("\nTotal locations retrieved:", len(locations))

    # Preserve original API response
    raw_file = RAW_DIR / "indian_locations.json"

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(locations, file, indent=2)

    # Build sensor catalog
    df = create_sensor_catalog(locations)

    catalog_file = (
        PROCESSED_DIR / "indian_sensor_catalog.csv"
    )

    df.to_csv(catalog_file, index=False)

    print("\nSENSOR CATALOG")
    print("=" * 45)

    print("Total sensor records:", len(df))

    print("\nAvailable parameters:")
    print(df["parameter"].value_counts().to_string())

    print("\nTop locations:")
    print(
        df["location_name"]
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nFiles saved:")
    print(raw_file)
    print(catalog_file)

    print("\nExploration completed.")


if __name__ == "__main__":
    main()
