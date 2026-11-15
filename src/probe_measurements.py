
import os
import time
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError("OPENAQ_API_KEY missing from .env")

INPUT = ROOT / "data" / "processed" / "sensor_coverage.csv"

BASE_URL = "https://api.openaq.org/v3/sensors"

HEADERS = {"X-API-Key": API_KEY}

# Pilot window: one calendar year
DATE_FROM = "2025-01-01"
DATE_TO = "2025-12-31"

session = requests.Session()
session.headers.update(HEADERS)


def fetch_daily(sensor_id):

    url = f"{BASE_URL}/{sensor_id}/days"

    params = {
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "limit": 1000,
        "page": 1
    }

    for attempt in range(5):

        try:
            response = session.get(
                url,
                params=params,
                timeout=60
            )

            if response.status_code == 429:
                wait = int(
                    response.headers.get("Retry-After", 10)
                )
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:

            if attempt == 4:
                return {"error": str(error)}

            time.sleep(2 ** attempt)

    return {"error": "Maximum retries exceeded"}


def main():

    df = pd.read_csv(INPUT)

    df["sensor_first_utc"] = pd.to_datetime(
        df["sensor_first_utc"],
        utc=True,
        errors="coerce"
    )

    df["sensor_last_utc"] = pd.to_datetime(
        df["sensor_last_utc"],
        utc=True,
        errors="coerce"
    )

    df["history_days"] = (
        df["sensor_last_utc"] - df["sensor_first_utc"]
    ).dt.total_seconds() / 86400

    candidates = df[
        (df["history_days"] >= 365)
        & df["sensor_first_utc"].notna()
        & df["sensor_last_utc"].notna()
    ].copy()

    # Test up to two sensors for every pollutant/unit combination
    # This lets us inspect both unit variants.
    samples = (
        candidates.sort_values(
            "history_days",
            ascending=False
        )
        .groupby(["parameter", "unit"], group_keys=False)
        .head(2)
    )

    print("\nOPENAQ DAILY MEASUREMENT PILOT")
    print("=" * 65)
    print("Date window:", DATE_FROM, "to", DATE_TO)
    print("Sensors selected:", len(samples))

    results = []

    for index, row in enumerate(
        samples.itertuples(index=False), 1
    ):

        print(
            f"\n[{index}/{len(samples)}] "
            f"{row.parameter} | {row.unit} | "
            f"Sensor {row.sensor_id}"
        )

        response = fetch_daily(row.sensor_id)

        if "error" in response:
            print("ERROR:", response["error"])
            continue

        records = response.get("results", [])
        meta = response.get("meta", {})

        print("API records found:", meta.get("found"))
        print("Records returned:", len(records))

        if records:

            first = records[0]

            print("First record:", first)

            valid_values = [
                r.get("value")
                for r in records
                if r.get("value") is not None
            ]

            print("Non-null daily values:", len(valid_values))

            if valid_values:
                print(
                    "Value range:",
                    min(valid_values),
                    "to",
                    max(valid_values)
                )

        results.append({
            "sensor_id": row.sensor_id,
            "location_id": row.location_id,
            "parameter": row.parameter,
            "unit": row.unit,
            "api_found": meta.get("found"),
            "records_returned": len(records),
            "non_null_values": sum(
                r.get("value") is not None for r in records
            ),
            "error": None
        })

        time.sleep(0.3)

    output = ROOT / "data" / "processed" / "measurement_pilot.csv"

    pd.DataFrame(results).to_csv(output, index=False)

    print("\nPILOT SUMMARY")
    print("=" * 65)
    print(pd.DataFrame(results).to_string(index=False))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
