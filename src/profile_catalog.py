
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "processed" / "indian_sensor_catalog.csv"
OUTPUT = ROOT / "data" / "processed"

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]


def main():

    df = pd.read_csv(INPUT)

    print("\nSTATION-LEVEL DATASET PROFILE")
    print("=" * 55)

    # 1. Basic statistics
    print("\nBasic statistics")
    print("Sensor records:", len(df))
    print("Unique stations:", df["location_id"].nunique())
    print("Unique sensors:", df["sensor_id"].nunique())

    # 2. Number of sensors per pollutant
    pollutant_counts = (
        df[df["parameter"].isin(POLLUTANTS)]
        .groupby("parameter")["sensor_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    print("\nSensors per pollutant:")
    print(pollutant_counts.to_string())

    # 3. Create station-pollutant matrix
    pollutant_df = df[df["parameter"].isin(POLLUTANTS)]

    matrix = pd.crosstab(
        pollutant_df["location_id"],
        pollutant_df["parameter"]
    )

    # Convert sensor counts into availability flags
    matrix = (matrix > 0).astype(int)

    # Ensure all six columns exist
    matrix = matrix.reindex(
        columns=POLLUTANTS,
        fill_value=0
    )

    matrix.to_csv(
        OUTPUT / "location_parameter_matrix.csv"
    )

    # 4. Number of pollutants available per station
    matrix["pollutant_count"] = matrix.sum(axis=1)

    print("\nStations by number of available pollutants:")
    print(
        matrix["pollutant_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # 5. Stations with multiple pollutants
    multi = matrix[matrix["pollutant_count"] >= 3]

    print(
        "\nStations with at least 3 pollutants:",
        len(multi)
    )

    print("\nTop stations by pollutant coverage:")
    print(
        matrix.sort_values(
            "pollutant_count",
            ascending=False
        ).head(20).to_string()
    )

    # 6. Station summary
    summary = (
        df.groupby("location_id")
        .agg(
            location_name=("location_name", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            timezone=("timezone", "first"),
            location_first_measurement=(
                "first_measurement", "first"
            ),
            location_last_measurement=(
                "last_measurement", "first"
            ),
            sensor_count=("sensor_id", "nunique")
        )
        .reset_index()
    )

    summary = summary.merge(
        matrix.reset_index(),
        on="location_id",
        how="left"
    )

    summary.to_csv(
        OUTPUT / "location_summary.csv",
        index=False
    )

    # 7. Parameter overlap
    print("\nPairwise pollutant station overlap:")

    for i, p1 in enumerate(POLLUTANTS):

        for p2 in POLLUTANTS[i + 1:]:

            overlap = (
                (matrix[p1] == 1) &
                (matrix[p2] == 1)
            ).sum()

            print(f"{p1} + {p2}: {overlap} stations")

    # 8. Date range information
    print("\nLocation-level date coverage:")

    print(
        "Earliest recorded first measurement:",
        summary["location_first_measurement"].min()
    )

    print(
        "Latest recorded last measurement:",
        summary["location_last_measurement"].max()
    )

    print(
        "\nStations without location date information:",
        summary["location_first_measurement"].isna().sum()
    )

    print("\nGenerated files:")
    print("location_parameter_matrix.csv")
    print("location_summary.csv")

    print("\nProfiling completed.")


if __name__ == "__main__":
    main()
