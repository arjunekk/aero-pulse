"""Central configuration: the assigned attributes and reproducibility settings."""

from __future__ import annotations

# Attributes fixed by the assigned topic.
FEATURES: list[str] = ["PM2.5", "PM10", "NO2", "TEMP"]
# Concentrations are strictly non-negative and heavily right-skewed -> log-transformed.
# Temperature is roughly symmetric and can be negative -> left as is.
POLLUTANTS: list[str] = ["PM2.5", "PM10", "NO2"]

RANDOM_STATE: int = 42

# Okabe-Ito colour-blind-safe palette (used for clusters), noise is always grey.
CLUSTER_COLORS: list[str] = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
]
NOISE_COLOR: str = "#8a8a8a"

SEASONS: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASON_ORDER: list[str] = ["Winter", "Spring", "Summer", "Autumn"]
