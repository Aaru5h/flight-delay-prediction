"""Path and schema configuration for the data-preparation pipeline.

Directories can be overridden with environment variables so the pipeline is
not hard-coded to one developer's machine (Phase 0 contract, artifact schemas).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _from_env(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, default))


RAW_DATA_DIR = _from_env("FLIGHT_DELAY_RAW_DIR", PROJECT_ROOT / "data" / "raw")
PROCESSED_DATA_DIR = _from_env(
    "FLIGHT_DELAY_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"
)
REPORTS_DIR = _from_env("FLIGHT_DELAY_REPORTS_DIR", PROJECT_ROOT / "reports")
CONFIG_DIR = PROJECT_ROOT / "configs"
CONTRACT_PATH = CONFIG_DIR / "project_contract.yaml"
DATA_MANIFEST_PATH = CONFIG_DIR / "data_manifest.json"

SOURCE_FILES = {
    "flights": "nyc_flights.csv",
    "airlines": "nyc_airlines.csv",
    "airports": "nyc_airports.csv",
    "planes": "nyc_planes.csv",
    "weather": "nyc_weather.csv",
}

REQUIRED_COLUMNS = {
    "flights": [
        "year", "month", "day", "dep_time", "sched_dep_time", "dep_delay",
        "arr_time", "sched_arr_time", "arr_delay", "carrier", "flight",
        "tailnum", "origin", "dest", "air_time", "distance", "hour",
        "minute", "time_hour",
    ],
    "airlines": ["carrier", "name"],
    "airports": ["faa", "name", "lat", "lon", "alt", "tz", "dst", "tzone"],
    "planes": [
        "tailnum", "year", "type", "manufacturer", "model", "engines",
        "seats", "speed", "engine",
    ],
    "weather": [
        "origin", "year", "month", "day", "hour", "temp", "dewp", "humid",
        "wind_dir", "wind_speed", "wind_gust", "precip", "pressure",
        "visib", "time_hour",
    ],
}

# Identifier-style columns read as strings so leading zeros and non-numeric
# codes survive (e.g. airport FAA code "04G").
STRING_COLUMNS = {
    "flights": ["carrier", "tailnum", "origin", "dest", "time_hour"],
    "airlines": ["carrier", "name"],
    "airports": ["faa", "name", "dst", "tzone"],
    "planes": ["tailnum", "type", "manufacturer", "model", "engine"],
    "weather": ["origin", "time_hour"],
}

# Integer columns that contain missing values in the supplied files.
NULLABLE_INT_COLUMNS = {
    "flights": ["dep_time", "sched_dep_time", "arr_time", "sched_arr_time"],
    "airports": ["alt", "tz"],
    "planes": ["year", "engines", "seats"],
    "weather": [],
    "airlines": [],
}
