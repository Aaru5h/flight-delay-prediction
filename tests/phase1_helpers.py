"""Shared helpers that build tiny synthetic raw CSVs for Phase 1 tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FLIGHT_HEADER = [
    "year", "month", "day", "dep_time", "sched_dep_time", "dep_delay",
    "arr_time", "sched_arr_time", "arr_delay", "carrier", "flight",
    "tailnum", "origin", "dest", "air_time", "distance", "hour", "minute",
    "time_hour",
]


def flights_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a flights frame with all 19 documented columns."""
    base = {col: pd.NA for col in FLIGHT_HEADER}
    out = []
    for row in rows:
        record = dict(base)
        record.update(row)
        out.append(record)
    return pd.DataFrame(out, columns=FLIGHT_HEADER)


def write_synthetic_sources(raw_dir: Path) -> None:
    """Write the five raw CSVs with a handful of coherent rows."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    flights = flights_rows([
        # on-time arrival (below threshold)
        dict(year=2013, month=1, day=1, sched_dep_time=515, sched_arr_time=819,
             arr_delay=11.0, carrier="UA", flight=1545, tailnum="N14228",
             origin="EWR", dest="IAH", distance=1400, hour=5, minute=15,
             time_hour="2013-01-01T10:00:00Z"),
        # substantial delay (>= 15)
        dict(year=2013, month=1, day=1, sched_dep_time=529, sched_arr_time=830,
             arr_delay=20.0, carrier="UA", flight=1714, tailnum="N24211",
             origin="LGA", dest="IAH", distance=1416, hour=5, minute=29,
             time_hour="2013-01-01T10:00:00Z"),
        # exactly at the 15-minute boundary
        dict(year=2013, month=2, day=2, sched_dep_time=600, sched_arr_time=730,
             arr_delay=15.0, carrier="AA", flight=1, tailnum="N1AA",
             origin="JFK", dest="MIA", distance=1090, hour=6, minute=0,
             time_hour="2013-02-02T11:00:00Z"),
        # unknown label (missing arr_delay) and an invalid scheduled time
        dict(year=2013, month=3, day=3, sched_dep_time=2400, sched_arr_time=930,
             carrier="B6", flight=25, tailnum=pd.NA,
             origin="JFK", dest="BQN", distance=1570, hour=7, minute=0,
             time_hour="2013-03-03T12:00:00Z"),
        # early arrival, negative delay
        dict(year=2013, month=3, day=3, sched_dep_time=130, sched_arr_time=2250,
             arr_delay=-7.0, carrier="AA", flight=2, tailnum="N1AA",
             origin="EWR", dest="STT", distance=1634, hour=1, minute=30,
             time_hour="2013-03-03T06:00:00Z"),
    ])
    flights.to_csv(raw_dir / "nyc_flights.csv", index=False)

    pd.DataFrame({
        "carrier": ["UA", "AA", "B6", "9E"],
        "name": ["United Air Lines Inc.", "American Airlines Inc.",
                 "JetBlue Airways", "Endeavor Air Inc."],
    }).to_csv(raw_dir / "nyc_airlines.csv", index=False)

    pd.DataFrame({
        "faa": ["EWR", "JFK", "LGA", "IAH", "MIA", "04G"],
        "name": ["Newark Liberty Intl", "John F Kennedy Intl",
                 "La Guardia", "George Bush Intercontinental",
                 "Miami Intl", "Lansdowne Airport"],
        "lat": [40.69, 40.64, 40.78, 29.98, 25.79, 41.13],
        "lon": [-74.17, -73.78, -73.87, -95.34, -80.29, -80.62],
        "alt": [18, 13, 21, 97, 8, 1044],
        "tz": [-5, -5, -5, -6, -5, -5],
        "dst": ["A"] * 6,
        "tzone": ["America/New_York"] * 5 + ["America/New_York"],
    }).to_csv(raw_dir / "nyc_airports.csv", index=False)

    pd.DataFrame({
        "tailnum": ["N14228", "N24211", "N1AA"],
        "year": [1999, 1999, 1998],
        "type": ["Fixed wing multi engine"] * 3,
        "manufacturer": ["BOEING", "BOEING", "AIRBUS INDUSTRIE"],
        "model": ["737-724", "737-824", "A320-214"],
        "engines": [2, 2, 2],
        "seats": [149, 178, 182],
        "speed": [pd.NA, pd.NA, pd.NA],
        "engine": ["Turbo-fan"] * 3,
    }).to_csv(raw_dir / "nyc_planes.csv", index=False)

    pd.DataFrame({
        "origin": ["EWR", "EWR", "JFK"],
        "year": [2013] * 3,
        "month": [1, 1, 2],
        "day": [1, 1, 2],
        "hour": [1, 2, 6],
        "temp": [39.02, 39.02, 41.0],
        "dewp": [26.06, 26.96, 28.0],
        "humid": [59.37, 61.63, 60.0],
        "wind_dir": [270, 250, 260],
        "wind_speed": [10.36, 8.06, 9.0],
        "wind_gust": [pd.NA, pd.NA, pd.NA],
        "precip": [0, 0, 0],
        "pressure": [1012.0, 1012.3, 1011.9],
        "visib": [10, 10, 10],
        "time_hour": ["2013-01-01T06:00:00Z", "2013-01-01T07:00:00Z",
                      "2013-02-02T11:00:00Z"],
    }).to_csv(raw_dir / "nyc_weather.csv", index=False)
