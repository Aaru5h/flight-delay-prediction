"""Canonical Phase 2 estimator allowlist, denylist and feature metadata."""

DENYLIST = ["dep_time", "dep_delay", "arr_time", "arr_delay", "air_time", "tailnum"]

CATEGORICAL_FEATURES = ["carrier", "origin", "dest", "route"]

NUMERIC_FEATURES = [
    "distance", "sched_dep_hour", "sched_day_of_week", "sched_month",
    "sched_day_of_year", "sched_hour_sin", "sched_hour_cos", "sched_dow_sin",
    "sched_dow_cos", "scheduled_duration_minutes", "origin_scheduled_hour_volume",
    "origin_airport_lat", "origin_airport_lon", "origin_airport_alt",
    "destination_airport_lat", "destination_airport_lon", "destination_airport_alt",
    "temp", "dewp", "humid", "wind_dir", "wind_speed", "wind_gust", "precip",
    "pressure", "visib", "weather_stale_or_missing", "carrier_prior_count",
    "carrier_delay_rate", "carrier_history_missing", "route_prior_count",
    "route_delay_rate", "route_history_missing",
]

ALLOWLIST = CATEGORICAL_FEATURES + NUMERIC_FEATURES
