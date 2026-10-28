# Phase 2 — Feature Manifest

## 1. Feature Allowlist and Specifications

| Feature | Type | Source | Availability at T-60 | Missing Policy |
|---|---|---|---|---|
| `carrier` | categorical | flights.carrier | Published airline schedule at ticket purchase (T-days) | never_missing |
| `origin` | categorical | flights.origin | Published departure airport (EWR/JFK/LGA) | never_missing |
| `dest` | categorical | flights.dest | Published destination airport code | never_missing |
| `sched_dep_hour` | integer | flights.sched_dep_time | Scheduled departure time (T-60) | never_missing |
| `sched_dep_minute_of_day` | integer | flights.sched_dep_minute | Minutes since midnight for scheduled departure | never_missing |
| `sched_dep_dayofweek` | integer | flights.sched_date | Calendar day of week (0=Mon, 6=Sun) | never_missing |
| `sched_dep_day` | integer | flights.sched_date | Day of month (1-31) | never_missing |
| `sched_dep_month` | integer | flights.sched_date | Month of year (1-12) | never_missing |
| `is_weekend` | binary | flights.sched_date | Weekend indicator (Saturday or Sunday) | never_missing |
| `sin_hour` | float | computed | Cyclical transformation of sched_dep_hour | never_missing |
| `cos_hour` | float | computed | Cyclical transformation of sched_dep_hour | never_missing |
| `sin_dayofweek` | float | computed | Cyclical transformation of day of week | never_missing |
| `cos_dayofweek` | float | computed | Cyclical transformation of day of week | never_missing |
| `sin_month` | float | computed | Cyclical transformation of month | never_missing |
| `cos_month` | float | computed | Cyclical transformation of month | never_missing |
| `distance` | float | flights.distance | Scheduled flight distance in miles | never_missing |
| `sched_flight_duration` | float | flights.sched_dep_ts, sched_arr_ts | Scheduled duration in minutes from timetable | never_missing |
| `origin_hourly_flight_volume` | integer | flights.origin, sched_dep_ts | Published timetable departure count for origin airport in same hour | never_missing |
| `dest_lat` | float | airports.lat | Airport geographic latitude lookup | median_imputed_or_indicator |
| `dest_lon` | float | airports.lon | Airport geographic longitude lookup | median_imputed_or_indicator |
| `dest_alt` | float | airports.alt | Airport elevation in feet lookup | median_imputed_or_indicator |
| `dest_tz` | float | airports.tz | Airport timezone offset lookup | median_imputed_or_indicator |
| `dest_unmatched` | binary | computed | Binary flag for destinations absent from FAA lookup | never_missing |
| `temp` | float | weather.temp | Latest origin weather observation with timestamp <= T-60 (<= 180m old) | median_imputed_in_pipeline |
| `dewp` | float | weather.dewp | Latest origin weather observation with timestamp <= T-60 (<= 180m old) | median_imputed_in_pipeline |
| `humid` | float | weather.humid | Latest origin weather observation with timestamp <= T-60 (<= 180m old) | median_imputed_in_pipeline |
| `wind_dir` | float | weather.wind_dir | Latest origin weather observation with timestamp <= T-60 (<= 180m old) | median_imputed_in_pipeline |
| `wind_speed` | float | weather.wind_speed | Latest origin weather observation with timestamp <= T-60 (<= 180m old) | median_imputed_in_pipeline |
| `wind_gust` | float | weather.wind_gust | Latest origin weather observation with timestamp <= T-60 (<= 180m old) | zero_or_median_imputed |
| `precip` | float | weather.precip | Latest origin weather observation with timestamp <= T-60 (<= 180m old) | zero_imputed |
| `pressure` | float | weather.pressure | Latest origin weather observation with timestamp <= T-60 (<= 180m old) | median_imputed_in_pipeline |
| `visib` | float | weather.visib | Latest origin weather observation with timestamp <= T-60 (<= 180m old) | median_imputed_in_pipeline |
| `weather_age_minutes` | float | computed | Age of origin weather observation at T-60 cutoff | median_imputed_or_flag |
| `weather_missing_or_stale` | binary | computed | Indicator if weather was missing or older than 180 minutes | never_missing |
| `carrier_prior_delay_rate` | float | historical_aggregates | Carrier prior delay rate from dates < target_date and actual arrival <= T-60 | smoothed_with_global_prior |
| `carrier_prior_count` | integer | historical_aggregates | Count of prior flights for carrier available at T-60 | zero_when_none |
| `has_prior_carrier_history` | binary | computed | Flag indicating carrier has >= 5 prior observations | never_missing |
| `route_prior_delay_rate` | float | historical_aggregates | Route prior delay rate from dates < target_date and actual arrival <= T-60 | smoothed_with_global_prior |
| `route_prior_count` | integer | historical_aggregates | Count of prior flights for route available at T-60 | zero_when_none |
| `has_prior_route_history` | binary | computed | Flag indicating route has >= 5 prior observations | never_missing |
| `origin_prior_delay_rate` | float | historical_aggregates | Origin prior delay rate from dates < target_date and actual arrival <= T-60 | smoothed_with_global_prior |
| `origin_prior_count` | integer | historical_aggregates | Count of prior flights for origin available at T-60 | zero_when_none |
| `has_prior_origin_history` | binary | computed | Flag indicating origin has >= 5 prior observations | never_missing |

## 2. Denylist (Prohibited Post-Departure Columns)

- `dep_time` (strictly prohibited; leakage-prone)
- `dep_delay` (strictly prohibited; leakage-prone)
- `arr_time` (strictly prohibited; leakage-prone)
- `arr_delay` (strictly prohibited; leakage-prone)
- `air_time` (strictly prohibited; leakage-prone)
- `tailnum` (strictly prohibited; leakage-prone)
