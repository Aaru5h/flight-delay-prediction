# Phase 2 — Feature manifest

Canonical manifest: `data/processed/phase2_feature_manifest.json` (version 2.2.0; SHA-256 `bf461f00978c9369de36b1847a1bf1ac9a248fa5f12664b4e582ac26c87d3e3b`).

37 estimator inputs. Audit fields and the target are outside the model allowlist.

| Feature | Type | Source | T−60 availability rule |
|---|---|---|---|
| `carrier` | categorical | published flight schedule | known by T-60 |
| `origin` | categorical | published flight schedule | known by T-60 |
| `dest` | categorical | published flight schedule | known by T-60 |
| `route` | categorical | published flight schedule | known by T-60 |
| `distance` | numeric | published flight schedule | known by T-60 |
| `sched_dep_hour` | numeric | published flight schedule | known by T-60 |
| `sched_day_of_week` | numeric | published flight schedule | known by T-60 |
| `sched_month` | numeric | published flight schedule | known by T-60 |
| `sched_day_of_year` | numeric | published flight schedule | known by T-60 |
| `sched_hour_sin` | numeric | published flight schedule | known by T-60 |
| `sched_hour_cos` | numeric | published flight schedule | known by T-60 |
| `sched_dow_sin` | numeric | published flight schedule | known by T-60 |
| `sched_dow_cos` | numeric | published flight schedule | known by T-60 |
| `scheduled_duration_minutes` | numeric | published schedule + destination airport timezone | convert scheduled New York departure and destination-local scheduled arrival to UTC; known by T-60 |
| `origin_scheduled_hour_volume` | numeric | published flight schedule | known by T-60 |
| `origin_airport_lat` | numeric | airports reference | static reference joined by scheduled origin/destination |
| `origin_airport_lon` | numeric | airports reference | static reference joined by scheduled origin/destination |
| `origin_airport_alt` | numeric | airports reference | static reference joined by scheduled origin/destination |
| `destination_airport_lat` | numeric | airports reference | static reference joined by scheduled origin/destination |
| `destination_airport_lon` | numeric | airports reference | static reference joined by scheduled origin/destination |
| `destination_airport_alt` | numeric | airports reference | static reference joined by scheduled origin/destination |
| `temp` | numeric | weather | latest origin observation at/before T-60, maximum age 180 minutes |
| `dewp` | numeric | weather | latest origin observation at/before T-60, maximum age 180 minutes |
| `humid` | numeric | weather | latest origin observation at/before T-60, maximum age 180 minutes |
| `wind_dir` | numeric | weather | latest origin observation at/before T-60, maximum age 180 minutes |
| `wind_speed` | numeric | weather | latest origin observation at/before T-60, maximum age 180 minutes |
| `wind_gust` | numeric | weather | latest origin observation at/before T-60, maximum age 180 minutes |
| `precip` | numeric | weather | latest origin observation at/before T-60, maximum age 180 minutes |
| `pressure` | numeric | weather | latest origin observation at/before T-60, maximum age 180 minutes |
| `visib` | numeric | weather | latest origin observation at/before T-60, maximum age 180 minutes |
| `weather_stale_or_missing` | numeric | weather | latest origin observation at/before T-60, maximum age 180 minutes |
| `carrier_prior_count` | numeric | label history | training-date outcomes only; earlier calendar dates and conservative availability: scheduled NYC departure + 24 hours + positive arrival delay <= T-60 |
| `carrier_delay_rate` | numeric | label history | training-date outcomes only; earlier calendar dates and conservative availability: scheduled NYC departure + 24 hours + positive arrival delay <= T-60 |
| `carrier_history_missing` | numeric | label history | training-date outcomes only; earlier calendar dates and conservative availability: scheduled NYC departure + 24 hours + positive arrival delay <= T-60 |
| `route_prior_count` | numeric | label history | training-date outcomes only; earlier calendar dates and conservative availability: scheduled NYC departure + 24 hours + positive arrival delay <= T-60 |
| `route_delay_rate` | numeric | label history | training-date outcomes only; earlier calendar dates and conservative availability: scheduled NYC departure + 24 hours + positive arrival delay <= T-60 |
| `route_history_missing` | numeric | label history | training-date outcomes only; earlier calendar dates and conservative availability: scheduled NYC departure + 24 hours + positive arrival delay <= T-60 |

## Limits

- Weather values are retrospective observations, not archived forecast vintages.
- Historical rates are frozen to training-date outcomes after 2013-08-31 so validation/test labels never become features.
- Historical arrival availability is conservatively estimated from scheduled NYC departure plus 24 hours and positive arrival delay; no release timestamp is supplied.
- Aircraft/tail-number features are excluded because assignment availability at T-60 is not established.
