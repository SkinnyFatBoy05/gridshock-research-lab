# Data sources and provenance

## Source register

| Data | Endpoint | Fields used | Licence / attribution |
|---|---|---|---|
| DE-LU day-ahead price | [Energy-Charts v2 price API](https://api.energy-charts.info/) | Hourly `day_ahead_price`, EUR/MWh | CC BY 4.0, Bundesnetzagentur / SMARD.de, delivered through Fraunhofer ISE Energy-Charts |
| Archived weather forecast | [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | `temperature_2m_previous_day2`, `wind_speed_100m_previous_day2`, `shortwave_radiation_previous_day2` | CC BY 4.0, attribution Open-Meteo.com |

Weather grid points represent Berlin (52.52, 13.41), Hamburg (53.55, 9.99), Frankfurt
(50.11, 8.68), and Munich (48.14, 11.58). Regional features are unweighted means of these
four points; they are weather proxies, not a physical load or renewable-generation model.

## Study window

The derived fixture covers delivery dates 2025-01-01 through 2025-09-30 and contains 6,551 hourly
rows after availability-safe feature construction. The common window was selected because archived
fixed-lag weather fields are complete there and because the European day-ahead product changed from
hourly to 15-minute market time units on 2025-10-01. Extending the dataset beyond that date requires
an explicit resampling and product-definition decision; the code does not silently mix resolutions.

## Acquisition controls

- One request spans at most 31 inclusive days.
- Responses must be UTF-8 JSON with the expected root, schema fields, timezone, and units.
- Transport timeouts and at most three attempts bound network behaviour.
- HTTP 429 honours a capped `Retry-After`; transient failures use bounded backoff.
- Each request stores canonical parameters, endpoint, retrieval timestamp, request fingerprint,
  payload SHA-256, source name, licence, timezone, unit, and resolution.
- Parsed observations store their retrieval identifier and pass uniqueness and point-in-time checks.

The exact 45-request register is committed in [`../data/demo/manifest.json`](../data/demo/manifest.json).
The derived gzip file has SHA-256
`be125fe72e9327e8af39ea03c6f0ff4f0b5f5cd9478b8466e6b8ba8f5b9de903`.

## Attribution and reuse

Project code is MIT-licensed. Source-data attribution and terms remain those of the providers and
are preserved per request in the manifest. Users redistributing or extending the data should review
the current provider terms themselves.
