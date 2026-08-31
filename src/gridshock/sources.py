"""Bounded adapters for official public market and weather APIs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pandas as pd

from gridshock import __version__
from gridshock.contracts import DataContractError, assert_point_in_time, validate_canonical
from gridshock.provenance import RequestManifest, sha256_bytes
from gridshock.time import decision_cutoff_utc, parse_source_times

ENERGY_CHARTS_PRICE_URL = "https://api.energy-charts.info/v2/price"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
WEATHER_FIELDS = (
    "temperature_2m_previous_day2",
    "wind_speed_100m_previous_day2",
    "shortwave_radiation_previous_day2",
)
WEATHER_FIELD_UNITS = {
    "temperature_2m_previous_day2": "°C",
    "wind_speed_100m_previous_day2": "km/h",
    "shortwave_radiation_previous_day2": "W/m²",
}


@dataclass(frozen=True)
class Location:
    """Named weather grid point."""

    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class SourcePayload:
    """Raw response bytes paired with an auditable manifest."""

    payload: bytes
    manifest: RequestManifest


def _validate_date_range(start: date, end: date) -> None:
    if end < start:
        raise ValueError("end date must not precede start date")
    inclusive_days = (end - start).days + 1
    if inclusive_days > 31:
        raise ValueError("a source request may cover at most 31 days")


class _HttpClient:
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._transport = transport
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def now(self) -> datetime:
        return self._clock()

    def get(self, url: str, params: Mapping[str, str]) -> bytes:
        headers = {"User-Agent": f"GridShock-Research-Lab/{__version__} (+research; no trading)"}
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                with httpx.Client(
                    timeout=httpx.Timeout(15.0),
                    headers=headers,
                    transport=self._transport,
                    follow_redirects=True,
                ) as client:
                    response = client.get(url, params=params)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500:
                    raise
                last_error = error
            except httpx.TransportError as error:
                last_error = error
        raise RuntimeError(f"source request failed after 3 attempts: {url}") from last_error


class EnergyChartsClient:
    """Client for the Energy-Charts v2 DE-LU price endpoint."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._http = _HttpClient(transport=transport, clock=clock)

    def fetch_prices(self, start: date, end: date) -> SourcePayload:
        _validate_date_range(start, end)
        params = {"bzn": "DE-LU", "start": start.isoformat(), "end": end.isoformat()}
        payload = self._http.get(ENERGY_CHARTS_PRICE_URL, params)
        document = _decode_document(payload)
        _require_equal(document, "schema_version", "2.0")
        _require_equal(document, "bidding_zone", "DE-LU")
        _require_equal(document, "resolution", "PT1H")
        unit = _normalise_price_unit(_require_string(document, "unit"))
        manifest = RequestManifest.from_payload(
            source="Fraunhofer ISE Energy-Charts / SMARD",
            url=ENERGY_CHARTS_PRICE_URL,
            params=params,
            payload=payload,
            retrieved_at=self._http.now,
            license=_require_string(document, "license"),
            timezone_name=_require_string(document, "timezone"),
            unit=unit,
            resolution="hourly",
        )
        return SourcePayload(payload=payload, manifest=manifest)


class OpenMeteoClient:
    """Client for point-in-time-safe fixed-lag archived weather forecasts."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._http = _HttpClient(transport=transport, clock=clock)

    def fetch_previous_runs(self, location: Location, start: date, end: date) -> SourcePayload:
        _validate_date_range(start, end)
        params = {
            "latitude": str(location.latitude),
            "longitude": str(location.longitude),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(WEATHER_FIELDS),
            "timezone": "UTC",
        }
        payload = self._http.get(OPEN_METEO_PREVIOUS_RUNS_URL, params)
        document = _decode_document(payload)
        _require_equal(document, "utc_offset_seconds", 0)
        timezone_name = _require_string(document, "timezone")
        if timezone_name not in {"GMT", "UTC"}:
            raise DataContractError(
                f"weather response timezone must be UTC/GMT, received {timezone_name}"
            )
        manifest = RequestManifest.from_payload(
            source="Open-Meteo Previous Runs API",
            url=OPEN_METEO_PREVIOUS_RUNS_URL,
            params=params,
            payload=payload,
            retrieved_at=self._http.now,
            license="CC BY 4.0; attribution: Open-Meteo.com",
            timezone_name="UTC",
            unit="multiple (see hourly_units)",
            resolution="hourly",
        )
        return SourcePayload(payload=payload, manifest=manifest)


def _decode_document(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataContractError("source payload is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise DataContractError("source payload root must be a JSON object")
    return value


def _require_string(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise DataContractError(f"source field {key} must be a non-empty string")
    return value


def _require_equal(document: Mapping[str, Any], key: str, expected: object) -> None:
    if document.get(key) != expected:
        raise DataContractError(
            f"source field {key} changed: expected {expected!r}, received {document.get(key)!r}"
        )


def _normalise_price_unit(raw_unit: str) -> str:
    normalised = raw_unit.replace(" ", "")
    if normalised != "EUR/MWh":
        raise DataContractError(f"unexpected price unit: {raw_unit}")
    return normalised


def _delivery_days(valid_times: pd.DatetimeIndex) -> list[date]:
    return [timestamp.date() for timestamp in valid_times.tz_convert("Europe/Berlin")]


def parse_energy_prices(payload: bytes) -> pd.DataFrame:
    """Parse an Energy-Charts v2 price payload into canonical observations."""

    document = _decode_document(payload)
    _require_equal(document, "schema_version", "2.0")
    _require_equal(document, "bidding_zone", "DE-LU")
    _require_equal(document, "resolution", "PT1H")
    unit = _normalise_price_unit(_require_string(document, "unit"))
    data = document.get("data")
    if not isinstance(data, list) or not data:
        raise DataContractError("price payload data must be a non-empty array")

    timestamps: list[str] = []
    values: list[float] = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("values"), dict):
            raise DataContractError("price data row has invalid shape")
        timestamps.append(_require_string(item, "timestamp"))
        price = item["values"].get("day_ahead_price")
        if not isinstance(price, int | float):
            raise DataContractError("price data row lacks numeric day_ahead_price")
        values.append(float(price))

    valid_times = parse_source_times(timestamps)
    cutoffs = pd.to_datetime(
        [decision_cutoff_utc(day) for day in _delivery_days(valid_times)], utc=True
    )
    retrieval_id = sha256_bytes(payload)
    frame = pd.DataFrame(
        {
            "valid_time_utc": valid_times,
            "available_at_utc": cutoffs + pd.Timedelta(hours=1),
            "cutoff_utc": cutoffs,
            "source": "Fraunhofer ISE Energy-Charts / SMARD",
            "series_id": "day_ahead_price",
            "value": values,
            "unit": unit,
            "retrieval_id": retrieval_id,
        }
    )
    validate_canonical(frame, expected_unit="EUR/MWh")
    return frame


def parse_weather_previous_runs(payload: bytes, location: Location) -> pd.DataFrame:
    """Parse Open-Meteo `previous_day2` fields into canonical long form."""

    document = _decode_document(payload)
    _require_equal(document, "utc_offset_seconds", 0)
    timezone_name = _require_string(document, "timezone")
    if timezone_name not in {"GMT", "UTC"}:
        raise DataContractError(
            f"weather response timezone must be UTC/GMT, received {timezone_name}"
        )
    hourly = document.get("hourly")
    units = document.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise DataContractError("weather payload requires hourly and hourly_units objects")
    raw_times = hourly.get("time")
    if not isinstance(raw_times, list) or not raw_times:
        raise DataContractError("weather hourly time must be a non-empty array")
    if not all(isinstance(value, str) for value in raw_times):
        raise DataContractError("weather hourly time values must be strings")
    valid_times = parse_source_times([f"{value}Z" for value in raw_times])
    cutoffs = pd.to_datetime(
        [decision_cutoff_utc(day) for day in _delivery_days(valid_times)], utc=True
    )
    availability = valid_times - pd.Timedelta(days=2)
    retrieval_id = sha256_bytes(payload)

    frames: list[pd.DataFrame] = []
    for field in WEATHER_FIELDS:
        expected_unit = WEATHER_FIELD_UNITS[field]
        if units.get(field) != expected_unit:
            raise DataContractError(
                f"weather unit changed for {field}: expected {expected_unit}, "
                f"received {units.get(field)}"
            )
        field_values = hourly.get(field)
        if not isinstance(field_values, list) or len(field_values) != len(valid_times):
            raise DataContractError(f"weather field {field} has incompatible length")
        if any(value is None for value in field_values):
            raise DataContractError(f"null weather values in {field}")
        if not all(isinstance(value, int | float) for value in field_values):
            raise DataContractError(f"weather field {field} must contain numeric values")
        series_id = field.removesuffix("_previous_day2") + f"@{location.name}"
        frame = pd.DataFrame(
            {
                "valid_time_utc": valid_times,
                "available_at_utc": availability,
                "cutoff_utc": cutoffs,
                "source": "Open-Meteo Previous Runs API",
                "series_id": series_id,
                "value": [float(value) for value in field_values],
                "unit": expected_unit,
                "retrieval_id": retrieval_id,
            }
        )
        validate_canonical(frame, expected_unit=expected_unit)
        assert_point_in_time(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
