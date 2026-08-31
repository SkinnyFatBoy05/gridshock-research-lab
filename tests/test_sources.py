from __future__ import annotations

import json
from datetime import UTC, date, datetime

import httpx
import pandas as pd
import pytest

from gridshock.contracts import DataContractError
from gridshock.sources import (
    EnergyChartsClient,
    Location,
    OpenMeteoClient,
    parse_energy_prices,
    parse_weather_previous_runs,
)

PRICE_RESPONSE = {
    "schema_version": "2.0",
    "endpoint": "price",
    "country": "de",
    "bidding_zone": "DE-LU",
    "timezone": "Europe/Berlin",
    "resolution": "PT1H",
    "interval_minutes": 60,
    "unit": "EUR / MWh",
    "generated_at": "2026-08-31T02:04:49+02:00",
    "available_from": "2025-01-01T00:00:00+01:00",
    "available_until": "2025-01-01T01:00:00+01:00",
    "series": [{"id": "day_ahead_price", "name": "Day-ahead spot market price"}],
    "data": [
        {"timestamp": "2025-01-01T00:00:00+01:00", "values": {"day_ahead_price": 30.0}},
        {"timestamp": "2025-01-01T01:00:00+01:00", "values": {"day_ahead_price": 35.0}},
    ],
    "attributes": {},
    "license": "CC BY 4.0 from Bundesnetzagentur | SMARD.de",
    "deprecated": False,
}

WEATHER_RESPONSE = {
    "latitude": 52.52,
    "longitude": 13.42,
    "generationtime_ms": 2.5,
    "utc_offset_seconds": 0,
    "timezone": "GMT",
    "timezone_abbreviation": "GMT",
    "elevation": 38.0,
    "hourly_units": {
        "time": "iso8601",
        "temperature_2m_previous_day2": "°C",
        "wind_speed_100m_previous_day2": "km/h",
        "shortwave_radiation_previous_day2": "W/m²",
    },
    "hourly": {
        "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
        "temperature_2m_previous_day2": [0.8, 1.2],
        "wind_speed_100m_previous_day2": [45.7, 47.3],
        "shortwave_radiation_previous_day2": [0.0, 0.0],
    },
}


def _payload(data: dict[str, object]) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()


def test_energy_client_uses_v2_de_lu_request_and_carries_license() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/price"
        assert dict(request.url.params) == {
            "bzn": "DE-LU",
            "start": "2025-01-01",
            "end": "2025-01-02",
        }
        assert request.headers["user-agent"].startswith("GridShock-Research-Lab/")
        return httpx.Response(200, content=_payload(PRICE_RESPONSE), request=request)

    client = EnergyChartsClient(
        transport=httpx.MockTransport(handler),
        clock=lambda: datetime(2026, 8, 31, tzinfo=UTC),
    )
    result = client.fetch_prices(date(2025, 1, 1), date(2025, 1, 2))

    assert result.manifest.license == "CC BY 4.0 from Bundesnetzagentur | SMARD.de"
    assert result.manifest.unit == "EUR/MWh"


def test_weather_client_requests_only_previous_day2_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        variables = request.url.params["hourly"].split(",")
        assert set(variables) == {
            "temperature_2m_previous_day2",
            "wind_speed_100m_previous_day2",
            "shortwave_radiation_previous_day2",
        }
        assert all("previous_day1" not in variable for variable in variables)
        assert request.url.params["timezone"] == "UTC"
        return httpx.Response(200, content=_payload(WEATHER_RESPONSE), request=request)

    client = OpenMeteoClient(
        transport=httpx.MockTransport(handler),
        clock=lambda: datetime(2026, 8, 31, tzinfo=UTC),
    )
    result = client.fetch_previous_runs(
        Location("Berlin", 52.52, 13.41), date(2025, 1, 1), date(2025, 1, 2)
    )

    assert result.manifest.timezone == "UTC"
    assert result.manifest.resolution == "hourly"


def test_energy_client_honours_rate_limit_then_retries() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, request=request)
        return httpx.Response(200, content=_payload(PRICE_RESPONSE), request=request)

    client = EnergyChartsClient(
        transport=httpx.MockTransport(handler),
        clock=lambda: datetime(2026, 8, 31, tzinfo=UTC),
        sleeper=sleeps.append,
    )

    result = client.fetch_prices(date(2025, 1, 1), date(2025, 1, 2))

    assert attempts == 2
    assert sleeps[0] == 0.0
    assert result.manifest.unit == "EUR/MWh"


@pytest.mark.parametrize(
    "client",
    [EnergyChartsClient(), OpenMeteoClient()],
)
def test_clients_reject_more_than_31_inclusive_days(client) -> None:
    with pytest.raises(ValueError, match="31 days"):
        if isinstance(client, EnergyChartsClient):
            client.fetch_prices(date(2025, 1, 1), date(2025, 2, 1))
        else:
            client.fetch_previous_runs(
                Location("Berlin", 52.52, 13.41), date(2025, 1, 1), date(2025, 2, 1)
            )


def test_price_parser_preserves_source_offsets_as_utc() -> None:
    frame = parse_energy_prices(_payload(PRICE_RESPONSE))

    assert frame["valid_time_utc"].tolist() == [
        pd.Timestamp("2024-12-31T23:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ]
    assert frame["value"].tolist() == [30.0, 35.0]
    assert set(frame["unit"]) == {"EUR/MWh"}


def test_weather_parser_emits_long_form_with_conservative_availability() -> None:
    frame = parse_weather_previous_runs(
        _payload(WEATHER_RESPONSE), Location("Berlin", 52.52, 13.41)
    )

    assert len(frame) == 6
    assert set(frame["series_id"]) == {
        "temperature_2m@Berlin",
        "wind_speed_100m@Berlin",
        "shortwave_radiation@Berlin",
    }
    temperature = frame.loc[frame["series_id"] == "temperature_2m@Berlin"].iloc[0]
    assert temperature["valid_time_utc"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert temperature["available_at_utc"] == pd.Timestamp("2024-12-30T00:00:00Z")


def test_weather_parser_rejects_null_fixed_lag_values() -> None:
    response = json.loads(json.dumps(WEATHER_RESPONSE))
    response["hourly"]["temperature_2m_previous_day2"][0] = None

    with pytest.raises(DataContractError, match="null weather values"):
        parse_weather_previous_runs(_payload(response), Location("Berlin", 52.52, 13.41))
