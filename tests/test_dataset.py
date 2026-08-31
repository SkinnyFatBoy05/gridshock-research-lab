from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from gridshock.contracts import DataContractError
from gridshock.dataset import build_feature_matrix
from gridshock.time import decision_cutoff_utc


def _price_frame(start: str, periods: int) -> pd.DataFrame:
    valid = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    delivery_days = [timestamp.tz_convert("Europe/Berlin").date() for timestamp in valid]
    cutoffs = pd.to_datetime([decision_cutoff_utc(day) for day in delivery_days], utc=True)
    return pd.DataFrame(
        {
            "valid_time_utc": valid,
            "available_at_utc": cutoffs + pd.Timedelta(hours=1),
            "cutoff_utc": cutoffs,
            "source": "price fixture",
            "series_id": "day_ahead_price",
            "value": [float(value) for value in range(periods)],
            "unit": "EUR/MWh",
            "retrieval_id": "price-fixture",
        }
    )


def _weather_frame(valid: pd.DatetimeIndex) -> pd.DataFrame:
    values = {
        "Berlin": {"temperature_2m": 10.0, "wind_speed_100m": 20.0, "shortwave_radiation": 100.0},
        "Hamburg": {"temperature_2m": 14.0, "wind_speed_100m": 40.0, "shortwave_radiation": 300.0},
    }
    units = {"temperature_2m": "°C", "wind_speed_100m": "km/h", "shortwave_radiation": "W/m²"}
    rows: list[dict[str, object]] = []
    for city, fields in values.items():
        for field, value in fields.items():
            for timestamp in valid:
                delivery_day = timestamp.tz_convert("Europe/Berlin").date()
                rows.append(
                    {
                        "valid_time_utc": timestamp,
                        "available_at_utc": timestamp - pd.Timedelta(days=2),
                        "cutoff_utc": pd.Timestamp(decision_cutoff_utc(delivery_day)),
                        "source": "weather fixture",
                        "series_id": f"{field}@{city}",
                        "value": value,
                        "unit": units[field],
                        "retrieval_id": "weather-fixture",
                    }
                )
    frame = pd.DataFrame(rows)
    for column in ("valid_time_utc", "available_at_utc", "cutoff_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


@pytest.fixture
def feature_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = _price_frame("2024-12-30T00:00:00Z", 120)
    weather_times = pd.date_range("2025-01-01T00:00:00Z", periods=72, freq="h", tz="UTC")
    return prices, _weather_frame(weather_times)


def test_price_lag_uses_48_utc_hours(feature_inputs) -> None:
    prices, weather = feature_inputs

    result = build_feature_matrix(prices, weather)
    row = result.set_index("valid_time_utc").loc[pd.Timestamp("2025-01-03T00:00:00Z")]

    assert row["price_lag_48h"] == 48.0


def test_regional_weather_features_have_hand_checked_values(feature_inputs) -> None:
    prices, weather = feature_inputs

    result = build_feature_matrix(prices, weather)
    row = result.iloc[0]

    assert row["wx_fcst_48h_temperature_2m_regional_mean"] == 12.0
    assert row["heating_degree_proxy"] == 6.0
    assert row["cooling_degree_proxy"] == 0.0
    assert row["wind_scarcity_proxy"] == 0.0
    assert row["solar_availability_proxy"] == 0.2


def test_predictive_weather_names_cannot_look_observed(feature_inputs) -> None:
    prices, weather = feature_inputs

    result = build_feature_matrix(prices, weather)
    weather_columns = [column for column in result if column.startswith("wx_")]

    assert weather_columns
    assert all("obs" not in column for column in weather_columns)


def test_feature_builder_rejects_weather_available_after_cutoff(feature_inputs) -> None:
    prices, weather = feature_inputs
    weather.loc[0, "available_at_utc"] = weather.loc[0, "cutoff_utc"] + pd.Timedelta(minutes=1)

    with pytest.raises(DataContractError, match="available after cutoff"):
        build_feature_matrix(prices, weather)


def test_autumn_delivery_intervals_remain_distinct() -> None:
    delivery_day = date(2024, 10, 27)
    valid = pd.date_range("2024-10-26T22:00:00Z", periods=25, freq="h", tz="UTC")
    prices = pd.concat(
        [_price_frame("2024-10-24T22:00:00Z", 48), _price_frame("2024-10-26T22:00:00Z", 25)],
        ignore_index=True,
    )
    prices["value"] = [float(value) for value in range(len(prices))]
    weather = _weather_frame(valid)

    result = build_feature_matrix(prices, weather)
    selected = result.loc[result["delivery_day"] == delivery_day]

    assert len(selected) == 25
    assert selected["valid_time_utc"].is_unique
    assert (selected["delivery_hour"] == 2).sum() == 2
