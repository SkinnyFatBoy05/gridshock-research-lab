"""Point-in-time-safe feature construction."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from gridshock.contracts import DataContractError, assert_point_in_time, validate_canonical

WEATHER_BASES = ("temperature_2m", "wind_speed_100m", "shortwave_radiation")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _validate_weather(weather: pd.DataFrame) -> None:
    if weather.empty:
        raise DataContractError("weather frame must not be empty")
    assert_point_in_time(weather)
    if not weather["series_id"].astype(str).str.contains("@", regex=False).all():
        raise DataContractError("weather series_id must include a location after '@'")
    for _series_id, series in weather.groupby("series_id", sort=False):
        units = series["unit"].dropna().astype(str).unique()
        if len(units) != 1:
            raise DataContractError("weather series must use exactly one unit")
        validate_canonical(series, expected_unit=str(units[0]))


def _weather_wide(weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    _validate_weather(weather)
    renamed = weather.copy()
    parts = renamed["series_id"].astype(str).str.split("@", n=1, expand=True)
    renamed["feature_name"] = [
        f"wx_fcst_48h_{base}_{_slug(city)}"
        for base, city in parts.itertuples(index=False, name=None)
    ]
    values = renamed.pivot(index="valid_time_utc", columns="feature_name", values="value")
    values.columns.name = None
    availability = renamed.groupby("valid_time_utc")["available_at_utc"].max()
    return values.sort_index(), availability.sort_index()


def _add_regional_weather(frame: pd.DataFrame) -> None:
    for base in WEATHER_BASES:
        city_columns = sorted(
            column
            for column in frame.columns
            if column.startswith(f"wx_fcst_48h_{base}_") and not column.endswith("regional_mean")
        )
        if not city_columns:
            raise DataContractError(f"no weather features found for {base}")
        frame[f"wx_fcst_48h_{base}_regional_mean"] = frame[city_columns].mean(axis=1)
        frame[f"wx_missing_{base}"] = frame[city_columns].isna().any(axis=1).astype(int)


def build_feature_matrix(prices: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join target, lags, and archived forecasts using UTC availability contracts."""

    validate_canonical(prices, expected_unit="EUR/MWh")
    target = prices.loc[prices["series_id"] == "day_ahead_price"].copy()
    if target.empty:
        raise DataContractError("price frame lacks day_ahead_price target")
    target = target.sort_values("valid_time_utc")

    weather_values, weather_availability = _weather_wide(weather)
    frame = target[
        ["valid_time_utc", "cutoff_utc", "value", "available_at_utc", "retrieval_id"]
    ].rename(
        columns={
            "value": "price_eur_mwh",
            "available_at_utc": "target_available_at_utc",
            "retrieval_id": "price_retrieval_id",
        }
    )
    frame = frame.set_index("valid_time_utc").join(weather_values, how="inner")
    frame["weather_available_at_utc"] = weather_availability.reindex(frame.index)

    price_history = target.set_index("valid_time_utc")[["value", "available_at_utc"]]
    lag_times = pd.DatetimeIndex(frame.index) - pd.Timedelta(hours=48)
    lagged = price_history.reindex(lag_times)
    frame["price_lag_48h"] = lagged["value"].to_numpy()
    frame["price_lag_available_at_utc"] = lagged["available_at_utc"].to_numpy()

    frame = frame.reset_index()
    _add_regional_weather(frame)
    required = [
        "price_lag_48h",
        "price_lag_available_at_utc",
        "weather_available_at_utc",
        *(f"wx_fcst_48h_{base}_regional_mean" for base in WEATHER_BASES),
    ]
    before = len(frame)
    frame = frame.dropna(subset=required).copy()
    frame.attrs["dropped_incomplete_rows"] = before - len(frame)
    if frame.empty:
        raise DataContractError("no complete point-in-time feature rows remain")

    available = pd.concat(
        [frame["weather_available_at_utc"], frame["price_lag_available_at_utc"]], axis=1
    ).max(axis=1)
    point_in_time = pd.DataFrame(
        {"available_at_utc": pd.to_datetime(available, utc=True), "cutoff_utc": frame["cutoff_utc"]}
    )
    assert_point_in_time(point_in_time)
    frame["feature_available_at_utc"] = point_in_time["available_at_utc"]

    local = frame["valid_time_utc"].dt.tz_convert("Europe/Berlin")
    frame["delivery_day"] = local.dt.date
    frame["delivery_hour"] = local.dt.hour.astype(int)
    frame["weekday"] = local.dt.weekday.astype(int)
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["delivery_hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["delivery_hour"] / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * frame["weekday"] / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * frame["weekday"] / 7)

    temperature = frame["wx_fcst_48h_temperature_2m_regional_mean"]
    wind = frame["wx_fcst_48h_wind_speed_100m_regional_mean"]
    solar = frame["wx_fcst_48h_shortwave_radiation_regional_mean"]
    frame["heating_degree_proxy"] = (18.0 - temperature).clip(lower=0.0)
    frame["cooling_degree_proxy"] = (temperature - 22.0).clip(lower=0.0)
    frame["wind_scarcity_proxy"] = (30.0 - wind).clip(lower=0.0)
    frame["solar_availability_proxy"] = solar.clip(lower=0.0) / 1000.0

    return frame.sort_values("valid_time_utc").reset_index(drop=True)
