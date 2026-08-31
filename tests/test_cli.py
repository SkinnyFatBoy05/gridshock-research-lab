from __future__ import annotations

import socket
from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest

from gridshock.cli import main, run_demo, write_demo_fixture
from gridshock.config import ExperimentConfig, ProjectPaths
from gridshock.contracts import DataContractError
from gridshock.time import decision_cutoff_utc


def test_version_command_is_offline(monkeypatch, capsys) -> None:
    """A version check must never open a network connection."""

    def forbid_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("version command attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbid_network)

    assert main(["version"]) == 0
    assert capsys.readouterr().out == "GridShock Research Lab 0.1.0\n"


def _offline_feature_frame(days: int = 12) -> pd.DataFrame:
    valid = pd.date_range("2024-12-31T23:00:00Z", periods=days * 24, freq="h", tz="UTC")
    local = valid.tz_convert("Europe/Berlin")
    delivery_days = [timestamp.date() for timestamp in local]
    cutoff = pd.to_datetime([decision_cutoff_utc(day) for day in delivery_days], utc=True)
    hour = local.hour.to_numpy(dtype=float)
    temperature = 9.0 + 4.0 * np.sin(2 * np.pi * hour / 24)
    wind = 30.0 + 3.0 * np.cos(2 * np.pi * hour / 24)
    solar = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
    price_lag = 45.0 + 10.0 * np.sin(2 * np.pi * (hour - 6.0) / 24)
    price = 10.0 + 0.7 * price_lag + np.maximum(18.0 - temperature, 0.0) - 0.2 * wind
    return pd.DataFrame(
        {
            "valid_time_utc": valid,
            "delivery_day": delivery_days,
            "cutoff_utc": cutoff,
            "feature_available_at_utc": cutoff - pd.Timedelta(hours=1),
            "price_eur_mwh": price,
            "price_lag_48h": price_lag,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_weekend": [int(timestamp.weekday() >= 5) for timestamp in local],
            "heating_degree_proxy": np.maximum(18.0 - temperature, 0.0),
            "wind_scarcity_proxy": np.maximum(30.0 - wind, 0.0),
            "solar_availability_proxy": solar,
        }
    )


def _small_config() -> ExperimentConfig:
    return ExperimentConfig(
        feature_columns=(
            "price_lag_48h",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "heating_degree_proxy",
            "wind_scarcity_proxy",
            "solar_availability_proxy",
        ),
        train_days=5,
        validation_days=2,
        step_days=2,
        holdout_days=2,
        random_seed=11,
    )


def test_demo_is_offline_and_deterministic(monkeypatch, tmp_path) -> None:
    paths = ProjectPaths.discover(tmp_path)
    write_demo_fixture(_offline_feature_frame(), [], paths.demo_data)

    def forbid_http(*args: object, **kwargs: object) -> httpx.Response:
        raise AssertionError("offline demo attempted HTTP")

    monkeypatch.setattr(httpx.Client, "send", forbid_http)
    first = run_demo(paths, _small_config())
    second = run_demo(paths, _small_config())

    assert first.input_sha256 == second.input_sha256
    assert first.summary() == second.summary()
    assert first.strategy_metrics["active_days"] == 2.0


def test_demo_rejects_fixture_that_no_longer_matches_manifest(tmp_path) -> None:
    paths = ProjectPaths.discover(tmp_path)
    write_demo_fixture(_offline_feature_frame(), [], paths.demo_data)
    paths.demo_data.write_bytes(paths.demo_data.read_bytes() + b"tampered")

    with pytest.raises(DataContractError, match="fixture hash"):
        run_demo(paths, _small_config())


def test_complete_normal_delivery_day_has_24_rows() -> None:
    frame = _offline_feature_frame(days=1)

    assert frame.loc[frame["delivery_day"] == date(2025, 1, 1)].shape[0] == 24
