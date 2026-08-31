from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridshock.config import ExperimentConfig
from gridshock.contracts import DataContractError
from gridshock.time import decision_cutoff_utc
from gridshock.validation import evaluate_models, rolling_origin_splits

FEATURE_COLUMNS = (
    "price_lag_48h",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "heating_degree_proxy",
    "wind_scarcity_proxy",
    "solar_availability_proxy",
)


def _feature_frame(days: int = 12) -> pd.DataFrame:
    valid = pd.date_range("2024-12-31T23:00:00Z", periods=days * 24, freq="h", tz="UTC")
    local = valid.tz_convert("Europe/Berlin")
    delivery_days = [timestamp.date() for timestamp in local]
    cutoff = pd.to_datetime([decision_cutoff_utc(day) for day in delivery_days], utc=True)
    hour = local.hour.to_numpy(dtype=float)
    temperature = 8.0 + 5.0 * np.sin(2 * np.pi * hour / 24)
    wind = 28.0 + 4.0 * np.cos(2 * np.pi * hour / 24)
    solar = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
    price_lag = 45.0 + 12.0 * np.sin(2 * np.pi * (hour - 6.0) / 24)
    price = 8.0 + 0.72 * price_lag + 0.9 * np.maximum(18.0 - temperature, 0.0) - 0.2 * wind
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


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        feature_columns=FEATURE_COLUMNS,
        train_days=5,
        validation_days=2,
        step_days=2,
        holdout_days=2,
        random_seed=23,
    )


def test_every_fold_trains_strictly_before_validation() -> None:
    frame = _feature_frame()

    folds = list(rolling_origin_splits(frame, train_days=5, validation_days=2, step_days=2))

    assert len(folds) == 3
    for fold in folds:
        assert fold.train["valid_time_utc"].max() < fold.validation["valid_time_utc"].min()
        assert set(fold.train["delivery_day"]).isdisjoint(fold.validation["delivery_day"])


def test_evaluation_reports_all_models_and_final_holdout() -> None:
    metrics, predictions = evaluate_models(_feature_frame(), _config())

    expected_models = {"seasonal_naive", "ridge", "hist_gradient_boosting"}
    assert set(metrics["model"]) == expected_models
    assert set(predictions["model"]) == expected_models
    holdout = predictions.loc[predictions["split_type"] == "holdout"]
    assert set(holdout["model"]) == expected_models
    assert holdout["delivery_day"].nunique() == 2


def test_evaluation_is_deterministic() -> None:
    first_metrics, first_predictions = evaluate_models(_feature_frame(), _config())
    second_metrics, second_predictions = evaluate_models(_feature_frame(), _config())

    pd.testing.assert_frame_equal(first_metrics, second_metrics)
    pd.testing.assert_frame_equal(first_predictions, second_predictions)


def test_evaluation_rejects_future_only_feature_rows() -> None:
    frame = _feature_frame()
    frame.loc[0, "feature_available_at_utc"] = frame.loc[0, "cutoff_utc"] + pd.Timedelta(seconds=1)

    with pytest.raises(DataContractError, match="available after cutoff"):
        evaluate_models(frame, _config())
