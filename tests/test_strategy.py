from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gridshock.strategy import (
    backtest_positions,
    build_positions,
    cost_sensitivity,
    shuffled_prediction_placebo,
)
from gridshock.time import decision_cutoff_utc, delivery_intervals_utc


def _prediction_day(delivery_day: date = date(2025, 1, 2)) -> pd.DataFrame:
    valid = delivery_intervals_utc(delivery_day)
    actual = np.linspace(10.0, 100.0, len(valid))
    prediction = actual.copy()
    return pd.DataFrame(
        {
            "valid_time_utc": valid,
            "delivery_day": delivery_day,
            "cutoff_utc": pd.Timestamp(decision_cutoff_utc(delivery_day)),
            "actual": actual,
            "prediction": prediction,
            "model": "fixture_model",
            "fold": "holdout",
            "split_type": "holdout",
        }
    )


def test_positions_are_neutral_and_select_predicted_extremes() -> None:
    predictions = _prediction_day()

    positions = build_positions(predictions, k=2, uncertainty_threshold=None)

    assert positions["position"].sum() == 0.0
    assert positions.nlargest(2, "prediction")["position"].tolist() == [1.0, 1.0]
    assert positions.nsmallest(2, "prediction")["position"].tolist() == [-1.0, -1.0]


def test_two_long_two_short_legs_pay_exact_cost() -> None:
    positions = build_positions(_prediction_day(), k=2, uncertainty_threshold=None)

    ledger, metrics = backtest_positions(positions, cost_eur_mwh=0.50)

    assert ledger.loc[0, "costs_eur"] == pytest.approx(2.0)
    assert ledger.loc[0, "net_eur"] == pytest.approx(ledger.loc[0, "gross_eur"] - 2.0)
    assert metrics["turnover_legs"] == 4.0


def test_incomplete_normal_day_has_zero_exposure() -> None:
    predictions = _prediction_day().iloc[:-1].copy()

    positions = build_positions(predictions, k=2, uncertainty_threshold=None)

    assert positions["position"].abs().sum() == 0.0
    assert set(positions["gate_reason"]) == {"incomplete_delivery_day"}


def test_complete_autumn_25_hour_day_is_tradeable() -> None:
    predictions = _prediction_day(date(2024, 10, 27))

    positions = build_positions(predictions, k=2, uncertainty_threshold=None)

    assert len(positions) == 25
    assert positions["position"].abs().sum() == 4.0
    assert set(positions["gate_reason"]) == {"active"}


def test_uncertainty_kill_switch_has_zero_exposure() -> None:
    predictions = _prediction_day()
    predictions["uncertainty_proxy"] = 12.0

    positions = build_positions(predictions, k=2, uncertainty_threshold=10.0)

    assert positions["position"].abs().sum() == 0.0
    assert set(positions["gate_reason"]) == {"uncertainty_above_threshold"}


def test_cost_sensitivity_decreases_net_result() -> None:
    positions = build_positions(_prediction_day(), k=2, uncertainty_threshold=None)

    sensitivity = cost_sensitivity(positions, [0.0, 0.5, 2.0])

    assert sensitivity["net_total_eur"].is_monotonic_decreasing
    assert sensitivity["cost_eur_mwh"].tolist() == [0.0, 0.5, 2.0]


def test_shuffled_placebo_is_seeded_and_preserves_actuals() -> None:
    predictions = _prediction_day()

    first = shuffled_prediction_placebo(predictions, seed=7)
    second = shuffled_prediction_placebo(predictions, seed=7)

    pd.testing.assert_series_equal(first["prediction"], second["prediction"])
    pd.testing.assert_series_equal(first["actual"], predictions["actual"])
    assert not first["prediction"].equals(predictions["prediction"])
