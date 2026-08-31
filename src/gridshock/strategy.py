"""Cost-aware hourly ranking research with explicit risk gates."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import numpy as np
import pandas as pd

from gridshock.contracts import DataContractError
from gridshock.time import delivery_intervals_utc

GROUP_COLUMNS = ["model", "fold", "split_type", "delivery_day"]


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str | np.datetime64):
        return pd.Timestamp(value).date()
    raise DataContractError(f"delivery_day has unsupported type: {type(value).__name__}")


def _gate_reason(
    group: pd.DataFrame, delivery_day: date, k: int, uncertainty_threshold: float | None
) -> str:
    expected_intervals = len(delivery_intervals_utc(delivery_day))
    if len(group) != expected_intervals:
        return "incomplete_delivery_day"
    if group["valid_time_utc"].duplicated().any():
        return "duplicate_delivery_interval"
    if 2 * k > expected_intervals:
        return "insufficient_intervals_for_k"
    numeric = group[["actual", "prediction"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        return "non_finite_market_value"
    if uncertainty_threshold is not None:
        if "uncertainty_proxy" not in group:
            raise DataContractError("uncertainty_threshold requires uncertainty_proxy")
        uncertainty = group["uncertainty_proxy"].to_numpy(dtype=float)
        if not np.isfinite(uncertainty).all():
            return "non_finite_uncertainty"
        if float(np.max(uncertainty)) > uncertainty_threshold:
            return "uncertainty_above_threshold"
    return "active"


def build_positions(
    predictions: pd.DataFrame, *, k: int, uncertainty_threshold: float | None
) -> pd.DataFrame:
    """Convert predicted within-day ranks into equal long/short research legs."""

    required = {
        "valid_time_utc",
        "delivery_day",
        "cutoff_utc",
        "actual",
        "prediction",
        "model",
        "fold",
        "split_type",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataContractError(f"prediction ledger missing columns: {', '.join(missing)}")
    if k <= 0:
        raise ValueError("k must be positive")
    if uncertainty_threshold is not None and uncertainty_threshold < 0:
        raise ValueError("uncertainty_threshold must be non-negative")

    result = predictions.sort_values([*GROUP_COLUMNS, "valid_time_utc"]).reset_index(drop=True)
    result["position"] = 0.0
    result["gate_reason"] = "not_evaluated"
    for _keys, group in result.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        day = _as_date(group["delivery_day"].iloc[0])
        reason = _gate_reason(group, day, k, uncertainty_threshold)
        result.loc[group.index, "gate_reason"] = reason
        if reason != "active":
            continue
        ordered = group.sort_values(["prediction", "valid_time_utc"], kind="mergesort")
        short_indices = ordered.index[:k]
        long_indices = ordered.index[-k:]
        result.loc[short_indices, "position"] = -1.0
        result.loc[long_indices, "position"] = 1.0
    return result


def _hours_with_position(group: pd.DataFrame, sign: int) -> str:
    selected = group.loc[np.sign(group["position"]) == sign, "valid_time_utc"]
    return ",".join(pd.Timestamp(value).isoformat() for value in selected)


def backtest_positions(
    positions: pd.DataFrame, *, cost_eur_mwh: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calculate an unlevered one-MWh-per-leg historical research ledger."""

    if cost_eur_mwh < 0:
        raise ValueError("cost_eur_mwh must be non-negative")
    if positions["model"].nunique() != 1:
        raise ValueError("backtest_positions requires exactly one model")

    rows: list[dict[str, object]] = []
    for keys, group in positions.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        model, fold, split_type, delivery_day = keys
        leg_count = float(group["position"].abs().sum())
        gross = float((group["position"] * group["actual"]).sum())
        costs = leg_count * float(cost_eur_mwh)
        active = bool(leg_count > 0)
        rows.append(
            {
                "model": model,
                "fold": fold,
                "split_type": split_type,
                "delivery_day": delivery_day,
                "long_hours_utc": _hours_with_position(group, 1),
                "short_hours_utc": _hours_with_position(group, -1),
                "gross_eur": gross,
                "costs_eur": costs,
                "net_eur": gross - costs,
                "turnover_legs": leg_count,
                "active": active,
                "gate_reason": "active" if active else str(group["gate_reason"].iloc[0]),
            }
        )
    ledger = pd.DataFrame(rows).sort_values(["model", "delivery_day"]).reset_index(drop=True)
    ledger["equity_eur"] = ledger.groupby("model")["net_eur"].cumsum()
    running_peak = ledger.groupby("model")["equity_eur"].cummax().clip(lower=0.0)
    ledger["drawdown_eur"] = ledger["equity_eur"] - running_peak

    daily_net = ledger["net_eur"].to_numpy(dtype=float)
    standard_deviation = float(np.std(daily_net, ddof=1)) if len(daily_net) > 1 else 0.0
    sharpe = (
        float(np.mean(daily_net) / standard_deviation * np.sqrt(365.0))
        if standard_deviation > 0
        else 0.0
    )
    tail_count = max(1, int(np.ceil(0.05 * len(daily_net))))
    expected_shortfall = float(np.mean(np.sort(daily_net)[:tail_count]))
    metrics = {
        "gross_total_eur": float(ledger["gross_eur"].sum()),
        "cost_total_eur": float(ledger["costs_eur"].sum()),
        "net_total_eur": float(ledger["net_eur"].sum()),
        "turnover_legs": float(ledger["turnover_legs"].sum()),
        "active_days": float(ledger["active"].sum()),
        "hit_rate": float((ledger["net_eur"] > 0).mean()),
        "mean_daily_net_eur": float(ledger["net_eur"].mean()),
        "annualised_sharpe_sqrt_365": sharpe,
        "max_drawdown_eur": float(-ledger["drawdown_eur"].min()),
        "worst_day_eur": float(ledger["net_eur"].min()),
        "expected_shortfall_5pct_eur": expected_shortfall,
    }
    return ledger, metrics


def cost_sensitivity(positions: pd.DataFrame, costs: Sequence[float]) -> pd.DataFrame:
    """Recalculate the same positions under declared per-leg costs."""

    rows: list[dict[str, float]] = []
    for cost in sorted(float(value) for value in costs):
        _ledger, metrics = backtest_positions(positions, cost_eur_mwh=cost)
        rows.append({"cost_eur_mwh": cost, **metrics})
    return pd.DataFrame(rows)


def shuffled_prediction_placebo(predictions: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Shuffle predictions within delivery day while preserving outcomes and timestamps."""

    result = predictions.copy()
    random = np.random.default_rng(seed)
    for _keys, group in result.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        result.loc[group.index, "prediction"] = random.permutation(
            group["prediction"].to_numpy(dtype=float)
        )
    return result
