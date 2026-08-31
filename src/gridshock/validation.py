"""Chronological model fitting, prediction ledgers, and forecast metrics."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from gridshock.config import ExperimentConfig
from gridshock.contracts import DataContractError, assert_point_in_time
from gridshock.models import ModelSpec, build_model_specs


@dataclass(frozen=True)
class Fold:
    """One expanding-window train/validation boundary."""

    label: str
    train: pd.DataFrame
    validation: pd.DataFrame


def rolling_origin_splits(
    frame: pd.DataFrame,
    train_days: int,
    validation_days: int,
    step_days: int,
) -> Iterator[Fold]:
    """Yield expanding training windows followed by disjoint validation days."""

    if min(train_days, validation_days, step_days) <= 0:
        raise ValueError("rolling-origin day counts must be positive")
    days = sorted(frame["delivery_day"].unique())
    fold_number = 1
    for validation_start in range(train_days, len(days) - validation_days + 1, step_days):
        train_set = set(days[:validation_start])
        validation_set = set(days[validation_start : validation_start + validation_days])
        train = frame.loc[frame["delivery_day"].isin(train_set)].copy()
        validation = frame.loc[frame["delivery_day"].isin(validation_set)].copy()
        if train.empty or validation.empty:
            continue
        if train["valid_time_utc"].max() >= validation["valid_time_utc"].min():
            raise DataContractError("rolling-origin training must precede validation")
        yield Fold(label=f"fold_{fold_number:02d}", train=train, validation=validation)
        fold_number += 1


def _predict(spec: ModelSpec, train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    if spec.estimator is None:
        return validation["price_lag_48h"].to_numpy(dtype=float)
    estimator: Any = clone(spec.estimator)
    estimator.fit(
        train.loc[:, spec.feature_columns],
        train["price_eur_mwh"].to_numpy(dtype=float),
    )
    prediction = estimator.predict(validation.loc[:, spec.feature_columns])
    return np.asarray(prediction, dtype=float)


def _rank_correlation(frame: pd.DataFrame) -> float:
    correlations: list[float] = []
    for _day, group in frame.groupby("delivery_day", sort=True):
        if group["actual"].nunique() < 2 or group["prediction"].nunique() < 2:
            continue
        value = group["actual"].rank().corr(group["prediction"].rank())
        if pd.notna(value):
            correlations.append(float(value))
    return float(np.mean(correlations)) if correlations else float("nan")


def _metric_row(predictions: pd.DataFrame) -> dict[str, object]:
    errors = predictions["prediction"] - predictions["actual"]
    return {
        "model": str(predictions["model"].iloc[0]),
        "fold": str(predictions["fold"].iloc[0]),
        "split_type": str(predictions["split_type"].iloc[0]),
        "n_observations": len(predictions),
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "median_absolute_error": float(errors.abs().median()),
        "rank_correlation": _rank_correlation(predictions),
    }


def _evaluate_fold(
    fold: Fold, split_type: str, config: ExperimentConfig
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in build_model_specs(config.feature_columns, random_seed=config.random_seed):
        prediction = _predict(spec, fold.train, fold.validation)
        if not np.isfinite(prediction).all():
            raise DataContractError(f"model {spec.name} emitted non-finite predictions")
        ledger = fold.validation[
            ["valid_time_utc", "delivery_day", "cutoff_utc", "price_eur_mwh"]
        ].copy()
        ledger = ledger.rename(columns={"price_eur_mwh": "actual"})
        ledger["model"] = spec.name
        ledger["fold"] = fold.label
        ledger["split_type"] = split_type
        ledger["prediction"] = prediction
        ledger["absolute_error"] = (ledger["prediction"] - ledger["actual"]).abs()
        metric_rows.append(_metric_row(ledger))
        prediction_frames.append(ledger)
    return metric_rows, prediction_frames


def evaluate_models(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all declared models on rolling validation folds and a final holdout."""

    required = {
        "valid_time_utc",
        "delivery_day",
        "cutoff_utc",
        "feature_available_at_utc",
        "price_eur_mwh",
        *config.feature_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"model frame missing columns: {', '.join(missing)}")
    ordered = frame.sort_values("valid_time_utc").reset_index(drop=True)
    assert_point_in_time(
        ordered[["feature_available_at_utc", "cutoff_utc"]].rename(
            columns={"feature_available_at_utc": "available_at_utc"}
        )
    )

    days = sorted(ordered["delivery_day"].unique())
    minimum_days = config.train_days + config.validation_days + config.holdout_days
    if len(days) < minimum_days:
        raise ValueError(f"evaluation requires at least {minimum_days} complete delivery days")
    holdout_set = set(days[-config.holdout_days :])
    development = ordered.loc[~ordered["delivery_day"].isin(holdout_set)].copy()
    holdout = ordered.loc[ordered["delivery_day"].isin(holdout_set)].copy()

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    folds = list(
        rolling_origin_splits(
            development,
            train_days=config.train_days,
            validation_days=config.validation_days,
            step_days=config.step_days,
        )
    )
    if not folds:
        raise ValueError("evaluation configuration produced no validation folds")
    for fold in folds:
        metrics, predictions = _evaluate_fold(fold, "validation", config)
        metric_rows.extend(metrics)
        prediction_frames.extend(predictions)

    holdout_fold = Fold(label="holdout", train=development, validation=holdout)
    metrics, predictions = _evaluate_fold(holdout_fold, "holdout", config)
    metric_rows.extend(metrics)
    prediction_frames.extend(predictions)

    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["split_type", "fold", "model"], ignore_index=True
    )
    prediction_frame = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["split_type", "fold", "model", "valid_time_utc"], ignore_index=True
    )
    return metrics_frame, prediction_frame
