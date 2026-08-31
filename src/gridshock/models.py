"""Transparent comparators for hourly power-price research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    """Named estimator and its point-in-time feature contract."""

    name: str
    estimator: Any | None
    feature_columns: tuple[str, ...]


def build_model_specs(
    feature_columns: tuple[str, ...], *, random_seed: int = 42
) -> list[ModelSpec]:
    """Create fresh, bounded-complexity model comparators."""

    if "price_lag_48h" not in feature_columns:
        raise ValueError("feature_columns must contain price_lag_48h for the baseline")
    ridge = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    gradient_boosting = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_depth=4,
            max_iter=160,
            l2_regularization=1.0,
            random_state=random_seed,
        ),
    )
    return [
        ModelSpec("seasonal_naive", None, ("price_lag_48h",)),
        ModelSpec("ridge", ridge, feature_columns),
        ModelSpec("hist_gradient_boosting", gradient_boosting, feature_columns),
    ]
