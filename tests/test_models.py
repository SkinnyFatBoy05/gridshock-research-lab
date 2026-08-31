from __future__ import annotations

from gridshock.models import build_model_specs


def test_model_specs_always_include_declared_comparators() -> None:
    specs = build_model_specs(("price_lag_48h", "hour_sin", "hour_cos"), random_seed=17)

    assert [spec.name for spec in specs] == [
        "seasonal_naive",
        "ridge",
        "hist_gradient_boosting",
    ]
    assert specs[0].estimator is None


def test_model_specs_create_independent_estimators() -> None:
    first = build_model_specs(("price_lag_48h",), random_seed=17)
    second = build_model_specs(("price_lag_48h",), random_seed=17)

    assert first[1].estimator is not second[1].estimator
    assert first[2].estimator is not second[2].estimator
