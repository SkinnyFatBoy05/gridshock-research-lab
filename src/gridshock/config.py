"""Typed project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_FEATURE_COLUMNS = (
    "price_lag_48h",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "wx_fcst_48h_temperature_2m_regional_mean",
    "wx_fcst_48h_wind_speed_100m_regional_mean",
    "wx_fcst_48h_shortwave_radiation_regional_mean",
    "heating_degree_proxy",
    "cooling_degree_proxy",
    "wind_scarcity_proxy",
    "solar_availability_proxy",
    "wx_missing_temperature_2m",
    "wx_missing_wind_speed_100m",
    "wx_missing_shortwave_radiation",
)


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by deterministic project commands."""

    root: Path
    demo_data: Path
    demo_manifest: Path
    reports: Path
    figures: Path
    experiment_manifest: Path

    @classmethod
    def discover(cls, root: Path | None = None) -> ProjectPaths:
        """Resolve paths from an explicit root or the installed source tree."""

        project_root = (root or Path(__file__).resolve().parents[2]).resolve()
        reports = project_root / "reports"
        return cls(
            root=project_root,
            demo_data=project_root / "data" / "demo" / "gridshock_demo.csv.gz",
            demo_manifest=project_root / "data" / "demo" / "manifest.json",
            reports=reports,
            figures=reports / "figures",
            experiment_manifest=reports / "experiment_manifest.json",
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters that define chronological model evaluation."""

    feature_columns: tuple[str, ...]
    train_days: int = 90
    validation_days: int = 30
    step_days: int = 30
    holdout_days: int = 30
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        for name in ("train_days", "validation_days", "step_days", "holdout_days"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def default_experiment_config() -> ExperimentConfig:
    """Return the predeclared public-demo evaluation contract."""

    return ExperimentConfig(
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        train_days=120,
        validation_days=30,
        step_days=30,
        holdout_days=45,
        random_seed=42,
    )
