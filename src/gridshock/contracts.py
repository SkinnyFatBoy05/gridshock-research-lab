"""Fail-closed contracts for canonical market observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_CANONICAL_COLUMNS = frozenset(
    {
        "valid_time_utc",
        "available_at_utc",
        "cutoff_utc",
        "source",
        "series_id",
        "value",
        "unit",
        "retrieval_id",
    }
)


class DataContractError(ValueError):
    """Raised when source data violate an auditable project contract."""


def _require_aware_utc(frame: pd.DataFrame, column: str) -> None:
    dtype = frame[column].dtype
    if not isinstance(dtype, pd.DatetimeTZDtype) or str(dtype.tz) != "UTC":
        raise DataContractError(f"{column} must be timezone-aware UTC")


def validate_canonical(frame: pd.DataFrame, *, expected_unit: str) -> None:
    """Validate the canonical long-form observation contract."""

    missing = sorted(REQUIRED_CANONICAL_COLUMNS.difference(frame.columns))
    if missing:
        raise DataContractError(f"missing canonical columns: {', '.join(missing)}")
    if frame.empty:
        raise DataContractError("canonical frame must not be empty")

    for column in ("valid_time_utc", "available_at_utc", "cutoff_utc"):
        _require_aware_utc(frame, column)

    units = set(frame["unit"].dropna().astype(str))
    if units != {expected_unit}:
        raise DataContractError(
            f"unexpected unit: expected {expected_unit}, received {sorted(units)}"
        )

    numeric_values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise DataContractError("canonical values must be finite numbers")

    duplicate_mask = frame.duplicated(["series_id", "valid_time_utc"], keep=False)
    if duplicate_mask.any():
        raise DataContractError("duplicate UTC timestamp within series")

    for _, series in frame.groupby("series_id", sort=False):
        if not series["valid_time_utc"].is_monotonic_increasing:
            raise DataContractError("valid_time_utc must be monotonic within each series")


def assert_point_in_time(frame: pd.DataFrame) -> None:
    """Reject any observation unavailable at its simulated decision cutoff."""

    for column in ("available_at_utc", "cutoff_utc"):
        if column not in frame:
            raise DataContractError(f"missing point-in-time column: {column}")
        _require_aware_utc(frame, column)
    late = frame["available_at_utc"] > frame["cutoff_utc"]
    if late.any():
        first = frame.index[late][0]
        raise DataContractError(f"row {first} was available after cutoff")
