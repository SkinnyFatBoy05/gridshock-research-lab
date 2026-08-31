from __future__ import annotations

import numpy as np
import pytest

from gridshock.contracts import DataContractError, assert_point_in_time, validate_canonical


def test_valid_canonical_frame_passes(canonical_frame) -> None:
    validate_canonical(canonical_frame, expected_unit="EUR/MWh")
    assert_point_in_time(canonical_frame)


def test_canonical_contract_rejects_wrong_unit(canonical_frame) -> None:
    canonical_frame.loc[1, "unit"] = "GBP/MWh"

    with pytest.raises(DataContractError, match="unexpected unit"):
        validate_canonical(canonical_frame, expected_unit="EUR/MWh")


def test_canonical_contract_rejects_duplicate_utc_series(canonical_frame) -> None:
    canonical_frame.loc[1, "valid_time_utc"] = canonical_frame.loc[0, "valid_time_utc"]

    with pytest.raises(DataContractError, match="duplicate"):
        validate_canonical(canonical_frame, expected_unit="EUR/MWh")


def test_canonical_contract_rejects_non_finite_values(canonical_frame) -> None:
    canonical_frame.loc[0, "value"] = np.inf

    with pytest.raises(DataContractError, match="finite"):
        validate_canonical(canonical_frame, expected_unit="EUR/MWh")


def test_point_in_time_rejects_feature_after_cutoff(canonical_frame) -> None:
    canonical_frame.loc[0, "available_at_utc"] = canonical_frame.loc[
        0, "cutoff_utc"
    ] + np.timedelta64(1, "m")

    with pytest.raises(DataContractError, match="available after cutoff"):
        assert_point_in_time(canonical_frame)
