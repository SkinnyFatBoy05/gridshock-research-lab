from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def canonical_frame() -> pd.DataFrame:
    valid = pd.to_datetime(["2024-01-02T00:00:00Z", "2024-01-02T01:00:00Z"], utc=True)
    cutoff = pd.to_datetime(["2024-01-01T11:00:00Z"] * 2, utc=True)
    return pd.DataFrame(
        {
            "valid_time_utc": valid,
            "available_at_utc": pd.to_datetime(
                ["2024-01-01T10:00:00Z", "2024-01-01T10:00:00Z"], utc=True
            ),
            "cutoff_utc": cutoff,
            "source": ["fixture", "fixture"],
            "series_id": ["day_ahead_price", "day_ahead_price"],
            "value": [42.0, 45.5],
            "unit": ["EUR/MWh", "EUR/MWh"],
            "retrieval_id": ["fixture-001", "fixture-001"],
        }
    )
