from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from gridshock.contracts import DataContractError
from gridshock.time import decision_cutoff_utc, delivery_intervals_utc, parse_source_times


@pytest.mark.parametrize(
    ("delivery_day", "expected_hours"),
    [(date(2024, 3, 31), 23), (date(2024, 10, 27), 25)],
)
def test_delivery_intervals_keep_dst_day_length(delivery_day: date, expected_hours: int) -> None:
    intervals = delivery_intervals_utc(delivery_day)

    assert len(intervals) == expected_hours
    assert intervals.is_unique
    assert str(intervals.tz) == "UTC"


def test_autumn_delivery_day_keeps_both_local_02_hours() -> None:
    intervals = delivery_intervals_utc(date(2024, 10, 27))
    local = intervals.tz_convert("Europe/Berlin")

    assert sum(timestamp.hour == 2 for timestamp in local) == 2
    assert {
        timestamp.utcoffset().total_seconds() for timestamp in local if timestamp.hour == 2
    } == {
        3600.0,
        7200.0,
    }


@pytest.mark.parametrize(
    ("delivery_day", "expected"),
    [
        (date(2024, 3, 31), pd.Timestamp("2024-03-30T11:00:00Z")),
        (date(2024, 10, 27), pd.Timestamp("2024-10-26T10:00:00Z")),
    ],
)
def test_cutoff_is_previous_local_noon(delivery_day: date, expected: pd.Timestamp) -> None:
    assert decision_cutoff_utc(delivery_day) == expected.to_pydatetime()


def test_parse_source_times_preserves_offset_instants() -> None:
    parsed = parse_source_times(["2024-10-27T02:00:00+02:00", "2024-10-27T02:00:00+01:00"])

    assert parsed.tolist() == [
        pd.Timestamp("2024-10-27T00:00:00Z"),
        pd.Timestamp("2024-10-27T01:00:00Z"),
    ]


def test_parse_source_times_rejects_naive_values() -> None:
    with pytest.raises(DataContractError, match="timezone offset"):
        parse_source_times(["2024-01-01T00:00:00"])
