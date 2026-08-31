"""Market-calendar and timezone handling."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from gridshock.contracts import DataContractError

BERLIN = ZoneInfo("Europe/Berlin")


def decision_cutoff_utc(delivery_day: date) -> datetime:
    """Return noon Berlin time on D-1 as an aware UTC datetime."""

    local_cutoff = datetime.combine(delivery_day - timedelta(days=1), time(12), tzinfo=BERLIN)
    return local_cutoff.astimezone(UTC)


def delivery_intervals_utc(delivery_day: date) -> pd.DatetimeIndex:
    """Return every valid hourly interval in one Berlin delivery day."""

    start_local = datetime.combine(delivery_day, time.min, tzinfo=BERLIN)
    end_local = datetime.combine(delivery_day + timedelta(days=1), time.min, tzinfo=BERLIN)
    return pd.date_range(
        pd.Timestamp(start_local).tz_convert("UTC"),
        pd.Timestamp(end_local).tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )


def parse_source_times(values: Sequence[str]) -> pd.DatetimeIndex:
    """Parse explicitly offset source timestamps without using host-local time."""

    parsed: list[datetime] = []
    for raw_value in values:
        normalised = raw_value[:-1] + "+00:00" if raw_value.endswith("Z") else raw_value
        try:
            timestamp = datetime.fromisoformat(normalised)
        except ValueError as error:
            raise DataContractError(f"invalid ISO-8601 timestamp: {raw_value}") from error
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise DataContractError(f"timestamp requires an explicit timezone offset: {raw_value}")
        parsed.append(timestamp.astimezone(UTC))
    return pd.DatetimeIndex(parsed, tz="UTC")
