from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gridshock.provenance import RequestManifest, request_fingerprint, sha256_bytes


def test_sha256_bytes_matches_published_vector() -> None:
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_request_fingerprint_is_independent_of_parameter_order() -> None:
    first = request_fingerprint(
        "https://example.test/data", {"end": "2024-01-02", "start": "2024-01-01"}
    )
    second = request_fingerprint(
        "https://example.test/data", {"start": "2024-01-01", "end": "2024-01-02"}
    )

    assert first == second


def test_manifest_serialisation_is_stable_and_carries_source_claims() -> None:
    manifest = RequestManifest.from_payload(
        source="Energy-Charts",
        url="https://api.energy-charts.info/v2/price",
        params={"start": "2024-01-01", "end": "2024-01-02"},
        payload=b"abc",
        retrieved_at=datetime(2026, 8, 31, 0, 0, tzinfo=UTC),
        license="CC BY 4.0",
        timezone_name="Europe/Berlin",
        unit="EUR/MWh",
        resolution="hourly",
    )

    serialised = manifest.to_json()
    assert serialised == manifest.to_json()
    assert '"license":"CC BY 4.0"' in serialised
    assert (
        '"payload_sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"'
        in serialised
    )


def test_manifest_rejects_naive_retrieval_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        RequestManifest.from_payload(
            source="fixture",
            url="https://example.test",
            params={},
            payload=b"x",
            retrieved_at=datetime(2026, 8, 31),
            license="CC0",
            timezone_name="UTC",
            unit="unitless",
            resolution="hourly",
        )
