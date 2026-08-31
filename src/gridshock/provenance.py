"""Stable source provenance and request fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for raw bytes."""

    return hashlib.sha256(payload).hexdigest()


def request_fingerprint(url: str, params: dict[str, str]) -> str:
    """Fingerprint a request independently of parameter insertion order."""

    canonical = json.dumps(
        {"params": dict(sorted(params.items())), "url": url},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(canonical)


@dataclass(frozen=True)
class RequestManifest:
    """Immutable metadata for one captured API response."""

    source: str
    url: str
    params: tuple[tuple[str, str], ...]
    request_fingerprint: str
    retrieved_at_utc: str
    payload_sha256: str
    license: str
    timezone: str
    unit: str
    resolution: str

    @classmethod
    def from_payload(
        cls,
        *,
        source: str,
        url: str,
        params: dict[str, str],
        payload: bytes,
        retrieved_at: datetime,
        license: str,
        timezone_name: str,
        unit: str,
        resolution: str,
    ) -> RequestManifest:
        """Construct a manifest after verifying retrieval-time awareness."""

        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        retrieved_utc = retrieved_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        return cls(
            source=source,
            url=url,
            params=tuple(sorted((str(key), str(value)) for key, value in params.items())),
            request_fingerprint=request_fingerprint(url, params),
            retrieved_at_utc=retrieved_utc,
            payload_sha256=sha256_bytes(payload),
            license=license,
            timezone=timezone_name,
            unit=unit,
            resolution=resolution,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with params as an object."""

        return {
            "source": self.source,
            "url": self.url,
            "params": dict(self.params),
            "request_fingerprint": self.request_fingerprint,
            "retrieved_at_utc": self.retrieved_at_utc,
            "payload_sha256": self.payload_sha256,
            "license": self.license,
            "timezone": self.timezone,
            "unit": self.unit,
            "resolution": self.resolution,
        }

    def to_json(self) -> str:
        """Serialise with stable keys and no insignificant whitespace."""

        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
