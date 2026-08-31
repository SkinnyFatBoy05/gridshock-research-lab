"""Command-line interface and deterministic GridShock workflow."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from gridshock import __version__
from gridshock.config import ExperimentConfig, ProjectPaths, default_experiment_config
from gridshock.contracts import DataContractError
from gridshock.dataset import build_feature_matrix
from gridshock.provenance import RequestManifest, sha256_bytes
from gridshock.sources import (
    EnergyChartsClient,
    Location,
    OpenMeteoClient,
    parse_energy_prices,
    parse_weather_previous_runs,
)
from gridshock.strategy import (
    backtest_positions,
    build_positions,
    cost_sensitivity,
    shuffled_prediction_placebo,
)
from gridshock.time import delivery_intervals_utc
from gridshock.validation import evaluate_models

GERMAN_LOCATIONS = (
    Location("Berlin", 52.52, 13.41),
    Location("Hamburg", 53.55, 9.99),
    Location("Frankfurt", 50.11, 8.68),
    Location("Munich", 48.14, 11.58),
)


@dataclass(frozen=True)
class DemoResult:
    """All reproducible outputs from one offline research run."""

    features: pd.DataFrame
    forecast_metrics: pd.DataFrame
    predictions: pd.DataFrame
    positions: pd.DataFrame
    strategy_ledger: pd.DataFrame
    strategy_metrics: dict[str, float]
    cost_table: pd.DataFrame
    placebo_metrics: dict[str, float]
    input_sha256: str

    def summary(self) -> dict[str, Any]:
        """Return a stable, human-readable result summary."""

        holdout = self.forecast_metrics.loc[
            self.forecast_metrics["split_type"] == "holdout"
        ].sort_values("model")
        holdout_metrics = {
            str(row.model): {
                "mae": round(float(str(row.mae)), 6),
                "rmse": round(float(str(row.rmse)), 6),
                "rank_correlation": round(float(str(row.rank_correlation)), 6),
            }
            for row in holdout.itertuples(index=False)
        }
        return {
            "dataset_rows": len(self.features),
            "delivery_start": str(self.features["delivery_day"].min()),
            "delivery_end": str(self.features["delivery_day"].max()),
            "input_sha256": self.input_sha256,
            "holdout_forecast": holdout_metrics,
            "strategy": {
                key: round(float(value), 6) for key, value in sorted(self.strategy_metrics.items())
            },
            "placebo_net_total_eur": round(float(self.placebo_metrics["net_total_eur"]), 6),
        }


def _stable_csv_bytes(frame: pd.DataFrame) -> bytes:
    serialisable = frame.copy()
    for column in serialisable.select_dtypes(include=["datetimetz"]).columns:
        serialisable[column] = serialisable[column].map(
            lambda value: pd.Timestamp(value).isoformat()
        )
    if "delivery_day" in serialisable:
        serialisable["delivery_day"] = serialisable["delivery_day"].astype(str)
    return serialisable.to_csv(index=False, lineterminator="\n", float_format="%.8f").encode(
        "utf-8"
    )


def write_demo_fixture(
    frame: pd.DataFrame,
    manifests: Sequence[RequestManifest],
    output_path: Path,
) -> str:
    """Write a deterministic compressed derived fixture and aggregate manifest."""

    if frame.empty:
        raise ValueError("demo fixture frame must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = _stable_csv_bytes(frame)
    with (
        output_path.open("wb") as raw_file,
        gzip.GzipFile(fileobj=raw_file, mode="wb", filename="", mtime=0) as archive,
    ):
        archive.write(csv_bytes)
    dataset_hash = sha256_bytes(output_path.read_bytes())
    manifest = {
        "schema_version": "1.0",
        "derived_dataset": output_path.name,
        "dataset_sha256": dataset_hash,
        "row_count": len(frame),
        "delivery_start": str(frame["delivery_day"].min()),
        "delivery_end": str(frame["delivery_day"].max()),
        "sources": [item.to_dict() for item in manifests],
    }
    output_path.with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dataset_hash


def _load_demo_fixture(paths: ProjectPaths) -> tuple[pd.DataFrame, str]:
    if not paths.demo_data.exists() or not paths.demo_manifest.exists():
        raise FileNotFoundError("demo fixture is missing; run the maintainer refresh workflow")
    manifest = json.loads(paths.demo_manifest.read_text(encoding="utf-8"))
    expected_hash = manifest.get("dataset_sha256")
    actual_hash = sha256_bytes(paths.demo_data.read_bytes())
    if expected_hash != actual_hash:
        raise DataContractError(
            f"demo fixture hash mismatch: expected {expected_hash}, received {actual_hash}"
        )
    frame = pd.read_csv(paths.demo_data, compression="gzip")
    if len(frame) != manifest.get("row_count"):
        raise DataContractError("demo fixture row count does not match manifest")
    for column in (
        "valid_time_utc",
        "cutoff_utc",
        "target_available_at_utc",
        "weather_available_at_utc",
        "price_lag_available_at_utc",
        "feature_available_at_utc",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.date
    return frame, actual_hash


def run_demo(
    paths: ProjectPaths | None = None,
    config: ExperimentConfig | None = None,
) -> DemoResult:
    """Run the complete modelling and strategy workflow without network access."""

    resolved_paths = paths or ProjectPaths.discover()
    experiment = config or default_experiment_config()
    features, input_hash = _load_demo_fixture(resolved_paths)
    forecast_metrics, predictions = evaluate_models(features, experiment)
    holdout = predictions.loc[
        (predictions["split_type"] == "holdout")
        & (predictions["model"] == "hist_gradient_boosting")
    ].copy()
    positions = build_positions(holdout, k=3, uncertainty_threshold=None)
    strategy_ledger, strategy_metrics = backtest_positions(positions, cost_eur_mwh=0.5)
    costs = cost_sensitivity(positions, [0.0, 0.5, 1.0, 2.0])
    placebo_predictions = shuffled_prediction_placebo(holdout, seed=experiment.random_seed)
    placebo_positions = build_positions(placebo_predictions, k=3, uncertainty_threshold=None)
    _placebo_ledger, placebo_metrics = backtest_positions(placebo_positions, cost_eur_mwh=0.5)
    return DemoResult(
        features=features,
        forecast_metrics=forecast_metrics,
        predictions=predictions,
        positions=positions,
        strategy_ledger=strategy_ledger,
        strategy_metrics=strategy_metrics,
        cost_table=costs,
        placebo_metrics=placebo_metrics,
        input_sha256=input_hash,
    )


def _date_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=30), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def refresh_demo_fixture(
    paths: ProjectPaths,
    *,
    start: date = date(2025, 1, 1),
    end: date = date(2025, 9, 30),
) -> Path:
    """Refresh the compact official-data fixture through tested bounded adapters."""

    if start < date(2025, 1, 1) or end > date(2025, 9, 30):
        raise ValueError("version 1 refresh is bounded to 2025-01-01 through 2025-09-30")
    if end < start:
        raise ValueError("end date must not precede start date")

    manifests: list[RequestManifest] = []
    price_frames: list[pd.DataFrame] = []
    price_client = EnergyChartsClient()
    for chunk_start, chunk_end in _date_chunks(start - timedelta(days=2), end):
        payload = price_client.fetch_prices(chunk_start, chunk_end)
        manifests.append(payload.manifest)
        price_frames.append(parse_energy_prices(payload.payload))
    prices = pd.concat(price_frames, ignore_index=True).sort_values("valid_time_utc")

    weather_frames: list[pd.DataFrame] = []
    weather_client = OpenMeteoClient()
    for location in GERMAN_LOCATIONS:
        for chunk_start, chunk_end in _date_chunks(start - timedelta(days=1), end):
            payload = weather_client.fetch_previous_runs(location, chunk_start, chunk_end)
            manifests.append(payload.manifest)
            weather_frames.append(parse_weather_previous_runs(payload.payload, location))
    weather = pd.concat(weather_frames, ignore_index=True)
    features = build_feature_matrix(prices, weather)
    features = features.loc[
        (features["delivery_day"] >= start) & (features["delivery_day"] <= end)
    ].copy()

    bad_days: list[str] = []
    for delivery_day, group in features.groupby("delivery_day", sort=True):
        day = (
            delivery_day
            if isinstance(delivery_day, date)
            else pd.Timestamp(str(delivery_day)).date()
        )
        expected = len(delivery_intervals_utc(day))
        if len(group) != expected:
            bad_days.append(f"{day}:{len(group)}/{expected}")
    if bad_days:
        raise DataContractError(f"incomplete delivery days in refreshed fixture: {bad_days}")
    write_demo_fixture(features, manifests, paths.demo_data)
    return paths.demo_data


def build_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free command parser."""

    parser = argparse.ArgumentParser(
        prog="gridshock",
        description="Point-in-time-safe DE-LU power-market research.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("version", help="show the package version")
    demo = subcommands.add_parser("demo", help="run the deterministic offline research demo")
    demo.add_argument("--root", type=Path, default=None)
    fetch = subcommands.add_parser("fetch", help="inspect one bounded live source window")
    fetch.add_argument("--start", type=date.fromisoformat, required=True)
    fetch.add_argument("--end", type=date.fromisoformat, required=True)
    return parser


def _fetch_summary(start: date, end: date) -> dict[str, Any]:
    price_payload = EnergyChartsClient().fetch_prices(start, end)
    weather_payload = OpenMeteoClient().fetch_previous_runs(GERMAN_LOCATIONS[0], start, end)
    prices = parse_energy_prices(price_payload.payload)
    weather = parse_weather_previous_runs(weather_payload.payload, GERMAN_LOCATIONS[0])
    return {
        "price_rows": len(prices),
        "weather_rows": len(weather),
        "price_license": price_payload.manifest.license,
        "weather_license": weather_payload.manifest.license,
    }


def main(args: Sequence[str] | None = None) -> int:
    """Execute a CLI command and return a process status."""

    parsed = build_parser().parse_args(args)
    if parsed.command == "version":
        print(f"GridShock Research Lab {__version__}")
        return 0
    if parsed.command == "demo":
        paths = ProjectPaths.discover(parsed.root)
        print(json.dumps(run_demo(paths).summary(), indent=2, sort_keys=True))
        return 0
    if parsed.command == "fetch":
        print(json.dumps(_fetch_summary(parsed.start, parsed.end), indent=2, sort_keys=True))
        return 0
    return 2


def entrypoint() -> None:
    """Console-script adapter."""

    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
