from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import pytest

from gridshock.cli import run_demo
from gridshock.config import ProjectPaths
from gridshock.reporting import build_figures, render_report, verify_artifacts


@pytest.fixture(scope="module")
def project_paths() -> ProjectPaths:
    return ProjectPaths.discover(Path(__file__).resolve().parents[1])


@pytest.fixture(scope="module")
def demo_result(project_paths: ProjectPaths):
    return run_demo(project_paths)


def test_figures_are_non_empty_readable_pngs(demo_result, tmp_path) -> None:
    figures = build_figures(demo_result, tmp_path)

    assert {path.name for path in figures} == {
        "forecast_vs_actual.png",
        "error_by_hour.png",
        "equity_drawdown.png",
        "cost_sensitivity.png",
    }
    for figure in figures:
        assert figure.stat().st_size > 10_000
        image = mpimg.imread(figure)
        assert image.shape[0] >= 600
        assert image.shape[1] >= 900


def test_report_contains_required_technical_sections(demo_result, project_paths) -> None:
    report = render_report(demo_result, project_paths)
    html = report.read_text(encoding="utf-8")

    for phrase in (
        "Technical summary",
        "Seasonal naive",
        "Transaction costs",
        "Limitations and robustness",
        "Not financial advice",
        "Extreme-weather case study",
        "Recommended next steps",
        "Further questions",
    ):
        assert phrase in html
    assert "data:image/png;base64," in html
    assert "cdn." not in html


def test_every_report_asset_matches_manifest(demo_result, project_paths) -> None:
    render_report(demo_result, project_paths)

    assert verify_artifacts(project_paths) == []


def test_verifier_detects_tampered_figure(demo_result, project_paths) -> None:
    render_report(demo_result, project_paths)
    figure = project_paths.figures / "cost_sensitivity.png"
    original = figure.read_bytes()
    try:
        figure.write_bytes(original + b"tampered")
        errors = verify_artifacts(project_paths)
        assert any("cost_sensitivity.png hash mismatch" in error for error in errors)
    finally:
        figure.write_bytes(original)
