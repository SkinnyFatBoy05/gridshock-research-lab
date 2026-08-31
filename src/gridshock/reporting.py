"""Deterministic technical figures, HTML report, and artifact verification."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, select_autoescape
from matplotlib.figure import Figure

from gridshock import __version__
from gridshock.config import ProjectPaths
from gridshock.provenance import sha256_bytes

if TYPE_CHECKING:
    from gridshock.cli import DemoResult

BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
GOLD = "#D97706"
INK = "#172033"
MUTED = "#667085"
GRID = "#E4E7EC"
PAPER = "#FFFFFF"
MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive (48-hour lag)",
    "ridge": "Ridge",
    "hist_gradient_boosting": "Histogram gradient boosting",
}
FIGURE_NAMES = (
    "forecast_vs_actual.png",
    "error_by_hour.png",
    "equity_drawdown.png",
    "cost_sensitivity.png",
)


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "figure.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def _save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(
        path,
        dpi=144,
        bbox_inches="tight",
        facecolor=PAPER,
        metadata={"Software": f"GridShock Research Lab {__version__}"},
    )
    plt.close(figure)


def _holdout_predictions(result: DemoResult) -> pd.DataFrame:
    return result.predictions.loc[result.predictions["split_type"] == "holdout"].copy()


def _extreme_weather_day(result: DemoResult) -> object:
    holdout_days = set(_holdout_predictions(result)["delivery_day"])
    frame = result.features.loc[result.features["delivery_day"].isin(holdout_days)].copy()
    columns = [
        "wx_fcst_48h_temperature_2m_regional_mean",
        "wx_fcst_48h_wind_speed_100m_regional_mean",
        "wx_fcst_48h_shortwave_radiation_regional_mean",
    ]
    reference = result.features[columns]
    scale = reference.std(ddof=0).replace(0.0, 1.0)
    anomaly = (frame[columns] - reference.mean()) / scale
    frame["weather_anomaly_score"] = anomaly.abs().mean(axis=1)
    daily = frame.groupby("delivery_day")["weather_anomaly_score"].mean()
    return daily.idxmax()


def _forecast_figure(result: DemoResult, output: Path) -> None:
    extreme_day = _extreme_weather_day(result)
    predictions = _holdout_predictions(result)
    selected = predictions.loc[predictions["delivery_day"] == extreme_day].copy()
    actual = selected.drop_duplicates("valid_time_utc").sort_values("valid_time_utc")
    figure, axis = plt.subplots(figsize=(10, 6))
    local_time = actual["valid_time_utc"].dt.tz_convert("Europe/Berlin")
    axis.plot(local_time, actual["actual"], color=INK, linewidth=2.4, label="Actual")
    styles = {
        "hist_gradient_boosting": (BLUE, "-", 2.2),
        "ridge": (GOLD, "-.", 1.8),
        "seasonal_naive": (MUTED, "--", 1.8),
    }
    for model, group in selected.groupby("model", sort=True):
        ordered = group.sort_values("valid_time_utc")
        color, linestyle, width = styles[str(model)]
        axis.plot(
            ordered["valid_time_utc"].dt.tz_convert("Europe/Berlin"),
            ordered["prediction"],
            color=color,
            linestyle=linestyle,
            linewidth=width,
            label=MODEL_LABELS[str(model)],
        )
    axis.set_title("Actual and forecast day-ahead prices", loc="left", pad=28)
    axis.text(
        0.0,
        1.02,
        f"Algorithmically selected weather-anomaly day: {extreme_day} · EUR/MWh",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10,
    )
    axis.set_ylabel("EUR/MWh")
    axis.set_xlabel("Delivery time (Europe/Berlin)")
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter(  # type: ignore[no-untyped-call]
            "%H:%M", tz=local_time.dt.tz
        )
    )
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, output)


def _error_by_hour_figure(result: DemoResult, output: Path) -> None:
    predictions = _holdout_predictions(result)
    predictions["delivery_hour"] = (
        predictions["valid_time_utc"].dt.tz_convert("Europe/Berlin").dt.hour
    )
    hourly = (
        predictions.groupby(["model", "delivery_hour"])["absolute_error"]
        .mean()
        .rename("mae")
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    styles = {
        "hist_gradient_boosting": (BLUE, "o", "-"),
        "ridge": (GOLD, "s", "-."),
        "seasonal_naive": (MUTED, "^", "--"),
    }
    for model, group in hourly.groupby("model", sort=True):
        color, marker, linestyle = styles[str(model)]
        axis.plot(
            group["delivery_hour"],
            group["mae"],
            color=color,
            marker=marker,
            markevery=3,
            linestyle=linestyle,
            linewidth=2.0,
            label=MODEL_LABELS[str(model)],
        )
    axis.set_title("Mean absolute error by delivery hour", loc="left", pad=28)
    axis.text(
        0.0,
        1.02,
        "Final 45-day chronological holdout · lower is better · EUR/MWh",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10,
    )
    axis.set_xlabel("Local delivery hour")
    axis.set_ylabel("MAE (EUR/MWh)")
    axis.set_xticks(range(0, 24, 2))
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, output)


def _equity_drawdown_figure(result: DemoResult, output: Path) -> None:
    ledger = result.strategy_ledger.copy()
    dates = pd.to_datetime(ledger["delivery_day"])
    figure, (equity_axis, drawdown_axis) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    equity_axis.plot(dates, ledger["equity_eur"], color=BLUE, linewidth=2.4)
    equity_axis.set_title("Cost-adjusted research-proxy equity and drawdown", loc="left", pad=28)
    equity_axis.text(
        0.0,
        1.02,
        "One MWh per selected leg · €0.50/MWh transaction cost · final holdout only",
        transform=equity_axis.transAxes,
        color=MUTED,
        fontsize=10,
    )
    equity_axis.set_ylabel("Cumulative proxy (€)")
    drawdown_axis.fill_between(dates, ledger["drawdown_eur"], 0.0, color=GOLD, alpha=0.28)
    drawdown_axis.plot(dates, ledger["drawdown_eur"], color=GOLD, linewidth=1.8)
    drawdown_axis.axhline(0.0, color=INK, linewidth=0.8)
    if np.allclose(ledger["drawdown_eur"].to_numpy(dtype=float), 0.0):
        drawdown_axis.text(
            0.5,
            0.5,
            "No peak-to-trough drawdown observed in this 45-day proxy sample",
            transform=drawdown_axis.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
    drawdown_axis.set_ylabel("Drawdown (€)")
    drawdown_axis.set_xlabel("Delivery day")
    for axis in (equity_axis, drawdown_axis):
        axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, output)


def _cost_sensitivity_figure(result: DemoResult, output: Path) -> None:
    table = result.cost_table.sort_values("cost_eur_mwh")
    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(
        table["cost_eur_mwh"].map(lambda value: f"€{value:.2f}"),
        table["net_total_eur"],
        color=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=1.2,
    )
    axis.bar_label(bars, labels=[f"€{value:,.0f}" for value in table["net_total_eur"]], padding=4)
    axis.set_title("Net research-proxy result by transaction-cost assumption", loc="left", pad=28)
    axis.text(
        0.0,
        1.02,
        "Same holdout positions · cost charged per entered hourly leg",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10,
    )
    axis.set_xlabel("Cost per leg (EUR/MWh)")
    axis.set_ylabel("Net proxy total (€)")
    axis.set_ylim(bottom=0)
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, output)


def build_figures(result: DemoResult, output_dir: Path) -> list[Path]:
    """Build four complementary, source-backed report figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    builders = (
        ("forecast_vs_actual.png", _forecast_figure),
        ("error_by_hour.png", _error_by_hour_figure),
        ("equity_drawdown.png", _equity_drawdown_figure),
        ("cost_sensitivity.png", _cost_sensitivity_figure),
    )
    paths: list[Path] = []
    for name, builder in builders:
        path = output_dir / name
        builder(result, path)
        paths.append(path)
    return paths


def _encoded_figure(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _model_rows(result: DemoResult) -> list[dict[str, object]]:
    holdout = result.forecast_metrics.loc[
        result.forecast_metrics["split_type"] == "holdout"
    ].sort_values("mae")
    return [
        {
            "model": MODEL_LABELS[str(row.model)],
            "mae": f"{float(str(row.mae)):.2f}",
            "rmse": f"{float(str(row.rmse)):.2f}",
            "rank": f"{float(str(row.rank_correlation)):.3f}",
        }
        for row in holdout.itertuples(index=False)
    ]


def _case_study(result: DemoResult) -> dict[str, object]:
    day = _extreme_weather_day(result)
    features = result.features.loc[result.features["delivery_day"] == day]
    predictions = _holdout_predictions(result)
    selected = predictions.loc[predictions["delivery_day"] == day]
    hgb = selected.loc[selected["model"] == "hist_gradient_boosting"]
    naive = selected.loc[selected["model"] == "seasonal_naive"]
    strategy = result.strategy_ledger.loc[result.strategy_ledger["delivery_day"] == day]
    return {
        "day": str(day),
        "temperature": float(features["wx_fcst_48h_temperature_2m_regional_mean"].mean()),
        "wind": float(features["wx_fcst_48h_wind_speed_100m_regional_mean"].mean()),
        "solar": float(features["wx_fcst_48h_shortwave_radiation_regional_mean"].mean()),
        "actual_mean": float(hgb["actual"].mean()),
        "actual_max": float(hgb["actual"].max()),
        "hgb_mae": float(hgb["absolute_error"].mean()),
        "naive_mae": float(naive["absolute_error"].mean()),
        "net_proxy": float(strategy["net_eur"].iloc[0]),
    }


def _write_experiment_manifest(
    result: DemoResult, paths: ProjectPaths, figures: list[Path]
) -> dict[str, Any]:
    manifest = {
        "schema_version": "1.0",
        "package_version": __version__,
        "input_sha256": result.input_sha256,
        "summary": result.summary(),
        "forecast_metrics": result.forecast_metrics.to_dict(orient="records"),
        "strategy_metrics": result.strategy_metrics,
        "placebo_metrics": result.placebo_metrics,
        "cost_sensitivity": result.cost_table.to_dict(orient="records"),
        "figures": {path.name: sha256_bytes(path.read_bytes()) for path in figures},
        "chart_map": [
            {
                "section": "Forecast evidence",
                "question": "How did each comparator track the selected stress day?",
                "family": "trend",
                "file": "forecast_vs_actual.png",
            },
            {
                "section": "Error diagnostics",
                "question": "Where does forecast error vary across delivery hours?",
                "family": "comparison",
                "file": "error_by_hour.png",
            },
            {
                "section": "Strategy proxy",
                "question": "How does cost-adjusted proxy value evolve and draw down?",
                "family": "progression",
                "file": "equity_drawdown.png",
            },
            {
                "section": "Cost robustness",
                "question": "How sensitive is the fixed position ledger to four cost scenarios?",
                "family": "discrete comparison",
                "file": "cost_sensitivity.png",
            },
        ],
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.experiment_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def render_report(result: DemoResult, paths: ProjectPaths | None = None) -> Path:
    """Render the complete self-contained technical research brief."""

    resolved = paths or ProjectPaths.discover()
    figures = build_figures(result, resolved.figures)
    experiment_manifest = _write_experiment_manifest(result, resolved, figures)
    holdout = result.forecast_metrics.loc[
        result.forecast_metrics["split_type"] == "holdout"
    ].set_index("model")
    hgb_mae = float(str(holdout.loc["hist_gradient_boosting", "mae"]))
    baseline_mae = float(str(holdout.loc["seasonal_naive", "mae"]))
    improvement = 100.0 * (baseline_mae - hgb_mae) / baseline_mae
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(("html", "xml")),
    )
    template = environment.get_template("report.html.j2")
    html = template.render(
        title="GridShock Research Lab",
        summary=result.summary(),
        model_rows=_model_rows(result),
        improvement=improvement,
        hgb_mae=hgb_mae,
        baseline_mae=baseline_mae,
        case=_case_study(result),
        strategy=result.strategy_metrics,
        placebo=result.placebo_metrics,
        costs=result.cost_table.to_dict(orient="records"),
        figures={path.name: _encoded_figure(path) for path in figures},
        input_hash=result.input_sha256,
        manifest_hash=sha256_bytes(resolved.experiment_manifest.read_bytes()),
        source_request_count=len(
            json.loads(resolved.demo_manifest.read_text(encoding="utf-8"))["sources"]
        ),
        experiment_manifest=experiment_manifest,
    )
    report_path = resolved.reports / "gridshock_research_brief.html"
    report_path.write_text(html, encoding="utf-8", newline="\n")
    return report_path


def verify_artifacts(paths: ProjectPaths | None = None) -> list[str]:
    """Return actionable integrity errors for committed report artifacts."""

    resolved = paths or ProjectPaths.discover()
    errors: list[str] = []
    for required in (
        resolved.demo_data,
        resolved.demo_manifest,
        resolved.experiment_manifest,
        resolved.reports / "gridshock_research_brief.html",
    ):
        if not required.exists():
            errors.append(f"missing artifact: {required.name}")
    if errors:
        return errors

    demo_manifest = json.loads(resolved.demo_manifest.read_text(encoding="utf-8"))
    actual_input_hash = sha256_bytes(resolved.demo_data.read_bytes())
    if actual_input_hash != demo_manifest.get("dataset_sha256"):
        errors.append("demo fixture hash mismatch")
    experiment = json.loads(resolved.experiment_manifest.read_text(encoding="utf-8"))
    if experiment.get("input_sha256") != actual_input_hash:
        errors.append("experiment manifest input hash mismatch")
    from gridshock.cli import run_demo

    replay_summary = run_demo(resolved).summary()
    if experiment.get("summary") != replay_summary:
        errors.append("experiment summary mismatch")
    figure_hashes = experiment.get("figures", {})
    if not isinstance(figure_hashes, dict) or set(figure_hashes) != set(FIGURE_NAMES):
        errors.append("experiment figure register mismatch")
        figure_hashes = {}
    for name, expected_hash in figure_hashes.items():
        path = resolved.figures / name
        if not path.exists():
            errors.append(f"missing figure: {name}")
            continue
        if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            errors.append(f"{name} is not a PNG")
        if sha256_bytes(path.read_bytes()) != expected_hash:
            errors.append(f"{name} hash mismatch")
    report = (resolved.reports / "gridshock_research_brief.html").read_text(encoding="utf-8")
    for phrase in (
        "Technical summary",
        "Limitations and robustness",
        "Not financial advice",
        "data:image/png;base64,",
    ):
        if phrase not in report:
            errors.append(f"report missing required content: {phrase}")
    manifest_hash = sha256_bytes(resolved.experiment_manifest.read_bytes())
    if manifest_hash not in report:
        errors.append("report manifest hash mismatch")
    return errors
