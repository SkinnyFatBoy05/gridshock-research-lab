# GridShock Research Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a reproducible Python research package that tests whether point-in-time-safe weather forecasts improve DE-LU day-ahead hourly power-price forecasts and a cost-aware hourly ranking strategy.

**Architecture:** Source adapters capture official API payloads and provenance, contract modules normalise timestamps and enforce availability, and a dataset builder emits a point-in-time feature matrix. Rolling-origin model evaluation feeds a deterministic strategy ledger, which in turn feeds a self-contained HTML research brief and recruiter-facing evidence.

**Tech Stack:** Python 3.11+, uv, pandas, NumPy, scikit-learn, httpx, Matplotlib, Seaborn, Jinja2, pytest, Ruff, mypy, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-08-31-gridshock-research-lab-design.md`

## Global Constraints

- The simulated decision cutoff is 12:00 `Europe/Berlin` on delivery day D-1.
- Version 1 delivery dates are 2024-01-01 through 2025-09-30 and use hourly DE-LU prices only.
- Open-Meteo predictive features use fixed-lag `previous_day2` fields; observed weather is never a same-day predictive feature.
- All internal joins and availability checks use aware UTC timestamps; local market labels use `Europe/Berlin` only at boundaries.
- Spring and autumn DST days retain 23 and 25 distinct UTC delivery intervals.
- Invalid schema, units, chronology, availability, or resolution fail closed with actionable errors.
- The default demo is deterministic and offline; live refresh is explicit and bounded.
- No API keys, paid data, live orders, autonomous trading, or claims of executable profit.
- No numerical performance threshold is required; baselines and negative results remain visible.
- Implementation is test-driven and each task ends with fresh focused tests plus a commit.

## File map

```text
.github/workflows/ci.yml                     CI quality gates
.gitignore                                   Python/build/generated exclusions
.gitattributes                               Stable text and fixture handling
LICENSE                                      MIT licence for project code
pyproject.toml                               Package, dependencies, tools, CLI
README.md                                    Five-minute recruiter path and claims
SECURITY.md                                  Supported workflow and reporting policy
docs/architecture.md                         System boundaries and decision timeline
docs/data-sources.md                         Attribution, licences, retrieval semantics
docs/methodology-card.md                     Model/backtest assumptions and limits
docs/claim-evidence.md                       Résumé claims mapped to artefacts/tests
docs/resume-bullets.md                       Metrics populated from verified run
src/gridshock/__init__.py                    Version export
src/gridshock/cli.py                         demo/fetch/train/backtest/report/verify commands
src/gridshock/config.py                      Typed paths and experiment configuration
src/gridshock/contracts.py                   Canonical schemas and validation errors
src/gridshock/time.py                        Cutoff and DST-safe conversion
src/gridshock/provenance.py                  Request fingerprints and manifests
src/gridshock/sources.py                     Energy-Charts/Open-Meteo HTTP adapters
src/gridshock/dataset.py                     Canonical point-in-time feature matrix
src/gridshock/models.py                      Baselines and fitted sklearn pipelines
src/gridshock/validation.py                  Rolling-origin splits and prediction ledger
src/gridshock/strategy.py                    Rank positions, costs, risk, kill switches
src/gridshock/reporting.py                   Figures, HTML, artefact manifest
src/gridshock/templates/report.html.j2       Self-contained report template
data/demo/gridshock_demo.csv.gz              Small derived offline research fixture
data/demo/manifest.json                      Source/provenance/fixture hash metadata
reports/gridshock_research_brief.html        Committed verified report
reports/figures/*.png                        Committed report figures
reports/experiment_manifest.json             Inputs, config, metrics, hashes
tests/conftest.py                             Reusable deterministic observations
tests/test_cli.py                             Offline and command-boundary tests
tests/test_contracts.py                       Schema/unit/failure tests
tests/test_time.py                            Cutoff and DST tests
tests/test_provenance.py                      Fingerprint/manifest tests
tests/test_sources.py                         Adapter tests with mocked transports
tests/test_dataset.py                         Availability and feature tests
tests/test_models.py                          Baseline and pipeline tests
tests/test_validation.py                      Fold and leakage tests
tests/test_strategy.py                        P&L/risk/kill-switch tests
tests/test_reporting.py                       Report and artefact tests
```

---

### Task 1: Reproducible package and offline command shell

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.gitattributes`
- Create: `LICENSE`
- Create: `src/gridshock/__init__.py`
- Create: `src/gridshock/config.py`
- Create: `src/gridshock/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `ProjectPaths.discover(root: Path | None = None) -> ProjectPaths`
- Produces: `main(args: Sequence[str] | None = None) -> int`
- Produces: console command `gridshock`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
from gridshock.cli import main


def test_cli_version_is_offline(monkeypatch, capsys):
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert main(["version"]) == 0
    assert "GridShock" in capsys.readouterr().out
```

- [ ] **Step 2: Run the focused test and confirm import failure**

Run: `uv run --with pytest pytest tests/test_cli.py::test_cli_version_is_offline -q`
Expected: FAIL because `gridshock` does not exist.

- [ ] **Step 3: Add package metadata and the minimal deterministic CLI**

Define Python `>=3.11`, runtime dependencies, the `gridshock = "gridshock.cli:entrypoint"` script, Ruff/mypy/pytest configuration, `__version__`, path discovery, an `argparse` parser, and `version`. `entrypoint()` raises `SystemExit(main())`; imports perform no I/O.

- [ ] **Step 4: Lock dependencies and run the smoke test**

Run: `uv lock && uv sync --extra dev && uv run pytest tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the independently runnable shell**

```bash
git add pyproject.toml uv.lock .gitignore .gitattributes LICENSE src tests/test_cli.py
git commit -m "build: bootstrap GridShock package"
```

### Task 2: Market calendar, canonical contracts, and provenance

**Files:**
- Create: `src/gridshock/time.py`
- Create: `src/gridshock/contracts.py`
- Create: `src/gridshock/provenance.py`
- Create: `tests/conftest.py`
- Create: `tests/test_time.py`
- Create: `tests/test_contracts.py`
- Create: `tests/test_provenance.py`

**Interfaces:**
- Produces: `decision_cutoff_utc(delivery_day: date) -> datetime`
- Produces: `parse_source_times(values: Sequence[str]) -> pd.DatetimeIndex`
- Produces: `validate_canonical(frame: pd.DataFrame, *, expected_unit: str) -> None`
- Produces: `assert_point_in_time(frame: pd.DataFrame) -> None`
- Produces: `RequestManifest.from_payload(...) -> RequestManifest`
- Produces: `sha256_bytes(payload: bytes) -> str`

- [ ] **Step 1: Write DST, naive-time, unit, cutoff, and hash tests**

```python
def test_autumn_delivery_day_keeps_both_02_hours():
    utc = pd.date_range("2024-10-26 22:00Z", periods=25, freq="h")
    local = utc.tz_convert("Europe/Berlin")
    assert len(local) == 25
    assert sum(ts.hour == 2 for ts in local) == 2
    assert local.tz_convert("UTC").is_unique


def test_point_in_time_rejects_feature_after_cutoff(canonical_frame):
    canonical_frame.loc[0, "available_at_utc"] = canonical_frame.loc[0, "cutoff_utc"] + pd.Timedelta(minutes=1)
    with pytest.raises(DataContractError, match="available after cutoff"):
        assert_point_in_time(canonical_frame)
```

- [ ] **Step 2: Run the contract tests and confirm missing-module failures**

Run: `uv run pytest tests/test_time.py tests/test_contracts.py tests/test_provenance.py -q`
Expected: FAIL because the modules are absent.

- [ ] **Step 3: Implement minimal aware-time and fail-closed contracts**

Use `zoneinfo.ZoneInfo`, explicit ISO parsing, frozen dataclasses, stable JSON serialisation, SHA-256, exact required-column sets, finite-value checks, uniqueness by UTC/series, monotonicity, and a `DataContractError` carrying the violated rule.

- [ ] **Step 4: Run focused tests and static checks**

Run: `uv run pytest tests/test_time.py tests/test_contracts.py tests/test_provenance.py -q && uv run mypy src/gridshock/time.py src/gridshock/contracts.py src/gridshock/provenance.py`
Expected: PASS.

- [ ] **Step 5: Commit the market-data boundary**

```bash
git add src/gridshock/time.py src/gridshock/contracts.py src/gridshock/provenance.py tests
git commit -m "feat: enforce point-in-time market contracts"
```

### Task 3: Official-source adapters with bounded refresh

**Files:**
- Create: `src/gridshock/sources.py`
- Modify: `src/gridshock/cli.py`
- Test: `tests/test_sources.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `EnergyChartsClient.fetch_prices(start: date, end: date) -> SourcePayload`
- Produces: `OpenMeteoClient.fetch_previous_runs(location: Location, start: date, end: date) -> SourcePayload`
- Produces: `parse_energy_prices(payload: bytes) -> pd.DataFrame`
- Produces: `parse_weather_previous_runs(payload: bytes, location: Location) -> pd.DataFrame`
- Consumes: `RequestManifest`, `parse_source_times`, `validate_canonical`

- [ ] **Step 1: Write mocked-response and request-bound tests**

```python
def test_weather_request_uses_previous_day2(mock_transport):
    client = OpenMeteoClient(transport=mock_transport)
    client.fetch_previous_runs(Location("Berlin", 52.52, 13.41), date(2024, 1, 1), date(2024, 1, 2))
    request = mock_transport.requests[0]
    assert "previous_day2" in request.url.params["hourly"]
    assert "previous_day1" not in request.url.params["hourly"]


def test_refresh_rejects_more_than_31_days(client):
    with pytest.raises(ValueError, match="31 days"):
        client.fetch_prices(date(2024, 1, 1), date(2024, 2, 2))
```

- [ ] **Step 2: Run tests and confirm adapter failures**

Run: `uv run pytest tests/test_sources.py tests/test_cli.py -q`
Expected: FAIL because source adapters and `fetch` are missing.

- [ ] **Step 3: Implement adapters without host-local timestamp conversion**

Use `httpx.Client` with 15-second timeout, three bounded retries, descriptive user agent, raw `bytes` capture, exact endpoints, 31-day maximum requests, and typed `SourcePayload(payload, manifest)`. Parse only from raw JSON text and preserve licence, unit, timezone, series id, and resolution.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_sources.py tests/test_cli.py -q`
Expected: PASS, including zero real network calls in tests.

- [ ] **Step 5: Commit source ingestion**

```bash
git add src/gridshock/sources.py src/gridshock/cli.py tests/test_sources.py tests/test_cli.py
git commit -m "feat: add bounded official data adapters"
```

### Task 4: Point-in-time feature dataset

**Files:**
- Create: `src/gridshock/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Produces: `build_feature_matrix(prices: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame`
- Produces columns: `valid_time_utc`, `delivery_day`, `cutoff_utc`, `price_eur_mwh`, `price_lag_48h`, calendar encodings, city weather forecasts, regional weather summaries, degree/scarcity proxies, missingness flags, and per-feature availability maxima
- Consumes: canonical source frames and `assert_point_in_time`

- [ ] **Step 1: Write leakage, lag, DST, and naming tests**

```python
def test_price_lag_uses_48_hours_not_delivery_row_position(price_frame, weather_frame):
    result = build_feature_matrix(price_frame, weather_frame)
    row = result.set_index("valid_time_utc").loc[pd.Timestamp("2024-01-03T00:00:00Z")]
    expected = price_frame.set_index("valid_time_utc").loc[pd.Timestamp("2024-01-01T00:00:00Z"), "value"]
    assert row["price_lag_48h"] == expected


def test_predictive_weather_names_are_forecasts(result):
    assert all("obs" not in name for name in result.filter(like="wx_").columns)
```

- [ ] **Step 2: Run the dataset tests and confirm failure**

Run: `uv run pytest tests/test_dataset.py -q`
Expected: FAIL because `build_feature_matrix` is missing.

- [ ] **Step 3: Implement UTC-keyed joins and compact features**

Pivot weather by city/variable, calculate equal-weight regional means, use 48-hour timestamp joins for lagged price, calculate cyclical hour/day features and heating/cooling/wind/solar proxies, retain availability columns through validation, and drop rows lacking target or pre-cutoff minimum features with an explicit coverage summary.

- [ ] **Step 4: Run focused and boundary tests**

Run: `uv run pytest tests/test_dataset.py tests/test_time.py tests/test_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the point-in-time dataset**

```bash
git add src/gridshock/dataset.py tests/test_dataset.py
git commit -m "feat: build leakage-safe power features"
```

### Task 5: Baselines and rolling-origin evaluation

**Files:**
- Create: `src/gridshock/models.py`
- Create: `src/gridshock/validation.py`
- Create: `tests/test_models.py`
- Create: `tests/test_validation.py`

**Interfaces:**
- Produces: `ModelSpec(name: str, estimator: RegressorMixin, feature_columns: tuple[str, ...])`
- Produces: `build_model_specs(feature_columns: Sequence[str]) -> list[ModelSpec]`
- Produces: `rolling_origin_splits(frame, train_days, validation_days, step_days) -> Iterator[Fold]`
- Produces: `evaluate_models(frame: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]`
- Prediction ledger columns: `model`, `fold`, `valid_time_utc`, `delivery_day`, `actual`, `prediction`, `cutoff_utc`

- [ ] **Step 1: Write chronological-fold, fit-boundary, and comparator tests**

```python
def test_every_fold_trains_strictly_before_validation(feature_frame):
    for fold in rolling_origin_splits(feature_frame, train_days=60, validation_days=14, step_days=14):
        assert fold.train["valid_time_utc"].max() < fold.validation["valid_time_utc"].min()


def test_evaluation_always_reports_all_predeclared_models(feature_frame, experiment_config):
    metrics, _ = evaluate_models(feature_frame, experiment_config)
    assert set(metrics["model"]) == {"seasonal_naive", "ridge", "hist_gradient_boosting"}
```

- [ ] **Step 2: Run model tests and confirm missing implementations**

Run: `uv run pytest tests/test_models.py tests/test_validation.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement deterministic fold-local pipelines**

Keep seasonal naive predictions explicit, fit `SimpleImputer`/`StandardScaler`/`Ridge` only inside each fold, fit bounded-complexity `HistGradientBoostingRegressor(random_state=...)`, calculate MAE/RMSE/median absolute error/Spearman rank by fold, and emit every prediction with its cutoff.

- [ ] **Step 4: Run focused tests and mypy**

Run: `uv run pytest tests/test_models.py tests/test_validation.py -q && uv run mypy src/gridshock/models.py src/gridshock/validation.py`
Expected: PASS.

- [ ] **Step 5: Commit modelling and evaluation**

```bash
git add src/gridshock/models.py src/gridshock/validation.py tests/test_models.py tests/test_validation.py
git commit -m "feat: add rolling-origin model evaluation"
```

### Task 6: Cost-aware research strategy and risk controls

**Files:**
- Create: `src/gridshock/strategy.py`
- Create: `tests/test_strategy.py`

**Interfaces:**
- Produces: `build_positions(predictions: pd.DataFrame, *, k: int, uncertainty_threshold: float | None) -> pd.DataFrame`
- Produces: `backtest_positions(positions: pd.DataFrame, *, cost_eur_mwh: float) -> tuple[pd.DataFrame, dict[str, float]]`
- Produces: `cost_sensitivity(positions, costs: Sequence[float]) -> pd.DataFrame`
- Ledger columns: day, model, long/short hours, gross, costs, net, equity, drawdown, active, gate_reason

- [ ] **Step 1: Write exact accounting, neutral-rank, and kill-switch tests**

```python
def test_two_long_two_short_legs_pay_exact_cost():
    ledger, _ = backtest_positions(four_leg_day(), cost_eur_mwh=0.50)
    assert ledger.loc[0, "costs_eur"] == pytest.approx(2.0)
    assert ledger.loc[0, "net_eur"] == pytest.approx(ledger.loc[0, "gross_eur"] - 2.0)


def test_failed_quality_gate_has_zero_exposure():
    result = build_positions(predictions_with_missing_required_hour(), k=2, uncertainty_threshold=None)
    assert result["position"].abs().sum() == 0
    assert set(result["gate_reason"]) == {"incomplete_delivery_day"}
```

- [ ] **Step 2: Run strategy tests and confirm failure**

Run: `uv run pytest tests/test_strategy.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement deterministic positions and risk metrics**

Rank within each actual 23/24/25-hour delivery day, use equal +1/-1 legs, enforce complete-day and finite-prediction gates, calculate explicit per-leg costs, daily equity/drawdown, hit rate, mean, stated annualised Sharpe, max drawdown, worst day, and 5% expected shortfall. Add a seeded shuffled-prediction placebo and sensitivity for costs `[0.0, 0.5, 1.0, 2.0]`.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_strategy.py tests/test_validation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit strategy research**

```bash
git add src/gridshock/strategy.py tests/test_strategy.py
git commit -m "feat: add cost-aware strategy research"
```

### Task 7: Source-attributed offline fixture and end-to-end demo

**Files:**
- Modify: `src/gridshock/cli.py`
- Modify: `src/gridshock/config.py`
- Create: `data/demo/gridshock_demo.csv.gz` (generated by the package from official API responses)
- Create: `data/demo/manifest.json` (generated by the package)
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `run_demo(paths: ProjectPaths) -> DemoResult`
- Produces: `write_demo_fixture(frame, manifests, output_path) -> None`
- Consumes: dataset, evaluation, strategy interfaces from Tasks 4-6

- [ ] **Step 1: Write an offline-network and deterministic-result test**

```python
def test_demo_is_offline_and_deterministic(monkeypatch, tmp_path, packaged_demo):
    monkeypatch.setattr("httpx.Client.send", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    first = run_demo(packaged_demo)
    second = run_demo(packaged_demo)
    assert first.metrics == second.metrics
    assert first.input_sha256 == second.input_sha256
```

- [ ] **Step 2: Run the demo test and confirm missing-fixture failure**

Run: `uv run pytest tests/test_cli.py::test_demo_is_offline_and_deterministic -q`
Expected: FAIL because the fixture and demo orchestrator are absent.

- [ ] **Step 3: Fetch bounded official slices through the tested adapters**

Use successive 31-day requests covering 2024-01-01 through 2025-09-30 for Energy-Charts prices and the four Open-Meteo locations. Preserve raw-response manifests, construct the canonical derived table, verify all availability rules, and write a compressed derived fixture plus aggregate manifest. If a requested weather field has incomplete coverage, retain the documented common subset instead of imputing future information.

- [ ] **Step 4: Implement and run the offline demo twice**

Run: `uv run gridshock demo && uv run gridshock demo`
Expected: both runs complete without HTTP and produce byte-stable JSON metrics for the same environment.

- [ ] **Step 5: Commit only the compact derived fixture and provenance**

```bash
git add src/gridshock/cli.py src/gridshock/config.py data/demo tests/test_cli.py
git commit -m "feat: ship reproducible official-data demo"
```

### Task 8: Figures, HTML research brief, and artefact verification

**Files:**
- Create: `src/gridshock/reporting.py`
- Create: `src/gridshock/templates/report.html.j2`
- Create: `tests/test_reporting.py`
- Create: `reports/gridshock_research_brief.html` (generated)
- Create: `reports/figures/forecast_vs_actual.png` (generated)
- Create: `reports/figures/error_by_hour.png` (generated)
- Create: `reports/figures/equity_drawdown.png` (generated)
- Create: `reports/figures/cost_sensitivity.png` (generated)
- Create: `reports/experiment_manifest.json` (generated)

**Interfaces:**
- Produces: `build_figures(result: DemoResult, output_dir: Path) -> list[Path]`
- Produces: `render_report(result: DemoResult, paths: ProjectPaths) -> Path`
- Produces: `verify_artifacts(paths: ProjectPaths) -> list[str]`

- [ ] **Step 1: Write report-content, PNG, hash, and no-cherry-picking tests**

```python
def test_report_contains_claim_controls(rendered_report):
    html = rendered_report.read_text(encoding="utf-8")
    for phrase in ["Seasonal naive", "Transaction costs", "Limitations", "Not financial advice", "Extreme-weather case study"]:
        assert phrase in html


def test_every_report_asset_matches_manifest(project_paths):
    assert verify_artifacts(project_paths) == []
```

- [ ] **Step 2: Run reporting tests and confirm failure**

Run: `uv run pytest tests/test_reporting.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement headless charts and escaped self-contained report**

Use the `Agg` backend, a deterministic project style, Matplotlib/Seaborn, PNG metadata suppression where practical, Jinja autoescape, embedded methodology tables, all comparator metrics, cost sensitivity, algorithmically selected weather anomaly day, limitations, source links, and artefact SHA-256 values.

- [ ] **Step 4: Generate and verify report artefacts**

Run: `uv run gridshock report && uv run gridshock verify && uv run pytest tests/test_reporting.py -q`
Expected: all commands exit 0 and verification returns no errors.

- [ ] **Step 5: Commit the research brief**

```bash
git add src/gridshock/reporting.py src/gridshock/templates tests/test_reporting.py reports
git commit -m "feat: publish auditable research brief"
```

### Task 9: Public documentation, CI, and claim evidence

**Files:**
- Create: `README.md`
- Create: `SECURITY.md`
- Create: `docs/architecture.md`
- Create: `docs/data-sources.md`
- Create: `docs/methodology-card.md`
- Create: `docs/claim-evidence.md`
- Create: `docs/resume-bullets.md` (generated metrics copied from verified manifest with command evidence)
- Create: `.github/workflows/ci.yml`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: a five-minute public reviewer path
- Consumes: verified metrics and hashes from `reports/experiment_manifest.json`

- [ ] **Step 1: Write documentation and installed-package acceptance tests**

```python
def test_readme_exposes_offline_path_and_limitations(project_root):
    text = (project_root / "README.md").read_text(encoding="utf-8")
    assert "uv run gridshock demo" in text
    assert "not financial advice" in text.lower()
    assert "2025-10-01" in text
```

- [ ] **Step 2: Run acceptance test and confirm missing-document failure**

Run: `uv run pytest tests/test_cli.py::test_readme_exposes_offline_path_and_limitations -q`
Expected: FAIL because README is absent.

- [ ] **Step 3: Write evidence-led documentation and CI**

Document architecture, exact cutoff, source licences, `previous_day2` rationale, 15-minute transition exclusion, reproducibility commands, results with baseline context, failure modes, limitations, security scope, and résumé bullets whose numbers exactly match the committed experiment manifest. Configure CI for Python 3.11 and 3.12 with `uv sync --extra dev`, Ruff, mypy, pytest, package build, offline demo, and artefact verification.

- [ ] **Step 4: Run the complete local quality gate from the locked environment**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -q && uv build && uv run gridshock demo && uv run gridshock verify`
Expected: every command exits 0.

- [ ] **Step 5: Commit public documentation and automation**

```bash
git add README.md SECURITY.md docs .github tests/test_cli.py reports/experiment_manifest.json
git commit -m "docs: prepare GridShock for public review"
```

### Task 10: Clean-room verification and GitHub publication

**Files:**
- Modify only if verification exposes a defect; every defect receives a failing regression test first

**Interfaces:**
- Produces: public repository `https://github.com/SkinnyFatBoy05/gridshock-research-lab`

- [ ] **Step 1: Inspect tracked files and secret exposure**

Run: `git status --short && git ls-files && git grep -n -I -E "(api[_-]?key|secret|token|password)[[:space:]]*[:=]" -- . ':!uv.lock'`
Expected: clean status, expected project files only, no credential values.

- [ ] **Step 2: Verify from a fresh uv environment**

Run: `uv sync --refresh --extra dev && uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -q && uv build && uv run gridshock demo && uv run gridshock report && uv run gridshock verify`
Expected: every command exits 0 with fresh evidence.

- [ ] **Step 3: Review the complete branch diff and report assets**

Run: `git diff main...HEAD --check && git log --oneline --decorate main..HEAD`
Expected: no whitespace errors and a task-oriented commit history.

- [ ] **Step 4: Merge the verified branch locally**

```bash
git switch main
git merge --ff-only codex/initial-build
```

- [ ] **Step 5: Create and push the public repository**

Run: `gh repo create SkinnyFatBoy05/gridshock-research-lab --public --source . --remote origin --push --description "Point-in-time-safe weather and DE-LU day-ahead power-market research with walk-forward evaluation and cost-aware backtesting."`
Expected: repository URL returned and `main` tracks `origin/main`.

- [ ] **Step 6: Verify the remote state**

Run: `gh repo view SkinnyFatBoy05/gridshock-research-lab --json name,url,visibility,defaultBranchRef && git status --short --branch && git ls-remote --heads origin main`
Expected: PUBLIC repository, default branch `main`, local branch tracking cleanly, and the remote main SHA matching local HEAD.
