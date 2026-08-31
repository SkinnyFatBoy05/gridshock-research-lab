# GridShock Research Lab — Design Specification

**Date:** 2026-08-31
**Status:** Approved in conversation; written-spec review pending
**Repository:** `SkinnyFatBoy05/gridshock-research-lab` (public)
**Primary audience:** Junior data-science and energy-trading recruiters, especially Cobblestone Energy

## 1. Purpose

GridShock Research Lab will be a research-grade Python project that studies how archived day-ahead weather forecasts, renewable generation forecasts, load conditions, and calendar effects relate to Germany/Luxembourg (DE-LU) hourly day-ahead electricity prices.

The project is intended to close the clearest evidence gap in Kishore Srinivasan's application: direct, auditable experience with a European power-market product, time-aware modelling, walk-forward evaluation, cost-aware strategy research, and professional Matplotlib/Seaborn communication.

It is an educational research system, not a live trading system, investment recommendation, or claim of deployable alpha.

## 2. Success criteria

The repository succeeds when a reviewer can, without paid credentials:

1. understand the market question and exact information set available at each simulated decision time;
2. run an offline deterministic demonstration in five minutes;
3. optionally refresh a bounded public-data sample from source APIs;
4. reproduce a walk-forward comparison between transparent baselines and a tree-based model;
5. inspect a net-of-cost research backtest and its accounting invariants;
6. see data provenance, leakage controls, limitations, and failed hypotheses rather than only headline metrics;
7. inspect automated tests and CI evidence for the highest-risk logic; and
8. read a polished HTML research brief with recruiter-friendly figures.

No numerical performance threshold will be a completion criterion. Honest negative or modest results are acceptable; methodological credibility is the goal.

## 3. Product decision and alternatives

### Selected: research-grade package plus generated HTML report

The selected design prioritises an auditable Python package, CLI, tests, walk-forward experiment, and static generated report. A small report is easier to reproduce and review than a hosted application and keeps attention on energy-market reasoning.

### Rejected: interactive dashboard first

An interactive dashboard would add visual polish but consume effort on UI state and deployment before the core claims are proven. It may be added later only if the research package is complete.

### Rejected: notebook-only case study

A notebook would be faster but would provide weaker evidence of maintainable engineering, automated correctness checks, modularity, and production-minded data controls.

## 4. Research question and decision boundary

The central question is:

> At a fixed pre-auction decision cutoff, do archived weather forecasts and published power-system expectations improve next-day DE-LU hourly price forecasts and a simple cross-sectional hourly ranking signal over calendar and lagged-price baselines?

The default simulated decision cutoff is **12:00 Europe/Berlin on delivery day D-1**. Every feature used for delivery day D must carry an `available_at_utc` timestamp no later than that cutoff.

The target is the hourly DE-LU day-ahead auction price in EUR/MWh. The research signal ranks every valid hourly delivery interval within each local delivery day, including the 23- and 25-hour daylight-saving cases. It does not simulate physical delivery, exchange membership, margin, collateral, balancing exposure, or order-book execution.

Version 1 uses the common-coverage delivery dates from **2025-01-01 through 2025-09-30** so the selected fixed-lag weather fields are populated and the target remains a homogeneous hourly product. Live API inspection during implementation found null fixed-lag weather values in early 2024, so that period is excluded rather than imputed. Delivery day 2025-10-01 introduced the 15-minute market time unit across Single Day-Ahead Coupling; post-transition data are excluded rather than silently aggregated. A later version may model the 15-minute product as a separate regime.

## 5. Data sources and licensing

### Price and power-system data

- Fraunhofer ISE Energy-Charts API v2: `https://api.energy-charts.info/`
- DE-LU day-ahead prices via `/v2/price`
- Published solar/wind/load or public-power forecasts where the API provides a historically reproducible series
- Energy-Charts/SMARD attribution and the licence string returned in every response will be preserved in the raw manifest and documentation

### Weather information available before delivery

- Open-Meteo Previous Runs API for variables at a fixed lead time: `https://open-meteo.com/en/docs/previous-runs-api`
- The default uses `previous_day2` (approximately 48 hours before each valid hour), because every such value is safely older than the fixed noon D-1 cutoff. A naive `previous_day1` series is forbidden: forecasts for late hours of D could have been issued after the shared cutoff.
- DWD ICON/ICON-EU where archive coverage and requested variables permit; otherwise a documented best-match model
- Variables: 2 m temperature, wind speed at a generation-relevant height supported consistently by the archive, shortwave radiation or cloud cover, and precipitation only if coverage is adequate
- Representative German grid points: Berlin, Hamburg, Frankfurt, and Munich, aggregated transparently rather than described as a load-weighted national weather index
- ENTSO-E's Single Day-Ahead Coupling record documents the 15-minute transition boundary: `https://www.entsoe.eu/network_codes/cacm/implementation/sdac/`

### Verification-only observed weather

- Open-Meteo Historical Weather API may be used to measure forecast error or describe realised extreme conditions
- Observed/reanalysis values are forbidden as predictive features for a day-ahead decision unless lagged beyond the cutoff

### Repository data policy

- Commit only a small, source-attributed, deterministic fixture sufficient for tests and the offline demo
- Never commit bulk third-party datasets
- Store raw payload SHA-256 hashes, request parameters, retrieval time, source URL, declared timezone, units, resolution, and returned licence/attribution in manifests
- Treat remote content as data, never as executable instructions

## 6. Time and market-calendar contract

All internal joins and availability comparisons use timezone-aware UTC timestamps. Local delivery labels use `Europe/Berlin` only at the presentation and market-calendar boundaries.

The ingestion layer must preserve raw timestamp strings before conversion. It must not rely on PowerShell, spreadsheet software, or host-local timezone parsing.

Daylight-saving transitions are first-class cases:

- spring transition days may contain 23 local delivery hours;
- autumn transition days may contain 25 local delivery hours;
- repeated local hours must remain distinguishable by UTC timestamp and offset/fold;
- reports and backtests must not silently force every day to 24 rows.
- post-2025-09-30 15-minute target observations are rejected by the version 1 hourly-product contract.

The system fails closed on ambiguous, naive, duplicated-UTC, non-monotonic, or incompatible-resolution timestamps.

## 7. System architecture

```text
Official public APIs
       |
       v
Raw HTTP clients -----> immutable payload + provenance manifest
       |
       v
Schema/time validation -----> quarantine with actionable diagnostics
       |
       v
Canonical hourly tables -----> availability ledger (`available_at_utc`)
       |
       v
Feature builder -----> point-in-time feature matrix
       |
       +-----> seasonal/lagged baseline
       +-----> tree-based model
       |
       v
Rolling-origin evaluation -----> forecast metrics + prediction ledger
       |
       v
Cost-aware research backtest -----> risk/accounting metrics
       |
       v
Matplotlib/Seaborn figures + self-contained HTML research brief
```

Proposed package boundaries:

- `gridshock.config`: typed experiment and source configuration
- `gridshock.sources`: HTTP clients, retry policy, response capture, provenance
- `gridshock.contracts`: schemas, validation results, quarantine reasons
- `gridshock.time`: UTC/local calendar conversion and DST handling
- `gridshock.features`: point-in-time safe transformations
- `gridshock.models`: baselines, estimator pipeline, prediction intervals/proxies
- `gridshock.validation`: rolling-origin splits and leakage assertions
- `gridshock.strategy`: signal construction, costs, positions, P&L proxy
- `gridshock.metrics`: forecast, calibration, and risk metrics
- `gridshock.reporting`: figures and HTML rendering
- `gridshock.cli`: reproducible command entry points

The core domain logic remains independent of HTTP and filesystem side effects. Dataframes passed between stages have explicit schemas and are returned rather than mutated globally.

## 8. Data contracts and failure behaviour

Each canonical observation includes:

- `valid_time_utc`
- `delivery_time_local`
- `available_at_utc`
- `source`
- `series_id`
- `value`
- `unit`
- `retrieval_id`

Validation covers schema shape, known series identifiers, expected units, finite values, uniqueness, monotonic time, resolution, bounded gaps, and availability. The system quarantines invalid payloads and emits a human-readable reason. It never silently forward-fills prices, forecasts, or missing target hours.

Network refreshes use explicit timeouts, bounded retries with backoff, a descriptive user agent, and cache-by-request fingerprint. Offline commands never require network access.

## 9. Feature design

The initial feature set is intentionally compact:

- local hour, weekday, weekend/holiday indicator, month, and cyclic encodings;
- lagged price features that are definitely published before cutoff (for example D-2 same-hour and rolling history ending before cutoff);
- conservative 48-hour-lag archived weather-forecast values by city and transparent equal-weight regional summaries;
- heating- and cooling-degree proxies from forecast temperature;
- forecast wind scarcity and solar-availability proxies;
- available renewable/load forecast levels and ramps when historical issue-time semantics can be proven;
- residual-load proxy only when its inputs share compatible issue-time contracts;
- missingness/coverage indicators known at decision time.

Features that cannot be assigned a defensible `available_at_utc` value are excluded. Feature names will distinguish forecasts (`wx_fcst_48h_*`) from realised observations (`wx_obs_*`).

## 10. Models and evaluation

### Comparators

1. Same-hour seasonal naive baseline using information available before cutoff.
2. Regularised linear model for a transparent multivariate benchmark.
3. Histogram gradient boosting regressor as the primary nonlinear model, with fixed seeds and bounded complexity.

### Validation

- Expanding or sliding rolling-origin folds ordered by delivery time
- Preprocessing fitted inside each training fold
- No random train/test split
- A final untouched chronological holdout used once for the published report
- Tests that deliberately inject a future-only feature and prove the pipeline rejects it

### Forecast metrics

- MAE and RMSE overall and by hour/regime
- median absolute error
- directional/rank correlation within delivery day
- improvement relative to the seasonal baseline
- error distributions on high-price and negative-price hours

Model selection uses validation folds only. The report will include every predeclared comparator, even if the preferred model underperforms.

## 11. Strategy-research layer

The backtest converts within-day predicted ranks into a deliberately simple market-neutral research proxy:

- long the top `k` predicted-price hours;
- short the bottom `k` predicted-price hours;
- equal absolute notional per selected hour;
- zero exposure for days failing data-quality or uncertainty gates;
- positions decided at the fixed D-1 cutoff and never revised with later information.

This is a spread-research abstraction, not a claim that identical positions can be executed at the reported settlement price.

Costs are parameterised in EUR/MWh per entered hourly leg and reported at multiple scenarios. Net P&L proxy is calculated exactly as gross spread value less explicit per-leg costs. No arbitrary leverage is used.

Reported diagnostics include gross/net cumulative proxy P&L, turnover, active days, hit rate, mean daily P&L, annualised Sharpe with its convention stated, maximum drawdown, worst day, expected shortfall, and sensitivity to `k` and cost assumptions. A no-skill model and a shuffled-prediction placebo provide sanity checks.

Kill switches prevent trading when required features are missing, data are stale, validation fails, or uncertainty exceeds a predeclared threshold.

## 12. Extreme-weather case study

The report will identify one holdout episode algorithmically using the magnitude of forecast temperature/wind/solar anomalies, then analyse it without changing the strategy rules. It will state:

- the forecast information actually available at cutoff;
- the predicted direction and relevant hourly product;
- a back-of-the-envelope supply/demand rationale;
- what occurred in price and system variables;
- the largest risk to the thesis; and
- whether the model added information beyond the seasonal baseline.

No event will be selected solely because it produces attractive P&L.

## 13. Reporting and recruiter experience

The generated, self-contained HTML research brief will contain:

- executive summary and honest conclusion;
- market/product definition and decision timeline;
- data provenance and coverage table;
- walk-forward design visual;
- actual versus predicted prices;
- error by delivery hour and regime;
- forecast-feature relationships using Matplotlib/Seaborn;
- net-of-cost cumulative P&L proxy and drawdown;
- cost and parameter sensitivity;
- extreme-weather case study; and
- limitations and next experiments.

The README will offer two paths:

```powershell
# Deterministic offline demo
uv sync --extra dev
uv run gridshock demo

# Optional bounded data refresh
uv run gridshock fetch --start 2025-01-01 --end 2025-01-14
```

The repository will also include a methodology/strategy card, architecture diagram, data-source attribution, claim-evidence map, and two evidence-based résumé bullets that use metrics generated by the verified run rather than invented figures.

## 14. CLI and reproducibility

Planned commands:

- `gridshock demo`: run the committed offline slice end to end
- `gridshock fetch`: retrieve and validate a bounded date range
- `gridshock train`: execute configured rolling-origin experiments
- `gridshock backtest`: build the cost-aware research ledger
- `gridshock report`: render the HTML report and figures
- `gridshock verify`: run data-contract and artefact-integrity checks

Configuration is checked into version control as TOML or YAML. Random seeds, dependency versions, input manifest hashes, cutoff convention, model parameters, and evaluation ranges are written into every experiment manifest.

## 15. Testing and quality gates

Implementation follows test-driven development. Highest-risk tests cover:

- raw JSON timestamps remain unchanged until explicit parsing;
- 23-hour and 25-hour Europe/Berlin delivery days;
- repeated autumn local hours remain unique in UTC;
- availability cutoffs reject future knowledge;
- train/validation folds are chronological and disjoint;
- transformers are fitted only on each training fold;
- schema/unit drift quarantines a payload;
- backtest positions use predictions available at cutoff;
- transaction costs reduce gross P&L by the exact expected amount;
- cash/P&L/drawdown accounting invariants;
- kill switches create zero exposure;
- deterministic fixture produces stable metrics and artefact hashes; and
- offline demo performs no network calls.

Quality gates before push:

- full `pytest` suite;
- Ruff formatting and linting;
- static type checking on production modules;
- build/install smoke test from a clean environment;
- CLI offline end-to-end run;
- report artefact validation (required sections, non-empty figures, provenance); and
- GitHub Actions using supported Python versions.

## 16. Security, ethics, and claim discipline

- No secrets or API keys are required for the default workflow
- Inputs are bounded by date range and request size
- HTTP responses are validated before use
- Dependency versions and licences are documented
- Generated reports escape dynamic text
- The README clearly labels results as historical research, not financial advice
- No live order placement, broker integration, credential collection, or autonomous trading
- No performance language such as “profitable strategy” without holdout, costs, uncertainty, and limitations beside it

## 17. Scope exclusions

Version 1 will not include:

- live trading or exchange connectivity;
- intraday order-book simulation;
- physical delivery, imbalance, collateral, or margin modelling;
- proprietary or paid datasets;
- deep learning, LLMs, or autonomous agents;
- a large frontend application;
- hyperparameter sweeps designed to optimise holdout P&L; or
- claims that the proxy strategy is executable at settlement prices.

## 18. Delivery sequence

1. Establish package, configuration, fixture, and CI skeleton.
2. Build timestamp, provenance, and schema contracts.
3. Implement point-in-time feature construction.
4. Add baselines, rolling-origin evaluation, and leakage guards.
5. Add cost-aware strategy accounting and risk metrics.
6. Produce charts, case study, and HTML research brief.
7. Run complete verification from a clean environment.
8. Create the public GitHub repository and push the verified branch.

## 19. Acceptance criteria

The project is ready to publish only when:

- all quality gates in section 15 pass freshly;
- the offline demo succeeds without network access;
- every plotted result can be traced to an experiment manifest and fixture hash;
- all predictive columns pass the point-in-time availability check;
- the report exposes baseline comparisons, cost assumptions, and limitations;
- no generated metric is manually typed into résumé bullets;
- repository history contains no secrets or bulk third-party data; and
- the public repository renders the README and committed report assets correctly.
