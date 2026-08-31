# Claim-to-evidence register

This register keeps application and portfolio claims narrower than the artifacts that support them.

| Claim | Evidence | Verification path | Safe wording |
|---|---|---|---|
| Built a point-in-time-safe energy-price research pipeline | Availability assertion, canonical contracts, DST tests | `src/gridshock/contracts.py`, `src/gridshock/time.py`, `tests/test_contracts.py`, `tests/test_time.py` | “Built and tested availability-time contracts for hourly energy research.” |
| Used real official-source data | 45 source request records with URLs, hashes, licences, and retrieval times | `data/demo/manifest.json`, `docs/data-sources.md` | “Integrated Energy-Charts/SMARD prices and Open-Meteo archived forecasts.” |
| Improved holdout MAE 51.6% | HGB 18.076435 versus seasonal naive 37.362935 on the same 1,080 observations | `reports/experiment_manifest.json`, `uv run gridshock train` | “Reduced MAE 51.6% versus a 48-hour seasonal-naive baseline on a fixed 45-day historical holdout.” |
| Evaluated chronologically | Three expanding validation folds and an untouched final holdout | `src/gridshock/validation.py`, `tests/test_validation.py` | “Used rolling-origin validation and a chronological holdout to limit temporal leakage.” |
| Tested ranking signal after costs | Fixed-leg ledger, four cost scenarios, shuffled-prediction placebo | `src/gridshock/strategy.py`, `tests/test_strategy.py`, HTML brief | “Built a cost-aware, non-executable ranking proxy and placebo diagnostic.” |
| Made the study reproducible | Offline fixture, dataset SHA-256, experiment and figure hashes, CI | `gridshock verify`, both manifests, `.github/workflows/ci.yml` | “Packaged a deterministic offline reproduction path with CI and artifact-integrity checks.” |

## Claims to avoid

- “Profitable strategy,” “production trading system,” or “live alpha.”
- “No leakage” without qualification; the code enforces the declared availability contract.
- Generalising the 51.6% improvement beyond this dataset and final holdout.
- Presenting the proxy EUR total, Sharpe, hit rate, or drawdown as executable performance.
- Calling four city weather points a complete representation of DE-LU fundamentals.

The dataset, code, tests, and generated report are the authoritative evidence. If a number changes,
update the artifact first and then revise any application wording.
