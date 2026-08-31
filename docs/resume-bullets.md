# Conservative résumé bullets

- Built a typed Python research pipeline joining **6,551** DE-LU hourly price intervals with
  fixed-lag archived weather forecasts, enforcing UTC availability, DST-aware delivery days,
  schema/unit contracts, and SHA-256 provenance across **45** source requests.
- Reduced MAE **51.6%** versus a 48-hour seasonal-naive baseline on a fixed **45-day historical
  holdout** (**18.08 vs 37.36 EUR/MWh**), using expanding rolling-origin validation and achieving
  **0.897** within-day rank correlation.
- Packaged the analysis as an offline-reproducible CLI and self-contained evidence brief with a
  cost-aware non-executable ranking proxy, shuffled placebo, automated tests, strict typing, CI, and
  artifact-integrity verification.

Use “historical holdout” and “research proxy” when space allows. Do not convert the proxy ledger into
a profitability claim.
