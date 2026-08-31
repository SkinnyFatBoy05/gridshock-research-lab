# Methodology card

## Research question

Do fixed-lag archived regional weather forecasts add useful out-of-sample information for ranking
hourly DE-LU day-ahead prices, compared with simple time-aware baselines?

## Population and target

- Unit: one DE-LU hourly day-ahead delivery interval.
- Period: 2025-01-01 through 2025-09-30.
- Target: published day-ahead price in EUR/MWh.
- Observations after feature availability and lag warm-up: 6,551.

## Information set

The decision cutoff is noon Europe/Berlin on the calendar day before delivery. Predictors comprise a
48-hour price lag, periodic calendar features, `previous_day2` archived weather forecasts, regional
means, degree/scarcity proxies, and weather missingness flags. Every predictive input must satisfy
`available_at_utc <= cutoff_utc`; target values are never treated as features.

## Comparators

1. Seasonal naive: price from 48 hours earlier.
2. Ridge regression: deterministic scaled linear comparator.
3. Histogram gradient boosting: nonlinear tree ensemble with a fixed seed and declared settings.

Model selection uses three expanding rolling-origin validation windows, each 30 days. The final 45
days remain untouched until the declared comparison. Metrics are MAE and RMSE in EUR/MWh plus
Spearman-style within-day rank correlation. There is no random train/test split.

## Final holdout results

| Model | MAE | RMSE | Rank correlation | Observations |
|---|---:|---:|---:|---:|
| Histogram gradient boosting | 18.076 | 29.860 | 0.897 | 1,080 |
| Ridge | 21.790 | 33.635 | 0.863 | 1,080 |
| Seasonal naive | 37.363 | 54.814 | 0.805 | 1,080 |

The 51.6% MAE improvement is `(37.362935 - 18.076435) / 37.362935`. It describes this one fixed
historical holdout, not expected future performance.

## Strategy-proxy boundary

For each complete holdout day, the proxy takes equal one-MWh positive positions in the three
highest-ranked hours and negative positions in the three lowest-ranked hours. It applies a fixed
cost to every leg and suppresses incomplete or invalid days. Cost scenarios are 0, 0.5, 1, and 2
EUR/MWh per leg. A seeded within-day prediction shuffle is the placebo.

This construction is a ranking diagnostic. It does not simulate an auction order, price impact,
bid acceptance, collateral, imbalance exposure, fees beyond the declared flat cost, transmission,
forecast publication latency beyond the fixed-lag contract, or constraints on taking an opposite
physical position. Accordingly, the proxy's EUR values, Sharpe statistic, hit rate, and drawdown are
not evidence of an executable trading opportunity.

## Known limitations

- Nine months is too short to establish regime robustness or annual seasonality.
- Four city points are crude proxies for a large bidding zone.
- Published prices and weather fields can be revised by their providers.
- One final holdout cannot support confidence intervals for deployment performance.
- The hourly sample stops before the 15-minute product transition on 2025-10-01.
- No load, generation, outages, fuels, carbon, cross-border flows, or auction-curve features are
  included.
- The exact availability assumption is testable and conservative within this project, but it is not
  a substitute for exchange-grade publication-timestamp data.

## Reproduction

Run `uv run gridshock verify`. Numerical outputs are in
[`../reports/experiment_manifest.json`](../reports/experiment_manifest.json); the visual narrative
is in [`../reports/gridshock_research_brief.html`](../reports/gridshock_research_brief.html).
