# Submission evidence tracker

Status: ACTIVE, not submission ready. Previous turn made concrete progress (files and runs); prior completion claims are not accepted as proof.

## Prior-result audit (2026-09-27)

- Numeric `flat` literally adds log(strength)*(len(rows)/8-1) to log odds. It is a stipulated faulty estimator, not empirical evidence of a new LLM failure mode. Retain only as analytic illustration.
- `source_average` currently duplicates belief mixing and deduplicates by ID. Prior comparisons do not isolate source grouping from explicit belief separation.
- Exact/semantic rows in the API subset are post-hoc item-score aggregates, not preprocessing plus the same flat readout. They cannot establish fair planner superiority.
- Item scorer sees every item together; it is not independent single-trajectory evaluation. Sample score cache was not frozen across duplication conditions.
- Constant per-item character width is not equal complete-context token length. No position/order-repeat controls were completed.
- API subset has no stable positive directional duplication effect. No real agent environment was previously executed.
- Protocol and code diverged during development. Prior files are exploratory, not preregistered tests.

