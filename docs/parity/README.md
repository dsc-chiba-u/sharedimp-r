# Cross-language verification scripts (record of 2026-11-13)

* `parity_check.py` — run from the `sharedimp-py` checkout (commit `bfcb2a5`) with
  its virtual environment, after exporting the R-side CSVs with the snippet in
  `docs/specification.md` Section 15 (the R values come from
  `shared_importance(..., learner = "linear", metric = "mse")` on the fixture and
  from `evaluate_metric()` for the metric table). It compares the Python
  package's output and scikit-learn's metric functions with the R values.
* `parity_output.txt` — the output of that run.
* `zero_column_learner_study.R` — the learner study behind the exact-collinearity
  rule (specification Section 15.3).

These are records, not package tests; the R-side expectations live in
`tests/testthat/test-common-spec.R` and `tests/testthat/test-metric-definitions.R`.
