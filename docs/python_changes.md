# Status of the Python package against the common specification

Reference state: `sharedimp-py` commit `5277017` (2026-11-13; read only,
nothing there was modified from this repository). The canonical list is
Section 16 of `docs/specification.md`; this file repeats it with the
implementing Python commits and what remains.

## Items raised by the R side (spec §16)

| # | item | Python commit | status |
|---|---|---|---|
| 1 | Direction diagnostic out of the v0.1 API | `febd7ea` | done |
| 2 | `p = 1` accepted; Reduced scored with the null model (`fit_null_model`: `DummyRegressor("mean")` / `DummyClassifier("prior")`) | `febd7ea` | done |
| 3 | Default metric MSE / multinomial log loss (`scoring=None` → `neg_mean_squared_error` / `common_scorer("log_loss")` reported as `neg_log_loss`) | `399bb20` | done |
| 4 | Learner failures: `NaN` + one warning by default, `on_error="error"` to raise; `error` column (`"<ExceptionType>: <message>"`), `n_failed` | `399bb20` | done |
| 5 | Python `docs/specification.md` as a pointer to the R spec plus Python notes and recorded deviations | `dd0bef0` | done |
| 6 | Scorer ↔ common-name mapping documented; `common_scorer()` for `mse, rmse, mae, r2, rsq, log_loss, accuracy, roc_auc` | `1bb2ce8`, `399bb20` | done |
| 7 | Parity fixture of spec §15.1 (and collinear variant) at `1e-8`, also with the default metric | `d612a06` | done |
| 7b | Regenerate `tests/fixtures/cross_language/r_reference.json` from the current R commit (generator adapted to `score_*` columns, `collinear` flags, no `drop_constant`), remove the compensating conversions in `tests/test_cross_language.py`, extend to the default metric and to a failing split with `on_error="warn"`; use an OLS learner with an explicit rank decision for the collinear design (spec §8.1, §15.4) | — | **open** (deferred by the Python side until the R spec revision was final; it is final as of this commit) |
| 8 | Docker policy (spec §17): `latest` + short SHA on `main`, semver on `vX.Y.Z`, no push on PRs, manual runs publish only from `main` or a tag | `692c3fe` | done |

## Recorded deviations (Python notes §4.3–4.4; no action required)

- Non-finite predictions are not inspected in Python (scorers receive the
  fitted model); they surface as a scorer exception (failure) or a non-finite
  score (excluded, empty `error`). Summary numbers agree with R.
- Fit-time `sample_weight` (`fit_params`) exists in Python only; metrics are
  unweighted in both.
- Overlapping training/assessment rows are accepted in Python (in-sample
  closed-form tests) and rejected in R.
- `split` is 0-based in Python, 1-based in R; unnamed features are `x0, …` in
  Python and `V1, …` in R.

## What the R side changed in response to the Python review (`r_side_changes.md`, third revision)

All ten text corrections were applied to `docs/specification.md` in this
commit: `p = 1` status (§3); implementing commits `399bb20` / `dd0bef0`
instead of `692c3fe` (§9.1, §10, §16); Python's `error` column and message
format (§10, §11.2); specification pointer done (§16 item 5); log-loss
clipping (`common_scorer("log_loss")` clips at `2^−52`; scikit-learn's
`log_loss` already did so at 1.3.2) (§9.1); `accuracy` on argmax vs
`predict` (§9.1); `common_scorer("rsq")` and its `NaN` cases (§9.1); rank
deficiency in the downstream OLS learner (§8.1, new §15.4,
`docs/parity/rank_deficiency.md`); manual Docker runs publish only from
`main` or a release tag, and the R workflow was aligned to that rule (§17).
No implementation change was requested of R and none was made, apart from the
workflow condition.

## Verified numerically

- Parity fixture (spec §15.1): R and Python agree to ≤ 6.8e-14 on every
  column of both tables; collinear variant to 2.9e-14 in the current
  environments (scikit-learn 1.9.1 / NumPy 2.4.6).
- All common metrics: yardstick 1.4.0 vs scikit-learn 1.9.1 to ≤ 3e-15.
- Rank deficiency (spec §15.4, `docs/parity/rank_deficiency.md`): every solver
  that recognises the machine-epsilon singular value matches R to ≤ 1e-13;
  a solver that does not changes per-split MSE by 0.02–0.4. Residualization is
  not involved.
