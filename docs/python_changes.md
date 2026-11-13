# Changes required in `sharedimp-py` to match the common specification

Reference state: `sharedimp-py` commit `bfcb2a5` plus the uncommitted working
tree observed at 12:50 JST on 2026-11-13 (edited concurrently on the Python
side; nothing there was modified from this repository). The canonical list is
Section 16 of `docs/specification.md`; this file repeats it with pointers into
the Python code and the status observed in that working tree.

## Items raised by the R side

| # | item | where in `sharedimp-py` | what to do | status in the working tree (12:50 JST) |
|---|---|---|---|---|
| 1 | Direction diagnostic is out of the v0.1 API | `_core.py` (`direction=` argument, `direction` columns), `_direction.py`, `_result.py`, docs | Remove the argument and the columns. Keep `_direction.py` only as internal, clearly marked experimental, or delete it. The R and Python definitions differed (Pearson on `pred_full − pred_reduced` vs Spearman on `pred_full`). | done |
| 2 | `p = 1` must be accepted | `_core.py::_validate_X` | Allow one column. When the reduced representation has zero columns, score the null model (training mean for regressors; training class frequencies for classifiers) instead of calling the estimator. R implements this in `null_learner()`. | done (`fit_null_model`) |
| 3 | Default scoring | `_scoring.py::resolve_scorer` (`scoring=None` → `estimator.score`) | Default to `neg_mean_squared_error` for regressors and `neg_log_loss` for classifiers (R defaults `mse` / `log_loss`), or keep `estimator.score` and document it. | open: documented, default unchanged; to be decided jointly |
| 4 | Learner failures | `_core.py::_evaluate_anchor` | Catch fit/predict errors (and a training split with a single class) per `(split, feature)`, record `NaN` scores, warn once with the count, aggregate over the remaining splits with `n_splits` = successful splits; offer a flag to raise instead (R: `on_error = "error"`). | open: exception propagates |
| 5 | Specification pointer | `docs/specification.md`, `CLAUDE.md` | Replace the Python-side spec by a pointer to `sharedimp-r/docs/specification.md` plus Python-specific notes. | open: waiting for the R revision, which is now committed |
| 6 | Metric documentation | README / docstrings | State the common-name ↔ scorer mapping: `r2` ↔ `r2`, `rsq` ↔ Pearson² (custom scorer), `log_loss` ↔ `neg_log_loss` (machine-epsilon clipping needs scikit-learn ≥ 1.5), `roc_auc` multiclass ↔ `roc_auc_ovo` with `average="macro"` (**not** `roc_auc_ovr`). | done (`common_scorer()`) |
| 7 | Parity test | `tests/` | Add the fixture of specification Section 15.1 (and its collinear variant) with tolerance `1e-8` on every column of both tables. | done; but `tests/fixtures/cross_language/r_reference.json` was generated from R commit `6c5631d` (old layout) and must be regenerated from the current R commit, removing the compensating conversions in `tests/test_cross_language.py` |

## Already consistent (no change)

* Fold-internal OLS residualization with intercept; exact-collinearity rule
  `SS_res ≤ 1e-10 · SS_tot` with the residual set to exactly 0 and the column
  kept; constant anchor → slopes 0.
* Full = anchor in its original position + residualized others; Reduced = Full
  without the anchor column.
* Scores oriented greater-is-better; `importance = score_full − score_reduced`;
  no truncation anywhere.
* Unweighted mean and sample standard deviation (`ddof = 1`) over splits.
* Summary columns `feature, importance, importance_std, score_full,
  score_reduced, n_splits`; per-split columns `feature, split, n_train, n_test,
  score_full, score_reduced, importance`; per-split rows ordered by split, then
  feature.
* Seeding policy and the statement that cross-language equality is limited to
  deterministic learners on explicit splits.
* Default names `x0, x1, …` for unnamed arrays (R uses `V1, V2, …`; parity
  tests use named columns).

## Verified numerically (2026-11-13)

With the Python working tree at 12:50 JST and the R package at this commit, on
the deterministic fixture with `LinearRegression` and `neg_mean_squared_error`:
every column of both tables agrees to ≤ 6.8e-14 (identical numbers in a run
about 20 minutes earlier). All common metrics agree
between yardstick 1.4.0 and scikit-learn 1.9.1 to ≤ 3e-15 (details:
`docs/specification.md` Section 15, `docs/parity/`).
