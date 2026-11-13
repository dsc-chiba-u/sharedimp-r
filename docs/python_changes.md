# Changes required in `sharedimp-py` to match the common specification

Reference state: `sharedimp-py` commit `bfcb2a5` (read on 2026-11-13; nothing
there was modified from this repository). The canonical list is Section 16 of
`docs/specification.md`; this file repeats it with pointers into the Python
code so the maintainers can act on it directly.

## Must change

| # | item | where in `sharedimp-py` | what to do |
|---|---|---|---|
| 1 | Direction diagnostic is out of the v0.1 API | `_core.py` (`direction=` argument, `direction` columns), `_direction.py`, `_result.py` (`direction` handling), docs | Remove the argument and the columns. Keep `_direction.py` only as internal, clearly marked experimental, or delete it. The R and Python definitions differed (Pearson on `pred_full − pred_reduced` vs Spearman on `pred_full`). |
| 2 | `p = 1` must be accepted | `_core.py::_validate_X` (`values.shape[1] < 2` → error) | Allow one column. When the reduced representation has zero columns, score the null model (training mean for regressors; training class frequencies for classifiers) instead of calling the estimator. R implements this in `null_learner()`. |
| 3 | Default scoring | `_scoring.py::resolve_scorer` (`scoring=None` → `estimator.score`) | Default to `neg_mean_squared_error` for regressors and `neg_log_loss` for classifiers (R defaults `mse` / `log_loss`). If `estimator.score` is kept as the default, document that parity tests must pass the scorer explicitly. |
| 4 | Learner failures | `_core.py::_evaluate_anchor` | Catch fit/predict errors (and a training split with a single class) per `(split, feature)`, record `NaN` scores, warn once with the count, aggregate over the remaining splits with `n_splits` = successful splits; offer a flag to raise instead (R: `on_error = "error"`). |
| 5 | Specification pointer | `docs/specification.md`, `CLAUDE.md` | Replace the provisional spec by a pointer to `sharedimp-r/docs/specification.md` plus Python-specific notes; remove the `0.1-provisional` label. |
| 6 | Metric documentation | README / docstrings | State the common-name ↔ scorer mapping: `r2` ↔ `r2`, `rsq` ↔ Pearson² (custom scorer), `log_loss` ↔ `neg_log_loss` (machine-epsilon clipping needs scikit-learn ≥ 1.5), `roc_auc` multiclass ↔ `roc_auc_ovo` with `average="macro"` (**not** `roc_auc_ovr`). |
| 7 | Parity test | `tests/` | Add the fixture of specification Section 15.1 (and its collinear variant) with tolerance `1e-8` on every column of both tables. A run against the current code already passes at `1e-13`. |

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

With the current Python code and the R package at this commit, on the
deterministic fixture with `LinearRegression` and `neg_mean_squared_error`:
every column of both tables agrees to ≤ 6.8e-14. All common metrics agree
between yardstick 1.4.0 and scikit-learn 1.9.1 to ≤ 3e-15 (details:
`docs/specification.md` Section 15, `docs/parity/`).
