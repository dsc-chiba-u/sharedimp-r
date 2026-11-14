# Status of the Python package against the common specification

Reference state: `sharedimp-py` commit `c082026` (2026-11-13; read only,
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
| 7b | Regenerate `r_reference.json` from the current R commit, compare directly, cover the default metric and failing splits, use a learner reproducing R's aliasing for rank-deficient designs (test-only `AliasingOLS`) | `2a34a02` | done (reference from R `92f9ecb`; recorded in `meta.sharedimp_r_commit`) |
| 9 | Non-finite predictions recorded as a learner failure, as in R | `98de522` | done (former deviation removed) |
| 8 | Docker policy (spec §17): `latest` + short SHA on `main`, semver on `vX.Y.Z`, no push on PRs, manual runs publish only from `main` or a tag | `692c3fe` | done |

## Section 8.2 rewording (2026-11-14): no Python change required

The random-forest table of spec §8.2 no longer pairs R's `min_n = 5` (ranger
`min.node.size = 5`) with scikit-learn's `min_samples_leaf = 5`; the closest
scikit-learn *definition* is `min_samples_split=6, min_samples_leaf=1`,
`max_features="sqrt"`, `n_estimators=500`, and the section states that the two
forests are not equivalent (node sizes counted in in-bag draws vs distinct
rows; tie-breaking). Source: `sharedimp-experiments`
`docs/package_change_proposal.md` and `docs/random_forest_comparison.md`. The
Python package has no built-in forest and its `docs/specification.md` only
refers to §8.2 for stochastic learners, so nothing in `sharedimp-py` needs to
change. If Python documentation ever shows a forest "like the R default", it
should use the settings above.

## Recorded deviations (Python notes §4.4; no action required)

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

Fourth revision of `r_side_changes.md` (against R `92f9ecb`): two text
corrections on rank-deficient OLS, applied to spec §8.1 and §15.4 and to
`docs/parity/rank_deficiency.md`: (1) solvers that recognise the rank
deficiency agree only where the assessment rows satisfy the training
dependency; in the constant-anchor case `lm.fit`'s column-order aliasing and
a minimum-norm solution predict differently (Python measurement: 1.5e-2);
(2) `numpy.linalg.lstsq` / SVD rank decisions are therefore not a counterpart
of `lm.fit`; parity tests use the test-only `AliasingOLS`. Again no
implementation change.

## Verified numerically

- Parity fixture (spec §15.1): R and Python agree to ≤ 6.8e-14 on every
  column of both tables; collinear variant to 2.9e-14 in the current
  environments (scikit-learn 1.9.1 / NumPy 2.4.6).
- All common metrics: yardstick 1.4.0 vs scikit-learn 1.9.1 to ≤ 3e-15.
- Rank deficiency (spec §15.4, `docs/parity/rank_deficiency.md`): on the
  collinear variant, solvers that recognised the machine-epsilon singular value
  matched R to ≤ 1e-13 in the R-side run and a solver that did not changed
  per-split MSE by 0.02–0.4. The Python side additionally measured (both
  stacks) one wrong cell for `scipy.linalg.lstsq` on that variant and a 1.5e-2
  difference for `numpy.linalg.lstsq` on a constant-anchor design, and ≤ 7.1e-14
  agreement with its `AliasingOLS` on all rank-deficient designs.
  Residualization is not involved.
- Python commit `2a34a02` regenerated its R reference by running R at `92f9ecb`
  and compares directly: residualization, representations, predictions, both
  tables, default metric, `p = 1`, learner failures all agree (Python notes §6).
