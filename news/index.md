# Changelog

## sharedimp 0.1.0

Initial release, following the common R/Python specification v0.1
(`docs/specification.md`, reconciled with `sharedimp-py` on 2026-11-13).

- [`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md):
  UMFI-LR with fold-internal residualization, paired full/reduced
  out-of-sample evaluation, scores oriented greater-is-better,
  `importance = score_full - score_reduced`, no truncation of negative
  values, per-split results, reproducible `seed`.
- Common output schema: `feature`, `importance`, `importance_std`,
  `score_full`, `score_reduced`, `n_splits`; per split `feature`,
  `split`, `n_train`, `n_test`, `score_full`, `score_reduced`,
  `importance` (+ `id`, `id2`, `error`).
- Full representation keeps the anchor in its original column position;
  exactly collinear columns get a residual of exactly zero and are kept.
- Regression, binary and multiclass classification; built-in
  `"rand_forest"` (ranger) and `"linear"` learners, any `parsnip` model
  specification, or a custom
  [`sharedimp_learner()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_learner.md).
  A single predictor is compared with the null model.
- Metrics via `yardstick` with common names `mse`, `rmse`, `mae`, `r2`,
  `rsq`, `log_loss`, `accuracy`, `roc_auc`, or any yardstick metric
  function.
- Low-level helpers
  [`residualize_fit()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md),
  [`residualize_apply()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md),
  [`build_representations()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md).
- The direction diagnostic is not part of the v0.1 API (definitions
  differed between the R and Python drafts); the code is kept internal
  only.
- Cross-language verification record in `docs/parity/`; comparison with
  the reference implementation in `docs/umfi_comparison.md`.
- Docker image (`Dockerfile`, `rocker/r-ver` base) published to
  `ghcr.io/dsc-chiba-u/sharedimp-r` and `koki/sharedimp-r` by
  `.github/workflows/build_test_push.yml`; the build runs the test suite
  and `docker/smoke-test.R`.
