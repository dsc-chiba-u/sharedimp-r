# Changelog

## sharedimp 0.1.0

Initial release.

- [`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md):
  UMFI-LR with fold-internal residualization, paired full/reduced
  out-of-sample evaluation, metric-direction-aware sign, no truncation
  of negative values, per-fold results, reproducible `seed`.
- Regression, binary and multiclass classification; built-in
  `"rand_forest"` (ranger) and `"linear"` learners, any `parsnip` model
  specification, or a custom
  [`sharedimp_learner()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_learner.md).
- Metrics via `yardstick` (`mse`, `rmse`, `mae`, `rsq`, `mn_log_loss`,
  `accuracy`, `roc_auc`, or any yardstick metric function).
- Low-level helpers
  [`residualize_fit()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md),
  [`residualize_apply()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md),
  [`build_representations()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md).
- Optional `direction = TRUE` diagnostic.
- Shared R/Python specification in `docs/specification.md`; comparison
  with the reference implementation in `docs/umfi_comparison.md`.
