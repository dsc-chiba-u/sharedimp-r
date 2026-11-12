# sharedimp 0.1.0

Initial release.

* `shared_importance()`: UMFI-LR with fold-internal residualization, paired
  full/reduced out-of-sample evaluation, metric-direction-aware sign, no
  truncation of negative values, per-fold results, reproducible `seed`.
* Regression, binary and multiclass classification; built-in `"rand_forest"`
  (ranger) and `"linear"` learners, any `parsnip` model specification, or a
  custom `sharedimp_learner()`.
* Metrics via `yardstick` (`mse`, `rmse`, `mae`, `rsq`, `mn_log_loss`,
  `accuracy`, `roc_auc`, or any yardstick metric function).
* Low-level helpers `residualize_fit()`, `residualize_apply()`,
  `build_representations()`.
* Optional `direction = TRUE` diagnostic.
* Shared R/Python specification in `docs/specification.md`; comparison with
  the reference implementation in `docs/umfi_comparison.md`.
