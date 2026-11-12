# CLAUDE.md — sharedimp (R)

## What this package is

`sharedimp` implements the linear-regression variant of Ultra-marginal Feature
Importance (UMFI-LR, Janssen, Guan & Robeva, AISTATS 2023) as a careful,
resampling-based R package built on tidymodels (parsnip / rsample / yardstick).
It is a companion to the Python package in `../sharedimp-py`; both follow
`docs/specification.md`, which is the single source of truth for the
algorithm, inputs, outputs and metric conventions. Paper benchmarks live in
`../sharedimp-experiments`, not here. Never edit files in the sibling
repositories from this one; report differences instead.

We do not claim a new importance measure. Do not describe the output as a
causal or conditional effect. Do not add confidence intervals, p-values, or
other formal inference. Keep the user-facing API small: `shared_importance()`
plus a few accessors and low-level helpers.

## Algorithm (summary; details in docs/specification.md)

For each anchor feature `i` and each resampling fold:

1. On the **training rows only**, regress every other feature `X_j` on `X_i`
   (OLS with intercept) and keep `(a_j, b_j)`.
2. Residualize training and assessment rows with the **same** coefficients:
   `R_j = X_j − a_j − b_j X_i`.
3. Full representation `[X_i, R_{-i}]`; reduced representation `[R_{-i}]`
   (residual columns constant on the training fold are dropped from both).
4. Fit the learner on both representations, predict the assessment rows,
   evaluate the metric on each.
5. `importance = s * (metric_full − metric_reduced)` with `s = +1` for
   maximize metrics and `−1` for minimize metrics, so positive always means
   "keeping the anchor helps". Negatives are kept.
6. Aggregate by the mean over folds; report fold-level results too.

Defaults: 5-fold CV (stratified for classification), random forest via ranger
with 500 trees / `min_n = 5`, MSE for regression and multinomial log loss for
classification.

## Code layout

- `R/residualize.R` — pure matrix core: `residualize_fit()`, `residualize_apply()`,
  `build_representations()`. No learner or tidymodels code here.
- `R/learners.R` — learner adapters: `sharedimp_learner()`, built-in `"linear"`
  and `"rand_forest"`, parsnip `model_spec` adapter, null model.
- `R/metrics.R` — metric adapters around yardstick; direction handling.
- `R/shared_importance.R` — orchestration (folds × anchors), seeding, errors.
- `R/result.R` — `sharedimp_result` class, `print()`, `fold_results()`.
- `R/direction.R` — optional direction diagnostic.
- `tests/testthat/` — one file per concern; see "Testing".
- `docs/specification.md` — shared spec. `docs/umfi_comparison.md` — record of
  the numerical comparison with the reference implementation.

## Development environment

R is not on the PATH on this machine. Use the conda environment:

```
R=/home/koki/anaconda3/envs/sharedimp-dev/bin/Rscript
$R -e 'devtools::document()'
$R -e 'devtools::test()'
$R -e 'devtools::check()'          # full R CMD check
$R -e 'devtools::build_vignettes()'
```

Run `devtools::document()` after changing roxygen comments; `NAMESPACE` and
`man/` are generated, do not hand-edit them.

## Constraints

- Numeric predictors only; no imputation; factors in `x` are an error.
- Residualization coefficients must be estimated on the training fold only.
  Any change touching `build_representations()` must keep the leakage tests
  passing.
- Fixed hyperparameters inside `shared_importance()`; no tuning.
- Negative importance is never truncated.
- Metric direction is taken from the yardstick `direction` attribute; never
  hard-code a sign for a named metric elsewhere.
- Learner failures on a fold produce `NA` for that fold plus a warning, not an
  abort (unless `on_error = "error"`).
- Dependencies: Imports are parsnip, rsample, yardstick, ranger, rlang, tibble,
  stats. Everything else (glmnet, kernlab, ggplot2, ...) is Suggests and must
  be guarded with `requireNamespace()`.
- Target CRAN quality: `R CMD check` with no ERRORs/WARNINGs; examples run in
  seconds; tests deterministic (always pass `seed`).

## Testing policy

Start from tiny deterministic synthetic data and verify against hand
computation before testing behaviour statistically.

1. Residualization coefficients equal `lm(X_j ~ X_i)` on the training rows.
2. Training and assessment rows are transformed with the same coefficients.
3. Assessment rows do not influence the coefficients (perturbing assessment
   rows leaves training representations unchanged).
4. Full / reduced representations have the specified columns and order.
5. With the linear learner and MSE, importance matches the analytic
   expectation on a linear-Gaussian design.
6. Exact duplicates and independent predictors behave as specified.
7. Comparison with the reference UMFI-LR preprocessing (function vendored in
   the test under its MIT licence) — document, never fabricate, agreement.
8. Smoke tests for regression / binary / multiclass with both built-in
   learners and a parsnip spec.
9. Negative importance survives; learner failures yield `NA` + warning.

Use `custom learners` that record the matrices they receive to test the
pipeline without depending on a particular model.

## Commit discipline

Small, focused commits with imperative messages. Run `devtools::test()` before
committing; run `devtools::check()` before pushing anything that changes
DESCRIPTION, NAMESPACE, or exported functions. Never report a test or check as
passing without having run it.
