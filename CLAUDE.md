# CLAUDE.md — sharedimp (R)

## What this package is

`sharedimp` implements the linear-regression variant of Ultra-marginal
Feature Importance (UMFI-LR, Janssen, Guan & Robeva, AISTATS 2023) as a
careful, resampling-based R package built on tidymodels (parsnip /
rsample / yardstick). It is a companion to the Python package in
`../sharedimp-py`; both follow `docs/specification.md` (v0.1, reconciled
2026-11-13, final revision after the Python-side review), which is the
single source of truth for the algorithm, inputs, outputs and metric
conventions; the Python repository’s `docs/specification.md` is a
pointer to it plus Python notes. The goal is that R and Python compute
the same UMFI-LR quantity; paper benchmarks live in
`../sharedimp-experiments`, not here. Never edit files in the sibling
repositories from this one; report differences instead
(`docs/python_changes.md` records the Python status per item with commit
SHAs; the Python side’s requests to R are in
`../sharedimp-py/docs/r_side_changes.md`).

We do not claim a new importance measure. Do not describe the output as
a causal or conditional effect. Do not add confidence intervals,
p-values, or other formal inference. Keep the user-facing API small:
[`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md)
plus a few accessors and low-level helpers.

## Algorithm (summary; details in docs/specification.md)

For each split and each anchor feature `i`:

1.  On the **training rows only**, regress every other feature `X_j` on
    `X_i` (OLS with intercept) and keep `(a_j, b_j)`.
2.  Residualize training and assessment rows with the **same**
    coefficients: `R_j = X_j − a_j − b_j X_i`. A column that is an exact
    affine function of the anchor on the training rows
    (`SS_res <= 1e-10 SS_tot`) gets a residual of exactly 0 and is
    **kept**. A constant anchor (`max == min`, exact) gets slopes 0 and
    a warning; a constant other column gets slope 0 and its value as
    intercept.
3.  Full representation = `X` with the other columns replaced by their
    residuals, the anchor raw and **in its original position**; reduced
    representation = full without the anchor column. Column count is
    always `p` and `p − 1`; nothing is dropped.
4.  Fit the learner on both representations, predict the assessment
    rows, evaluate the metric on each.
5.  Scores are oriented greater-is-better (`score = metric` for maximize
    metrics, `−metric` for minimize metrics);
    `importance = score_full − score_reduced`, so positive always means
    “keeping the anchor helps”. Negatives are kept.
6.  Aggregate by the mean over splits; report split-level results too.

Defaults: 5-fold CV (stratified for classification), random forest via
ranger with 500 trees / `min_n = 5`, MSE for regression and log loss for
classification. `p = 1` is allowed: the reduced model is the null model.

## Output schema (shared with Python)

- `$importance`:
  `feature, importance, importance_std, score_full, score_reduced, n_splits`
  — one row per anchor in `features` order.
- [`fold_results()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_result.md):
  `feature, split, n_train, n_test, score_full, score_reduced, importance`
  (common) + `id, id2, error` (R only) — ordered by split, then feature.

The direction diagnostic is **not** part of the v0.1 API (the R and
Python drafts defined different statistics). `R/direction.R` is kept
internal and unexported; do not expose it without an agreed definition.

## Code layout

- `R/residualize.R` — pure matrix core:
  [`residualize_fit()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md),
  [`residualize_apply()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md),
  [`build_representations()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md).
  No learner or tidymodels code here.
- `R/learners.R` — learner adapters:
  [`sharedimp_learner()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_learner.md),
  built-in `"linear"` (`lm.fit` / `glm.fit` /
  [`nnet::multinom`](https://rdrr.io/pkg/nnet/man/multinom.html),
  aliased coefficients → 0) and `"rand_forest"`, parsnip `model_spec`
  adapter, null model.
- `R/metrics.R` — metric adapters around yardstick; direction handling;
  `mse`.
- `R/shared_importance.R` — orchestration (splits × anchors), seeding,
  errors.
- `R/result.R` — `sharedimp_result` class,
  [`print()`](https://rdrr.io/r/base/print.html),
  [`fold_results()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_result.md).
- `R/direction.R` — experimental, unexported, unreachable from the API.
- `tests/testthat/` — one file per concern; see “Testing”.
- `docs/STATUS.md` — running work log and TODO list across the three
  repos; read it first when resuming, update it at the end of a session.
- `docs/specification.md` — common spec. `docs/python_changes.md` —
  Python status per spec item. `docs/parity/rank_deficiency.md` — why
  rank-deficient OLS designs need a learner that reproduces `lm.fit`’s
  aliasing in parity tests (with its later correction). `docs/parity/` —
  cross-language verification record. `docs/umfi_comparison.md` —
  comparison with the reference implementation.

## Citation

`CITATION.cff` (Rbuildignored) holds the citation metadata: author from
DESCRIPTION, the UMFI paper as `preferred-citation`, the Python package
under `references`. Only verified facts go in it: no affiliation or
ORCID has been confirmed, and no DOI for the (unsubmitted) SoftwareX
paper. Add the DOI only once the paper is published.

## Development environment

R is not on the PATH on this machine. Use the conda environment:

    R=/home/koki/anaconda3/envs/sharedimp-dev/bin/Rscript
    $R -e 'devtools::document()'
    $R -e 'devtools::test()'
    $R -e 'devtools::check()'          # full R CMD check
    $R -e 'devtools::build_vignettes()'

Run `devtools::document()` after changing roxygen comments; `NAMESPACE`
and `man/` are generated, do not hand-edit them. The Python package can
be *run* for comparisons from `../sharedimp-py` with its
`.venv/bin/python` (read-only; never edit that repository from here).

## Docker image

`Dockerfile` (base `rocker/r-ver:<R_VERSION>`, pinned stable R; not
Bioconductor, which sharedimp does not need) installs the Imports plus
`nnet`, `testthat`, `withr`, installs the package with
`--install-tests`, runs
[`tools::testInstalledPackage()`](https://rdrr.io/r/tools/testInstalledPackage.html)
and `docker/smoke-test.R` during the build, and removes the sources.
`.dockerignore` keeps docs, CI files and build artifacts out of the
context; `Dockerfile`, `.dockerignore` and `docker/` are in
`.Rbuildignore`.

`.github/workflows/build_test_push.yml` (policy from
`dsc-chiba-u/fbrglm` with refinements): pull requests build and test
only; pushes to `main` publish the same build to
`ghcr.io/dsc-chiba-u/sharedimp-r` and `koki/sharedimp-r` with tags
`latest` + short SHA via `docker/metadata-action` and one
`docker/build-push-action` call; `vX.Y.Z` tags publish semver tags
without `latest`; manual runs publish only from `main` or a `vX.Y.Z`
tag; missing `DOCKER_USERNAME` / `DOCKER_PASSWORD` secrets fail the
publishing run early with an explicit message; GHCR uses `GITHUB_TOKEN`.
A manual run with `publish=false`
(`gh workflow run build_test_push.yml -f publish=false`) builds and
tests only, which validates the Dockerfile in CI without registry
credentials. Docker is not usable from the sandbox on this machine (snap
socket owned by root), so verify image changes through the Actions run,
or run `sudo docker build .` yourself. Never report a registry push as
successful without seeing the workflow succeed.

## Constraints

- Numeric predictors only; no imputation; factors in `x` are an error.
- Residualization coefficients must be estimated on the training rows
  only. Any change touching
  [`build_representations()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md)
  must keep the leakage tests and the cross-language fixture
  (`test-common-spec.R`) passing.
- Fixed hyperparameters inside
  [`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md);
  no tuning.
- Negative importance is never truncated.
- Metric direction is taken from the yardstick `direction` attribute;
  never hard-code a sign for a named metric elsewhere. `r2` means
  `rsq_trad` (1 − SS_res/SS_tot), `rsq` means squared Pearson
  correlation; keep them distinct.
- Do not claim that arbitrary learners give identical predictions across
  languages on rank-deficient designs: OLS coefficients are not unique
  there, and predictions on assessment rows that leave the training
  dependency depend on the solver’s convention. `lm.fit` aliases in
  column order; parity tests on such designs must reproduce that
  convention (Python: test-only `AliasingOLS`), not merely recognise the
  rank deficiency.
- Learner failures on a split produce `NA` for that split plus a
  warning, not an abort (unless `on_error = "error"`).
- Dependencies: Imports are parsnip, rsample, yardstick, ranger, rlang,
  tibble, stats. Everything else (glmnet, kernlab, ggplot2, nnet, …) is
  Suggests and must be guarded with
  [`requireNamespace()`](https://rdrr.io/r/base/ns-load.html).
- Target CRAN quality: `R CMD check` with no ERRORs/WARNINGs; examples
  run in seconds; tests deterministic (always pass `seed` or explicit
  `resamples`).

## Testing policy

Start from tiny deterministic synthetic data and verify against hand
computation before testing behaviour statistically.

1.  Residualization coefficients equal `lm(X_j ~ X_i)` on the training
    rows.
2.  Training and assessment rows are transformed with the same
    coefficients.
3.  Assessment rows do not influence the coefficients (perturbing
    assessment rows leaves training representations unchanged).
4.  Full / reduced representations have the specified columns and order
    (anchor in place); importance is invariant to the column order of
    `x`.
5.  With the linear learner and MSE, importance matches the analytic
    expectation on a linear-Gaussian design.
6.  Exact duplicates (zero residual column, kept), independent
    predictors and `p = 1` (null model) behave as specified.
7.  Comparison with the reference UMFI-LR preprocessing (function
    re-implemented in the test) — document, never fabricate, agreement.
8.  Smoke tests for regression / binary / multiclass with both built-in
    learners and a parsnip spec.
9.  Negative importance survives; learner failures yield `NA` + warning.
10. Output schema (both tables) and the deterministic cross-language
    fixture of specification Section 15.1 reproduce the recorded values;
    metric definitions match the specification table.

Use custom learners that record the matrices they receive to test the
pipeline without depending on a particular model.

## Commit discipline

Small, focused commits with imperative messages. Run `devtools::test()`
before committing; run `devtools::check()` before pushing anything that
changes DESCRIPTION, NAMESPACE, or exported functions. Never report a
test or check as passing without having run it.
