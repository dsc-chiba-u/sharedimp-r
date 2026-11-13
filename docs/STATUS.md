# sharedimp-r — work log and TODO

Last updated: 2026-11-13 (commit after `eaafaee`). This file records what has
been done, what was verified and how, and what remains, across the three
sibling repositories. The algorithm itself is specified in
`docs/specification.md`; this file is about project state.

## 1. Done (R package, this repository)

### Package

- `shared_importance()` implementing UMFI-LR with fold-internal OLS
  residualization, paired full/reduced out-of-sample evaluation, scores
  oriented greater-is-better, `importance = score_full − score_reduced`, no
  truncation of negatives, per-split results, local `seed`, explicit
  `resamples` (rset or fold labels), `features` subset, `on_error`.
- Regression, binary and multiclass classification. Built-in learners
  `"rand_forest"` (ranger, 500 trees, `min_n = 5`) and `"linear"`
  (`lm.fit` / `glm.fit` / `nnet::multinom`, aliased coefficients → 0); any
  parsnip spec; `sharedimp_learner()`.
- Metrics via yardstick with common names `mse`, `rmse`, `mae`, `r2`
  (`rsq_trad`), `rsq`, `log_loss` (`mn_log_loss`), `accuracy`, `roc_auc`;
  defaults `mse` / `log_loss`.
- Low-level exports `residualize_fit()`, `residualize_apply()`,
  `build_representations()`.
- Common R/Python output schema (`feature, importance, importance_std,
  score_full, score_reduced, n_splits`; per split `feature, split, n_train,
  n_test, score_full, score_reduced, importance` + R-only `id, id2, error`).
- Degenerate rules: constant anchor / constant feature by exact `max == min`;
  exactly collinear features (`SS_res ≤ 1e-10 SS_tot`) get residual exactly 0
  and are kept; `p = 1` uses the null model.
- Direction diagnostic removed from the public API (kept internal,
  unexported, experimental).

### Verification (all actually run)

- `devtools::test()`: 567 tests pass, 0 failures, 0 warnings.
- `R CMD check --as-cran --no-manual`: 0 errors, 0 warnings, 2 NOTEs ("New
  submission"; "unable to verify current time" — environmental).
- GitHub Actions on `main`: R-CMD-check (macOS / Windows / Ubuntu devel,
  release, oldrel-1), test-coverage, pkgdown, DockerHub/GHCR all green on the
  latest commits.
- Cross-language parity (2026-11-13, `docs/parity/`): deterministic fixture
  (OLS + mse, explicit splits, including an exactly collinear column) agrees
  with `sharedimp-py` to ≤ 7e-14 on every column of both tables; all common
  metrics agree between yardstick 1.4.0 and scikit-learn 1.9.1 to ≤ 3e-15.
- Comparison with the official UMFI code (`docs/umfi_comparison.md`):
  residuals identical where the official p-value gate fires; rankings agree on
  the paper's synthetic designs.
- Docker image built in CI on `rocker/r-ver:4.5.3`; the build runs the full
  test suite and `docker/smoke-test.R`; published to
  `ghcr.io/dsc-chiba-u/sharedimp-r` (public) and `koki/sharedimp-r` with tags
  `latest` + short SHA, identical digest on both; the pushed image was pulled
  and `library(sharedimp)` run in CI.
- pkgdown site live at https://dsc-chiba-u.github.io/sharedimp-r/ (Pages
  enabled from `gh-pages` on 2026-11-13).

### Documentation and infrastructure

- `docs/specification.md` v0.1 (common spec, reconciled with the Python side;
  Sections 15–17: verification record, Python change list, Docker policy).
- `docs/python_changes.md`, `docs/parity/`, `docs/umfi_comparison.md`.
- `CLAUDE.md`, `README.md` (badges, Docker usage), vignette, `NEWS.md`,
  `CITATION.cff` (author from DESCRIPTION; UMFI paper as preferred citation;
  no affiliation/ORCID, no paper DOI — none verified).
- Workflows: `R-CMD-check.yaml`, `test-coverage.yaml`, `pkgdown.yaml`
  (`dest_dir = "pkgdown_site"`), `build_test_push.yml` (PR = build/test
  only; `main` = publish `latest` + SHA; `vX.Y.Z` = semver tags, no `latest`;
  manual `publish=false` = build only; missing Docker Hub secrets fail early).
- Repository secrets `DOCKER_USERNAME` / `DOCKER_PASSWORD` registered by the
  owner on 2026-11-13.

## 2. TODO — R package

- [ ] Decide the default learner: `"rand_forest"` (current) vs `"linear"`
      (fast, deterministic). One-line change in `shared_importance()`.
- [ ] Optional: `CODECOV_TOKEN` secret so coverage uploads (currently
      skipped, non-fatal). Then add a coverage badge.
- [ ] Decide whether `vX.Y.Z` tags should also build an R release (GitHub
      Release + tarball); the Docker workflow already handles version tags.
- [ ] Before any CRAN submission: `Description` wording, `\value` sections,
      spell check, `cran-comments.md`, run `devtools::check()` on R-devel
      and `rhub`/win-builder.
- [ ] Add the SoftwareX DOI to `CITATION.cff`, `DESCRIPTION` (`Description`
      / `URL`) and README **only after** publication.
- [ ] Keep `docs/specification.md` as the single source of truth; any change
      must be mirrored in `sharedimp-py` (never edit that repo from here).

## 3. TODO — Python side (`sharedimp-py`, for its maintainers; see `docs/python_changes.md`)

State observed at commit `692c3fe`:

- [x] Direction removed from the API; `p = 1` with null model;
      `common_scorer()`; parity fixture test; Docker workflow with the same
      tag policy; default metrics `neg_mean_squared_error` / common
      `log_loss`; `on_error = "warn" | "error"` — all implemented in code.
- [ ] Update the Python `docs/specification.md` §10/§11 and README: they
      still describe `estimator.score` as default and "exception propagates".
- [ ] Replace the Python spec by a pointer to the R spec + Python notes (or
      keep the restatement but mark the R file as authoritative).
- [ ] Regenerate `tests/fixtures/cross_language/r_reference.json` from the
      current R commit (old reference came from `6c5631d`: anchor first,
      dropped columns, raw metric values, feature-major rows) and remove the
      compensating conversions in `tests/test_cross_language.py`.

## 4. TODO — `sharedimp-experiments` (not started; do not touch until asked)

- [ ] Benchmarks for the SoftwareX paper: comparison with the official UMFI
      implementations (R and Python) on the paper's synthetic designs;
      R-vs-Python agreement on a real data set with explicit splits and the
      `"linear"` learner; runtime scaling in `n`, `p`, folds; effect of the
      learner (linear vs random forest) on rankings.
- [ ] Reproducible environment (Docker images above, pinned versions).

## 5. TODO — SoftwareX manuscript

- [ ] Positioning text shared with the Python repo: software contribution
      (leakage-free fold-internal residualization, paired evaluation,
      cross-language parity, reproducibility), explicitly *not* a new
      importance method, no causal/conditional claims, no inference.
- [ ] Figures/tables from `sharedimp-experiments`; the parity numbers from
      `docs/specification.md` Section 15.
- [ ] Zenodo DOIs for both packages at the version used in the paper; add
      them to the CITATION files afterwards.

## 6. Decisions taken (with dates)

| date | decision |
|---|---|
| 2026-11-12 | Always residualize (no p-value / R² gate); keep negatives; fold-internal fit; paired per-split metrics. |
| 2026-11-13 | Full keeps the anchor in its original column position; Reduced = Full without it. |
| 2026-11-13 | Exactly collinear features: residual fixed at 0, column kept (learner study in spec §15.3). |
| 2026-11-13 | Constant detection by exact `max == min`, not a computed mean. |
| 2026-11-13 | Scores oriented greater-is-better; common column names; split-major row order. |
| 2026-11-13 | Direction diagnostic out of v0.1 (R Pearson-on-difference vs Python Spearman-on-prediction differ). |
| 2026-11-13 | Default metrics MSE / multinomial log loss; `on_error` default `"warn"`; both normative. |
| 2026-11-13 | Docker: `rocker/r-ver` base (no Bioconductor); `latest` only from `main`; semver tags from `vX.Y.Z`. |

## 7. Environment notes

- R is not on PATH; use `/home/koki/anaconda3/envs/sharedimp-dev/bin/Rscript`
  (R 4.4.3). `devtools::check()` fails there ("compile tools"); use
  `R CMD build` + `R CMD check` with the env's `bin` on PATH instead.
- Docker on this machine is the snap package with a root-owned socket and no
  passwordless sudo, so images are verified through GitHub Actions
  (`gh workflow run build_test_push.yml -f publish=false` for a build-only
  run).
- The Python package is installed editable in `../sharedimp-py/.venv`; running
  it uses that repository's working tree.
