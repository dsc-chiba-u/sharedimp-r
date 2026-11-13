# Rank deficiency in the downstream OLS learner (verification record, 2026-11-13)

> **Correction (same day, after the Python side's fourth review).** The
> conclusion below originally claimed that every solver which recognises the
> rank deficiency gives the same predictions and that a minimum-norm fit is
> therefore a counterpart of R's `lm.fit`. That holds only when the assessment
> rows satisfy the same linear dependency as the training rows, which is the
> case for the design measured here. It fails when they do not (an anchor
> constant on the training rows but varying on the assessment rows), where the
> Python side measured a 1.5e-2 difference between `numpy.linalg.lstsq` and R.
> The measurements in this file are unchanged and correct; the conclusion
> section was rewritten. See `sharedimp-py/docs/rank_deficiency.md` and the
> specification Sections 8.1 and 15.4.

## The report

The Python side reported that on the collinear variant of the parity fixture
(specification Section 15.1: `x1_dup = 3·x1 − 2` added to the three-feature
fixture, OLS learner, `mse`, three explicit splits) `LinearRegression` gave an
importance of 41.79 instead of 45.55 for `x3` under scikit-learn 1.3.2 /
NumPy 1.23, while the current stack (scikit-learn 1.9.1 / NumPy 2.4.6) and R
agree to 2.9e-14. Question: is this a problem of the UMFI-LR residualization
or of the downstream learner?

## What the design looks like

For anchors `x2` and `x3` the exact-collinearity rule D3 does **not** fire
(`x1` and `x1_dup` are not collinear with the anchor), so both are
residualized on the anchor. In exact arithmetic `r(x1_dup) = 3·r(x1)`: the two
residual columns are mutually collinear and the Full design `[1, r(x1), x_anchor,
r(x3), r(x1_dup)]` has rank 4, not 5. In floating point the two columns differ
by round-off only:

| quantity (training rows, all anchors/splits) | value |
|---|---|
| `max |r(x1_dup) − 3·r(x1)|` | 1.8e-15 … 7.1e-15 |
| smallest / largest singular value of the Full design | 2.0e-17 … 1.6e-16 (R and NumPy) |
| machine epsilon | 2.2e-16 |

The residualization itself is identical in R and Python (the representations
agree to ~1e-15, Section 15.1), so the representation is not the cause.

## What the solvers do with it

The smallest singular value sits **at** machine epsilon. Every least-squares
solver decides whether to treat it as zero:

| solver | rank decision | result (all 6 anchor × split cells) |
|---|---|---|
| R `lm.fit` / `lm` (pivoted QR, `tol = 1e-7`) | aliases `x1_dup` (coefficient `NA` → 0 in sharedimp) | reference |
| NumPy `lstsq`, `rcond = None` (default: `eps·max(M, N)`) | drops the tiny singular value | equal to reference (≤ 1.3e-14) |
| NumPy `lstsq`, `rcond = -1` (pre-1.14 default: machine epsilon) | drops it *in this environment* | equal to reference (≤ 1.3e-14) |
| NumPy `lstsq`, `rcond = 1e-7` (R's tolerance) | drops it | equal to reference (≤ 1.3e-14) |
| scikit-learn 1.9.1 `LinearRegression` (SciPy `lstsq`, `cond = None`) | drops it *in this environment* | equal to reference (≤ 9.6e-14) |
| minimum-norm SVD solve in R, tolerance `1e-7` or machine epsilon | drops it | equal to reference (≤ 1.4e-14) |
| **any solver that keeps it** (`rcond = 1e-20` in NumPy; untruncated SVD solve in R) | treats the design as full rank | coefficients of order 1e13–1e14; test MSE changes by 0.02–0.4 per split (e.g. 0.286 → 0.547) |

Fitted values on the training rows are a projection onto the column space, so
every solver that recognises the rank deficiency gives the same *fitted*
values. The *predictions* on new rows coincide as well **in this design**,
because the assessment rows satisfy the same dependency (`r(x1_dup) = 3·r(x1)`
holds everywhere); they need not coincide in general (see the correction above
and the rewritten conclusion). A solver that does not recognise it
inverts a singular value of ~1e-16 and amplifies round-off into the
predictions. Because the critical singular value is within a factor of two of
the cut-off used by "machine precision" rules, the decision can go either way
depending on BLAS/LAPACK build, centring of the design (`fit_intercept`), and
library version. That is what the older scikit-learn / NumPy stack showed; it
could not be reproduced in the current environments, where all solvers agree.

## Conclusion (rewritten after the Python review)

- **Not a residualization problem.** UMFI-LR's step is exact and identical in
  both languages; the design it produces is legitimately rank-deficient when
  features other than the anchor are mutually collinear, or when an anchor is
  constant on the training rows (rule D1), and the specification deliberately
  does not alter such columns (rule D3 concerns collinearity with the anchor
  only).
- **A downstream learner problem, in two parts.** (i) On a rank-deficient
  training design the OLS coefficients are not unique; whether a solver even
  recognises a singular value at machine epsilon is implementation-dependent
  (this file). (ii) Even among solvers that recognise it, the predictions on
  assessment rows that *leave* the training dependency depend on which
  least-squares solution is returned: R's `lm.fit` aliases in column order
  (the dependent column gets coefficient `NA`, set to 0 by sharedimp), a
  minimum-norm solution spreads weight over the dependent columns. The
  constant-anchor case shows the difference (Python-side measurement, 1.5e-2
  in a Full score); the design measured here does not, because its assessment
  rows satisfy the dependency.
- **Rule adopted** (specification Section 8.1): no identity claim for
  arbitrary learners on rank-deficient designs. Cross-language parity tests on
  such designs reproduce R's aliasing convention: R `lm.fit`; Python the
  test-only `AliasingOLS` of `tests/test_cross_language.py`. Generic
  least-squares routines (`LinearRegression`, `numpy.linalg.lstsq` with any
  `rcond`, SVD minimum-norm fits) are not counterparts of `lm.fit` on every
  rank-deficient design. The full-rank parity fixture is unaffected and uses
  `LinearRegression`.

Scripts and raw output: `rankdef.py`, `rankdef_mechanism.py`, `rankdef.R`,
`rankdef_mechanism.R`, `rank_deficiency_output.txt` in this directory.
Environment: R 4.4.3; Python 3.11 with NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.9.1 (the `sharedimp-py` virtual environment at commit
`5277017`, read-only).
