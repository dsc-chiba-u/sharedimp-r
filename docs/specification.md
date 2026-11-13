# sharedimp — common algorithm specification (R / Python)

Status: **v0.1, reconciled 2026-11-13; final revision after the Python-side
review (`sharedimp-py/docs/r_side_changes.md`, third revision).** This
document is the single source of truth for the R package `sharedimp` (this
repository) and the Python package `sharedimp` (`../sharedimp-py`, whose
`docs/specification.md` is a pointer to this file plus Python-specific notes
and recorded deviations since Python commit `dd0bef0`). Both implementations
must follow this document; any intentional deviation must be recorded in the
deviating repository, never absorbed silently. Section 16 records the status
of the changes that were asked of the Python package; both packages implement
everything below.

## 1. Scope and non-claims

`sharedimp` is a careful, resampling-based implementation of the
linear-regression variant of Ultra-marginal Feature Importance (UMFI-LR;
Janssen, Guan & Robeva, AISTATS 2023, PMLR 206). We do not claim a new feature
importance measure. The contribution is engineering: fold-internal
residualization, paired out-of-sample evaluation, learner/metric adapters for
modern ML workflows (tidymodels in R, scikit-learn in Python), reproducible
random state, and a result object that exposes per-split results.

The reported numbers are *predictive* importances: how much out-of-sample
predictive performance is gained by keeping the anchor feature next to a
representation of the other features from which its linear component has been
removed. They are not causal effects, not conditional effects, and no
confidence intervals, p-values, or other formal inference are provided.

Out of scope for v0.1: inference of any kind, optimal-transport dependence
removal (UMFI-OT), hyperparameter tuning, non-numeric features, missing values,
and the direction diagnostic (Section 13).

## 2. Notation

- `n` observations, `p` numeric predictors `X` (n × p), response `y`.
- `i` is the *anchor* feature (the feature whose importance is being
  computed); `-i` denotes the remaining `p − 1` features.
- A *split* `s` is a pair (training index set `T_s`, assessment index set
  `A_s`) with `T_s ∩ A_s = ∅`. "Fold" is used as a synonym.
- `ν` (nu) is an evaluation metric; `score` is `ν` oriented so that greater is
  better (Section 9).

## 3. Inputs

| Input | Requirement |
|---|---|
| `x` / `X` | numeric matrix or data frame whose columns are all numeric (integer/double; logical/boolean coerced to 0/1); `p ≥ 1`; no missing or non-finite values; unique column names. Factor/character columns are an error (encode them yourself). Unnamed matrices get language-default names (R: `V1..Vp`; Python: `x0..x{p-1}`); parity tests therefore use named columns. |
| `y` | regression: numeric vector; classification: factor (R) / categorical labels (Python) with `K ≥ 2` levels. Character `y` is coerced to a factor with levels in sorted order. Level order is preserved and is significant for the column order of probability matrices. |
| `task` | R: `"auto"` (default; numeric `y` → regression, factor `y` → classification), `"regression"`, `"classification"`. Python: taken from the estimator (`is_classifier`). |
| missing values | not allowed in `x` or `y`; the caller must impute or drop first. |

Rows of `x` and `y` are aligned by position. `n ≥ 2`.

**Single predictor (`p = 1`).** Allowed. The reduced representation then has
zero columns and the reduced model is the null model of Section 8.3. The
importance is the out-of-sample gain of the learner over the null model.
(Both packages accept `p = 1`; Python since commit `febd7ea`.)

## 4. Resampling

Default: `v`-fold cross-validation with `v = 5`, one repeat, produced by the
host language's standard tool (R: `rsample::vfold_cv()`; Python: shuffled
`KFold` / `StratifiedKFold`). For classification the folds are stratified by
`y`. The same splits are used for every anchor feature, and for the full and
reduced models within an anchor, so the comparison is paired.

Both implementations also accept explicit splits supplied by the caller (R: an
`rset` object or a vector of fold labels of length `n`; Python: a scikit-learn
splitter or an iterable of `(train_idx, test_idx)` pairs). Explicit splits are
the mechanism for cross-language equality tests, because the two languages'
random fold generators differ.

Every assessment set must contain at least one observation; every training set
at least two. For classification, a training set with a single class is a
learner failure for that split (Section 10).

## 5. Fold-internal residualization (the UMFI-LR preprocessing)

For a split `s` and anchor `i`, and each `j ≠ i`, fit by ordinary least squares
on the **training rows only**:

```
X_j = a_j + b_j X_i + e_j          (rows in T_s)
b_j = S_ij / S_ii                  S_ij = Σ_T (X_i − mean_T X_i)(X_j − mean_T X_j)
a_j = mean_T(X_j) − b_j · mean_T(X_i)
```

Then the residualized feature is

```
R_j = X_j − a_j − b_j X_i
```

applied with the **same** `(a_j, b_j)` to the training rows and to the
assessment rows. Assessment rows never enter the computation of `(a_j, b_j)`.

Unlike the reference implementation we always residualize: there is no p-value
or R² gate on `b_j` (Section 12).

### 5.1 Degenerate cases (deterministic rules, computed on `T_s` only)

| Case | Rule |
|---|---|
| D1. Anchor constant: `S_ii = 0`, tested exactly as `max_T(X_i) = min_T(X_i)` | `b_j = 0`, `a_j = mean_T(X_j)` for all `j` (the other columns are only centred). A warning names the anchor. |
| D2. `X_j` constant: `S_jj = 0`, tested exactly as `max_T(X_j) = min_T(X_j)` | `b_j = 0`, `a_j =` that constant; `R_j = 0` exactly on the training rows and `X_j − a_j` on the assessment rows. Not flagged as collinear. |
| D3. `X_j` exactly collinear with the anchor: `S_jj > 0` and `SS_res_j ≤ tol · S_jj`, `tol = 1e-10`, where `SS_res_j = Σ_T (X_j − a_j − b_j X_i)²` | `R_j` is set to **exactly 0** for training and assessment rows, so that floating-point round-off of the residual cannot carry the anchor into the reduced representation. The column is **kept**. |

`tol = 1e-10` is the only numerical tolerance in the algorithm; D1 and D2 are
exact equalities of `max` and `min` (not of a computed mean, whose
floating-point value for a constant vector is not guaranteed to equal that
constant), and all three rules are invariant to changes of units and origin.
No random noise is ever injected and no column is ever dropped; the
representations always have exactly `p` and `p − 1` columns.

Rationale for keeping (not dropping) collinear columns, and the learner study
behind it, are in Section 15.3.

## 6. Full and reduced representations

```
Full_i    = X with every column j ≠ i replaced by R_j; column i is the raw X_i   (p columns)
Reduced_i = Full_i without column i                                             (p − 1 columns)
```

**Column order:** the original order of `X` in both; the anchor stays in its
original position in `Full_i`. Column names are kept (residualized columns are
representations of those columns, not new features). Importance must not
depend on the column order of `X`, and this is tested.

If `p = 1`, `Reduced_i` has zero columns and the learner is replaced by the
null model (Section 8.3).

## 7. (Reserved)

The former tolerant constant-column test was replaced by the exact rules of
Section 5.1 and this section number is kept so that references in both
code bases stay valid.

## 8. Learners

A learner is an object with two operations:

- `fit(X_train, y_train) → model`
- `predict(model, X_new) → prediction`

where `prediction` is a numeric vector (regression) or an `n × K` matrix of
class probabilities with columns in the level order of `y` (classification).
Predicted class labels are **always** derived as the argmax of the probability
matrix (ties → first level); the learner's own class prediction, if any, is
ignored. Non-finite predictions are a learner failure in R
(`check_prediction()`); Python does not inspect predictions (scikit-learn
scorers receive the fitted model), so a non-finite prediction surfaces either
as an exception from the scorer (recorded as a failure) or as a non-finite
score that is excluded from the aggregation with an empty `error` — the
summary numbers agree, only the `error` text can differ (recorded deviation
in the Python notes §4.3). If a level of `y` is absent from a training set
(but at least two levels are present), the learner is fitted on the levels
present and the missing level receives probability 0.

Hyperparameters are fixed: no tuning is performed inside `shared_importance()`.

### 8.1 Built-in learner `"linear"` (R) / reference estimators (Python)

| task | model |
|---|---|
| regression | ordinary least squares (R: `stats::lm.fit()` on `[1, X]`; Python: `sklearn.linear_model.LinearRegression`) |
| binary | unpenalized logistic regression (R: `stats::glm.fit()` binomial, event = first level; Python: `LogisticRegression(penalty=None)`) |
| multiclass | unpenalized multinomial logistic regression (R: `nnet::multinom(maxit = 500)`; Python: `LogisticRegression(penalty=None)` multinomial) |

**Rank deficiency.** R's `lm.fit` detects rank deficiency by pivoted QR with
relative tolerance `1e-7` and the built-in learner sets aliased coefficients
to zero, so predictions stay finite. A design can be rank-deficient even
though rule D3 did not fire: D3 removes collinearity *with the anchor* only,
and two other features that are collinear *with each other* (e.g. a feature
and its affine copy, when neither is the anchor) stay collinear after
residualization. Fitted values are a projection onto the column space, so
every least-squares solver that **recognises** the deficiency gives the same
predictions whatever its coefficient convention (R aliases a column;
NumPy/SciPy return the minimum-norm solution). A solver that does **not**
recognise a singular value at machine epsilon inverts it and the predictions
change by amounts of order 0.1 (Section 15.4). Whether such a singular value
is recognised depends on the library and its version; the specification
therefore makes **no claim that arbitrary learners give identical predictions
on rank-deficient designs**. Cross-language parity tests on rank-deficient
designs must use an OLS learner with an explicit rank decision: R
`lm.fit`; Python a least-squares solve with an explicit relative tolerance
(e.g. `numpy.linalg.lstsq(rcond = 1e-7)` or an SVD minimum-norm fit truncating
singular values below `1e-7` of the largest), not `LinearRegression` with
library defaults. Non-convergence or separation is a learner failure only if
it produces non-finite predictions.

### 8.2 Built-in learner `"rand_forest"` (R default)

Random forest with fixed hyperparameters shared across languages:

| parameter | value |
|---|---|
| trees | 500 |
| mtry / max_features | `floor(sqrt(q))`, `q` = number of representation columns |
| min_n / min_samples_leaf | 5 |
| classification output | class probabilities |
| bootstrap | yes, sample fraction 1 |

R engine: `ranger` through `parsnip::rand_forest()`. Python: scikit-learn
`RandomForestRegressor` / `RandomForestClassifier`. Forest implementations
differ between languages; cross-language equality is **not** expected for this
learner, only qualitative agreement. Note that an exactly-zero column occupies
one of the `mtry` candidate slots at every split; both languages are affected
identically (Section 15.3).

### 8.3 Null model

Used when a representation has zero columns. Regression: predict the training
mean of `y`. Classification: predict the training class frequencies as
probabilities (class = most frequent; ties → first level).

### 8.4 User-supplied learners

R: any `parsnip` model specification with a mode matching the task, or a
`sharedimp_learner` object built from a `fit` and a `predict` function.
Python: any scikit-learn compatible estimator (`fit`, `predict`,
`predict_proba`). The adapter translates to this section's contract.

## 9. Metrics, scores and the sign of importance

A metric `ν` has a `direction` ∈ {`maximize`, `minimize`}. Its **score** is

```
score = ν            if direction = maximize
score = −ν           if direction = minimize
```

so that greater is always better. This is exactly the scikit-learn scorer
convention (`neg_mean_squared_error`, `neg_log_loss`, ...); in R the
orientation is taken from the yardstick `direction` attribute. For split `s`
and anchor `i`:

```
score_full[s,i]    = score(y_A, pred_full)
score_reduced[s,i] = score(y_A, pred_reduced)
importance[s,i]    = score_full[s,i] − score_reduced[s,i]
```

so `importance > 0` always means "keeping the anchor predicts better out of
sample". Negative values are **kept**, never truncated or floored. Scores are
not truncated either. Metrics are evaluated per assessment set, never on
pooled predictions. Both `score_*` columns are reported in this oriented form
in both languages, so every column of both output tables is comparable.

### 9.1 Common metric names and their exact definitions

Checked numerically on 2026-11-13 with yardstick 1.4.0 and scikit-learn 1.9.1
on identical inputs (Section 15.2): every pair below agrees to ≤ 3e-15.

| common name | task | direction | definition | R (yardstick) | Python (scikit-learn scorer) |
|---|---|---|---|---|---|
| `mse` (**default for regression**) | regression | minimize | mean squared error | `sharedimp::mse` | `neg_mean_squared_error` |
| `rmse` | regression | minimize | sqrt(mse) | `rmse` | `neg_root_mean_squared_error` |
| `mae` | regression | minimize | mean absolute error | `mae` | `neg_mean_absolute_error` |
| `r2` | regression | maximize | `1 − SS_res / SS_tot` (coefficient of determination) | `rsq_trad` | `r2` |
| `rsq` | regression | maximize | squared Pearson correlation of truth and estimate; undefined (`NA`/`NaN`) when truth or estimate is constant | `rsq` | `common_scorer("rsq")` (no scikit-learn name) |
| `log_loss` (**default for classification**) | classification | minimize | multinomial log loss: mean of `−log p_true`, probabilities clipped to `[ε, 1 − ε]` with `ε = 2^−52` (machine epsilon) | `mn_log_loss` | `neg_log_loss` or `common_scorer("log_loss")`, which clips at `2^−52` itself (scikit-learn's own `log_loss` has clipped at machine epsilon at least since 1.3.2, the minimum version the Python package supports; a zero probability contributes `−log(2^−52) = 36.04`) |
| `accuracy` | classification | maximize | fraction of correct **argmax** labels (ties → first level) | `accuracy` on argmax labels | `common_scorer("accuracy")`. scikit-learn's `"accuracy"` scorer uses the estimator's `predict`, which equals the argmax of `predict_proba` for most but not all classifiers (not for `SVC(probability=True)`) |
| `roc_auc` | binary | maximize | Mann–Whitney AUC (ties count ½) | `roc_auc` (event = first level) | `roc_auc` (positive = second sorted class). Identical value: swapping the event class and its probability column leaves the AUC unchanged |
| `roc_auc` | multiclass | maximize | Hand & Till (2001): mean over class pairs of ½(A(i\|j) + A(j\|i)) | `roc_auc` (`hand_till`, the default) | `roc_auc_ovo` with `average="macro"`. **Not** `roc_auc_ovr` (one-vs-rest differs) |

Important distinctions:

- `r2` ≠ `rsq`. yardstick's `rsq` is the squared Pearson correlation, which
  ignores bias and scale; `rsq_trad` / scikit-learn `r2_score` is the usual
  coefficient of determination. The common name `r2` means the latter. R
  accepts `"rsq_trad"` and `"mn_log_loss"` as aliases of `"r2"` and
  `"log_loss"`. `rsq` is undefined (`NA`) when the predictions are constant on
  the assessment rows, which is always the case for the null model (`p = 1`);
  `r2` is undefined only when `y` is constant there.
- Binary probability metrics never depend on which level is called the event
  as long as the matching probability column is used; R uses the first level,
  scikit-learn the second sorted class.
- **Default metric (common).** When the caller does not name a metric, the
  implementation uses `mse` for regression and multinomial `log_loss` for
  classification, in both languages. Both are smooth, strictly proper scores
  that decompose over observations. `r2` and `accuracy` remain available by
  name. (Python implements this since commit `399bb20`: `scoring=None`
  resolves to `neg_mean_squared_error` for regressors and to
  `common_scorer("log_loss")`, reported as `neg_log_loss`, for classifiers.)
  Parity tests pass the metric explicitly regardless.

R additionally accepts any yardstick metric function; Python any scikit-learn
scorer name or callable.

### 9.2 Aggregation

Over the splits whose fit succeeded (Section 10):

```
importance_i      = mean_s importance[s,i]
importance_std_i  = sd_s   importance[s,i]      (sample sd, denominator n − 1; NA with one split)
score_full_i      = mean_s score_full[s,i]
score_reduced_i   = mean_s score_reduced[s,i]
n_splits_i        = number of splits with a successful fit
```

Unweighted arithmetic mean over all splits including repeats; not pooled
predictions, not weighted by assessment size. `importance_std` is a
descriptive dispersion of split-level values. It is not a standard error and
must not be presented as inference.

## 10. Learner failures

If fitting or predicting fails (error, or non-finite predictions, or a training
set with a single class) for split `s` and anchor `i`, the **default
behaviour in both languages** is:

- the split row is recorded with `score_full`, `score_reduced` and
  `importance` equal to `NA` / `NaN`, and the error message is stored in an
  `error` column appended after the common columns (R: the condition message;
  Python: `"<ExceptionType>: <message>"`);
- one warning is emitted summarising how many (feature, split) pairs failed;
- aggregation uses the remaining splits; `n_splits` counts the successful
  ones; if no split succeeds, the feature's summary values are `NA` and
  `n_splits = 0`.

An option switches to **aborting at the first failure** instead
(`on_error = "error"` in both languages; the default is `"warn"`).
Cross-language comparisons use the aborting mode so that a failure cannot be
mistaken for a value. (Python implements this since commit `399bb20`, with the
`error` column and `n_failed` in its result.)

## 11. Output schema

`shared_importance()` returns an object (R: class `sharedimp_result`; Python:
`SharedImportanceResult`) holding two tables with **common column names**.

### 11.1 Summary table — one row per evaluated anchor (R: `$importance`; Python: `.importances`)

| column | type | meaning |
|---|---|---|
| `feature` | label | feature name |
| `importance` | double | mean over splits of `score_full − score_reduced` |
| `importance_std` | double | sample sd of the per-split importances; `NA` with one split; descriptive only |
| `score_full` | double | mean full score over splits |
| `score_reduced` | double | mean reduced score over splits |
| `n_splits` | integer | number of splits that contributed (all splits when no fit failed) |

Row order: the order of `features` (default: input column order); never
sorted by importance.

### 11.2 Per-split table — one row per (split, anchor) (R: `fold_results()` / `$folds`; Python: `.fold_results`)

Common columns, in this order:

| column | type | meaning |
|---|---|---|
| `feature` | label | anchor feature |
| `split` | integer | index of the split in generation order (R: 1-based; Python: 0-based) |
| `n_train`, `n_test` | integer | sizes of `T_s` and `A_s` |
| `score_full`, `score_reduced` | double | oriented scores on `A_s` |
| `importance` | double | `score_full − score_reduced` |

Rows are ordered by `split`, then by feature in `features` order. Parity tests
compare rows by position, not by the `split` label. Implementations may append
language-specific columns **after** the common ones; R appends `id`, `id2`
(`rsample` labels) and `error`; Python appends `error` (Section 10).

### 11.3 Metadata

Metric name and direction, task, learner description, `n`, `p`, number of
splits, seed, class levels, and a textual description of the resampling
scheme. Names are language-idiomatic (R: `metric`, `task`, `learner`, `n`,
`p`, `v`, `repeats`, `seed`, `levels`, `resampling`, `call`).

## 12. Differences from the reference UMFI-LR implementation

Reference: https://github.com/HydroML/UMFI (`R/preprocess_lr.R`, `R/umfi.R`) and
https://github.com/guanton/umfi_python (`umfi/preprocess_LR.py`, `umfi/UMFI.py`),
both inspected on 2026-11-12.

| aspect | reference UMFI-LR | sharedimp | why |
|---|---|---|---|
| where residualization is fit | whole data set | training split only; coefficients applied to assessment rows | avoid leaking assessment rows into the representation |
| evaluation | ranger out-of-bag R² / accuracy on the whole data (Python: `oob_score_`) | paired per-split assessment scores | explicit, learner-agnostic out-of-sample evaluation; OOB is forest-specific |
| residualize or not | only if slope p-value < 0.01 (R) / in-sample R² > 0.01 (Python); otherwise raw `X_j` kept | always residualize | a data-dependent gate makes the representation discontinuous and differs between the two reference implementations |
| negative importance | `max(score, 0)` floors (0.5 for classification accuracy) and `fi[fi < 0] <- 0` | kept as is | negatives are informative (noise features, overfitting) and truncation biases averages |
| constant / collinear columns | Gaussian noise injected | constant anchor → slope 0 with warning; exactly collinear columns → residual fixed at 0, column kept | determinism; identical column count for every anchor |
| learner | ranger, 100 trees, defaults; Python: ExtraTrees by default | any learner; built-in RF with 500 trees, `min_n = 5`; built-in linear | fixed, documented hyperparameters |
| metric | R² (regression), accuracy (classification) | configurable; R defaults MSE / log loss | smooth, decomposable defaults; R² and accuracy still available |
| random state | not controlled | `seed` / `random_state`; explicit splits accepted | reproducibility |

Because of these differences, numerical identity with the reference
implementation is **not** expected. What *is* tested (see
`docs/umfi_comparison.md`): (a) the residualization coefficients are identical
to `lm(X_j ~ X_i)` on the same rows when the gate would have fired; (b) on
deterministic synthetic data with a clear ground truth, the rankings agree.

## 13. Direction diagnostic — removed from the v0.1 API

The two drafts defined different quantities:

- R draft: Pearson correlation, over assessment rows, between `X_i` and
  `pred_full − pred_reduced`.
- Python draft: Spearman rank correlation, over assessment rows, between `X_i`
  and the Full model's prediction (probability of the second class level for
  binary targets).

These are not the same statistic, so the diagnostic is **not part of the v0.1
common API**: it is not an argument of `shared_importance()` in either
language and not a column of either output table. Existing code may be kept
internally, marked experimental and unreachable from the public API (R keeps
an unexported `compute_direction()`). If a direction diagnostic is adopted
later, its statistical definition will be agreed first and it will be
described as a descriptive summary of a fitted model, never as a causal,
conditional or regression effect.

## 14. Reproducibility

- `seed` (R) / `random_state` (Python): when set, split generation and any
  stochastic learner run under a local RNG state seeded with it; R restores the
  caller's RNG state afterwards. The full and reduced models of a pair receive
  the same seed where the learner is stochastic.
- Bitwise equality between R and Python is **not** expected for stochastic
  learners or randomly generated splits (different RNGs). Cross-language parity
  is defined for deterministic learners on explicit splits: given identical
  splits, the `"linear"` learner / `LinearRegression` and any common metric
  must give identical results to about `1e-8` in every column of both tables
  (Section 15.1 achieves `1e-13`).

## 15. Cross-language verification record (2026-11-13)

Run against the `sharedimp-py` **working tree** at 12:50 JST (commit
`bfcb2a5` plus uncommitted changes that were being made concurrently on the
Python side; the package is installed in editable mode, so the working tree is
what ran; scikit-learn 1.9.1, numpy 2.4.6). The Python code was only executed
from here, never modified. R side: this package with yardstick 1.4.0. A first
run about 20 minutes earlier, during those Python edits, gave the same numbers.
Those working-tree changes were later committed on the Python side as
`399bb20` (default metric, `on_error`) and `dd0bef0` (specification pointer);
the package state compared here corresponds to that code. R expectations: `tests/testthat/test-common-spec.R` and
`tests/testthat/test-metric-definitions.R`. Comparison script, its output and
the learner study: `docs/parity/`.

### 15.1 Parity fixture (deterministic, no RNG)

```
i  = 1, ..., 12
x1 = i
x2 = (i mod 4) + 0.5 i
x3 = 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11
y  = 2 x1 − x2 + 0.5 x3 + (0.3, −0.2, 0.1, 0.4, −0.5, 0.2, −0.1, 0.3, −0.4, 0.1, 0.2, −0.3)
   = 1.8, 1.3, 3.6, 7.9, 9.0, 9.7, 11.4, 15.8, 17.1, 17.6, 19.7, 23.2
splits: row i is assessed in split ((i − 1) mod 3) + 1   (3 splits, 8 train / 4 test each)
learner: OLS with intercept; metric: mse (score = −mse)
```

Expected summary table (R, `learner = "linear"`, `metric = "mse"`):

| feature | importance | importance_std | score_full | score_reduced | n_splits |
|---|---|---|---|---|---|
| x1 | 48.3228798721808 | 6.17147177856012 | −0.276470061784757 | −48.5993499339656 | 3 |
| x2 | 34.4600258422062 | 29.5939201894648 | −0.276470061784757 | −34.7364959039909 | 3 |
| x3 | 45.5507348310811 | 22.4063405773906 | −0.276470061784757 | −45.8272048928659 | 3 |

Per-split importances (split 1, 2, 3): x1 53.5652538573545, 49.8820278139317,
41.5213579452562; x2 61.4017818839248, 39.1937200003847, 2.78457564230898;
x3 36.9725234594595, 70.9787812500000, 28.7008997837838.

Python (`LinearRegression()`, `scoring="neg_mean_squared_error"`, the same
splits as an explicit `cv` list) reproduced every column: maximum absolute
difference 4.3e-14 (summary table) and 6.8e-14 (per-split table); `n_splits`,
`n_train`, `n_test` identical. A variant with an added exactly collinear
column `x1_dup = 3 x1 − 2` (exercising the zero-residual rule in both
languages) also agreed to 2.9e-14, with `x1_dup` receiving the same importance
as `x1` — **in the current environments**. For the anchors `x2` and `x3` this
variant is rank-deficient downstream (`r(x1)` and `r(x1_dup)` are collinear),
and the Python side observed 41.79 instead of 45.55 for `x3` with scikit-learn
1.3.2 / NumPy 1.23; Section 15.4 explains why and states the rule for such
tests.

### 15.2 Metric definitions

On one random data set (n = 40) the R and Python values of every common metric
agreed to ≤ 3e-15: mse, rmse, mae, r2 (`rsq_trad` vs `r2_score`), rsq
(`rsq` vs `corrcoef²`), binary log loss (including a clipped zero probability),
binary accuracy, binary AUC (first-level event in R vs second-class positive
in scikit-learn), multiclass log loss, multiclass accuracy, and multiclass AUC
(`hand_till` vs `multi_class="ovo", average="macro"`). scikit-learn's
`multi_class="ovr"` gave a different value (0.4624 vs 0.4679) and must not be
used as the counterpart of `roc_auc`.

### 15.3 Zero columns and learners (basis for the collinearity rule)

The Python draft fixed collinear residuals at zero; the R draft dropped them.
Before adopting the zero rule, R learners were run with and without an added
exactly-zero column (n = 120, 90/30 split), in a "zero column only"
representation (the reduced representation of a duplicated pair with `p = 2`)
and in a "zero column plus one real column" representation:

| learner | predictions with vs without zero column | zero column only | note |
|---|---|---|---|
| `lm`, `glm` (binomial), `nnet::multinom`, `xgboost`, `rpart` | identical | fit | `predict.lm` warns about rank deficiency; the built-in `"linear"` learner therefore uses `lm.fit`/`glm.fit` with aliased coefficients set to 0 |
| `glmnet` | identical | error: needs ≥ 2 columns | pre-existing glmnet limitation for any `p = 2` input; recorded as a learner failure |
| `ranger` | differ (random candidate draws; the zero column takes an `mtry` slot) | fit | same effect in scikit-learn forests; equality across runs is not expected for forests anyway |
| `kernlab::ksvm` | differ (scaling of a constant column) | error: no support vectors | recorded as a learner failure (`NA` + warning) |
| `nnet` MLP | differ (random initial weights) | fit | stochastic learner |

Decision: adopt the zero rule (deterministic learners give identical
predictions, so the statistical quantity is unchanged; the column count is the
same for every anchor; both languages behave identically). Learners that cannot
handle a constant column fail on that split through the ordinary
learner-failure path. The alternative (dropping the column, the R draft) is
documented here as rejected; it is not available as an option in either
package.

### 15.4 Rank deficiency in the downstream OLS learner

Record: `docs/parity/rank_deficiency.md` (scripts and raw output alongside).
On the collinear variant of Section 15.1, for anchors `x2` and `x3`, the
residualized `x1` and `x1_dup` differ by round-off only (≤ 7.1e-15) and the
Full design has a singular value ratio of 2e-17 … 1.6e-16 — at machine
epsilon (2.2e-16). The residualization is identical in R and Python; the
question is purely how the least-squares solver treats that singular value:

- recognised as zero (R `lm.fit`, NumPy `lstsq` with its default or with
  `rcond = 1e-7`, SciPy `lstsq` in scikit-learn 1.9.1, SVD minimum-norm solve):
  all predictions agree with R to ≤ 1e-13;
- not recognised (forced in the record with `rcond = 1e-20`, or an untruncated
  SVD solve in R): coefficients of order 1e14 and per-split test MSE changes
  of 0.02–0.4, the behaviour the older scikit-learn / NumPy stack showed.

Consequences, now part of Section 8.1: no identity claim for arbitrary
learners on rank-deficient designs; parity tests on such designs use an OLS
learner with an explicit rank decision. The full-rank fixture of Section 15.1
is unaffected.

## 16. Required changes on the Python side

Listed for the maintainers of `sharedimp-py`; nothing in that repository was
modified from here. Status: as of Python commit `5277017` (2026-11-13). The
implementing commits are named per item.

1. **Direction diagnostic.** Remove the `direction` argument of
   `shared_importance()` and the `direction` columns; keep `_direction.py`
   internal and marked experimental, or delete it (Section 13).
   *Status: done (`febd7ea`).*
2. **Single predictor.** Accept `p = 1`: the reduced representation has zero
   columns and must be scored with the null model (training mean / training
   class frequencies), not with the user's estimator (Sections 3, 8.3).
   *Status: done (`febd7ea`, `fit_null_model`).*
3. **Default scoring.** Default to `neg_mean_squared_error` for regressors and
   `neg_log_loss` for classifiers instead of `estimator.score`. This is now a
   **normative** part of the common specification (Section 9.1).
   *Status: done (`399bb20`; documented in the Python notes §4.2). Earlier
   revisions of this document attributed it to `692c3fe`, where it was only
   observed in the editable-install working tree before being committed.*
4. **Learner failures.** Record a failed `(split, feature)` fit as `NaN` with a
   warning and aggregate over the remaining splits (`n_splits` counts the
   successful ones) **by default**, with an option to raise instead
   (Section 10, now normative). A training split with a single class is such
   a failure.
   *Status: done (`399bb20`; `on_error = "warn" | "error"`, default `"warn"`,
   `error` column with `"<ExceptionType>: <message>"`, `n_failed`; Python
   notes §4.3).*
5. **Specification pointer.** Replace the provisional
   `docs/specification.md` by a pointer to this file plus Python-specific
   notes; drop the provisional label.
   *Status: done (`dd0bef0`): the Python `docs/specification.md` is a pointer
   to this file plus a conformance table, Python notes and recorded
   deviations.*
6. **Metric documentation.** State the scorer ↔ common-name mapping of
   Section 9.1 (`r2` = `r2`, `roc_auc` multiclass = `roc_auc_ovo` macro,
   `common_scorer("log_loss")` clipping at `2^−52`) and provide the Pearson²
   scorer for `rsq`.
   *Status: done (`common_scorer()` for all eight common names, including
   `rsq` and an argmax-based `accuracy`).*
7. **Parity test.** Add the fixture of Section 15.1 (including the collinear
   variant) as a test with tolerance `1e-8` on every column of both tables.
   *Status: done (`test_cross_language.py`), also with the default metric.
   Still open (deferred by the Python side until this revision is declared
   final, which it now is): regenerate the second reference set
   `tests/fixtures/cross_language/r_reference.json` from the current R commit
   (its generator must be adapted to `score_*` columns, `collinear` flags and
   the removed `drop_constant` argument), remove the compensating conversions
   in `tests/test_cross_language.py`, and extend the comparison to the default
   metric and to a failing split with `on_error = "warn"` in both languages.
   For the collinear design use an OLS learner with an explicit rank decision
   (Section 8.1).*

Already consistent with this specification (no change needed): fold-internal
OLS residualization with the exact-collinearity rule (`tol = 1e-10`), Full
with the anchor in place, Reduced without it, scorer orientation and
`importance = score_full − score_reduced`, no truncation, unweighted mean and
sample sd over splits, the common column names of Section 11, split-major row
order of the per-split table, seeding policy, `x0..` default names, Docker
tag policy (Section 17).

Recorded deviations on the Python side (its notes §4.3–4.4): non-finite
predictions are not inspected (see Section 8); fit-time `sample_weight` is
supported in Python only; overlapping training/assessment rows are accepted in
Python (for in-sample closed-form tests) and rejected in R.

## 17. Distribution: Docker images

Both packages publish a Docker image from GitHub Actions
(`.github/workflows/build_test_push.yml` in each repository) with the same
policy:

| event | tags pushed | registries |
|---|---|---|
| push to `main` | `latest`, short commit SHA (7 characters) | GHCR `ghcr.io/dsc-chiba-u/sharedimp-r` / `ghcr.io/dsc-chiba-u/sharedimp-py`; Docker Hub `koki/sharedimp-r` / `koki/sharedimp-py` |
| release tag `vX.Y.Z` | `X.Y.Z`, `X.Y`, short commit SHA (**no** `latest`) | same |
| pull request | none (build and test only) | — |
| manual run (`workflow_dispatch`) | as for the ref it runs on, **only** when that ref is `main` or a `vX.Y.Z` tag and the `publish` input is true; otherwise build and test only | same |

`latest` therefore always points at the last successful build of `main`, and a
version tag at the release it names. One `docker/build-push-action` build is
tagged for both registries, so they hold the identical image digest. The
build itself installs the package, runs its test suite and a smoke test that
calls `shared_importance()`; a failing test fails the build and nothing is
pushed. The R image is based on `rocker/r-ver` (pinned stable R release).
Missing Docker Hub credentials fail a publishing run early with an explicit
message; GHCR authenticates with the workflow's `GITHUB_TOKEN`.
