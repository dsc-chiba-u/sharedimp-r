# sharedimp — shared algorithm specification (R / Python)

Status: v0.1 draft, 2026-11-12. This document is the single source of truth for
the R package `sharedimp` (this repository) and the Python package `sharedimp`
(`../sharedimp-py`). Both implementations must follow it; any intentional
deviation must be recorded in the "Implementation notes" section of the
deviating repository.

At the time of writing `sharedimp-py` contains only a README and a LICENSE, so
no pre-existing Python specification could be consulted. If the Python side
later adopts different conventions, this file must be updated and the
difference reported, not silently absorbed.

## 1. Scope and non-claims

`sharedimp` is a careful, resampling-based implementation of the
linear-regression variant of Ultra-marginal Feature Importance (UMFI-LR;
Janssen, Guan & Robeva, AISTATS 2023, PMLR 206). We do not claim a new feature
importance measure. The contribution is engineering: fold-internal
residualization, paired out-of-sample evaluation, learner/metric adapters for
modern ML workflows (tidymodels in R, scikit-learn in Python), reproducible
random state, and a result object that exposes per-fold results.

The reported numbers are *predictive* importances: how much out-of-sample
predictive performance is gained by keeping the anchor feature next to a
representation of the other features from which its linear component has been
removed. They are not causal effects, not conditional effects, and no
confidence intervals, p-values, or other formal inference are provided.

## 2. Notation

- `n` observations, `p` numeric predictors `X` (n × p), response `y`.
- `i` is the *anchor* feature (the feature whose importance is being
  computed); `-i` denotes the remaining `p − 1` features.
- A *fold* `f` is a pair (training index set `T_f`, assessment index set
  `A_f`) with `T_f ∩ A_f = ∅`.
- `ν` (nu) is an evaluation metric computed on assessment predictions.

## 3. Inputs

| Input | Requirement |
|---|---|
| `x` | numeric matrix or data frame whose columns are all numeric (integer/double); `p ≥ 1`; column names required (data frame) or generated as `V1..Vp` (unnamed matrix). Logical columns are coerced to 0/1 with a message. Factor/character columns are an error (encode them yourself). |
| `y` | regression: numeric vector; classification: factor (R) / categorical (Python) with `K ≥ 2` levels. Character `y` is coerced to a factor with levels in sorted order. Level order is preserved and is significant (first level is the event level for binary probability metrics). |
| `task` | `"auto"` (default; numeric `y` → regression, factor `y` → classification), `"regression"`, `"classification"`. |
| missing values | not allowed in `x` or `y`; the caller must impute or drop first. |

Rows of `x` and `y` are aligned by position.

## 4. Resampling

Default: `v`-fold cross-validation with `v = 5`, `repeats = 1`, produced by the
host language's standard tool (R: `rsample::vfold_cv()`; Python:
`sklearn.model_selection.KFold` / `StratifiedKFold`). For classification the
folds are stratified by `y`. The same folds are used for every anchor feature,
and for the full and reduced models within an anchor, so the comparison is
paired.

Both implementations must also accept an explicit fold assignment supplied by
the caller (R: an `rset` object or an integer vector of fold labels of length
`n`; Python: an iterable of (train_idx, test_idx) pairs or an integer vector).
Explicit folds are the mechanism for cross-language equality tests, because the
two languages' random fold generators differ.

Every assessment fold must contain at least one observation; every training
fold must contain at least two observations. For classification, a training
fold must contain at least two classes; otherwise the fold is treated as a
learner failure (Section 9).

## 5. Fold-internal residualization (the UMFI-LR preprocessing)

For a fold `f` and anchor `i`, and each `j ≠ i`, fit by ordinary least squares
on the **training rows only**:

```
X_j = a_j + b_j X_i + e_j          (rows in T_f)
b_j = Cov_T(X_i, X_j) / Var_T(X_i)
a_j = mean_T(X_j) − b_j · mean_T(X_i)
```

where `Cov_T`, `Var_T`, `mean_T` are computed over `T_f` (any consistent
normalisation; `b_j` does not depend on it). Then the residualized feature is

```
R_j = X_j − a_j − b_j X_i
```

applied with the **same** `(a_j, b_j)` to the training rows and to the
assessment rows. Assessment rows never enter the computation of `(a_j, b_j)`.

Constant-anchor rule: if `X_i` is constant on `T_f` (see Section 7 for the
test) then `b_j = 0` and `a_j = mean_T(X_j)` for all `j`, and the implementation
emits a warning naming the anchor. No noise is injected.

Unlike the reference implementation we always residualize: there is no p-value
gate on `b_j` (see Section 12).

## 6. Full and reduced representations

```
Full_i    = [ X_i , R_{-i} ]     (p columns; X_i is the raw, un-centred anchor)
Reduced_i = [ R_{-i} ]           (p − 1 columns)
```

Column order: anchor first, then the residualized columns in original order.
Column names: the anchor keeps its name; residualized columns keep their
original names (they are representations of those columns, not new features).

Columns of `R_{-i}` that are constant on the training rows (e.g. the residual
of an exact duplicate of the anchor) are dropped from **both** `Full_i` and
`Reduced_i` using the training-fold test only. The anchor column itself is
never dropped.

If `p = 1`, `Reduced_i` has zero columns and the learner is replaced by the
null model (Section 8.3).

## 7. Constant-column test

A vector `v` is treated as constant when

```
max(v) − min(v) <= 1e-10 * max(1, max(|v|))
```

This is the only numerical tolerance in the algorithm.

## 8. Learners

A learner is an object with two operations:

- `fit(X_train, y_train) → model`
- `predict(model, X_new) → prediction`

where `prediction` is a numeric vector (regression) or an `n × K` matrix of
class probabilities with columns in the level order of `y` (classification).
Predicted class labels are **always** derived as the argmax of the probability
matrix (ties → first level); the learner's own class prediction, if any, is
ignored. Rows with NA predictions are a learner failure. If a level of `y` is
absent from a training fold (but at least two levels are present), the
learner is fitted on the levels present and the missing level receives
probability 0 in the output matrix.

Hyperparameters are fixed: no tuning is performed inside `shared_importance()`.

### 8.1 Built-in learner `"linear"`

| task | model |
|---|---|
| regression | ordinary least squares (R: `stats::lm` via `parsnip::linear_reg()`; Python: `sklearn.linear_model.LinearRegression`) |
| binary | unpenalized logistic regression (R: `stats::glm` binomial via `parsnip::logistic_reg()`; Python: `LogisticRegression(penalty=None)`) |
| multiclass | unpenalized multinomial logistic regression (R: `nnet::multinom` called directly with `maxit = 500`, because the `parsnip::multinom_reg()`/nnet pairing mis-formats probabilities when a training fold holds only two levels; Python: `LogisticRegression(penalty=None, multi_class="multinomial")`) |

Rank deficiency in OLS: aliased coefficients are set to zero (R's `lm`
convention) so predictions remain finite. Non-convergence or separation is a
learner failure only if it produces non-finite predictions.

### 8.2 Built-in learner `"rand_forest"` (default)

Random forest with fixed hyperparameters shared across languages:

| parameter | value |
|---|---|
| trees | 500 |
| mtry / max_features | `floor(sqrt(q))`, `q` = number of representation columns |
| min_n / min_samples_leaf | 5 |
| classification output | class probabilities (R: `ranger(probability = TRUE)`; Python: `predict_proba`) |
| bootstrap | yes, sample fraction 1 |

R engine: `ranger` through `parsnip::rand_forest()`. Python: scikit-learn
`RandomForestRegressor` / `RandomForestClassifier`. Forest implementations
differ between languages; cross-language equality is **not** expected for this
learner, only qualitative agreement.

### 8.3 Null model

Used when a representation has zero columns. Regression: predict the training
mean of `y`. Classification: predict the training class frequencies as
probabilities (class = most frequent; ties → first level).

### 8.4 User-supplied learners

R: any `parsnip` model specification with a mode matching the task, or a
`sharedimp_learner` object built from a `fit` and a `predict` function.
Python: any scikit-learn compatible estimator (`fit`, `predict`,
`predict_proba`). The adapter translates to Section 8's contract.

## 9. Metrics and the sign of importance

A metric `ν` has a `direction` ∈ {`maximize`, `minimize`}. Let
`s = +1` if `maximize`, `s = −1` if `minimize`. For fold `f` and anchor `i`:

```
imp_{i,f} = s · ( ν(y_A, pred_full_{i,f}) − ν(y_A, pred_reduced_{i,f}) )
```

so `imp > 0` always means "keeping the anchor predicts better out of sample".
Negative values are **kept**, never truncated or floored. Metrics are evaluated
per assessment fold, never on pooled predictions.

Built-in metric names and definitions (R: yardstick; Python: scikit-learn):

| name | task | direction | definition |
|---|---|---|---|
| `mse` (default, regression) | regression | minimize | mean squared error |
| `rmse` | regression | minimize | sqrt(mse) |
| `mae` | regression | minimize | mean absolute error |
| `rsq` | regression | maximize | squared Pearson correlation between truth and estimate (yardstick `rsq`); Python must use the same definition, **not** `r2_score` |
| `mn_log_loss` (default, classification) | classification | minimize | mean multinomial log loss, probabilities clipped to `[1e-15, 1 − 1e-15]` |
| `accuracy` | classification | maximize | fraction of correct argmax labels |
| `roc_auc` | classification | maximize | binary: AUC with the **first** level as the event; multiclass: Hand & Till (2001) average of pairwise AUCs (yardstick `hand_till`) |

R additionally accepts any yardstick metric function; Python any callable with
a declared direction.

### 9.1 Aggregation

```
importance_i = mean_f imp_{i,f}          over folds with a successful fit
sd_i         = sd_f   imp_{i,f}          (sample sd; NA if < 2 folds)
n_folds_i    = number of successful folds
```

`sd` is a descriptive dispersion of fold-level values. It is not a standard
error and must not be presented as inference.

## 10. Learner failures

If fitting or predicting fails (error, or non-finite predictions, or a training
fold with a single class) for a fold `f` and anchor `i`:

- the fold result is recorded with `imp_{i,f} = NA`, the metric values `NA`, and
  the error message in a `error` column;
- a warning is emitted summarising how many (anchor, fold) pairs failed;
- aggregation uses the remaining folds; if no fold succeeds, `importance_i` is
  `NA`.

With `on_error = "error"` the first failure aborts the whole computation.

## 11. Output schema

`shared_importance()` returns an object (R: class `sharedimp_result`; Python:
`SharedImportanceResult`) with at least:

| field | content |
|---|---|
| `importance` | table with columns `feature` (chr), `importance` (dbl), `sd` (dbl), `n_folds` (int), plus `direction` (dbl) when the diagnostic was requested; one row per anchor, in the order of `features` (default: input column order) |
| `folds` | table with columns `feature`, `id` (fold label), `id2` (repeat label, or `NA`), `metric_full`, `metric_reduced`, `importance`, `error` (chr or NA), plus `direction` when requested; rows ordered anchor-major, fold-minor |
| `metric` | metric name and direction |
| `task` | `"regression"` or `"classification"` |
| `learner` | human-readable learner description |
| `direction` | `NULL`, or a table `feature`, `direction`, `sd`, `n_folds` (Section 13) |
| `n`, `p`, `v`, `repeats`, `seed`, `levels`, `resampling` (human-readable description of the folds), `call` | bookkeeping |

The `importance` table is the headline output; `folds` is always retained.

## 12. Differences from the reference UMFI-LR implementation

Reference: https://github.com/HydroML/UMFI (`R/preprocess_lr.R`, `R/umfi.R`) and
https://github.com/guanton/umfi_python (`umfi/preprocess_LR.py`, `umfi/UMFI.py`),
both inspected on 2026-11-12.

| aspect | reference UMFI-LR | sharedimp | why |
|---|---|---|---|
| where residualization is fit | whole data set | training fold only; coefficients applied to assessment rows | avoid leaking assessment rows into the representation |
| evaluation | ranger out-of-bag R² / accuracy on the whole data (Python: `oob_score_`) | paired per-fold assessment metrics | explicit, learner-agnostic out-of-sample evaluation; OOB is forest-specific |
| residualize or not | only if slope p-value < 0.01 (R) / in-sample R² > 0.01 (Python); otherwise raw `X_j` kept | always residualize | a data-dependent gate makes the representation discontinuous and differs between the two reference implementations |
| negative importance | `max(score, 0)` floors (0.5 for classification accuracy) and `fi[fi < 0] <- 0` | kept as is | negatives are informative (noise features, overfitting) and truncation biases averages |
| constant / near-constant columns | Gaussian noise injected | constant anchor → slope 0 with warning; constant residual columns dropped | determinism |
| learner | ranger, 100 trees, defaults; Python: ExtraTrees by default | any learner; built-in RF with 500 trees, `min_n = 5`; built-in linear | fixed, documented hyperparameters |
| metric | R² (regression), accuracy (classification) | configurable; defaults MSE / log loss | smooth, decomposable defaults; R² and accuracy still available |
| random state | not controlled | `seed` argument; explicit folds accepted | reproducibility |

Because of these differences, numerical identity with the reference
implementation is **not** expected. What *is* tested (see
`docs/umfi_comparison.md`): (a) the residualization coefficients are identical
to `lm(X_j ~ X_i)` on the same rows when the gate would have fired; (b) on
deterministic synthetic data with a clear ground truth, the rankings agree.

## 13. Optional direction diagnostic (implemented after the core)

For regression and binary classification, `direction = TRUE` additionally
reports, per anchor,

```
dir_{i,f} = cor( X_i[A_f] , pred_full_{i,f} − pred_reduced_{i,f} )
```

(for binary classification the predictions are the probabilities of the first
level), averaged over folds. It is a descriptive summary of whether the
anchor's contribution to the prediction increases or decreases with the
anchor's value. It is **not** a regression coefficient, not a conditional
effect, and not a causal effect, and it is `NA` for multiclass tasks and when
`X_i` or the prediction difference is constant on the assessment fold.

## 14. Reproducibility

- `seed`: when non-`NULL`, the whole computation (fold generation and any
  stochastic learner) runs under a local RNG state seeded with `seed`; the
  caller's RNG state is restored afterwards. R: `set.seed(seed)` with the
  default generators; Python: `numpy.random.default_rng(seed)` /
  `random_state=seed`.
- Given identical folds, the `"linear"` learner and any built-in metric must
  give identical results (to floating-point tolerance) across the two
  languages. This is the cross-language acceptance test.
