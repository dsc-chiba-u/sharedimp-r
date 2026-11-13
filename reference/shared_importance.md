# Shared predictive importance with fold-internal residualization (UMFI-LR)

For every anchor feature, `shared_importance()` residualizes the other
predictors on the anchor using ordinary least squares coefficients
estimated on the training rows of each resampling split, applies the
same coefficients to the assessment rows, fits the learner to the *full*
representation (anchor plus residualized others) and to the *reduced*
representation (residualized others only), and compares their
assessment-set performance. This is the linear-regression variant of
ultra-marginal feature importance (UMFI-LR) of Janssen, Guan and Robeva
(2023), evaluated out of sample with paired splits. The R and Python
packages follow one common specification (`docs/specification.md` in the
source repository).

## Usage

``` r
shared_importance(
  x,
  y,
  learner = "rand_forest",
  metric = NULL,
  task = c("auto", "regression", "classification"),
  v = 5,
  repeats = 1,
  resamples = NULL,
  features = NULL,
  seed = NULL,
  on_error = c("warn", "error"),
  verbose = FALSE
)
```

## Arguments

- x:

  A numeric matrix or a data frame whose columns are all numeric.
  Missing values are not allowed. A single column is allowed; the
  reduced model is then the null model (training mean or class
  frequencies).

- y:

  The response: a numeric vector for regression or a factor for
  classification (binary or multiclass; the first level is the event
  level for probability metrics). Character vectors are converted to
  factors.

- learner:

  `"rand_forest"` (default), `"linear"`, a `parsnip` model
  specification, or a
  [`sharedimp_learner()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_learner.md).
  Hyperparameters are fixed; no tuning is performed. See
  [`sharedimp_learner()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_learner.md)
  for details.

- metric:

  A built-in metric name or a `yardstick` metric function. The defaults
  are `"mse"` for regression and `"log_loss"` for classification. See
  [`as_metric()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/as_metric.md).

- task:

  `"auto"` (default: numeric `y` is regression, factor `y` is
  classification), `"regression"`, or `"classification"`.

- v, repeats:

  Number of folds and repeats for
  [`rsample::vfold_cv()`](https://rsample.tidymodels.org/reference/vfold_cv.html),
  used when `resamples` is `NULL`. Classification folds are stratified
  by `y`.

- resamples:

  Optional explicit resampling: an `rsample` `rset` built on data with
  `nrow(x)` rows, or an integer vector of length `nrow(x)` whose unique
  values label the assessment fold of each row. Overrides `v` and
  `repeats`. Explicit splits are the mechanism for comparisons with the
  Python implementation.

- features:

  Optional names or indices of the anchor features to evaluate. Default:
  all columns of `x`.

- seed:

  Optional integer. When supplied, fold generation and any stochastic
  learner run under a local RNG state seeded with `seed`, and the
  caller's RNG state is restored afterwards.

- on_error:

  `"warn"` (default) records a learner failure on a split as `NA`,
  aggregates over the remaining splits and warns; `"error"` stops at the
  first failure.

- verbose:

  Print progress messages?

## Value

An object of class `sharedimp_result` (see
[sharedimp_result](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_result.md))
with elements `importance` (tibble with the common columns `feature`,
`importance`, `importance_std`, `score_full`, `score_reduced`,
`n_splits`), `folds` (per-split paired scores; see
[`fold_results()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_result.md)),
`metric`, `task`, `learner`, and bookkeeping fields `n`, `p`, `v`,
`repeats`, `seed`, `levels`, `resampling`, `call`.

## Scores and sign convention

Every metric is reported as a *score* oriented so that greater is
better: the metric value itself for a metric to maximize (`"r2"`,
`"accuracy"`, `"roc_auc"`, ...), and its negative for a metric to
minimize (`"mse"`, `"rmse"`, `"mae"`, `"log_loss"`, ...). For each
split, `importance = score_full - score_reduced`, so a positive value
always means that keeping the anchor predicts better than the
residualized others alone. Negative values are reported as they are;
they are never truncated to zero. The reported importance is the mean
over splits; `importance_std` is the sample standard deviation of the
split values and is a descriptive quantity only, not a standard error.

## What the numbers are not

The values are out-of-sample predictive gains for a particular learner,
metric and resampling scheme. They are not causal effects, not
regression coefficients and not conditional effects, and the package
deliberately provides no confidence intervals or p-values.

## References

Janssen, J., Guan, V. and Robeva, E. (2023). Ultra-marginal Feature
Importance: Learning from Data with Causal Guarantees. *AISTATS 2023*,
PMLR 206:10782-10814.
<https://proceedings.mlr.press/v206/janssen23a.html>

## Examples

``` r
set.seed(1)
n <- 150
x <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
x$x4 <- x$x1 + rnorm(n, sd = 0.1) # near duplicate of x1
y <- 2 * x$x1 + x$x2 + rnorm(n)

# Fast, deterministic linear learner with the MSE metric
res <- shared_importance(x, y, learner = "linear", v = 5, seed = 1)
res
#> <sharedimp_result>
#>   task:       regression
#>   learner:    linear
#>   metric:     mse (minimize); score = -mse, importance = score_full - score_reduced
#>   resampling: 5-fold cross-validation (1 repeat)
#> 
#> # A tibble: 4 × 6
#>   feature importance importance_std score_full score_reduced n_splits
#>   <chr>        <dbl>          <dbl>      <dbl>         <dbl>    <int>
#> 1 x1         3.18            0.809       -1.33         -4.51        5
#> 2 x2         0.920           0.383       -1.33         -2.25        5
#> 3 x3         0.00634         0.0250      -1.33         -1.34        5
#> 4 x4         3.13            0.865       -1.33         -4.46        5
fold_results(res)
#> # A tibble: 20 × 10
#>    feature split n_train n_test score_full score_reduced importance id    id2  
#>    <chr>   <int>   <int>  <int>      <dbl>         <dbl>      <dbl> <chr> <chr>
#>  1 x1          1     120     30      -1.02         -4.31    3.29    Fold1 NA   
#>  2 x2          1     120     30      -1.02         -2.22    1.20    Fold1 NA   
#>  3 x3          1     120     30      -1.02         -1.03    0.00905 Fold1 NA   
#>  4 x4          1     120     30      -1.02         -4.49    3.47    Fold1 NA   
#>  5 x1          2     120     30      -1.85         -4.00    2.15    Fold2 NA   
#>  6 x2          2     120     30      -1.85         -2.53    0.688   Fold2 NA   
#>  7 x3          2     120     30      -1.85         -1.89    0.0398  Fold2 NA   
#>  8 x4          2     120     30      -1.85         -3.69    1.85    Fold2 NA   
#>  9 x1          3     120     30      -1.25         -3.92    2.67    Fold3 NA   
#> 10 x2          3     120     30      -1.25         -1.61    0.364   Fold3 NA   
#> 11 x3          3     120     30      -1.25         -1.24   -0.00375 Fold3 NA   
#> 12 x4          3     120     30      -1.25         -3.92    2.67    Fold3 NA   
#> 13 x1          4     120     30      -1.37         -4.89    3.52    Fold4 NA   
#> 14 x2          4     120     30      -1.37         -2.46    1.09    Fold4 NA   
#> 15 x3          4     120     30      -1.37         -1.34   -0.0282  Fold4 NA   
#> 16 x4          4     120     30      -1.37         -5.03    3.66    Fold4 NA   
#> 17 x1          5     120     30      -1.18         -5.44    4.26    Fold5 NA   
#> 18 x2          5     120     30      -1.18         -2.45    1.26    Fold5 NA   
#> 19 x3          5     120     30      -1.18         -1.20    0.0148  Fold5 NA   
#> 20 x4          5     120     30      -1.18         -5.17    3.99    Fold5 NA   
#> # ℹ 1 more variable: error <chr>

# Binary classification with accuracy
yb <- factor(ifelse(y > median(y), "high", "low"))
shared_importance(x, yb, learner = "linear", metric = "accuracy", v = 5, seed = 1)
#> <sharedimp_result>
#>   task:       classification (2 classes)
#>   learner:    linear
#>   metric:     accuracy (maximize); score = accuracy, importance = score_full - score_reduced
#>   resampling: 5-fold cross-validation (1 repeat), stratified by y
#> 
#> # A tibble: 4 × 6
#>   feature importance importance_std score_full score_reduced n_splits
#>   <chr>        <dbl>          <dbl>      <dbl>         <dbl>    <int>
#> 1 x1          0.253          0.110       0.893         0.64         5
#> 2 x2          0.0667         0.113       0.893         0.827        5
#> 3 x3         -0.0133         0.0298      0.893         0.907        5
#> 4 x4          0.253          0.0606      0.893         0.64         5

# \donttest{
# Default random forest learner (ranger, 500 trees)
shared_importance(x, y, v = 5, seed = 1)
#> <sharedimp_result>
#>   task:       regression
#>   learner:    rand_forest
#>   metric:     mse (minimize); score = -mse, importance = score_full - score_reduced
#>   resampling: 5-fold cross-validation (1 repeat)
#> 
#> # A tibble: 4 × 6
#>   feature importance importance_std score_full score_reduced n_splits
#>   <chr>        <dbl>          <dbl>      <dbl>         <dbl>    <int>
#> 1 x1          3.70           0.455       -1.72         -5.42        5
#> 2 x2          0.877          0.276       -1.45         -2.33        5
#> 3 x3          0.0109         0.0948      -1.50         -1.51        5
#> 4 x4          3.62           0.414       -1.70         -5.32        5
# }
```
