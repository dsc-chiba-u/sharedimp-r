# Shared predictive importance with fold-internal residualization (UMFI-LR)

For every anchor feature, `shared_importance()` residualizes the other
predictors on the anchor using ordinary least squares coefficients
estimated on the training rows of each resampling fold, applies the same
coefficients to the assessment rows, fits the learner to the *full*
representation (anchor plus residualized others) and to the *reduced*
representation (residualized others only), and compares their
assessment-set performance. This is the linear-regression variant of
ultra-marginal feature importance (UMFI-LR) of Janssen, Guan and Robeva
(2023), evaluated out of sample with paired folds.

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
  direction = FALSE,
  on_error = c("warn", "error"),
  verbose = FALSE
)
```

## Arguments

- x:

  A numeric matrix or a data frame whose columns are all numeric.
  Missing values are not allowed.

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
  are `"mse"` for regression and `"mn_log_loss"` for classification. See
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
  `repeats`.

- features:

  Optional names or indices of the anchor features to evaluate. Default:
  all columns of `x`.

- seed:

  Optional integer. When supplied, fold generation and any stochastic
  learner run under a local RNG state seeded with `seed`, and the
  caller's RNG state is restored afterwards.

- direction:

  Logical. If `TRUE`, also report the direction diagnostic: the
  correlation, on each assessment fold, between the anchor's values and
  the difference between the full and reduced predictions (probability
  of the first level for binary classification; `NA` for multiclass). It
  is a descriptive summary only and must not be read as a causal or
  conditional effect.

- on_error:

  `"warn"` (default) records a learner failure on a fold as `NA`,
  aggregates over the remaining folds and warns; `"error"` stops at the
  first failure.

- verbose:

  Print progress messages?

## Value

An object of class `sharedimp_result` (see
[sharedimp_result](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_result.md))
with elements `importance` (tibble: `feature`, `importance`, `sd`,
`n_folds`, and `direction` when requested), `folds` (per-fold paired
metrics; see
[`fold_results()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_result.md)),
`metric`, `task`, `learner`, `direction`, and bookkeeping fields `n`,
`p`, `v`, `repeats`, `seed`, `levels`, `resampling`, `call`.

## Sign convention

Let `s = +1` if the metric is to be maximized and `s = -1` if it is to
be minimized. For each fold,
`importance = s * (metric_full - metric_reduced)`, so a positive value
always means that keeping the anchor predicts better than the
residualized others alone. Negative values are reported as they are;
they are never truncated to zero. The reported importance is the mean
over folds; `sd` is the sample standard deviation of the fold values and
is a descriptive quantity only, not a standard error.

## What the numbers are not

The values are out-of-sample predictive gains for a particular learner,
metric and resampling scheme. They are not causal effects, not
regression coefficients and not conditional effects, and the package
deliberately provides no confidence intervals or p-values. See
`docs/specification.md` in the source repository for the complete
specification, including how the computation differs from the reference
UMFI-LR implementation.

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
#>   metric:     mse (minimize); importance = -(full - reduced)
#>   resampling: 5-fold cross-validation (1 repeat)
#> 
#> # A tibble: 4 × 4
#>   feature importance     sd n_folds
#>   <chr>        <dbl>  <dbl>   <int>
#> 1 x1         3.18    0.809        5
#> 2 x2         0.920   0.383        5
#> 3 x3         0.00634 0.0250       5
#> 4 x4         3.13    0.865        5
fold_results(res)
#> # A tibble: 20 × 7
#>    feature id    id2   metric_full metric_reduced importance error
#>    <chr>   <chr> <chr>       <dbl>          <dbl>      <dbl> <chr>
#>  1 x1      Fold1 NA           1.02           4.31    3.29    NA   
#>  2 x1      Fold2 NA           1.85           4.00    2.15    NA   
#>  3 x1      Fold3 NA           1.25           3.92    2.67    NA   
#>  4 x1      Fold4 NA           1.37           4.89    3.52    NA   
#>  5 x1      Fold5 NA           1.18           5.44    4.26    NA   
#>  6 x2      Fold1 NA           1.02           2.22    1.20    NA   
#>  7 x2      Fold2 NA           1.85           2.53    0.688   NA   
#>  8 x2      Fold3 NA           1.25           1.61    0.364   NA   
#>  9 x2      Fold4 NA           1.37           2.46    1.09    NA   
#> 10 x2      Fold5 NA           1.18           2.45    1.26    NA   
#> 11 x3      Fold1 NA           1.02           1.03    0.00905 NA   
#> 12 x3      Fold2 NA           1.85           1.89    0.0398  NA   
#> 13 x3      Fold3 NA           1.25           1.24   -0.00375 NA   
#> 14 x3      Fold4 NA           1.37           1.34   -0.0282  NA   
#> 15 x3      Fold5 NA           1.18           1.20    0.0148  NA   
#> 16 x4      Fold1 NA           1.02           4.49    3.47    NA   
#> 17 x4      Fold2 NA           1.85           3.69    1.85    NA   
#> 18 x4      Fold3 NA           1.25           3.92    2.67    NA   
#> 19 x4      Fold4 NA           1.37           5.03    3.66    NA   
#> 20 x4      Fold5 NA           1.18           5.17    3.99    NA   

# Binary classification with log loss
yb <- factor(ifelse(y > median(y), "high", "low"))
shared_importance(x, yb, learner = "linear", metric = "accuracy", v = 5, seed = 1)
#> <sharedimp_result>
#>   task:       classification (2 classes)
#>   learner:    linear
#>   metric:     accuracy (maximize); importance = +(full - reduced)
#>   resampling: 5-fold cross-validation (1 repeat), stratified by y
#> 
#> # A tibble: 4 × 4
#>   feature importance     sd n_folds
#>   <chr>        <dbl>  <dbl>   <int>
#> 1 x1          0.253  0.110        5
#> 2 x2          0.0667 0.113        5
#> 3 x3         -0.0133 0.0298       5
#> 4 x4          0.253  0.0606       5

# \donttest{
# Default random forest learner (ranger, 500 trees)
shared_importance(x, y, v = 5, seed = 1)
#> <sharedimp_result>
#>   task:       regression
#>   learner:    rand_forest
#>   metric:     mse (minimize); importance = -(full - reduced)
#>   resampling: 5-fold cross-validation (1 repeat)
#> 
#> # A tibble: 4 × 4
#>   feature importance     sd n_folds
#>   <chr>        <dbl>  <dbl>   <int>
#> 1 x1         3.73    0.448        5
#> 2 x2         0.860   0.319        5
#> 3 x3         0.00795 0.0194       5
#> 4 x4         3.60    0.409        5
# }
```
