# Shared predictive importance with sharedimp

`sharedimp` implements the linear-regression variant of ultra-marginal
feature importance (UMFI-LR; Janssen, Guan and Robeva, AISTATS 2023) in
a form that fits modern resampling-based workflows. The package does not
introduce a new importance measure. It makes an existing one safe to
use: the residualization step is estimated inside each training fold,
the two models being compared are evaluated on the same assessment rows,
the sign of the result does not depend on the metric, and negative
values are not hidden.

## The idea in one paragraph

Take an *anchor* feature $`X_i`$. Regress every other feature $`X_j`$ on
$`X_i`$ (ordinary least squares with an intercept) and keep the
residuals $`R_j = X_j - a_j - b_j X_i`$. By construction the residuals
are uncorrelated with the anchor on the data used to fit them. Now
compare two models of $`y`$:

- **full**: $`[X_i, R_{-i}]`$ — the anchor plus the residualized others;
- **reduced**: $`[R_{-i}]`$ — the residualized others only.

The out-of-sample gain of the full model over the reduced model is the
importance of $`X_i`$. Because the linear part of $`X_i`$ has been
removed from the other features, information that $`X_i`$*shares* with
correlated features is credited to $`X_i`$ rather than being split or
lost, which is what distinguishes UMFI from ablation and permutation
importance for correlated predictors.

## A worked example

``` r

library(sharedimp)
set.seed(2026)
n <- 300
x <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  x3 = rnorm(n),
  x4 = rnorm(n)
)
x$x5 <- x$x1 + rnorm(n, sd = 0.2) # a noisy copy of x1
y <- 3 * x$x1 + 2 * x$x2 + 1 * x$x3 + rnorm(n)
```

`x4` is pure noise and `x5` is a near-duplicate of `x1`. With the fast,
deterministic linear learner and the mean squared error:

``` r

res <- shared_importance(x, y, learner = "linear", v = 5, seed = 1)
res
#> <sharedimp_result>
#>   task:       regression
#>   learner:    linear
#>   metric:     mse (minimize); importance = -(full - reduced)
#>   resampling: 5-fold cross-validation (1 repeat)
#> 
#> # A tibble: 5 × 4
#>   feature importance     sd n_folds
#>   <chr>        <dbl>  <dbl>   <int>
#> 1 x1          9.81   1.41         5
#> 2 x2          6.67   0.572        5
#> 3 x3          1.38   0.352        5
#> 4 x4          0.0742 0.0793       5
#> 5 x5          9.58   1.38         5
```

Importance is `-(MSE_full - MSE_reduced)`, i.e. the increase in MSE
caused by dropping the anchor while keeping the residualized others. For
an independent linear-Gaussian design it is approximately $`\beta_i^2
\mathrm{Var}(X_i)`$: roughly 9, 4 and 1 for `x1`, `x2`, `x3`, and 0 for
the noise feature. The near-duplicate `x5` receives almost the same
importance as `x1`: once `x1` is residualized on `x5`, little of `x1` is
left, so `x5` carries the shared signal. This is the
duplication-symmetry property of UMFI.

The per-fold paired metrics are always kept:

``` r

fold_results(res)
#> # A tibble: 25 × 7
#>    feature id    id2   metric_full metric_reduced importance error
#>    <chr>   <chr> <chr>       <dbl>          <dbl>      <dbl> <chr>
#>  1 x1      Fold1 NA          1.29           10.1        8.82 NA   
#>  2 x1      Fold2 NA          1.04           12.2       11.2  NA   
#>  3 x1      Fold3 NA          0.708          10.0        9.32 NA   
#>  4 x1      Fold4 NA          1.03            9.35       8.32 NA   
#>  5 x1      Fold5 NA          0.904          12.3       11.4  NA   
#>  6 x2      Fold1 NA          1.29            8.36       7.08 NA   
#>  7 x2      Fold2 NA          1.04            7.34       6.31 NA   
#>  8 x2      Fold3 NA          0.708           6.90       6.20 NA   
#>  9 x2      Fold4 NA          1.03            8.50       7.48 NA   
#> 10 x2      Fold5 NA          0.904           7.20       6.30 NA   
#> # ℹ 15 more rows
```

Negative fold values for `x4` are expected and are reported as they are.

## Choosing the learner and the metric

The default learner is a random forest (`ranger`, 500 trees,
`min_n = 5`) with fixed hyperparameters. Any `parsnip` model
specification can be used instead, as can a pair of `fit`/`predict`
functions wrapped with
[`sharedimp_learner()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/sharedimp_learner.md).

``` r

res_rf <- shared_importance(x, y, v = 5, seed = 1)
res_rf$importance
#> # A tibble: 5 × 4
#>   feature importance     sd n_folds
#>   <chr>        <dbl>  <dbl>   <int>
#> 1 x1           9.23  1.95         5
#> 2 x2           6.37  0.823        5
#> 3 x3           1.01  0.345        5
#> 4 x4          -0.124 0.0872       5
#> 5 x5           9.05  1.81         5
```

Metrics come from `yardstick`. The direction attribute of the metric
decides the sign convention, so a metric to maximize (`"rsq"`,
`"accuracy"`, `"roc_auc"`) and a metric to minimize (`"mse"`, `"rmse"`,
`"mae"`, `"mn_log_loss"`) both yield positive importance when the anchor
helps.

``` r

shared_importance(x, y, learner = "linear", metric = "rsq", v = 5, seed = 1)$importance
#> # A tibble: 5 × 4
#>   feature importance      sd n_folds
#>   <chr>        <dbl>   <dbl>   <int>
#> 1 x1         0.572   0.107         5
#> 2 x2         0.419   0.118         5
#> 3 x3         0.0897  0.0379        5
#> 4 x4         0.00595 0.00664       5
#> 5 x5         0.552   0.118         5
```

## Classification

For a factor response the task is classification. The first factor level
is the event level for probability metrics, and folds are stratified by
`y`.

``` r

yb <- factor(ifelse(y > median(y), "high", "low"), levels = c("high", "low"))
res_b <- shared_importance(x, yb, learner = "linear", metric = "mn_log_loss", v = 5, seed = 1)
#> Warning: The learner emitted 16 warning(s) during fitting or prediction:
#> • [16 times] glm.fit: fitted probabilities numerically 0 or 1 occurred
res_b$importance
#> # A tibble: 5 × 4
#>   feature importance     sd n_folds
#>   <chr>        <dbl>  <dbl>   <int>
#> 1 x1          0.406  0.129        5
#> 2 x2          0.329  0.134        5
#> 3 x3          0.0792 0.0372       5
#> 4 x4         -0.0174 0.0444       5
#> 5 x5          0.400  0.144        5
```

Multiclass responses work the same way; `"roc_auc"` then uses the
Hand-Till estimator and `"mn_log_loss"` the multinomial log loss.

## Reproducibility and explicit folds

`seed` controls fold generation and any stochastic learner under a local
RNG state, leaving the caller’s random stream untouched. For comparisons
across software (for example with the Python implementation of the same
specification) pass explicit folds, either an `rsample` `rset` or a
vector of fold labels:

``` r

labels <- rep(1:5, length.out = n)
res_fixed <- shared_importance(x, y, learner = "linear", resamples = labels)
res_fixed$resampling
#> [1] "user-supplied fold labels (5 folds)"
```

## What the numbers are and are not

The values are out-of-sample predictive gains for a given learner,
metric and resampling scheme. They are not causal effects, not
conditional effects, and not regression coefficients. The `sd` column is
the dispersion of the fold-level values and is purely descriptive; the
package deliberately provides no confidence intervals or p-values.

The optional `direction = TRUE` diagnostic reports the correlation
between the anchor and the difference between the full and reduced
predictions on the assessment rows. It summarises whether the anchor’s
contribution to the prediction increases or decreases with the anchor’s
value and nothing more.

``` r

shared_importance(x, y, learner = "linear", v = 5, seed = 1, direction = TRUE)$importance
#> # A tibble: 5 × 5
#>   feature importance     sd n_folds direction
#>   <chr>        <dbl>  <dbl>   <int>     <dbl>
#> 1 x1          9.81   1.41         5         1
#> 2 x2          6.67   0.572        5         1
#> 3 x3          1.38   0.352        5         1
#> 4 x4          0.0742 0.0793       5        -1
#> 5 x5          9.58   1.38         5         1
```

## Relation to the reference UMFI implementation

The reference implementation (<https://github.com/HydroML/UMFI>) fits
the residualization on the whole data set, gates it on a slope p-value,
scores random forests by out-of-bag $`R^2`$ or accuracy with floors, and
sets negative values to zero. `sharedimp` residualizes inside each
training fold, always residualizes, uses paired per-fold assessment
metrics with any learner, and keeps negative values. Numerical identity
is therefore not expected; the specification and a recorded comparison
are in the `docs/` directory of the source repository.
