# Results of shared_importance()

[`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md)
returns an object of class `sharedimp_result`. The headline table is
`$importance`; `fold_results()` (or `$folds`) gives the paired per-split
scores from which it was aggregated. Both tables use the column names
shared with the Python package.

## Usage

``` r
fold_results(x)

# S3 method for class 'sharedimp_result'
print(x, ...)

# S3 method for class 'sharedimp_result'
as.data.frame(x, row.names = NULL, optional = FALSE, ...)

# S3 method for class 'sharedimp_result'
summary(object, ...)
```

## Arguments

- x, object:

  A `sharedimp_result`.

- ...:

  Passed on to the tibble print method.

- row.names, optional:

  Ignored; present for compatibility with the generic.

## Value

`fold_results()` returns a tibble with one row per (split, feature),
ordered by split and then by feature, with the common columns `feature`,
`split` (1-based index in generation order), `n_train`, `n_test`,
`score_full`, `score_reduced`, `importance`, followed by the R-specific
columns `id`, `id2` (`rsample` labels) and `error` (`NA` unless the
learner failed on that split).
[`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html) returns
the `importance` table as a plain data frame.

## Details

`$importance` has one row per anchor feature, in the order of
`features`, with columns `feature`, `importance` (mean over splits of
`score_full - score_reduced`), `importance_std` (sample standard
deviation of the per-split importances, `NA` with one split;
descriptive, not a standard error), `score_full`, `score_reduced` (mean
scores) and `n_splits` (number of splits with a successful fit).

Scores are oriented so that greater is better: the metric value for a
metric to maximize and its negative for a metric to minimize (see
[`as_metric()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/as_metric.md)).

## Examples

``` r
set.seed(1)
x <- matrix(rnorm(300), 100, 3, dimnames = list(NULL, c("a", "b", "c")))
y <- 2 * x[, "a"] + rnorm(100)
res <- shared_importance(x, y, learner = "linear", v = 3, seed = 1)
res
#> <sharedimp_result>
#>   task:       regression
#>   learner:    linear
#>   metric:     mse (minimize); score = -mse, importance = score_full - score_reduced
#>   resampling: 3-fold cross-validation (1 repeat)
#> 
#> # A tibble: 3 × 6
#>   feature importance importance_std score_full score_reduced n_splits
#>   <chr>        <dbl>          <dbl>      <dbl>         <dbl>    <int>
#> 1 a           3.09           1.39        -1.10         -4.19        3
#> 2 b          -0.0276         0.0342      -1.10         -1.07        3
#> 3 c          -0.0477         0.134       -1.10         -1.05        3
fold_results(res)
#> # A tibble: 9 × 10
#>   feature split n_train n_test score_full score_reduced importance id    id2  
#>   <chr>   <int>   <int>  <int>      <dbl>         <dbl>      <dbl> <chr> <chr>
#> 1 a           1      66     34      -1.15        -4.61    3.45     Fold1 NA   
#> 2 b           1      66     34      -1.15        -1.14   -0.0173   Fold1 NA   
#> 3 c           1      66     34      -1.15        -1.23    0.0739   Fold1 NA   
#> 4 a           2      67     33      -1.04        -5.30    4.26     Fold2 NA   
#> 5 b           2      67     33      -1.04        -0.979  -0.0657   Fold2 NA   
#> 6 c           2      67     33      -1.04        -1.02   -0.0252   Fold2 NA   
#> 7 a           3      67     33      -1.10        -2.65    1.55     Fold3 NA   
#> 8 b           3      67     33      -1.10        -1.10    0.000251 Fold3 NA   
#> 9 c           3      67     33      -1.10        -0.906  -0.192    Fold3 NA   
#> # ℹ 1 more variable: error <chr>
```
