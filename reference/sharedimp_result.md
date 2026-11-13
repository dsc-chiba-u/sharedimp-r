# Results of shared_importance()

[`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md)
returns an object of class `sharedimp_result`. The headline table is
`$importance`; `fold_results()` (or `$folds`) gives the paired per-fold
metrics from which it was aggregated.

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

`fold_results()` returns a tibble with one row per (feature, fold) and
columns `feature`, `id`, `id2`, `metric_full`, `metric_reduced`,
`importance`, `error`, and `direction` when it was requested.
[`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html) returns
the `importance` table as a plain data frame.

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
#>   metric:     mse (minimize); importance = -(full - reduced)
#>   resampling: 3-fold cross-validation (1 repeat)
#> 
#> # A tibble: 3 × 4
#>   feature importance     sd n_folds
#>   <chr>        <dbl>  <dbl>   <int>
#> 1 a           3.09   1.39         3
#> 2 b          -0.0276 0.0342       3
#> 3 c          -0.0477 0.134        3
fold_results(res)
#> # A tibble: 9 × 7
#>   feature id    id2   metric_full metric_reduced importance error
#>   <chr>   <chr> <chr>       <dbl>          <dbl>      <dbl> <chr>
#> 1 a       Fold1 NA           1.15          4.61    3.45     NA   
#> 2 a       Fold2 NA           1.04          5.30    4.26     NA   
#> 3 a       Fold3 NA           1.10          2.65    1.55     NA   
#> 4 b       Fold1 NA           1.15          1.14   -0.0173   NA   
#> 5 b       Fold2 NA           1.04          0.979  -0.0657   NA   
#> 6 b       Fold3 NA           1.10          1.10    0.000251 NA   
#> 7 c       Fold1 NA           1.15          1.23    0.0739   NA   
#> 8 c       Fold2 NA           1.04          1.02   -0.0252   NA   
#> 9 c       Fold3 NA           1.10          0.906  -0.192    NA   
```
