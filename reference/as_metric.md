# Metrics available in shared_importance()

[`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md)
evaluates predictions with `yardstick` metrics. The `metric` argument
accepts a built-in metric name or any `yardstick` metric function (class
`numeric_metric`, `class_metric`, or `prob_metric`) whose direction is
`"maximize"` or `"minimize"`. The direction determines the sign
convention: importance is `+(full - reduced)` for metrics to maximize
and `-(full - reduced)` for metrics to minimize, so that a positive
value always means that keeping the anchor feature predicts better.

## Usage

``` r
as_metric(metric, task, label = NULL)

mse(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...)
```

## Arguments

- metric:

  A metric name or a `yardstick` metric function.

- task:

  `"regression"` or `"classification"`.

- label:

  Optional display name for a metric passed as a function.

- data, truth, estimate, na_rm, case_weights, ...:

  Standard `yardstick` metric arguments.

## Value

`as_metric()` returns an object of class `sharedimp_metric` with
elements `name`, `direction`, `kind` (`"numeric"`, `"class"` or
`"prob"`), and `fn`. `mse` is the `yardstick`-style metric function
itself.

## Details

Built-in names:

|                           |                |           |
|---------------------------|----------------|-----------|
| name                      | task           | direction |
| `"mse"` (default)         | regression     | minimize  |
| `"rmse"`                  | regression     | minimize  |
| `"mae"`                   | regression     | minimize  |
| `"rsq"`                   | regression     | maximize  |
| `"mn_log_loss"` (default) | classification | minimize  |
| `"accuracy"`              | classification | maximize  |
| `"roc_auc"`               | classification | maximize  |

`"mse"` is the mean squared error, implemented as a `yardstick` metric
because `yardstick` does not ship one. For binary tasks the first level
of `y` is the event level of probability metrics; for multiclass tasks
`"roc_auc"` uses the Hand-Till estimator (the `yardstick` default).

## Examples

``` r
m <- as_metric("rmse", task = "regression")
m$direction
#> [1] "minimize"
m2 <- as_metric(yardstick::mae, task = "regression")
m2$name
#> [1] "custom metric"
mse(data.frame(t = c(1, 2, 3), e = c(1, 2, 5)), t, e)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 mse     standard        1.33
```
