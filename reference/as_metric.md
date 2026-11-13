# Metrics available in shared_importance()

[`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md)
evaluates predictions with `yardstick` metrics. The `metric` argument
accepts a built-in metric name or any `yardstick` metric function (class
`numeric_metric`, `class_metric`, or `prob_metric`) whose direction is
`"maximize"` or `"minimize"`. The direction determines the orientation
of the reported *scores*: the metric value for metrics to maximize and
its negative for metrics to minimize, so that
`importance = score_full - score_reduced` is positive whenever keeping
the anchor feature predicts better.

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

Built-in names (the common names of the R/Python specification, with the
`yardstick` function and the matching scikit-learn scorer):

|  |  |  |  |  |
|----|----|----|----|----|
| name | task | direction | yardstick | scikit-learn |
| `"mse"` (default) | regression | minimize | `mse` (this package) | `neg_mean_squared_error` |
| `"rmse"` | regression | minimize | `rmse` | `neg_root_mean_squared_error` |
| `"mae"` | regression | minimize | `mae` | `neg_mean_absolute_error` |
| `"r2"` | regression | maximize | `rsq_trad` (1 - SS_res / SS_tot) | `r2` |
| `"rsq"` | regression | maximize | `rsq` (squared Pearson correlation) | none built in |
| `"log_loss"` (default) | classification | minimize | `mn_log_loss` | `neg_log_loss` |
| `"accuracy"` | classification | maximize | `accuracy` | `accuracy` |
| `"roc_auc"` | classification | maximize | `roc_auc` (binary; Hand-Till for multiclass) | `roc_auc` / `roc_auc_ovo` |

The `yardstick` names `"rsq_trad"` and `"mn_log_loss"` are accepted as
aliases of `"r2"` and `"log_loss"`. `"mse"` is implemented here because
`yardstick` does not ship a mean squared error. For binary tasks the
first level of `y` is the event level of probability metrics (log loss
and AUC do not depend on this choice); for multiclass tasks `"roc_auc"`
uses the Hand-Till estimator.

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
