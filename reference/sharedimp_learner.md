# Learner adapters for shared_importance()

A learner is a pair of functions, `fit(x, y)` and `predict(model, x)`,
where `x` is a numeric matrix with column names (a representation built
by
[`build_representations()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/residualize_fit.md)).
For regression, [`predict()`](https://rdrr.io/r/stats/predict.html) must
return a numeric vector with one value per row of `x`. For
classification it must return a numeric matrix of class probabilities
with one column per level of `y`, in level order (column names are
optional). Predicted class labels are always derived by
[`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md)
as the argmax of that matrix.

## Usage

``` r
sharedimp_learner(fit, predict, label = "custom learner")
```

## Arguments

- fit:

  A function `function(x, y)` returning a fitted model object.

- predict:

  A function `function(model, x)` returning predictions in the format
  described above.

- label:

  A short description used in printing.

## Value

An object of class `sharedimp_learner`.

## Details

[`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md)
accepts three kinds of `learner`:

- a string: `"rand_forest"` (default; `ranger` with 500 trees and
  `min_n = 5`, `mtry = floor(sqrt(ncol(x)))`) or `"linear"` (ordinary
  least squares, logistic regression, or multinomial logistic regression
  via `nnet`, depending on the task);

- a `parsnip` model specification whose mode matches the task, used as
  is (its hyperparameters are fixed; nothing is tuned);

- an object created by `sharedimp_learner()`.

## Examples

``` r
ols <- sharedimp_learner(
  fit = function(x, y) stats::lm.fit(cbind(1, x), y),
  predict = function(model, x) {
    beta <- model$coefficients
    beta[is.na(beta)] <- 0
    drop(cbind(1, x) %*% beta)
  },
  label = "OLS via lm.fit"
)
set.seed(1)
x <- matrix(rnorm(200), 50, 4, dimnames = list(NULL, paste0("x", 1:4)))
y <- x[, 1] + rnorm(50)
shared_importance(x, y, learner = ols, v = 3, seed = 1)
#> <sharedimp_result>
#>   task:       regression
#>   learner:    OLS via lm.fit
#>   metric:     mse (minimize); importance = -(full - reduced)
#>   resampling: 3-fold cross-validation (1 repeat)
#> 
#> # A tibble: 4 × 4
#>   feature importance     sd n_folds
#>   <chr>        <dbl>  <dbl>   <int>
#> 1 x1          0.365  0.874        3
#> 2 x2         -0.0707 0.0569       3
#> 3 x3          0.0961 0.0603       3
#> 4 x4         -0.189  0.455        3
```
