# sharedimp

`sharedimp` is an R implementation of the linear-regression variant of
**ultra-marginal feature importance** (UMFI-LR; Janssen, Guan & Robeva,
AISTATS 2023) designed for modern, resampling-based machine-learning
workflows. It is a companion to the Python package of the same name
(`sharedimp-py`); both follow the specification in
[`docs/specification.md`](https://dsc-chiba-u.github.io/sharedimp-r/docs/specification.md).

We do not claim a new importance measure. The package makes an existing
one safe and convenient:

- **fold-internal residualization**: the OLS coefficients used to remove
  the anchor’s linear component from the other features are estimated on
  the training rows of each fold and applied, unchanged, to the
  assessment rows;
- **paired full/reduced evaluation**: the model with the anchor and the
  model without it are evaluated on the same assessment rows, fold by
  fold;
- **metric-independent sign**: importance is positive whenever keeping
  the anchor predicts better, for metrics to maximize and to minimize
  alike;
- **no truncation**: negative values are reported, not clipped to zero;
- **tidymodels adapters**: `parsnip` learners, `rsample` resamples and
  `yardstick` metrics, with fixed hyperparameters and a `seed` argument;
- **per-fold results** are always retained.

## Installation

``` r

# install.packages("pak")
pak::pak("dsc-chiba-u/sharedimp-r")
```

## Usage

``` r

library(sharedimp)
set.seed(1)
n <- 300
x <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n))
x$x5 <- x$x1 + rnorm(n, sd = 0.2)        # a noisy copy of x1
y <- 3 * x$x1 + 2 * x$x2 + x$x3 + rnorm(n)

# Default: random forest (ranger, 500 trees), 5-fold CV, MSE
res <- shared_importance(x, y, seed = 1)
res
#> <sharedimp_result>
#>   task:       regression
#>   learner:    rand_forest
#>   metric:     mse (minimize); score = -mse, importance = score_full - score_reduced
#>   resampling: 5-fold cross-validation (1 repeat)
#>
#> # A tibble: 5 × 6
#>   feature importance importance_std score_full score_reduced n_splits
#>   ...

# Fast deterministic linear learner, explicit metric (r2 = 1 - SS_res / SS_tot)
shared_importance(x, y, learner = "linear", metric = "r2", v = 5, seed = 1)

# Per-split paired scores (common schema with the Python package)
fold_results(res)

# Classification (first level = event level), any yardstick metric
yb <- factor(y > median(y), labels = c("low", "high"))
shared_importance(x, yb, metric = "roc_auc", seed = 1)

# Any parsnip specification with fixed hyperparameters (here: lasso via glmnet)
spec <- parsnip::linear_reg(penalty = 0.01, mixture = 1) |>
  parsnip::set_engine("glmnet")
shared_importance(x, y, learner = spec, seed = 1)
```

See
[`vignette("sharedimp")`](https://dsc-chiba-u.github.io/sharedimp-r/articles/sharedimp.md)
for a longer walk-through.

## How it works

For each anchor feature `i` and each split:

1.  On the training rows, regress every other feature on the anchor
    (`X_j = a_j + b_j X_i + e`) and residualize both training and
    assessment rows with the same `(a_j, b_j)`. A column that is an
    exact affine function of the anchor gets a residual of exactly zero
    (and is kept).
2.  Fit the learner to the *full* representation (the anchor, raw and in
    its original position, plus the residualized others) and to the
    *reduced* representation (the residualized others only).
3.  Score both on the assessment rows, with scores oriented
    greater-is-better (`-mse`, `-log_loss`, `r2`, `accuracy`, …), and
    take `importance = score_full - score_reduced`.
4.  Average over splits; keep the split-level table.

The R and Python packages share one specification
([`docs/specification.md`](https://dsc-chiba-u.github.io/sharedimp-r/docs/specification.md))
and produce identical tables on deterministic learners with explicit
splits.

## What the numbers are not

They are out-of-sample predictive gains for a particular learner, metric
and resampling scheme. They are **not** causal effects, conditional
effects or regression coefficients, and the package provides no
confidence intervals or p-values; `importance_std` is a descriptive
between-split dispersion.

## Differences from the reference UMFI-LR implementation

| aspect | [HydroML/UMFI](https://github.com/HydroML/UMFI) | sharedimp |
|----|----|----|
| residualization fit | whole data set | training fold only |
| gate | slope p-value \< 0.01 | always residualize |
| evaluation | ranger out-of-bag R² / accuracy with floors | paired per-fold metrics, any learner |
| negative values | set to 0 | kept |
| learner / metric | fixed (ranger 100 trees; R², accuracy) | configurable, fixed hyperparameters |

Details and a recorded numerical comparison:
[`docs/specification.md`](https://dsc-chiba-u.github.io/sharedimp-r/docs/specification.md),
[`docs/umfi_comparison.md`](https://dsc-chiba-u.github.io/sharedimp-r/docs/umfi_comparison.md).

## Reference

Janssen, J., Guan, V., & Robeva, E. (2023). Ultra-marginal Feature
Importance: Learning from Data with Causal Guarantees. *Proceedings of
the 26th International Conference on Artificial Intelligence and
Statistics (AISTATS)*, PMLR 206:10782–10814.
<https://proceedings.mlr.press/v206/janssen23a.html>

## License

MIT © 2026 Data Science Core. See
[LICENSE.md](https://dsc-chiba-u.github.io/sharedimp-r/LICENSE.md).
