# Fold-internal residualization (the UMFI-LR preprocessing)

`residualize_fit()` regresses every column of `x` except the anchor on
the anchor by ordinary least squares with an intercept and returns the
coefficients. `residualize_apply()` applies previously estimated
coefficients to (possibly different) rows, so that assessment data can
be transformed with coefficients estimated on training data only.
`build_representations()` combines the two to produce the full and
reduced representations used by
[`shared_importance()`](https://dsc-chiba-u.github.io/sharedimp-r/reference/shared_importance.md).

## Usage

``` r
residualize_fit(x, anchor)

residualize_apply(x, coefs)

build_representations(x_train, x_assess, anchor, drop_constant = TRUE)
```

## Arguments

- x:

  A numeric matrix (or data frame of numeric columns) with column names.
  For `residualize_fit()` these are the training rows.

- anchor:

  Column name or index of the anchor feature.

- coefs:

  An object returned by `residualize_fit()`.

- x_train, x_assess:

  Training and assessment rows with identical columns.

- drop_constant:

  Should residualized columns that are constant on the training rows
  (for example the residual of an exact duplicate of the anchor) be
  dropped from both representations? Default `TRUE`, as in the
  specification.

## Value

`residualize_fit()` returns a list of class `sharedimp_residualization`
with elements `anchor` (column name), `features` (names of the other
columns), `intercept` and `slope` (named numeric vectors, one entry per
other column), and `constant_anchor` (logical).

`residualize_apply()` returns a numeric matrix with one column per entry
of `coefs$features`, in that order.

`build_representations()` returns a list with `full_train`,
`full_assess` (anchor column first, then residualized columns),
`reduced_train`, `reduced_assess` (residualized columns only), `coefs`,
and `dropped` (names of residualized columns removed because they were
constant on the training rows).

## Details

For anchor `i` and every other column `j`,
`b_j = Cov(x_i, x_j) / Var(x_i)` and
`a_j = mean(x_j) - b_j * mean(x_i)`, both computed on the rows passed to
`residualize_fit()`. The residualized column is `x_j - a_j - b_j * x_i`.
If the anchor is constant on those rows, all slopes are set to zero, the
intercepts become column means, and a warning is raised. No noise is
ever injected.

## Examples

``` r
set.seed(1)
x <- cbind(a = rnorm(20), b = rnorm(20), c = rnorm(20))
x[, "b"] <- 2 * x[, "a"] + x[, "b"]
cf <- residualize_fit(x[1:15, ], anchor = "a")
cf$slope
#>         b         c 
#> 1.6878654 0.3595172 
r <- residualize_apply(x[16:20, ], cf)
rep <- build_representations(x[1:15, ], x[16:20, ], anchor = "a")
colnames(rep$full_train)
#> [1] "a" "b" "c"
colnames(rep$reduced_train)
#> [1] "b" "c"
```
