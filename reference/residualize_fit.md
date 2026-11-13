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

build_representations(x_train, x_assess, anchor)
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

## Value

`residualize_fit()` returns a list of class `sharedimp_residualization`
with elements `anchor` (column name), `features` (names of the other
columns, in their original order), `intercept` and `slope` (named
numeric vectors, one entry per other column), `collinear` (named logical
vector, `TRUE` for columns whose residual is fixed at zero),
`constant_anchor` (logical) and `n_train`.

`residualize_apply()` returns a numeric matrix with one column per entry
of `coefs$features`, in that order.

`build_representations()` returns a list with `full_train`,
`full_assess` (all `p` columns in the original order, the anchor raw and
in its original position, the others residualized), `reduced_train`,
`reduced_assess` (the same without the anchor column) and `coefs`.

## Details

For anchor `i` and every other column `j`,
`b_j = Cov(x_i, x_j) / Var(x_i)` and
`a_j = mean(x_j) - b_j * mean(x_i)`, both computed on the rows passed to
`residualize_fit()`. The residualized column is `x_j - a_j - b_j * x_i`.

Two deterministic rules cover degenerate cases (specification Section
5.1):

- If the anchor has zero sum of squares about its mean on the fitting
  rows (it is constant), all slopes are set to zero, the intercepts
  become the column means, and a warning of class
  `sharedimp_constant_anchor` is raised.

- If a column `j` is (numerically) an exact affine function of the
  anchor on the fitting rows, i.e. its residual sum of squares is at
  most `1e-10` times its total sum of squares, it is flagged in
  `collinear` and its residual is set to exactly `0` in
  `residualize_apply()`, for fitting and new rows alike. The column is
  kept, so the representations always have `p` and `p - 1` columns.

No noise is ever injected and no column is ever dropped.

## Examples

``` r
set.seed(1)
x <- cbind(a = rnorm(20), b = rnorm(20), c = rnorm(20))
x[, "b"] <- 2 * x[, "a"] + x[, "b"]
cf <- residualize_fit(x[1:15, ], anchor = "b")
cf$slope
#>         a         c 
#> 0.4696116 0.1309606 
r <- residualize_apply(x[16:20, ], cf)
rep <- build_representations(x[1:15, ], x[16:20, ], anchor = "b")
colnames(rep$full_train) # anchor "b" stays in the second position
#> [1] "a" "b" "c"
colnames(rep$reduced_train)
#> [1] "a" "c"
```
