#' Fold-internal residualization (the UMFI-LR preprocessing)
#'
#' `residualize_fit()` regresses every column of `x` except the anchor on the
#' anchor by ordinary least squares with an intercept and returns the
#' coefficients. `residualize_apply()` applies previously estimated
#' coefficients to (possibly different) rows, so that assessment data can be
#' transformed with coefficients estimated on training data only.
#' `build_representations()` combines the two to produce the full and reduced
#' representations used by [shared_importance()].
#'
#' For anchor `i` and every other column `j`,
#' `b_j = Cov(x_i, x_j) / Var(x_i)` and `a_j = mean(x_j) - b_j * mean(x_i)`,
#' both computed on the rows passed to `residualize_fit()`. The residualized
#' column is `x_j - a_j - b_j * x_i`.
#'
#' Two deterministic rules cover degenerate cases (specification Section 5.1):
#'
#' * If the anchor has zero sum of squares about its mean on the fitting rows
#'   (it is constant), all slopes are set to zero, the intercepts become the
#'   column means, and a warning of class `sharedimp_constant_anchor` is
#'   raised.
#' * If a column `j` is (numerically) an exact affine function of the anchor on
#'   the fitting rows, i.e. its residual sum of squares is at most `1e-10`
#'   times its total sum of squares, it is flagged in `collinear` and its
#'   residual is set to exactly `0` in `residualize_apply()`, for fitting and
#'   new rows alike. The column is kept, so the representations always have
#'   `p` and `p - 1` columns.
#'
#' No noise is ever injected and no column is ever dropped.
#'
#' @param x A numeric matrix (or data frame of numeric columns) with column
#'   names. For `residualize_fit()` these are the training rows.
#' @param anchor Column name or index of the anchor feature.
#' @param coefs An object returned by `residualize_fit()`.
#' @param x_train,x_assess Training and assessment rows with identical columns.
#'
#' @return
#' `residualize_fit()` returns a list of class `sharedimp_residualization`
#' with elements `anchor` (column name), `features` (names of the other
#' columns, in their original order), `intercept` and `slope` (named numeric
#' vectors, one entry per other column), `collinear` (named logical vector,
#' `TRUE` for columns whose residual is fixed at zero), `constant_anchor`
#' (logical) and `n_train`.
#'
#' `residualize_apply()` returns a numeric matrix with one column per entry
#' of `coefs$features`, in that order.
#'
#' `build_representations()` returns a list with `full_train`, `full_assess`
#' (all `p` columns in the original order, the anchor raw and in its original
#' position, the others residualized), `reduced_train`, `reduced_assess` (the
#' same without the anchor column) and `coefs`.
#'
#' @examples
#' set.seed(1)
#' x <- cbind(a = rnorm(20), b = rnorm(20), c = rnorm(20))
#' x[, "b"] <- 2 * x[, "a"] + x[, "b"]
#' cf <- residualize_fit(x[1:15, ], anchor = "b")
#' cf$slope
#' r <- residualize_apply(x[16:20, ], cf)
#' rep <- build_representations(x[1:15, ], x[16:20, ], anchor = "b")
#' colnames(rep$full_train) # anchor "b" stays in the second position
#' colnames(rep$reduced_train)
#' @export
residualize_fit <- function(x, anchor) {
  x <- check_x(x)
  i <- resolve_anchor(anchor, x)
  residualize_fit_impl(x, i)
}

# Relative tolerance of the exact-collinearity rule (specification Section 5.1).
collinear_tol <- 1e-10

# Unvalidated core used in the main loop; `x` is a checked numeric matrix.
residualize_fit_impl <- function(x, i) {
  n <- nrow(x)
  xi <- x[, i]
  others <- x[, -i, drop = FALSE]
  features <- colnames(others)
  mu_other <- colMeans(others)
  mu_i <- mean(xi)
  xic <- xi - mu_i
  ss_i <- sum(xic^2)
  constant_anchor <- ss_i == 0
  if (constant_anchor) {
    warn(
      sprintf(
        "Anchor `%s` is constant on the training rows; its slopes were set to zero.",
        colnames(x)[i]
      ),
      class = "sharedimp_constant_anchor"
    )
    slope <- setNames(rep(0, length(features)), features)
  } else {
    slope <- as.vector(crossprod(xic, others)) / ss_i
    names(slope) <- features
  }
  intercept <- mu_other - slope * mu_i
  # Exact-collinearity rule: residual sum of squares relative to total.
  others_c <- sweep(others, 2L, mu_other)
  ss_tot <- colSums(others_c^2)
  resid <- others_c - outer(xic, slope)
  ss_res <- colSums(resid^2)
  collinear <- ss_tot > 0 & ss_res <= collinear_tol * ss_tot
  names(collinear) <- features
  structure(
    list(
      anchor = colnames(x)[i],
      features = features,
      intercept = intercept,
      slope = slope,
      collinear = collinear,
      constant_anchor = constant_anchor,
      n_train = n
    ),
    class = "sharedimp_residualization"
  )
}

#' @rdname residualize_fit
#' @export
residualize_apply <- function(x, coefs) {
  if (!inherits(coefs, "sharedimp_residualization")) {
    abort("`coefs` must be the result of `residualize_fit()`.")
  }
  x <- check_x(x)
  i <- resolve_anchor(coefs$anchor, x)
  missing_cols <- setdiff(coefs$features, colnames(x))
  if (length(missing_cols) > 0L) {
    abort(sprintf(
      "`x` lacks column(s) present at fit time: %s.",
      paste(sprintf("`%s`", missing_cols), collapse = ", ")
    ))
  }
  residualize_apply_impl(x, coefs, i)
}

residualize_apply_impl <- function(x, coefs, i) {
  xi <- x[, i]
  others <- x[, coefs$features, drop = FALSE]
  res <- others - outer(xi, coefs$slope) -
    matrix(coefs$intercept, nrow = nrow(x), ncol = length(coefs$features), byrow = TRUE)
  if (any(coefs$collinear)) {
    res[, coefs$collinear] <- 0
  }
  colnames(res) <- coefs$features
  res
}

#' @rdname residualize_fit
#' @export
build_representations <- function(x_train, x_assess, anchor) {
  x_train <- check_x(x_train)
  x_assess <- check_x_assess(x_assess, x_train)
  i <- resolve_anchor(anchor, x_train)
  build_representations_impl(x_train, x_assess, i)
}

build_representations_impl <- function(x_train, x_assess, i) {
  coefs <- residualize_fit_impl(x_train, i)
  full_train <- x_train
  full_assess <- x_assess
  if (length(coefs$features) > 0L) {
    full_train[, coefs$features] <- residualize_apply_impl(x_train, coefs, i)
    full_assess[, coefs$features] <- residualize_apply_impl(x_assess, coefs, i)
  }
  list(
    full_train = full_train,
    full_assess = full_assess,
    reduced_train = full_train[, -i, drop = FALSE],
    reduced_assess = full_assess[, -i, drop = FALSE],
    coefs = coefs
  )
}

# Assessment rows may be few (even one), so relax the row check.
check_x_assess <- function(x_assess, x_train) {
  if (is.data.frame(x_assess)) {
    nm <- names(x_assess)
    x_assess <- as.matrix(data.frame(lapply(x_assess, as.numeric), check.names = FALSE))
    colnames(x_assess) <- nm
  }
  if (!is.matrix(x_assess) || !is.numeric(x_assess)) {
    abort("`x_assess` must be a numeric matrix or data frame.")
  }
  if (is.null(colnames(x_assess))) {
    colnames(x_assess) <- paste0("V", seq_len(ncol(x_assess)))
  }
  if (!identical(colnames(x_assess), colnames(x_train))) {
    abort("`x_assess` must have the same columns, in the same order, as `x_train`.")
  }
  if (anyNA(x_assess) || any(!is.finite(x_assess))) {
    abort("`x_assess` must not contain missing or non-finite values.")
  }
  storage.mode(x_assess) <- "double"
  x_assess
}
