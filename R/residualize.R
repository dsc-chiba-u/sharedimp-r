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
#' column is `x_j - a_j - b_j * x_i`. If the anchor is constant on those rows,
#' all slopes are set to zero, the intercepts become column means, and a
#' warning is raised. No noise is ever injected.
#'
#' @param x A numeric matrix (or data frame of numeric columns) with column
#'   names. For `residualize_fit()` these are the training rows.
#' @param anchor Column name or index of the anchor feature.
#' @param coefs An object returned by `residualize_fit()`.
#' @param x_train,x_assess Training and assessment rows with identical columns.
#' @param drop_constant Should residualized columns that are constant on the
#'   training rows (for example the residual of an exact duplicate of the
#'   anchor) be dropped from both representations? Default `TRUE`, as in the
#'   specification.
#'
#' @return
#' `residualize_fit()` returns a list of class `sharedimp_residualization`
#' with elements `anchor` (column name), `features` (names of the other
#' columns), `intercept` and `slope` (named numeric vectors, one entry per
#' other column), and `constant_anchor` (logical).
#'
#' `residualize_apply()` returns a numeric matrix with one column per entry
#' of `coefs$features`, in that order.
#'
#' `build_representations()` returns a list with `full_train`, `full_assess`
#' (anchor column first, then residualized columns), `reduced_train`,
#' `reduced_assess` (residualized columns only), `coefs`, and `dropped` (names
#' of residualized columns removed because they were constant on the training
#' rows).
#'
#' @examples
#' set.seed(1)
#' x <- cbind(a = rnorm(20), b = rnorm(20), c = rnorm(20))
#' x[, "b"] <- 2 * x[, "a"] + x[, "b"]
#' cf <- residualize_fit(x[1:15, ], anchor = "a")
#' cf$slope
#' r <- residualize_apply(x[16:20, ], cf)
#' rep <- build_representations(x[1:15, ], x[16:20, ], anchor = "a")
#' colnames(rep$full_train)
#' colnames(rep$reduced_train)
#' @export
residualize_fit <- function(x, anchor) {
  x <- check_x(x)
  i <- resolve_anchor(anchor, x)
  residualize_fit_impl(x, i)
}

# Unvalidated core used in the main loop; `x` is a checked numeric matrix.
residualize_fit_impl <- function(x, i) {
  n <- nrow(x)
  xi <- x[, i]
  others <- x[, -i, drop = FALSE]
  features <- colnames(others)
  mu_other <- colMeans(others)
  mu_i <- mean(xi)
  constant_anchor <- is_constant(xi)
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
    xic <- xi - mu_i
    slope <- as.vector(crossprod(xic, others)) / sum(xic^2)
    names(slope) <- features
  }
  intercept <- mu_other - slope * mu_i
  structure(
    list(
      anchor = colnames(x)[i],
      features = features,
      intercept = intercept,
      slope = slope,
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
  colnames(res) <- coefs$features
  res
}

#' @rdname residualize_fit
#' @export
build_representations <- function(x_train, x_assess, anchor, drop_constant = TRUE) {
  x_train <- check_x(x_train)
  x_assess <- check_x_assess(x_assess, x_train)
  i <- resolve_anchor(anchor, x_train)
  build_representations_impl(x_train, x_assess, i, drop_constant = drop_constant)
}

build_representations_impl <- function(x_train, x_assess, i, drop_constant = TRUE) {
  coefs <- residualize_fit_impl(x_train, i)
  r_train <- residualize_apply_impl(x_train, coefs, i)
  r_assess <- residualize_apply_impl(x_assess, coefs, i)
  dropped <- character(0)
  if (drop_constant && ncol(r_train) > 0L) {
    const <- apply(r_train, 2L, is_constant)
    if (any(const)) {
      dropped <- colnames(r_train)[const]
      r_train <- r_train[, !const, drop = FALSE]
      r_assess <- r_assess[, !const, drop = FALSE]
    }
  }
  anchor_name <- colnames(x_train)[i]
  full_train <- cbind(x_train[, i, drop = FALSE], r_train)
  full_assess <- cbind(x_assess[, i, drop = FALSE], r_assess)
  colnames(full_train)[1L] <- anchor_name
  colnames(full_assess)[1L] <- anchor_name
  list(
    full_train = full_train,
    full_assess = full_assess,
    reduced_train = r_train,
    reduced_assess = r_assess,
    coefs = coefs,
    dropped = dropped
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
