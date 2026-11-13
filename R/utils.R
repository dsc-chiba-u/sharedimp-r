# Internal helpers: input validation, constant detection, local RNG scope.

#' Tolerant constant-vector test
#'
#' Used only by the experimental, non-exported direction diagnostic. The
#' algorithm itself uses the exact rules of `docs/specification.md`,
#' Section 5.1 (zero sum of squares for the anchor; relative residual sum of
#' squares for collinear columns), implemented in `residualize_fit_impl()`.
#'
#' @param v A numeric vector without missing values.
#' @return A single logical.
#' @keywords internal
#' @noRd
is_constant <- function(v) {
  if (length(v) <= 1L) {
    return(TRUE)
  }
  r <- range(v)
  (r[2L] - r[1L]) <= 1e-10 * max(1, abs(r[1L]), abs(r[2L]))
}

#' Validate and coerce predictors to a numeric matrix with column names
#' @noRd
check_x <- function(x, call = rlang::caller_env()) {
  if (is.data.frame(x)) {
    bad <- !vapply(x, function(col) is.numeric(col) || is.logical(col), logical(1))
    if (any(bad)) {
      abort(
        c(
          "All columns of `x` must be numeric.",
          "x" = sprintf(
            "Non-numeric column(s): %s.",
            paste(sprintf("`%s`", names(x)[bad]), collapse = ", ")
          ),
          "i" = "Encode categorical predictors numerically before calling `shared_importance()`."
        ),
        call = call
      )
    }
    if (any(vapply(x, is.logical, logical(1)))) {
      inform("Logical columns of `x` were coerced to 0/1.")
    }
    nm <- names(x)
    x <- as.matrix(data.frame(lapply(x, as.numeric), check.names = FALSE))
    colnames(x) <- nm
  } else if (is.matrix(x)) {
    if (is.logical(x)) {
      inform("Logical `x` was coerced to 0/1.")
      x <- x + 0
    }
    if (!is.numeric(x)) {
      abort("`x` must be a numeric matrix or a data frame of numeric columns.", call = call)
    }
    if (is.null(colnames(x))) {
      colnames(x) <- paste0("V", seq_len(ncol(x)))
    }
    storage.mode(x) <- "double"
  } else {
    abort("`x` must be a numeric matrix or a data frame of numeric columns.", call = call)
  }
  if (ncol(x) < 1L) {
    abort("`x` must have at least one column.", call = call)
  }
  if (nrow(x) < 2L) {
    abort("`x` must have at least two rows.", call = call)
  }
  if (anyNA(x) || any(!is.finite(x))) {
    abort(
      c(
        "`x` must not contain missing or non-finite values.",
        "i" = "Impute or drop incomplete rows before calling `shared_importance()`."
      ),
      call = call
    )
  }
  if (anyDuplicated(colnames(x))) {
    abort("Column names of `x` must be unique.", call = call)
  }
  x
}

#' Validate the response and resolve the task
#' @return list(y = <numeric or factor>, task = chr, levels = chr or NULL)
#' @noRd
check_y <- function(y, task, n, call = rlang::caller_env()) {
  if (is.data.frame(y) || is.matrix(y)) {
    if (NCOL(y) != 1L) {
      abort("`y` must be a vector (or a single-column data frame).", call = call)
    }
    y <- y[[1L]]
  }
  if (length(y) != n) {
    abort(sprintf("`y` has length %d but `x` has %d rows.", length(y), n), call = call)
  }
  if (anyNA(y)) {
    abort("`y` must not contain missing values.", call = call)
  }
  if (is.character(y) || is.logical(y)) {
    y <- factor(y)
  }
  if (task == "auto") {
    task <- if (is.factor(y)) "classification" else "regression"
  }
  if (task == "regression") {
    if (!is.numeric(y)) {
      abort("For `task = \"regression\"`, `y` must be numeric.", call = call)
    }
    if (any(!is.finite(y))) {
      abort("`y` must not contain non-finite values.", call = call)
    }
    y <- as.numeric(y)
    levels <- NULL
  } else {
    if (!is.factor(y)) {
      abort(
        c(
          "For `task = \"classification\"`, `y` must be a factor.",
          "i" = "Convert numeric class labels with `factor()` so the level order is explicit."
        ),
        call = call
      )
    }
    y <- droplevels(y)
    levels <- levels(y)
    if (length(levels) < 2L) {
      abort("`y` must have at least two classes.", call = call)
    }
  }
  list(y = y, task = task, levels = levels)
}

#' Evaluate code under a local RNG state, restoring the caller's state
#' @noRd
with_local_seed <- function(seed, code) {
  if (is.null(seed)) {
    return(force(code))
  }
  if (!is.numeric(seed) || length(seed) != 1L || is.na(seed)) {
    abort("`seed` must be a single integer or `NULL`.")
  }
  genv <- globalenv()
  had_seed <- exists(".Random.seed", envir = genv, inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", envir = genv, inherits = FALSE) else NULL
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = genv)
    } else if (exists(".Random.seed", envir = genv, inherits = FALSE)) {
      rm(".Random.seed", envir = genv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed))
  force(code)
}

#' Resolve an anchor given as name or index to an integer index
#' @noRd
resolve_anchor <- function(anchor, x, call = rlang::caller_env()) {
  if (length(anchor) != 1L) {
    abort("`anchor` must be a single column name or index.", call = call)
  }
  if (is.character(anchor)) {
    idx <- match(anchor, colnames(x))
    if (is.na(idx)) {
      abort(sprintf("`anchor` \"%s\" is not a column of `x`.", anchor), call = call)
    }
    return(idx)
  }
  if (is.numeric(anchor) && anchor == as.integer(anchor) && anchor >= 1 && anchor <= ncol(x)) {
    return(as.integer(anchor))
  }
  abort("`anchor` must be a column name or an integer between 1 and `ncol(x)`.", call = call)
}
