#' Results of shared_importance()
#'
#' `shared_importance()` returns an object of class `sharedimp_result`. The
#' headline table is `$importance`; `fold_results()` (or `$folds`) gives the
#' paired per-split scores from which it was aggregated. Both tables use the
#' column names shared with the Python package.
#'
#' `$importance` has one row per anchor feature, in the order of `features`,
#' with columns `feature`, `importance` (mean over splits of
#' `score_full - score_reduced`), `importance_std` (sample standard deviation
#' of the per-split importances, `NA` with one split; descriptive, not a
#' standard error), `score_full`, `score_reduced` (mean scores) and
#' `n_splits` (number of splits with a successful fit).
#'
#' Scores are oriented so that greater is better: the metric value for a
#' metric to maximize and its negative for a metric to minimize (see
#' [as_metric()]).
#'
#' @param x,object A `sharedimp_result`.
#' @param ... Passed on to the tibble print method.
#'
#' @return `fold_results()` returns a tibble with one row per (split, feature),
#'   ordered by split and then by feature, with the common columns `feature`,
#'   `split` (1-based index in generation order), `n_train`, `n_test`,
#'   `score_full`, `score_reduced`, `importance`, followed by the R-specific
#'   columns `id`, `id2` (`rsample` labels) and `error` (`NA` unless the
#'   learner failed on that split).
#'   `as.data.frame()` returns the `importance` table as a plain data frame.
#' @name sharedimp_result
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(300), 100, 3, dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, "a"] + rnorm(100)
#' res <- shared_importance(x, y, learner = "linear", v = 3, seed = 1)
#' res
#' fold_results(res)
NULL

new_sharedimp_result <- function(importance, folds, metric, task, learner,
                                 n, p, v, repeats, seed, levels, resampling, call) {
  structure(
    list(
      importance = importance,
      folds = folds,
      metric = metric,
      task = task,
      learner = learner,
      n = n,
      p = p,
      v = v,
      repeats = repeats,
      seed = seed,
      levels = levels,
      resampling = resampling,
      call = call
    ),
    class = "sharedimp_result"
  )
}

#' @rdname sharedimp_result
#' @export
fold_results <- function(x) {
  if (!inherits(x, "sharedimp_result")) {
    abort("`x` must be a `sharedimp_result`.")
  }
  x$folds
}

#' @rdname sharedimp_result
#' @export
print.sharedimp_result <- function(x, ...) {
  cat("<sharedimp_result>\n")
  cat(sprintf("  task:       %s", x$task))
  if (!is.null(x$levels)) {
    cat(sprintf(" (%d classes)", length(x$levels)))
  }
  cat("\n")
  cat(sprintf("  learner:    %s\n", x$learner))
  cat(sprintf(
    "  metric:     %s (%s); score = %s%s, importance = score_full - score_reduced\n",
    x$metric$name, x$metric$direction,
    if (x$metric$direction == "maximize") "" else "-", x$metric$name
  ))
  cat(sprintf("  resampling: %s\n", x$resampling))
  n_fail <- sum(!is.na(x$folds$error))
  if (n_fail > 0L) {
    cat(sprintf("  failures:   %d of %d (feature, split) fits failed\n", n_fail, nrow(x$folds)))
  }
  cat("\n")
  print(x$importance, ...)
  invisible(x)
}

#' @rdname sharedimp_result
#' @param row.names,optional Ignored; present for compatibility with the
#'   generic.
#' @export
as.data.frame.sharedimp_result <- function(x, row.names = NULL, optional = FALSE, ...) {
  as.data.frame(x$importance, stringsAsFactors = FALSE)
}

#' @rdname sharedimp_result
#' @export
summary.sharedimp_result <- function(object, ...) {
  print(object, ...)
}
