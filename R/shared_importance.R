#' Shared predictive importance with fold-internal residualization (UMFI-LR)
#'
#' For every anchor feature, `shared_importance()` residualizes the other
#' predictors on the anchor using ordinary least squares coefficients estimated
#' on the training rows of each resampling split, applies the same coefficients
#' to the assessment rows, fits the learner to the *full* representation
#' (anchor plus residualized others) and to the *reduced* representation
#' (residualized others only), and compares their assessment-set performance.
#' This is the linear-regression variant of ultra-marginal feature importance
#' (UMFI-LR) of Janssen, Guan and Robeva (2023), evaluated out of sample with
#' paired splits. The R and Python packages follow one common specification
#' (`docs/specification.md` in the source repository).
#'
#' @section Scores and sign convention:
#' Every metric is reported as a *score* oriented so that greater is better:
#' the metric value itself for a metric to maximize (`"r2"`, `"accuracy"`,
#' `"roc_auc"`, ...), and its negative for a metric to minimize (`"mse"`,
#' `"rmse"`, `"mae"`, `"log_loss"`, ...). For each split,
#' `importance = score_full - score_reduced`, so a positive value always means
#' that keeping the anchor predicts better than the residualized others alone.
#' Negative values are reported as they are; they are never truncated to zero.
#' The reported importance is the mean over splits; `importance_std` is the
#' sample standard deviation of the split values and is a descriptive quantity
#' only, not a standard error.
#'
#' @section What the numbers are not:
#' The values are out-of-sample predictive gains for a particular learner,
#' metric and resampling scheme. They are not causal effects, not regression
#' coefficients and not conditional effects, and the package deliberately
#' provides no confidence intervals or p-values.
#'
#' @param x A numeric matrix or a data frame whose columns are all numeric.
#'   Missing values are not allowed. A single column is allowed; the reduced
#'   model is then the null model (training mean or class frequencies).
#' @param y The response: a numeric vector for regression or a factor for
#'   classification (binary or multiclass; the first level is the event level
#'   for probability metrics). Character vectors are converted to factors.
#' @param learner `"rand_forest"` (default), `"linear"`, a `parsnip` model
#'   specification, or a [sharedimp_learner()]. Hyperparameters are fixed; no
#'   tuning is performed. See [sharedimp_learner()] for details.
#' @param metric A built-in metric name or a `yardstick` metric function. The
#'   defaults are `"mse"` for regression and `"log_loss"` for classification.
#'   See [as_metric()].
#' @param task `"auto"` (default: numeric `y` is regression, factor `y` is
#'   classification), `"regression"`, or `"classification"`.
#' @param v,repeats Number of folds and repeats for `rsample::vfold_cv()`,
#'   used when `resamples` is `NULL`. Classification folds are stratified by
#'   `y`.
#' @param resamples Optional explicit resampling: an `rsample` `rset` built on
#'   data with `nrow(x)` rows, or an integer vector of length `nrow(x)` whose
#'   unique values label the assessment fold of each row. Overrides `v` and
#'   `repeats`. Explicit splits are the mechanism for comparisons with the
#'   Python implementation.
#' @param features Optional names or indices of the anchor features to
#'   evaluate. Default: all columns of `x`.
#' @param seed Optional integer. When supplied, fold generation and any
#'   stochastic learner run under a local RNG state seeded with `seed`, and the
#'   caller's RNG state is restored afterwards.
#' @param on_error `"warn"` (default) records a learner failure on a split as
#'   `NA`, aggregates over the remaining splits and warns; `"error"` stops at
#'   the first failure.
#' @param verbose Print progress messages?
#'
#' @return An object of class `sharedimp_result` (see [sharedimp_result]) with
#'   elements `importance` (tibble with the common columns `feature`,
#'   `importance`, `importance_std`, `score_full`, `score_reduced`,
#'   `n_splits`), `folds` (per-split paired scores; see [fold_results()]),
#'   `metric`, `task`, `learner`, and bookkeeping fields `n`, `p`, `v`,
#'   `repeats`, `seed`, `levels`, `resampling`, `call`.
#'
#' @references
#' Janssen, J., Guan, V. and Robeva, E. (2023). Ultra-marginal Feature
#' Importance: Learning from Data with Causal Guarantees. *AISTATS 2023*,
#' PMLR 206:10782-10814. <https://proceedings.mlr.press/v206/janssen23a.html>
#'
#' @examples
#' set.seed(1)
#' n <- 150
#' x <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
#' x$x4 <- x$x1 + rnorm(n, sd = 0.1) # near duplicate of x1
#' y <- 2 * x$x1 + x$x2 + rnorm(n)
#'
#' # Fast, deterministic linear learner with the MSE metric
#' res <- shared_importance(x, y, learner = "linear", v = 5, seed = 1)
#' res
#' fold_results(res)
#'
#' # Binary classification with accuracy
#' yb <- factor(ifelse(y > median(y), "high", "low"))
#' shared_importance(x, yb, learner = "linear", metric = "accuracy", v = 5, seed = 1)
#'
#' \donttest{
#' # Default random forest learner (ranger, 500 trees)
#' shared_importance(x, y, v = 5, seed = 1)
#' }
#' @export
shared_importance <- function(x, y,
                              learner = "rand_forest",
                              metric = NULL,
                              task = c("auto", "regression", "classification"),
                              v = 5,
                              repeats = 1,
                              resamples = NULL,
                              features = NULL,
                              seed = NULL,
                              on_error = c("warn", "error"),
                              verbose = FALSE) {
  call <- match.call()
  task <- rlang::arg_match(task)
  on_error <- rlang::arg_match(on_error)
  metric_label <- if (is.null(metric) || is.character(metric)) NULL else rlang::as_label(substitute(metric))
  if (!rlang::is_bool(verbose)) {
    abort("`verbose` must be `TRUE` or `FALSE`.")
  }

  x <- check_x(x)
  n <- nrow(x)
  p <- ncol(x)
  yinfo <- check_y(y, task, n)
  y <- yinfo$y
  task <- yinfo$task
  levels <- yinfo$levels

  metric <- as_metric(metric %||% default_metric(task), task, label = metric_label)
  lrn <- as_learner(learner, task, levels)
  null_lrn <- null_learner(task, levels)
  anchors <- resolve_features(features, x)

  with_local_seed(seed, {
    folds <- make_folds(resamples, v, repeats, y, task, n)
    run_shared_importance(
      x, y, task, levels, lrn, null_lrn, metric, anchors, folds,
      on_error, verbose, seed, v, repeats, call
    )
  })
}

resolve_features <- function(features, x) {
  if (is.null(features)) {
    return(seq_len(ncol(x)))
  }
  if (is.character(features)) {
    idx <- match(features, colnames(x))
    if (anyNA(idx)) {
      abort(sprintf(
        "Unknown feature(s) in `features`: %s.",
        paste(sprintf("`%s`", features[is.na(idx)]), collapse = ", ")
      ))
    }
  } else if (is.numeric(features)) {
    idx <- as.integer(features)
    if (any(idx < 1L | idx > ncol(x)) || any(idx != features)) {
      abort("`features` indices must be integers between 1 and `ncol(x)`.")
    }
  } else {
    abort("`features` must be `NULL`, column names, or column indices.")
  }
  if (anyDuplicated(idx)) {
    abort("`features` must not contain duplicates.")
  }
  idx
}

# Build a list of folds: each element has train, assess, id, id2.
make_folds <- function(resamples, v, repeats, y, task, n, call = rlang::caller_env()) {
  if (is.null(resamples)) {
    if (!is.numeric(v) || length(v) != 1L || v < 2 || v != as.integer(v)) {
      abort("`v` must be a single integer >= 2.", call = call)
    }
    if (!is.numeric(repeats) || length(repeats) != 1L || repeats < 1 || repeats != as.integer(repeats)) {
      abort("`repeats` must be a single integer >= 1.", call = call)
    }
    if (v > n) {
      abort(sprintf("`v` (%d) cannot exceed the number of rows (%d).", v, n), call = call)
    }
    df <- data.frame(.row = seq_len(n))
    if (task == "classification") {
      df$.y <- y
      rs <- rsample::vfold_cv(df, v = v, repeats = repeats, strata = ".y")
    } else {
      rs <- rsample::vfold_cv(df, v = v, repeats = repeats)
    }
    description <- sprintf(
      "%d-fold cross-validation (%d repeat%s)%s",
      v, repeats, if (repeats > 1) "s" else "",
      if (task == "classification") ", stratified by y" else ""
    )
    return(folds_from_rset(rs, n, description, call))
  }
  if (inherits(resamples, "rset")) {
    description <- sprintf("user-supplied %s (%d resamples)", class(resamples)[1L], nrow(resamples))
    return(folds_from_rset(resamples, n, description, call))
  }
  if (is.numeric(resamples) || is.factor(resamples) || is.character(resamples)) {
    if (length(resamples) != n) {
      abort(sprintf("Fold labels in `resamples` must have length %d.", n), call = call)
    }
    if (anyNA(resamples)) {
      abort("Fold labels in `resamples` must not contain missing values.", call = call)
    }
    labs <- unique(resamples)
    if (is.numeric(labs)) {
      labs <- sort(labs)
    }
    if (length(labs) < 2L) {
      abort("Fold labels in `resamples` must define at least two folds.", call = call)
    }
    folds <- lapply(labs, function(l) {
      assess <- which(resamples == l)
      list(train = setdiff(seq_len(n), assess), assess = assess, id = paste0("Fold", l), id2 = NA_character_)
    })
    check_folds(folds, call)
    return(structure(folds, description = sprintf("user-supplied fold labels (%d folds)", length(labs))))
  }
  abort("`resamples` must be `NULL`, an rsample `rset`, or a vector of fold labels.", call = call)
}

folds_from_rset <- function(rs, n, description, call) {
  n_data <- nrow(rs$splits[[1L]]$data)
  if (n_data != n) {
    abort(sprintf("`resamples` was built on %d rows but `x` has %d rows.", n_data, n), call = call)
  }
  has_id2 <- "id2" %in% names(rs)
  folds <- lapply(seq_len(nrow(rs)), function(k) {
    s <- rs$splits[[k]]
    list(
      train = as.integer(s$in_id),
      assess = as.integer(rsample::complement(s)),
      id = as.character(rs$id[k]),
      id2 = if (has_id2) as.character(rs$id2[k]) else NA_character_
    )
  })
  check_folds(folds, call)
  structure(folds, description = description)
}

check_folds <- function(folds, call) {
  for (f in folds) {
    if (length(f$assess) < 1L) {
      abort("Every assessment fold must contain at least one observation.", call = call)
    }
    if (length(f$train) < 2L) {
      abort("Every training fold must contain at least two observations.", call = call)
    }
    if (length(intersect(f$train, f$assess)) > 0L) {
      abort("Training and assessment rows of a fold must not overlap.", call = call)
    }
  }
  invisible(TRUE)
}

# Main loop: split-major, anchor-minor, matching the row order of the common
# per-split table (specification Section 11.2).
run_shared_importance <- function(x, y, task, levels, lrn, null_lrn, metric, anchors, folds,
                                  on_error, verbose, seed, v, repeats, call) {
  feature_names <- colnames(x)[anchors]
  n_anchors <- length(anchors)
  n_folds <- length(folds)
  n_rows <- n_anchors * n_folds

  out_feature <- rep(feature_names, times = n_folds)
  out_split <- rep(seq_len(n_folds), each = n_anchors)
  out_n_train <- rep(vapply(folds, function(f) length(f$train), integer(1)), each = n_anchors)
  out_n_test <- rep(vapply(folds, function(f) length(f$assess), integer(1)), each = n_anchors)
  out_id <- rep(vapply(folds, `[[`, character(1), "id"), each = n_anchors)
  out_id2 <- rep(vapply(folds, `[[`, character(1), "id2"), each = n_anchors)
  out_full <- rep(NA_real_, n_rows)
  out_reduced <- rep(NA_real_, n_rows)
  out_error <- rep(NA_character_, n_rows)

  learner_warnings <- character(0)
  constant_anchors <- character(0)
  # Scores are oriented so that greater is better (specification Section 9).
  sign <- if (metric$direction == "maximize") 1 else -1

  fit_predict <- function(x_train, y_train, x_assess) {
    l <- if (ncol(x_train) == 0L) null_lrn else lrn
    model <- l$fit(x_train, y_train)
    pred <- l$predict(model, x_assess)
    check_prediction(pred, task, levels, nrow(x_assess))
  }

  row <- 0L
  for (k in seq_len(n_folds)) {
    f <- folds[[k]]
    if (verbose) {
      inform(sprintf("Split %d/%d (%s)", k, n_folds, f$id))
    }
    x_train <- x[f$train, , drop = FALSE]
    x_assess <- x[f$assess, , drop = FALSE]
    y_train <- y[f$train]
    y_assess <- y[f$assess]
    single_class <- task == "classification" && length(unique(y_train)) < 2L

    for (a in seq_len(n_anchors)) {
      row <- row + 1L
      i <- anchors[a]

      rep <- withCallingHandlers(
        build_representations_impl(x_train, x_assess, i),
        sharedimp_constant_anchor = function(w) {
          constant_anchors <<- union(constant_anchors, feature_names[a])
          invokeRestart("muffleWarning")
        }
      )

      result <- tryCatch(
        withCallingHandlers(
          {
            if (single_class) {
              abort("training fold contains a single class")
            }
            pred_full <- fit_predict(rep$full_train, y_train, rep$full_assess)
            pred_reduced <- fit_predict(rep$reduced_train, y_train, rep$reduced_assess)
            list(
              full = sign * evaluate_metric(metric, y_assess, pred_full, levels),
              reduced = sign * evaluate_metric(metric, y_assess, pred_reduced, levels)
            )
          },
          warning = function(w) {
            learner_warnings <<- c(learner_warnings, conditionMessage(w))
            invokeRestart("muffleWarning")
          }
        ),
        error = function(e) e
      )

      if (inherits(result, "error")) {
        msg <- conditionMessage(result)
        if (on_error == "error") {
          abort(
            c(
              sprintf("Learner failed for feature `%s`, split %d (%s).", feature_names[a], k, f$id),
              "x" = msg
            ),
            call = call
          )
        }
        out_error[row] <- msg
      } else {
        out_full[row] <- result$full
        out_reduced[row] <- result$reduced
      }
    }
  }

  folds_tbl <- tibble::tibble(
    feature = out_feature,
    split = out_split,
    n_train = out_n_train,
    n_test = out_n_test,
    score_full = out_full,
    score_reduced = out_reduced,
    importance = out_full - out_reduced,
    id = out_id,
    id2 = out_id2,
    error = out_error
  )
  importance_tbl <- aggregate_by_feature(folds_tbl, feature_names)

  n_fail <- sum(!is.na(out_error))
  if (n_fail > 0L) {
    warn(c(
      sprintf("%d of %d (feature, split) fits failed and were recorded as NA.", n_fail, n_rows),
      "i" = "See `fold_results()` column `error` for details."
    ))
  }
  if (length(constant_anchors) > 0L) {
    warn(sprintf(
      "Anchor feature(s) constant on at least one training fold (slopes set to zero): %s.",
      paste(sprintf("`%s`", constant_anchors), collapse = ", ")
    ))
  }
  if (length(learner_warnings) > 0L) {
    tab <- table(learner_warnings)
    warn(c(
      sprintf("The learner emitted %d warning(s) during fitting or prediction:", length(learner_warnings)),
      setNames(sprintf("[%d times] %s", as.integer(tab), names(tab)), rep("*", length(tab)))
    ))
  }

  new_sharedimp_result(
    importance = importance_tbl,
    folds = folds_tbl,
    metric = list(name = metric$name, direction = metric$direction),
    task = task,
    learner = lrn$label,
    n = nrow(x),
    p = ncol(x),
    v = length(folds),
    repeats = repeats,
    seed = seed,
    levels = levels,
    resampling = attr(folds, "description") %||% "unknown",
    call = call
  )
}

# One row per anchor, in `feature_names` order, aggregated over the splits
# whose fit succeeded (specification Section 9.1).
aggregate_by_feature <- function(folds_tbl, feature_names) {
  grp <- factor(folds_tbl$feature, levels = feature_names)
  ok <- !is.na(folds_tbl$importance)
  mean_ok <- function(value) {
    vapply(split(value[ok], grp[ok]), function(v) if (length(v) == 0L) NA_real_ else mean(v), numeric(1))
  }
  imp <- folds_tbl$importance
  std <- vapply(split(imp[ok], grp[ok]), function(v) if (length(v) < 2L) NA_real_ else stats::sd(v), numeric(1))
  n_ok <- vapply(split(ok, grp), sum, integer(1))
  tibble::tibble(
    feature = feature_names,
    importance = unname(mean_ok(imp)),
    importance_std = unname(std),
    score_full = unname(mean_ok(folds_tbl$score_full)),
    score_reduced = unname(mean_ok(folds_tbl$score_reduced)),
    n_splits = unname(n_ok)
  )
}
