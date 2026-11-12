#' Shared predictive importance with fold-internal residualization (UMFI-LR)
#'
#' For every anchor feature, `shared_importance()` residualizes the other
#' predictors on the anchor using ordinary least squares coefficients estimated
#' on the training rows of each resampling fold, applies the same coefficients
#' to the assessment rows, fits the learner to the *full* representation
#' (anchor plus residualized others) and to the *reduced* representation
#' (residualized others only), and compares their assessment-set performance.
#' This is the linear-regression variant of ultra-marginal feature importance
#' (UMFI-LR) of Janssen, Guan and Robeva (2023), evaluated out of sample with
#' paired folds.
#'
#' @section Sign convention:
#' Let `s = +1` if the metric is to be maximized and `s = -1` if it is to be
#' minimized. For each fold, `importance = s * (metric_full - metric_reduced)`,
#' so a positive value always means that keeping the anchor predicts better
#' than the residualized others alone. Negative values are reported as they
#' are; they are never truncated to zero. The reported importance is the mean
#' over folds; `sd` is the sample standard deviation of the fold values and is
#' a descriptive quantity only, not a standard error.
#'
#' @section What the numbers are not:
#' The values are out-of-sample predictive gains for a particular learner,
#' metric and resampling scheme. They are not causal effects, not regression
#' coefficients and not conditional effects, and the package deliberately
#' provides no confidence intervals or p-values. See `docs/specification.md`
#' in the source repository for the complete specification, including how the
#' computation differs from the reference UMFI-LR implementation.
#'
#' @param x A numeric matrix or a data frame whose columns are all numeric.
#'   Missing values are not allowed.
#' @param y The response: a numeric vector for regression or a factor for
#'   classification (binary or multiclass; the first level is the event level
#'   for probability metrics). Character vectors are converted to factors.
#' @param learner `"rand_forest"` (default), `"linear"`, a `parsnip` model
#'   specification, or a [sharedimp_learner()]. Hyperparameters are fixed; no
#'   tuning is performed. See [sharedimp_learner()] for details.
#' @param metric A built-in metric name or a `yardstick` metric function. The
#'   defaults are `"mse"` for regression and `"mn_log_loss"` for
#'   classification. See [as_metric()].
#' @param task `"auto"` (default: numeric `y` is regression, factor `y` is
#'   classification), `"regression"`, or `"classification"`.
#' @param v,repeats Number of folds and repeats for `rsample::vfold_cv()`,
#'   used when `resamples` is `NULL`. Classification folds are stratified by
#'   `y`.
#' @param resamples Optional explicit resampling: an `rsample` `rset` built on
#'   data with `nrow(x)` rows, or an integer vector of length `nrow(x)` whose
#'   unique values label the assessment fold of each row. Overrides `v` and
#'   `repeats`.
#' @param features Optional names or indices of the anchor features to
#'   evaluate. Default: all columns of `x`.
#' @param seed Optional integer. When supplied, fold generation and any
#'   stochastic learner run under a local RNG state seeded with `seed`, and the
#'   caller's RNG state is restored afterwards.
#' @param direction Logical. If `TRUE`, also report the direction diagnostic:
#'   the correlation, on each assessment fold, between the anchor's values and
#'   the difference between the full and reduced predictions (probability of
#'   the first level for binary classification; `NA` for multiclass). It is a
#'   descriptive summary only and must not be read as a causal or conditional
#'   effect.
#' @param on_error `"warn"` (default) records a learner failure on a fold as
#'   `NA`, aggregates over the remaining folds and warns; `"error"` stops at
#'   the first failure.
#' @param verbose Print progress messages?
#'
#' @return An object of class `sharedimp_result` (see [sharedimp_result]) with
#'   elements `importance` (tibble: `feature`, `importance`, `sd`, `n_folds`,
#'   and `direction` when requested), `folds` (per-fold paired metrics;
#'   see [fold_results()]), `metric`, `task`, `learner`, `direction`, and
#'   bookkeeping fields `n`, `p`, `v`, `repeats`, `seed`, `levels`,
#'   `resampling`, `call`.
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
#' # Binary classification with log loss
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
                              direction = FALSE,
                              on_error = c("warn", "error"),
                              verbose = FALSE) {
  call <- match.call()
  task <- rlang::arg_match(task)
  on_error <- rlang::arg_match(on_error)
  metric_label <- if (is.null(metric) || is.character(metric)) NULL else rlang::as_label(substitute(metric))
  if (!rlang::is_bool(direction)) {
    abort("`direction` must be `TRUE` or `FALSE`.")
  }
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
      direction, on_error, verbose, seed, v, repeats, call
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

run_shared_importance <- function(x, y, task, levels, lrn, null_lrn, metric, anchors, folds,
                                  direction, on_error, verbose, seed, v, repeats, call) {
  feature_names <- colnames(x)[anchors]
  n_folds <- length(folds)
  n_rows <- length(anchors) * n_folds

  out_feature <- rep(feature_names, each = n_folds)
  out_id <- rep(vapply(folds, `[[`, character(1), "id"), times = length(anchors))
  out_id2 <- rep(vapply(folds, `[[`, character(1), "id2"), times = length(anchors))
  out_full <- rep(NA_real_, n_rows)
  out_reduced <- rep(NA_real_, n_rows)
  out_error <- rep(NA_character_, n_rows)
  out_dir <- rep(NA_real_, n_rows)

  learner_warnings <- character(0)
  constant_anchors <- character(0)

  fit_predict <- function(x_train, x_assess) {
    l <- if (ncol(x_train) == 0L) null_lrn else lrn
    model <- l$fit(x_train, y_train)
    pred <- l$predict(model, x_assess)
    check_prediction(pred, task, levels, nrow(x_assess))
  }

  row <- 0L
  for (a in seq_along(anchors)) {
    i <- anchors[a]
    if (verbose) {
      inform(sprintf("Feature %d/%d: %s", a, length(anchors), feature_names[a]))
    }
    for (k in seq_len(n_folds)) {
      row <- row + 1L
      f <- folds[[k]]
      y_train <- y[f$train]
      y_assess <- y[f$assess]

      rep <- withCallingHandlers(
        build_representations_impl(x[f$train, , drop = FALSE], x[f$assess, , drop = FALSE], i),
        sharedimp_constant_anchor = function(w) {
          constant_anchors <<- union(constant_anchors, feature_names[a])
          invokeRestart("muffleWarning")
        }
      )

      result <- tryCatch(
        withCallingHandlers(
          {
            if (task == "classification" && length(unique(y_train)) < 2L) {
              abort("training fold contains a single class")
            }
            pred_full <- fit_predict(rep$full_train, rep$full_assess)
            pred_reduced <- fit_predict(rep$reduced_train, rep$reduced_assess)
            m_full <- evaluate_metric(metric, y_assess, pred_full, levels)
            m_reduced <- evaluate_metric(metric, y_assess, pred_reduced, levels)
            d <- if (direction) {
              compute_direction(rep$full_assess[, 1L], pred_full, pred_reduced, task, levels)
            } else {
              NA_real_
            }
            list(full = m_full, reduced = m_reduced, direction = d)
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
              sprintf("Learner failed for feature `%s`, fold %s.", feature_names[a], f$id),
              "x" = msg
            ),
            call = call
          )
        }
        out_error[row] <- msg
      } else {
        out_full[row] <- result$full
        out_reduced[row] <- result$reduced
        out_dir[row] <- result$direction
      }
    }
  }

  sign <- if (metric$direction == "maximize") 1 else -1
  out_imp <- sign * (out_full - out_reduced)

  folds_tbl <- tibble::tibble(
    feature = out_feature,
    id = out_id,
    id2 = out_id2,
    metric_full = out_full,
    metric_reduced = out_reduced,
    importance = out_imp,
    error = out_error
  )
  if (direction) {
    folds_tbl$direction <- out_dir
  }

  importance_tbl <- aggregate_by_feature(out_feature, feature_names, out_imp, "importance")
  direction_tbl <- NULL
  if (direction) {
    direction_tbl <- aggregate_by_feature(out_feature, feature_names, out_dir, "direction")
    importance_tbl$direction <- direction_tbl$direction
  }

  n_fail <- sum(!is.na(out_error))
  if (n_fail > 0L) {
    warn(c(
      sprintf("%d of %d (feature, fold) fits failed and were recorded as NA.", n_fail, n_rows),
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
    direction = direction_tbl,
    n = nrow(x),
    p = ncol(x),
    v = if (is.null(attr(folds, "description"))) NA_integer_ else length(folds),
    repeats = repeats,
    seed = seed,
    levels = levels,
    resampling = attr(folds, "description") %||% "unknown",
    call = call
  )
}

aggregate_by_feature <- function(feature, feature_names, value, name) {
  split_vals <- split(value, factor(feature, levels = feature_names))
  mean_v <- vapply(split_vals, function(v) if (all(is.na(v))) NA_real_ else mean(v, na.rm = TRUE), numeric(1))
  sd_v <- vapply(split_vals, function(v) {
    v <- v[!is.na(v)]
    if (length(v) < 2L) NA_real_ else stats::sd(v)
  }, numeric(1))
  n_v <- vapply(split_vals, function(v) sum(!is.na(v)), integer(1))
  out <- tibble::tibble(feature = feature_names, value = unname(mean_v), sd = unname(sd_v), n_folds = unname(n_v))
  names(out)[2L] <- name
  out
}
