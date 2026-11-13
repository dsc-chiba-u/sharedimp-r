#' Metrics available in shared_importance()
#'
#' `shared_importance()` evaluates predictions with `yardstick` metrics. The
#' `metric` argument accepts a built-in metric name or any `yardstick` metric
#' function (class `numeric_metric`, `class_metric`, or `prob_metric`) whose
#' direction is `"maximize"` or `"minimize"`. The direction determines the
#' orientation of the reported *scores*: the metric value for metrics to
#' maximize and its negative for metrics to minimize, so that
#' `importance = score_full - score_reduced` is positive whenever keeping the
#' anchor feature predicts better.
#'
#' Built-in names (the common names of the R/Python specification, with the
#' `yardstick` function and the matching scikit-learn scorer):
#'
#' | name | task | direction | yardstick | scikit-learn |
#' |---|---|---|---|---|
#' | `"mse"` (default) | regression | minimize | `mse` (this package) | `neg_mean_squared_error` |
#' | `"rmse"` | regression | minimize | `rmse` | `neg_root_mean_squared_error` |
#' | `"mae"` | regression | minimize | `mae` | `neg_mean_absolute_error` |
#' | `"r2"` | regression | maximize | `rsq_trad` (1 - SS_res / SS_tot) | `r2` |
#' | `"rsq"` | regression | maximize | `rsq` (squared Pearson correlation) | none built in |
#' | `"log_loss"` (default) | classification | minimize | `mn_log_loss` | `neg_log_loss` |
#' | `"accuracy"` | classification | maximize | `accuracy` | `accuracy` |
#' | `"roc_auc"` | classification | maximize | `roc_auc` (binary; Hand-Till for multiclass) | `roc_auc` / `roc_auc_ovo` |
#'
#' The `yardstick` names `"rsq_trad"` and `"mn_log_loss"` are accepted as
#' aliases of `"r2"` and `"log_loss"`. `"mse"` is implemented here because
#' `yardstick` does not ship a mean squared error. For binary tasks the first
#' level of `y` is the event level of probability metrics (log loss and AUC do
#' not depend on this choice); for multiclass tasks `"roc_auc"` uses the
#' Hand-Till estimator.
#'
#' @param metric A metric name or a `yardstick` metric function.
#' @param task `"regression"` or `"classification"`.
#' @param label Optional display name for a metric passed as a function.
#'
#' @return `as_metric()` returns an object of class `sharedimp_metric` with
#'   elements `name`, `direction`, `kind` (`"numeric"`, `"class"` or `"prob"`),
#'   and `fn`. `mse` is the `yardstick`-style metric function itself.
#' @examples
#' m <- as_metric("rmse", task = "regression")
#' m$direction
#' m2 <- as_metric(yardstick::mae, task = "regression")
#' m2$name
#' mse(data.frame(t = c(1, 2, 3), e = c(1, 2, 5)), t, e)
#' @export
as_metric <- function(metric, task, label = NULL) {
  task <- rlang::arg_match0(task, c("regression", "classification"))
  if (is.character(metric)) {
    if (length(metric) != 1L) {
      abort("`metric` must be a single metric name or a yardstick metric function.")
    }
    fn <- builtin_metric(metric)
    label <- metric
  } else if (is.function(metric) && inherits(metric, "metric")) {
    fn <- metric
    label <- label %||% "custom metric"
  } else {
    abort("`metric` must be a metric name or a yardstick metric function.")
  }
  kind <- if (inherits(fn, "numeric_metric")) {
    "numeric"
  } else if (inherits(fn, "class_metric")) {
    "class"
  } else if (inherits(fn, "prob_metric")) {
    "prob"
  } else {
    abort("Unsupported yardstick metric class; use a numeric, class or probability metric.")
  }
  if (task == "regression" && kind != "numeric") {
    abort(sprintf("Metric \"%s\" is a classification metric but the task is regression.", label))
  }
  if (task == "classification" && kind == "numeric") {
    abort(sprintf("Metric \"%s\" is a regression metric but the task is classification.", label))
  }
  direction <- attr(fn, "direction")
  if (is.null(direction) || !direction %in% c("maximize", "minimize")) {
    abort(c(
      sprintf("Metric \"%s\" must have direction \"maximize\" or \"minimize\".", label),
      "i" = sprintf("Found: %s.", direction %||% "no direction attribute")
    ))
  }
  structure(
    list(name = label, direction = direction, kind = kind, fn = fn),
    class = "sharedimp_metric"
  )
}

#' @export
print.sharedimp_metric <- function(x, ...) {
  cat("<sharedimp_metric> ", x$name, " (", x$kind, ", ", x$direction, ")\n", sep = "")
  invisible(x)
}

builtin_metric <- function(name) {
  switch(name,
    mse = mse,
    rmse = yardstick::rmse,
    mae = yardstick::mae,
    r2 = yardstick::rsq_trad,
    rsq_trad = yardstick::rsq_trad,
    rsq = yardstick::rsq,
    log_loss = yardstick::mn_log_loss,
    mn_log_loss = yardstick::mn_log_loss,
    accuracy = yardstick::accuracy,
    roc_auc = yardstick::roc_auc,
    abort(c(
      sprintf("Unknown metric \"%s\".", name),
      "i" = "Built-in metrics: mse, rmse, mae, r2, rsq, log_loss, accuracy, roc_auc."
    ))
  )
}

default_metric <- function(task) {
  if (task == "regression") "mse" else "log_loss"
}

mse_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
  if (na_rm) {
    ok <- !(is.na(truth) | is.na(estimate))
    truth <- truth[ok]
    estimate <- estimate[ok]
  } else if (anyNA(truth) || anyNA(estimate)) {
    return(NA_real_)
  }
  mean((truth - estimate)^2)
}

#' @rdname as_metric
#' @param data,truth,estimate,na_rm,case_weights,... Standard `yardstick`
#'   metric arguments.
#' @export
mse <- function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
  yardstick::numeric_metric_summarizer(
    name = "mse",
    fn = mse_vec,
    data = data,
    truth = !!rlang::enquo(truth),
    estimate = !!rlang::enquo(estimate),
    na_rm = na_rm,
    case_weights = !!rlang::enquo(case_weights)
  )
}
mse <- yardstick::new_numeric_metric(mse, direction = "minimize")

# Evaluate a sharedimp_metric on assessment truth and a prediction object.
evaluate_metric <- function(metric, truth, pred, levels) {
  if (metric$kind == "numeric") {
    df <- data.frame(truth = truth, estimate = pred)
    res <- rlang::inject(metric$fn(df, !!rlang::sym("truth"), !!rlang::sym("estimate")))
  } else if (metric$kind == "class") {
    cls <- factor(levels[max.col(pred, ties.method = "first")], levels = levels)
    df <- data.frame(truth = factor(truth, levels = levels), estimate = cls)
    res <- rlang::inject(metric$fn(df, !!rlang::sym("truth"), !!rlang::sym("estimate")))
  } else {
    cols <- paste0(".pred_", levels)
    df <- data.frame(truth = factor(truth, levels = levels), pred, check.names = FALSE)
    names(df) <- c("truth", cols)
    if (length(levels) == 2L) {
      res <- rlang::inject(metric$fn(df, !!rlang::sym("truth"), !!rlang::sym(cols[1L])))
    } else {
      res <- rlang::inject(metric$fn(df, !!rlang::sym("truth"), !!!rlang::syms(cols)))
    }
  }
  val <- res$.estimate
  if (length(val) != 1L) {
    abort("metric did not return a single value")
  }
  as.numeric(val)
}
