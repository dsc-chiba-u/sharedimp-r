#' Learner adapters for shared_importance()
#'
#' A learner is a pair of functions, `fit(x, y)` and `predict(model, x)`,
#' where `x` is a numeric matrix with column names (a representation built by
#' [build_representations()]). For regression, `predict()` must return a
#' numeric vector with one value per row of `x`. For classification it must
#' return a numeric matrix of class probabilities with one column per level of
#' `y`, in level order (column names are optional). Predicted class labels are
#' always derived by [shared_importance()] as the argmax of that matrix.
#'
#' `shared_importance()` accepts three kinds of `learner`:
#'
#' * a string: `"rand_forest"` (default; `ranger` with 500 trees and
#'   `min_n = 5`, which is ranger's `min.node.size`: nodes with more than 5
#'   rows are split, and terminal nodes may hold a single row; `mtry =
#'   floor(sqrt(ncol(x)))`, the ranger default) or `"linear"` (ordinary least
#'   squares via `stats::lm.fit()`, unpenalized logistic regression via
#'   `stats::glm.fit()`, or multinomial logistic regression via
#'   `nnet::multinom()`, depending on the task; aliased coefficients are set
#'   to zero);
#' * a `parsnip` model specification whose mode matches the task, used as is
#'   (its hyperparameters are fixed; nothing is tuned);
#' * an object created by `sharedimp_learner()`.
#'
#' @param fit A function `function(x, y)` returning a fitted model object.
#' @param predict A function `function(model, x)` returning predictions in the
#'   format described above.
#' @param label A short description used in printing.
#'
#' @return An object of class `sharedimp_learner`.
#' @examples
#' ols <- sharedimp_learner(
#'   fit = function(x, y) stats::lm.fit(cbind(1, x), y),
#'   predict = function(model, x) {
#'     beta <- model$coefficients
#'     beta[is.na(beta)] <- 0
#'     drop(cbind(1, x) %*% beta)
#'   },
#'   label = "OLS via lm.fit"
#' )
#' set.seed(1)
#' x <- matrix(rnorm(200), 50, 4, dimnames = list(NULL, paste0("x", 1:4)))
#' y <- x[, 1] + rnorm(50)
#' shared_importance(x, y, learner = ols, v = 3, seed = 1)
#' @export
sharedimp_learner <- function(fit, predict, label = "custom learner") {
  if (!is.function(fit) || !is.function(predict)) {
    abort("`fit` and `predict` must be functions.")
  }
  if (!is.character(label) || length(label) != 1L) {
    abort("`label` must be a single string.")
  }
  structure(
    list(fit = fit, predict = predict, label = label),
    class = "sharedimp_learner"
  )
}

#' @export
print.sharedimp_learner <- function(x, ...) {
  cat("<sharedimp_learner> ", x$label, "\n", sep = "")
  invisible(x)
}

# Resolve the `learner` argument of shared_importance() to a sharedimp_learner.
as_learner <- function(learner, task, levels, call = rlang::caller_env()) {
  if (inherits(learner, "sharedimp_learner")) {
    return(learner)
  }
  if (inherits(learner, "model_spec")) {
    return(parsnip_learner(learner, task, levels, call = call))
  }
  if (is.character(learner) && length(learner) == 1L) {
    spec <- switch(learner,
      rand_forest = builtin_rand_forest(task),
      linear = builtin_linear(task, levels),
      abort(
        c(
          sprintf("Unknown learner \"%s\".", learner),
          "i" = "Use \"rand_forest\", \"linear\", a parsnip model specification, or `sharedimp_learner()`."
        ),
        call = call
      )
    )
    if (inherits(spec, "sharedimp_learner")) {
      return(spec)
    }
    return(parsnip_learner(spec, task, levels, label = learner, call = call))
  }
  abort(
    "`learner` must be a string, a parsnip model specification, or a `sharedimp_learner()`.",
    call = call
  )
}

builtin_rand_forest <- function(task) {
  parsnip::set_engine(
    parsnip::rand_forest(mode = task, trees = 500, min_n = 5),
    "ranger"
  )
}

builtin_linear <- function(task, levels) {
  if (task == "regression") {
    return(ols_learner_builtin())
  }
  if (length(levels) == 2L) {
    return(logistic_learner_builtin(levels))
  }
  if (!requireNamespace("nnet", quietly = TRUE)) {
    abort("The \"linear\" learner for multiclass tasks needs the 'nnet' package.")
  }
  multinom_learner(levels)
}

# Coefficients of a rank-deficient fit are NA in R; the specification sets
# aliased coefficients to zero so that predictions stay finite. A residual
# column fixed at zero (exact collinearity rule) is the common case.
coef_or_zero <- function(beta) {
  beta[is.na(beta)] <- 0
  beta
}

# Ordinary least squares through stats::lm.fit() (same estimator as lm(),
# without the formula interface and without predict.lm()'s rank-deficiency
# warning, which the exact-collinearity rule would trigger routinely).
ols_learner_builtin <- function() {
  sharedimp_learner(
    fit = function(x, y) coef_or_zero(stats::lm.fit(cbind(1, x), y)$coefficients),
    predict = function(model, x) drop(cbind(1, x) %*% model),
    label = "linear"
  )
}

# Unpenalized logistic regression through stats::glm.fit(); the first level
# of `y` is the event, so the first probability column is P(first level).
logistic_learner_builtin <- function(levels) {
  sharedimp_learner(
    fit = function(x, y) {
      event <- as.numeric(y == levels[1L])
      fit <- stats::glm.fit(cbind(1, x), event, family = stats::binomial())
      coef_or_zero(fit$coefficients)
    },
    predict = function(model, x) {
      p1 <- stats::plogis(drop(cbind(1, x) %*% model))
      matrix(c(p1, 1 - p1), ncol = 2L, dimnames = list(NULL, levels))
    },
    label = "linear"
  )
}

# Multinomial logistic regression through nnet::multinom() called directly.
# parsnip's multinom_reg()/nnet pairing mis-formats probabilities when the
# training fold contains only two of the levels, so the built-in learner
# handles the two-class case itself.
multinom_learner <- function(levels) {
  sharedimp_learner(
    fit = function(x, y) {
      df <- as.data.frame(x, check.names = FALSE)
      df$.y <- droplevels(y)
      nnet::multinom(.y ~ ., data = df, maxit = 500, trace = FALSE)
    },
    predict = function(model, x) {
      df <- as.data.frame(x, check.names = FALSE)
      p <- stats::predict(model, newdata = df, type = "probs")
      lev <- model$lev
      if (length(lev) == 2L) {
        p <- cbind(1 - p, p)
      } else if (!is.matrix(p)) {
        p <- matrix(p, nrow = 1L)
      }
      colnames(p) <- lev
      align_prob_columns(p, levels)
    },
    label = "linear"
  )
}

# Adapter from a parsnip model_spec to the sharedimp learner contract.
parsnip_learner <- function(spec, task, levels, label = NULL, call = rlang::caller_env()) {
  if (identical(spec$mode, "unknown")) {
    spec <- parsnip::set_mode(spec, task)
  }
  if (!identical(spec$mode, task)) {
    abort(
      sprintf("The parsnip specification has mode \"%s\" but the task is \"%s\".", spec$mode, task),
      call = call
    )
  }
  engine <- spec$engine %||% "default engine"
  label <- label %||% sprintf("%s (%s)", class(spec)[1L], engine)
  if (is.null(spec$engine)) {
    # Make the engine explicit so the label and behaviour are stable.
    spec <- parsnip::set_engine(spec, parsnip::show_engines(class(spec)[1L])$engine[1L])
  }
  fit_fn <- function(x, y) {
    if (is.factor(y)) {
      # Levels absent from a training fold would make some engines (e.g.
      # nnet::multinom) return malformed probabilities; drop them here and
      # restore the full set of columns in predict_fn.
      y <- droplevels(y)
    }
    parsnip::fit_xy(spec, x = as.data.frame(x, check.names = FALSE), y = y)
  }
  predict_fn <- if (task == "regression") {
    function(model, x) {
      stats::predict(model, new_data = as.data.frame(x, check.names = FALSE), type = "numeric")$.pred
    }
  } else {
    function(model, x) {
      p <- stats::predict(model, new_data = as.data.frame(x, check.names = FALSE), type = "prob")
      p <- as.matrix(p)
      colnames(p) <- sub("^\\.pred_", "", colnames(p))
      align_prob_columns(p, levels)
    }
  }
  sharedimp_learner(fit_fn, predict_fn, label = label)
}

# Ensure a probability matrix has one column per level, in level order,
# filling levels absent from the training fold with zero probability.
align_prob_columns <- function(p, levels) {
  out <- matrix(0, nrow = nrow(p), ncol = length(levels), dimnames = list(NULL, levels))
  common <- intersect(levels, colnames(p))
  out[, common] <- p[, common, drop = FALSE]
  out
}

# Null model used when a representation has zero columns.
null_learner <- function(task, levels) {
  if (task == "regression") {
    sharedimp_learner(
      fit = function(x, y) mean(y),
      predict = function(model, x) rep(model, nrow(x)),
      label = "null model (training mean)"
    )
  } else {
    sharedimp_learner(
      fit = function(x, y) {
        tab <- table(factor(y, levels = levels))
        as.numeric(tab) / sum(tab)
      },
      predict = function(model, x) {
        matrix(model, nrow = nrow(x), ncol = length(levels), byrow = TRUE, dimnames = list(NULL, levels))
      },
      label = "null model (training class frequencies)"
    )
  }
}

# Validate a prediction against the learner contract; abort on violation so the
# caller can record a learner failure.
check_prediction <- function(pred, task, levels, n) {
  if (task == "regression") {
    if (!is.numeric(pred) || length(pred) != n) {
      abort(sprintf("learner returned %d prediction(s) for %d assessment rows", length(pred), n))
    }
    if (any(!is.finite(pred))) {
      abort("learner returned non-finite predictions")
    }
    return(as.numeric(pred))
  }
  if (is.data.frame(pred)) {
    pred <- as.matrix(pred)
  }
  if (!is.matrix(pred) || nrow(pred) != n || ncol(pred) != length(levels)) {
    abort(sprintf(
      "learner returned a %s probability object for %d rows and %d classes",
      paste(dim(pred), collapse = "x"), n, length(levels)
    ))
  }
  if (!is.numeric(pred) || any(!is.finite(pred))) {
    abort("learner returned non-finite class probabilities")
  }
  if (any(pred < -1e-8) || any(pred > 1 + 1e-8)) {
    abort("learner returned class probabilities outside [0, 1]")
  }
  pred <- pmin(pmax(pred, 0), 1)
  colnames(pred) <- levels
  pred
}
