# Small deterministic designs shared by several test files.

# Tiny hand-checkable matrix: 6 rows, 3 columns.
tiny_x <- function() {
  matrix(
    c(
      1, 2, 3, 4, 5, 6, # a
      2, 1, 4, 3, 7, 5, # b
      1, 1, 2, 2, 3, 3  # c
    ),
    nrow = 6, ncol = 3,
    dimnames = list(NULL, c("a", "b", "c"))
  )
}

# Linear-Gaussian design with known coefficients and independent predictors.
linear_design <- function(n = 400, seed = 42) {
  set.seed(seed)
  x <- cbind(
    x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n)
  )
  y <- 3 * x[, "x1"] + 2 * x[, "x2"] + 1 * x[, "x3"] + rnorm(n, sd = 0.5)
  list(x = x, y = y, beta = c(x1 = 3, x2 = 2, x3 = 1, x4 = 0))
}

# OLS learner that does not rely on parsnip; aliased coefficients -> 0.
ols_learner <- function() {
  sharedimp_learner(
    fit = function(x, y) stats::lm.fit(cbind(1, x), y),
    predict = function(model, x) {
      beta <- model$coefficients
      beta[is.na(beta)] <- 0
      drop(cbind(1, x) %*% beta)
    },
    label = "OLS (lm.fit)"
  )
}

# Learner that records every matrix it is given, for leakage tests.
recording_learner <- function(store) {
  sharedimp_learner(
    fit = function(x, y) {
      store$fits[[length(store$fits) + 1L]] <- list(x = x, y = y)
      list(mean = mean(y), ncol = ncol(x))
    },
    predict = function(model, x) {
      store$preds[[length(store$preds) + 1L]] <- x
      rep(model$mean, nrow(x))
    },
    label = "recording learner"
  )
}

# Explicit fold labels so tests do not depend on rsample's RNG.
fixed_folds <- function(n, v) {
  rep(seq_len(v), length.out = n)
}

# Trivial probabilistic classifier for tests that only need the contract:
# predicts training class frequencies (ignores x).
ols_learner_cls <- function() {
  sharedimp_learner(
    fit = function(x, y) {
      lv <- levels(y)
      list(p = as.numeric(table(factor(y, levels = lv))) / length(y), lv = lv)
    },
    predict = function(model, x) {
      matrix(model$p, nrow(x), length(model$lv), byrow = TRUE, dimnames = list(NULL, model$lv))
    },
    label = "frequency classifier"
  )
}
