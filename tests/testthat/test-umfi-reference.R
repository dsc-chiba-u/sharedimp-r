# Comparison with the reference UMFI-LR implementation (HydroML/UMFI).
#
# The reference functions are *re-implemented* here, following
# R/preprocess_lr.R and R/umfi.R of https://github.com/HydroML/UMFI line by
# line (inspected 2026-11-12), so that the package does not vendor third-party
# code. A run against the actual upstream files is recorded in
# docs/umfi_comparison.md.

reference_preprocess_lr <- function(dat, protect) {
  modified <- dat
  for (i in setdiff(seq_len(ncol(dat)), protect)) {
    mod <- stats::lm(dat[, i] ~ dat[, protect])
    if (summary(mod)$coefficients[2, 4] < 0.01) modified[, i] <- mod$residuals
  }
  modified
}

reference_umfi_lr <- function(x, y, num.trees = 100) {
  fi <- numeric(ncol(x))
  for (i in seq_len(ncol(x))) {
    newx <- reference_preprocess_lr(x, i)
    with <- ranger::ranger(x = newx, y = y, num.trees = num.trees)
    without <- ranger::ranger(x = newx[, -i, drop = FALSE], y = y, num.trees = num.trees)
    fi[i] <- max(with$r.squared, 0) - max(without$r.squared, 0)
  }
  fi[fi < 0] <- 0
  fi
}

test_that("whole-data residualization equals the reference where its p-value gate fires", {
  set.seed(1)
  n <- 200
  x <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  x$x3 <- x$x1 + rnorm(n, sd = 0.1)
  x$x4 <- rnorm(n)
  for (anchor in 1:4) {
    ref <- reference_preprocess_lr(x, anchor)
    cf <- residualize_fit(x, anchor)
    ours <- residualize_apply(x, cf)
    for (j in cf$features) {
      p <- summary(stats::lm(x[[j]] ~ x[[anchor]]))$coefficients[2, 4]
      if (p < 0.01) {
        expect_equal(unname(ours[, j]), unname(ref[[j]]), tolerance = 1e-10)
      } else {
        # reference keeps the raw column; we residualize (documented deviation)
        expect_equal(unname(ref[[j]]), x[[j]])
        expect_false(isTRUE(all.equal(unname(ours[, j]), x[[j]])))
      }
    }
  }
})

test_that("rankings agree with the reference algorithm on a clear-cut design", {
  set.seed(11)
  n <- 400
  x <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n))
  y <- 3 * x$x1 + 2 * x$x2 + 1 * x$x3 + rnorm(n, sd = 0.5)
  set.seed(12)
  ref <- reference_umfi_lr(x, y, num.trees = 100)
  spec <- parsnip::set_engine(parsnip::rand_forest(mode = "regression", trees = 100, min_n = 5), "ranger")
  ours <- shared_importance(x, y, learner = spec, metric = yardstick::rsq_trad, v = 5, seed = 13)$importance$importance
  expect_identical(order(ref, decreasing = TRUE), 1:4)
  expect_identical(order(ours, decreasing = TRUE), 1:4)
  expect_gte(stats::cor(ref, ours, method = "spearman"), 0.99)
  # both are differences in R^2, so magnitudes are on the same scale
  expect_equal(ours[1:3], ref[1:3], tolerance = 0.35)
  expect_lt(abs(ours[4]), 0.05)
})

test_that("the blood-relation design of the paper gives near-zero importance to non-relatives in both", {
  set.seed(21)
  n <- 500
  x1 <- rnorm(n)
  x2 <- 3 * x1 + runif(n, -1, 1)
  S <- rnorm(n)
  x3 <- x2 + S
  y <- S + runif(n, -0.5, 0.5)
  x4 <- y + rexp(n)
  x <- data.frame(x1, x2, x3, x4)
  set.seed(22)
  ref <- reference_umfi_lr(x, y, num.trees = 100)
  spec <- parsnip::set_engine(parsnip::rand_forest(mode = "regression", trees = 100, min_n = 5), "ranger")
  ours <- shared_importance(x, y, learner = spec, metric = yardstick::rsq_trad, v = 5, seed = 23)$importance$importance
  # x1 and x2 are not blood-related to y: both methods give them ~0
  # (observed: ref 0.000 / 0.015, ours -0.004 / 0.019)
  expect_true(all(ref[1:2] < 0.05))
  expect_true(all(abs(ours[1:2]) < 0.05))
  # x4 (a child of y) dominates; x3 (sibling via S) is clearly positive
  expect_identical(which.max(ref), 4L)
  expect_identical(which.max(ours), 4L)
  expect_gt(ref[3], 0.03)
  expect_gt(ours[3], 0.03)
  expect_gt(ref[4], 0.2)
  expect_gt(ours[4], 0.2)
})
