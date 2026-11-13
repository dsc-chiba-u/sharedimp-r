test_that("exact duplicates receive identical importance (duplication symmetry)", {
  d <- linear_design(n = 120)
  x <- cbind(d$x, x1_copy = d$x[, "x1"])
  folds <- fixed_folds(120, 4)
  res <- shared_importance(x, d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  imp <- res$importance
  expect_equal(
    imp$importance[imp$feature == "x1"],
    imp$importance[imp$feature == "x1_copy"],
    tolerance = 1e-10
  )
  # the other features are unaffected by the presence of a duplicate
  res0 <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  expect_equal(
    imp$importance[imp$feature %in% c("x2", "x3", "x4")],
    res0$importance$importance[2:4],
    tolerance = 1e-10
  )
  expect_true(all(is.na(fold_results(res)$error)))
})

test_that("an independent noise predictor has importance near zero and may be negative", {
  d <- linear_design(n = 2000, seed = 3)
  res <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", v = 5, seed = 11)
  x4 <- res$importance[res$importance$feature == "x4", ]
  expect_lt(abs(x4$importance), 0.02)
  expect_lt(abs(x4$importance), min(res$importance$importance[1:3]))
})

test_that("negative importance is never truncated", {
  # Learner that is deliberately worse whenever the anchor column is present.
  bad_learner <- sharedimp_learner(
    fit = function(x, y) list(mean = mean(y), has_anchor = ncol(x) == 4L),
    predict = function(model, x) rep(model$mean, nrow(x)) + if (model$has_anchor) 100 else 0,
    label = "deliberately bad"
  )
  d <- linear_design(n = 40)
  res <- shared_importance(d$x, d$y, learner = bad_learner, metric = "mse", resamples = fixed_folds(40, 4))
  expect_true(all(res$importance$importance < 0))
  expect_true(all(fold_results(res)$importance < 0))
})

test_that("a single predictor is handled with the null model", {
  set.seed(5)
  x <- matrix(rnorm(200), ncol = 1, dimnames = list(NULL, "only"))
  y <- 2 * x[, 1] + rnorm(200, sd = 0.1)
  res <- shared_importance(x, y, learner = ols_learner(), metric = "mse", v = 4, seed = 1)
  expect_identical(nrow(res$importance), 1L)
  # reduced model predicts the training mean: MSE ~ Var(y) ~ 4; full ~ 0.01
  expect_gt(res$importance$importance, 3)
  fr <- fold_results(res)
  expect_true(all(-fr$score_reduced > 3))
  expect_true(all(-fr$score_full < 0.05))
})

test_that("a constant anchor is reported once and does not abort", {
  d <- linear_design(n = 50)
  x <- cbind(d$x, const = 1)
  expect_warning(
    res <- shared_importance(x, d$y, learner = ols_learner(), metric = "mse", resamples = fixed_folds(50, 2)),
    "constant on at least one training fold"
  )
  expect_true(all(is.na(fold_results(res)$error)))
  # a constant anchor cannot help OLS: its importance is ~0
  expect_lt(abs(res$importance$importance[res$importance$feature == "const"]), 1e-8)
})

test_that("rank-deficient designs do not break the linear learner", {
  d <- linear_design(n = 60)
  x <- cbind(d$x, sum12 = d$x[, "x1"] + d$x[, "x2"])
  res <- suppressWarnings(
    shared_importance(x, d$y, learner = ols_learner(), metric = "mse", resamples = fixed_folds(60, 3))
  )
  expect_true(all(is.finite(res$importance$importance)))
})
