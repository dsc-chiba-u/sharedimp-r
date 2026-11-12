test_that("as_metric() resolves built-ins and respects yardstick direction", {
  m <- as_metric("mse", "regression")
  expect_s3_class(m, "sharedimp_metric")
  expect_identical(m$direction, "minimize")
  expect_identical(m$kind, "numeric")
  expect_identical(as_metric("rsq", "regression")$direction, "maximize")
  expect_identical(as_metric("accuracy", "classification")$kind, "class")
  expect_identical(as_metric("roc_auc", "classification")$kind, "prob")
  expect_identical(as_metric("mn_log_loss", "classification")$direction, "minimize")
  expect_error(as_metric("nope", "regression"), "Unknown metric")
  expect_error(as_metric("accuracy", "regression"), "classification metric")
  expect_error(as_metric("mse", "classification"), "regression metric")
  expect_error(as_metric(mean, "regression"), "yardstick metric")
})

test_that("mse() is a valid yardstick metric", {
  df <- data.frame(t = c(1, 2, 3), e = c(1, 2, 5))
  out <- mse(df, t, e)
  expect_equal(out$.estimate, 4 / 3)
  expect_identical(out$.metric, "mse")
  expect_identical(attr(mse, "direction"), "minimize")
})

test_that("user-supplied yardstick metrics work, including custom ones", {
  d <- linear_design(n = 80)
  folds <- fixed_folds(80, 4)
  r_mae <- shared_importance(d$x, d$y, learner = ols_learner(), metric = yardstick::mae, resamples = folds)
  expect_identical(r_mae$metric$name, "yardstick::mae")
  fr <- fold_results(r_mae)
  expect_equal(fr$importance, fr$metric_reduced - fr$metric_full)

  # a maximize metric defined by the user: negative MAE
  neg_mae <- yardstick::new_numeric_metric(
    function(data, truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
      yardstick::numeric_metric_summarizer(
        name = "neg_mae",
        fn = function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) -mean(abs(truth - estimate)),
        data = data, truth = !!rlang::enquo(truth), estimate = !!rlang::enquo(estimate),
        na_rm = na_rm, case_weights = !!rlang::enquo(case_weights)
      )
    },
    direction = "maximize"
  )
  r_neg <- shared_importance(d$x, d$y, learner = ols_learner(), metric = neg_mae, resamples = folds)
  expect_equal(r_neg$importance$importance, r_mae$importance$importance)
  expect_identical(r_neg$metric$direction, "maximize")

  # metrics with direction "zero" are rejected
  expect_error(as_metric(yardstick::mpe, "regression"), "maximize.*minimize")
})

test_that("learner failures become NA with a warning, or abort on request", {
  flaky <- sharedimp_learner(
    fit = function(x, y) {
      # fail only on the *full* representation of anchor x2 (4 columns, x2 first)
      if (ncol(x) == 4L && colnames(x)[1] == "x2") stop("boom on anchor x2")
      mean(y)
    },
    predict = function(model, x) rep(model, nrow(x)),
    label = "flaky"
  )
  d <- linear_design(n = 40)
  folds <- fixed_folds(40, 4)
  expect_warning(
    res <- shared_importance(d$x, d$y, learner = flaky, metric = "mse", resamples = folds),
    "4 of 16 .* failed"
  )
  imp <- res$importance
  expect_true(is.na(imp$importance[imp$feature == "x2"]))
  expect_identical(imp$n_folds[imp$feature == "x2"], 0L)
  expect_identical(imp$n_folds[imp$feature == "x1"], 4L)
  fr <- fold_results(res)
  expect_identical(sum(!is.na(fr$error)), 4L)
  expect_match(fr$error[fr$feature == "x2"][1], "boom on anchor x2")
  expect_true(all(is.na(fr$importance[fr$feature == "x2"])))
  expect_output(print(res), "failures:   4 of 16")

  expect_error(
    shared_importance(d$x, d$y, learner = flaky, metric = "mse", resamples = folds, on_error = "error"),
    "Learner failed for feature `x2`"
  )
})

test_that("non-finite or malformed predictions are treated as learner failures", {
  nan_learner <- sharedimp_learner(
    fit = function(x, y) NULL,
    predict = function(model, x) rep(NaN, nrow(x)),
    label = "nan"
  )
  d <- linear_design(n = 20)
  expect_warning(
    res <- shared_importance(d$x, d$y, learner = nan_learner, metric = "mse", resamples = fixed_folds(20, 2)),
    "failed"
  )
  expect_true(all(is.na(res$importance$importance)))
  expect_match(fold_results(res)$error[1], "non-finite")

  short_learner <- sharedimp_learner(
    fit = function(x, y) NULL,
    predict = function(model, x) 1,
    label = "short"
  )
  expect_warning(
    res2 <- shared_importance(d$x, d$y, learner = short_learner, metric = "mse", resamples = fixed_folds(20, 2)),
    "failed"
  )
  expect_match(fold_results(res2)$error[1], "assessment rows")
})

test_that("learner warnings are collected and reported once", {
  noisy <- sharedimp_learner(
    fit = function(x, y) {
      warning("something mildly wrong")
      mean(y)
    },
    predict = function(model, x) rep(model, nrow(x)),
    label = "noisy"
  )
  d <- linear_design(n = 20)
  w <- testthat::capture_warnings(
    shared_importance(d$x, d$y, learner = noisy, metric = "mse", resamples = fixed_folds(20, 2))
  )
  expect_length(w, 1L)
  expect_match(w, "16 warning")
  expect_match(w, "something mildly wrong")
})

test_that("parsnip model specifications are accepted and mode is checked", {
  d <- linear_design(n = 80)
  folds <- fixed_folds(80, 4)
  spec <- parsnip::set_engine(parsnip::linear_reg(), "lm")
  r1 <- shared_importance(d$x, d$y, learner = spec, metric = "mse", resamples = folds)
  r2 <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  expect_equal(r1$importance$importance, r2$importance$importance, tolerance = 1e-10)
  expect_identical(r1$learner, "linear_reg (lm)")

  yb <- factor(d$y > 0)
  expect_error(
    shared_importance(d$x, yb, learner = spec, resamples = folds),
    "mode \"regression\" but the task is \"classification\""
  )
  # unknown-mode specs are given the task's mode
  rf <- parsnip::rand_forest(trees = 20)
  r3 <- shared_importance(d$x, d$y, learner = rf, resamples = folds, seed = 1)
  expect_identical(r3$learner, "rand_forest (ranger)")
  expect_true(all(is.finite(r3$importance$importance)))
})

test_that("glmnet specifications work when the package is installed", {
  skip_if_not_installed("glmnet")
  d <- linear_design(n = 80)
  spec <- parsnip::set_engine(parsnip::linear_reg(penalty = 0.01, mixture = 1), "glmnet")
  res <- shared_importance(d$x, d$y, learner = spec, resamples = fixed_folds(80, 4))
  expect_true(all(is.finite(res$importance$importance)))
  expect_gt(res$importance$importance[1], res$importance$importance[4])
})

test_that("invalid learner arguments are rejected", {
  d <- linear_design(n = 20)
  expect_error(shared_importance(d$x, d$y, learner = "nope"), "Unknown learner")
  expect_error(shared_importance(d$x, d$y, learner = 42), "must be a string")
  expect_error(sharedimp_learner(fit = 1, predict = identity), "functions")
  expect_error(sharedimp_learner(fit = identity, predict = identity, label = 1), "single string")
})

test_that("custom learner contract errors are informative for classification", {
  set.seed(2)
  x <- matrix(rnorm(120), 40, 3, dimnames = list(NULL, c("a", "b", "c")))
  y <- factor(rep(c("u", "v"), each = 20))
  wrong_cols <- sharedimp_learner(
    fit = function(x, y) NULL,
    predict = function(model, x) matrix(0.5, nrow(x), 3),
    label = "wrong"
  )
  expect_warning(res <- shared_importance(x, y, learner = wrong_cols, resamples = fixed_folds(40, 2)), "failed")
  expect_match(fold_results(res)$error[1], "probability object")
  out_of_range <- sharedimp_learner(
    fit = function(x, y) NULL,
    predict = function(model, x) matrix(c(1.5, -0.5), nrow(x), 2, byrow = TRUE),
    label = "range"
  )
  expect_warning(res2 <- shared_importance(x, y, learner = out_of_range, resamples = fixed_folds(40, 2)), "failed")
  expect_match(fold_results(res2)$error[1], "outside")
})
