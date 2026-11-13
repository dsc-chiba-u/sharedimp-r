make_cls <- function(n = 120, k = 2, seed = 9) {
  set.seed(seed)
  x <- cbind(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  score <- 2 * x[, "x1"] + x[, "x2"]
  if (k == 2) {
    y <- factor(ifelse(score + rnorm(n, sd = 1.5) > 0, "yes", "no"), levels = c("yes", "no"))
  } else {
    y <- cut(score + rnorm(n, sd = 1.5), breaks = c(-Inf, -1, 1, Inf), labels = c("lo", "mid", "hi"))
  }
  list(x = x, y = y)
}

test_that("regression smoke test: linear and random forest", {
  d <- linear_design(n = 150)
  r_lin <- shared_importance(d$x, d$y, learner = "linear", v = 3, seed = 1)
  r_rf <- shared_importance(d$x, d$y, learner = "rand_forest", v = 3, seed = 1)
  for (r in list(r_lin, r_rf)) {
    expect_identical(r$task, "regression")
    expect_true(all(is.finite(r$importance$importance)))
    expect_true(all(is.na(r$folds$error)))
    # x1 is the strongest signal, x4 is noise
    expect_identical(which.max(r$importance$importance), 1L)
    expect_lt(r$importance$importance[4], r$importance$importance[3])
  }
  expect_identical(r_rf$learner, "rand_forest")
})

test_that("binary classification smoke test across learners and metrics", {
  d <- make_cls(k = 2)
  for (lrn in c("linear", "rand_forest")) {
    for (m in c("mn_log_loss", "accuracy", "roc_auc")) {
      # glm may warn about separation on small folds; that is the learner's business
      r <- suppressWarnings(shared_importance(d$x, d$y, learner = lrn, metric = m, v = 3, seed = 1))
      expect_identical(r$task, "classification")
      expect_identical(r$levels, c("yes", "no"))
      expect_true(all(is.finite(r$importance$importance)), info = paste(lrn, m))
      expect_identical(which.max(r$importance$importance), 1L, info = paste(lrn, m))
      expect_match(r$resampling, "stratified")
    }
  }
})

test_that("multiclass classification smoke test across learners and metrics", {
  skip_if_not_installed("nnet")
  d <- make_cls(n = 180, k = 3)
  for (lrn in c("linear", "rand_forest")) {
    for (m in c("mn_log_loss", "accuracy", "roc_auc")) {
      r <- suppressWarnings(shared_importance(d$x, d$y, learner = lrn, metric = m, v = 3, seed = 1))
      expect_identical(r$levels, c("lo", "mid", "hi"))
      expect_true(all(is.finite(r$importance$importance)), info = paste(lrn, m))
      expect_identical(which.max(r$importance$importance), 1L, info = paste(lrn, m))
    }
  }
})

test_that("a class missing from a training fold does not break probability alignment", {
  skip_if_not_installed("nnet")
  d <- make_cls(n = 90, k = 3)
  # put every "hi" row into the assessment fold 1 -> training fold 1 lacks "hi"
  labs <- rep(2:3, length.out = 90)
  labs[d$y == "hi"] <- 1
  r <- suppressWarnings(shared_importance(d$x, d$y, learner = "linear", metric = "accuracy", resamples = labs))
  expect_true(all(is.na(r$folds$error)))
  expect_true(all(is.finite(r$importance$importance)))
})

test_that("the direction diagnostic is not exported and not an argument of shared_importance()", {
  expect_false("direction" %in% names(formals(shared_importance)))
  expect_false("compute_direction" %in% getNamespaceExports("sharedimp"))
  # the internal function still works, for future reconciliation with Python
  a <- 1:10
  expect_equal(sharedimp:::compute_direction(a, 2 * a, rep(0, 10), "regression", NULL), 1)
  expect_true(is.na(sharedimp:::compute_direction(a, 2 * a, 2 * a, "regression", NULL)))
})
