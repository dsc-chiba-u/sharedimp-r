# These tests drive shared_importance() with a learner that records every
# matrix it receives, and check the representations against hand computation.

test_that("the learner receives training representations built from training rows only", {
  x <- tiny_x()
  y <- c(1, 3, 2, 5, 4, 6)
  folds <- c(1, 1, 1, 2, 2, 2) # rows 4:6 assessed in fold 2, trained on 1:3
  store <- new.env()
  store$fits <- list()
  store$preds <- list()
  res <- shared_importance(x, y, learner = recording_learner(store), resamples = folds, features = "a")
  # 2 folds x (full, reduced) = 4 fits and 4 predictions for anchor "a"
  expect_length(store$fits, 4L)
  expect_length(store$preds, 4L)

  # Fold 1: assessment rows 1:3, training rows 4:6.
  train <- x[4:6, ]
  assess <- x[1:3, ]
  cf <- residualize_fit(train, "a")
  full_train <- store$fits[[1]]$x
  reduced_train <- store$fits[[2]]$x
  full_assess <- store$preds[[1]]
  reduced_assess <- store$preds[[2]]

  expect_identical(colnames(full_train), c("a", "b", "c"))
  expect_equal(unname(full_train[, "a"]), unname(train[, "a"]))
  for (j in c("b", "c")) {
    expect_equal(unname(full_train[, j]), unname(train[, j] - cf$intercept[j] - cf$slope[j] * train[, "a"]))
    # same coefficients applied to assessment rows
    expect_equal(unname(full_assess[, j]), unname(assess[, j] - cf$intercept[j] - cf$slope[j] * assess[, "a"]))
  }
  expect_equal(reduced_train, full_train[, c("b", "c"), drop = FALSE])
  expect_equal(reduced_assess, full_assess[, c("b", "c"), drop = FALSE])
  expect_equal(store$fits[[1]]$y, y[4:6])
  expect_equal(store$fits[[2]]$y, y[4:6])
})

test_that("perturbing assessment rows leaves training representations unchanged", {
  d <- linear_design(n = 60)
  folds <- fixed_folds(60, 3)
  run <- function(x) {
    store <- new.env()
    store$fits <- list()
    store$preds <- list()
    shared_importance(x, d$y, learner = recording_learner(store), resamples = folds, features = "x2")
    store
  }
  s1 <- run(d$x)
  x2 <- d$x
  assess1 <- which(folds == 1)
  x2[assess1, ] <- x2[assess1, ] * 3 + 100
  s2 <- run(x2)
  # fold 1 training matrices identical (fits 1 and 2 belong to fold 1)
  expect_equal(s1$fits[[1]]$x, s2$fits[[1]]$x)
  expect_equal(s1$fits[[2]]$x, s2$fits[[2]]$x)
  # fold 1 assessment matrices differ (as they must)
  expect_false(isTRUE(all.equal(s1$preds[[1]], s2$preds[[1]])))
  # fold 2's training set contains the perturbed rows, so it differs
  expect_false(isTRUE(all.equal(s1$fits[[3]]$x, s2$fits[[3]]$x)))
})

test_that("the same folds are used for every anchor and for full/reduced", {
  d <- linear_design(n = 30)
  store <- new.env()
  store$fits <- list()
  store$preds <- list()
  folds <- fixed_folds(30, 3)
  shared_importance(d$x, d$y, learner = recording_learner(store), resamples = folds)
  # 4 anchors x 3 folds x 2 models
  expect_length(store$fits, 24L)
  ys <- lapply(store$fits, `[[`, "y")
  # fits are ordered split-major, anchor-minor, (full, reduced): y vectors repeat
  for (k in 1:3) {
    for (a in 0:3) {
      idx <- (k - 1) * 8 + a * 2 + 1
      expect_equal(ys[[idx]], d$y[folds != k])
      expect_equal(ys[[idx + 1]], ys[[idx]])
    }
  }
})
