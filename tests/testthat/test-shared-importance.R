test_that("result object has the documented structure", {
  d <- linear_design(n = 60)
  res <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", v = 3, seed = 1)
  expect_s3_class(res, "sharedimp_result")
  expect_identical(names(res$importance), c("feature", "importance", "sd", "n_folds"))
  expect_identical(res$importance$feature, colnames(d$x))
  expect_identical(
    names(res$folds),
    c("feature", "id", "id2", "metric_full", "metric_reduced", "importance", "error")
  )
  expect_identical(nrow(res$folds), 12L)
  expect_identical(res$task, "regression")
  expect_identical(res$metric, list(name = "mse", direction = "minimize"))
  expect_identical(res$n, 60L)
  expect_identical(res$p, 4L)
  expect_identical(res$seed, 1)
  expect_null(res$levels)
  expect_null(res$direction)
  expect_match(res$resampling, "3-fold cross-validation")
  expect_s3_class(fold_results(res), "tbl_df")
  expect_identical(as.data.frame(res), as.data.frame(res$importance))
  expect_output(print(res), "<sharedimp_result>")
  expect_output(print(res), "importance = -\\(full - reduced\\)")
  expect_output(summary(res), "learner:")
  expect_error(fold_results(1), "sharedimp_result")
})

test_that("seed makes results reproducible and restores the caller's RNG state", {
  d <- linear_design(n = 60)
  set.seed(123)
  before <- runif(1)
  set.seed(123)
  r1 <- shared_importance(d$x, d$y, learner = "rand_forest", v = 3, seed = 99)
  after <- runif(1)
  expect_identical(before, after)
  r2 <- shared_importance(d$x, d$y, learner = "rand_forest", v = 3, seed = 99)
  expect_identical(r1$importance, r2$importance)
  expect_identical(r1$folds, r2$folds)
  r3 <- shared_importance(d$x, d$y, learner = "rand_forest", v = 3, seed = 100)
  expect_false(identical(r1$folds$metric_full, r3$folds$metric_full))

  # without a seed, the global stream is consumed (no restoration)
  set.seed(1)
  shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", v = 3)
  x1 <- runif(1)
  set.seed(1)
  x2 <- runif(1)
  expect_false(identical(x1, x2))
})

test_that("explicit resamples are honoured (rset and fold labels)", {
  d <- linear_design(n = 40)
  labs <- fixed_folds(40, 4)
  r_lab <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = labs)
  expect_identical(unique(r_lab$folds$id), paste0("Fold", 1:4))
  expect_match(r_lab$resampling, "fold labels")

  rs <- rsample::vfold_cv(data.frame(i = 1:40), v = 4)
  r_rs <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = rs)
  expect_identical(unique(r_rs$folds$id), rs$id)
  # the rset's assessment rows determine metric_full: recompute for fold 1
  te <- rsample::complement(rs$splits[[1]])
  tr <- rs$splits[[1]]$in_id
  fit <- stats::lm.fit(cbind(1, d$x[tr, ]), d$y[tr])
  mse1 <- mean((d$y[te] - cbind(1, d$x[te, ]) %*% fit$coefficients)^2)
  expect_equal(r_rs$folds$metric_full[r_rs$folds$id == rs$id[1]][1], mse1)

  rs_rep <- rsample::vfold_cv(data.frame(i = 1:40), v = 2, repeats = 2)
  r_rep <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = rs_rep)
  expect_identical(sort(unique(r_rep$folds$id2)), c("Fold1", "Fold2"))
  expect_identical(nrow(r_rep$folds), 16L)

  r_v <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", v = 2, repeats = 2, seed = 1)
  expect_identical(nrow(r_v$folds), 16L)
  expect_match(r_v$resampling, "2 repeats")

  expect_error(shared_importance(d$x, d$y, resamples = labs[-1]), "length 40")
  expect_error(shared_importance(d$x, d$y, resamples = rep(1, 40)), "at least two folds")
  expect_error(shared_importance(d$x, d$y, resamples = rsample::vfold_cv(data.frame(i = 1:30), v = 3)), "30 rows")
  expect_error(shared_importance(d$x, d$y, resamples = "a"), "length 40")
  expect_error(shared_importance(d$x, d$y, v = 1), ">= 2")
  expect_error(shared_importance(d$x, d$y, v = 41), "cannot exceed")
  expect_error(shared_importance(d$x, d$y, repeats = 0), ">= 1")
})

test_that("features= restricts the anchors but keeps all columns in the representations", {
  d <- linear_design(n = 60)
  folds <- fixed_folds(60, 3)
  r_all <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  r_sub <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds, features = c("x3", "x1"))
  expect_identical(r_sub$importance$feature, c("x3", "x1"))
  expect_equal(r_sub$importance$importance[2], r_all$importance$importance[1])
  expect_equal(r_sub$importance$importance[1], r_all$importance$importance[3])
  r_idx <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds, features = 2)
  expect_identical(r_idx$importance$feature, "x2")
  expect_error(shared_importance(d$x, d$y, features = "zz"), "Unknown feature")
  expect_error(shared_importance(d$x, d$y, features = 9), "between 1")
  expect_error(shared_importance(d$x, d$y, features = c(1, 1)), "duplicates")
})

test_that("input validation covers x, y and task", {
  d <- linear_design(n = 30)
  expect_error(shared_importance(d$x, d$y[-1]), "length 29")
  expect_error(shared_importance(d$x, c(NA, d$y[-1])), "missing")
  expect_error(shared_importance(d$x, d$y, task = "classification"), "must be a factor")
  expect_error(shared_importance(d$x, factor(d$y > 0), task = "regression"), "must be numeric")
  expect_error(shared_importance(d$x, factor(rep("a", 30))), "at least two classes")
  expect_error(shared_importance(data.frame(a = 1:30, b = letters[rep(1:3, 10)]), d$y), "Non-numeric")
  expect_error(shared_importance(d$x[1, , drop = FALSE], d$y[1]), "two rows")
  expect_error(shared_importance(d$x, d$y, direction = "yes"), "TRUE")
  xm <- d$x
  xm[2, 3] <- Inf
  expect_error(shared_importance(xm, d$y), "non-finite")
  xd <- d$x
  colnames(xd) <- c("a", "a", "b", "c")
  expect_error(shared_importance(xd, d$y), "unique")
  # character y becomes a factor; logical columns are coerced with a message
  xdf <- as.data.frame(d$x)
  xdf$flag <- xdf$x1 > 0
  expect_message(
    res <- shared_importance(xdf, ifelse(d$y > 0, "pos", "neg"), learner = ols_learner_cls(), resamples = fixed_folds(30, 2)),
    "coerced"
  )
  expect_identical(res$task, "classification")
  expect_identical(res$levels, c("neg", "pos"))
})

test_that("verbose prints progress", {
  d <- linear_design(n = 20)
  expect_message(
    shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = fixed_folds(20, 2), verbose = TRUE, features = "x1"),
    "Feature 1/1: x1"
  )
})
