test_that("with an OLS learner, fold importances equal hand-computed MSE differences", {
  d <- linear_design(n = 80)
  folds <- fixed_folds(80, 4)
  res <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  fr <- fold_results(res)
  for (a in colnames(d$x)) {
    for (k in 1:4) {
      tr <- folds != k
      te <- !tr
      rep <- build_representations(d$x[tr, ], d$x[te, ], a)
      fit_full <- stats::lm.fit(cbind(1, rep$full_train), d$y[tr])
      fit_red <- stats::lm.fit(cbind(1, rep$reduced_train), d$y[tr])
      mse_full <- mean((d$y[te] - cbind(1, rep$full_assess) %*% fit_full$coefficients)^2)
      mse_red <- mean((d$y[te] - cbind(1, rep$reduced_assess) %*% fit_red$coefficients)^2)
      row <- fr[fr$feature == a & fr$split == k, ]
      expect_identical(row$id, paste0("Fold", k))
      expect_equal(row$score_full, -mse_full) # minimize metric: score = -mse
      expect_equal(row$score_reduced, -mse_red)
      expect_equal(row$importance, mse_red - mse_full)
    }
  }
})

test_that("with a linear learner the full-model metric is invariant to the anchor", {
  # [x_i, R_{-i}] spans the same column space as [1, X], so OLS predictions coincide.
  d <- linear_design(n = 60)
  folds <- fixed_folds(60, 3)
  res <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  fr <- fold_results(res)
  for (k in 1:3) {
    vals <- fr$score_full[fr$split == k]
    expect_equal(vals, rep(vals[1], length(vals)), tolerance = 1e-10)
  }
  # and equals the plain OLS out-of-sample MSE
  tr <- folds != 1
  fit <- stats::lm.fit(cbind(1, d$x[tr, ]), d$y[tr])
  mse <- mean((d$y[!tr] - cbind(1, d$x[!tr, ]) %*% fit$coefficients)^2)
  expect_equal(fr$score_full[fr$split == 1][1], -mse)
})

test_that("importance approximates beta_i^2 * Var(x_i) for an independent linear-Gaussian design", {
  # Population: y = 3 x1 + 2 x2 + x3 + e, Var(x_i) = 1. Dropping x_i while
  # keeping residualized others (orthogonal in population) raises MSE by beta_i^2.
  d <- linear_design(n = 4000, seed = 7)
  res <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", v = 5, seed = 1)
  imp <- res$importance
  expect_equal(imp$feature, c("x1", "x2", "x3", "x4"))
  expect_equal(imp$importance, unname(d$beta^2), tolerance = 0.08)
  expect_lt(abs(imp$importance[4]), 0.02)
  expect_identical(imp$n_splits, rep(5L, 4))
  expect_true(all(is.finite(imp$importance_std)))
})

test_that("built-in linear learner reproduces the OLS learner", {
  d <- linear_design(n = 100)
  folds <- fixed_folds(100, 4)
  r1 <- shared_importance(d$x, d$y, learner = "linear", metric = "mse", resamples = folds)
  r2 <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  expect_equal(r1$importance$importance, r2$importance$importance, tolerance = 1e-10)
  expect_identical(r1$learner, "linear")
})

test_that("metric direction flips the sign convention consistently", {
  d <- linear_design(n = 100)
  folds <- fixed_folds(100, 4)
  r_mse <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  r_rsq <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "rsq", resamples = folds)
  r_rmse <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "rmse", resamples = folds)
  f_mse <- fold_results(r_mse)
  f_rsq <- fold_results(r_rsq)
  f_rmse <- fold_results(r_rmse)
  # minimize metric: score = -metric, importance = score_full - score_reduced
  expect_equal(f_mse$importance, f_mse$score_full - f_mse$score_reduced)
  expect_equal(f_rmse$importance, f_rmse$score_full - f_rmse$score_reduced)
  expect_equal(-f_rmse$score_full, sqrt(-f_mse$score_full))
  # maximize metric: score = metric
  expect_equal(f_rsq$importance, f_rsq$score_full - f_rsq$score_reduced)
  expect_true(all(f_rsq$score_full >= 0 & f_rsq$score_full <= 1))
  expect_identical(r_rsq$metric$direction, "maximize")
  expect_identical(r_mse$metric$direction, "minimize")
  # informative features are positive under both conventions
  expect_true(all(r_mse$importance$importance[1:3] > 0))
  expect_true(all(r_rsq$importance$importance[1:3] > 0))
})
