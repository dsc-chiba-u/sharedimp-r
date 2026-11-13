# Tests of the points settled in the common R/Python specification (v0.1):
# column order, exact collinearity, single predictor, sign, output schema,
# and the deterministic cross-language fixture.

# Deterministic fixture without any RNG: 12 rows, 3 features, explicit splits.
# The same numbers are listed in docs/specification.md (Section 15) so the
# Python package can reproduce them.
fixture_x <- function() {
  i <- 1:12
  cbind(
    x1 = i,
    x2 = (i %% 4) + 0.5 * i,
    x3 = c(2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11)
  )
}
fixture_y <- function() {
  x <- fixture_x()
  2 * x[, "x1"] - x[, "x2"] + 0.5 * x[, "x3"] + c(0.3, -0.2, 0.1, 0.4, -0.5, 0.2, -0.1, 0.3, -0.4, 0.1, 0.2, -0.3)
}
fixture_splits <- function() rep(1:3, times = 4) # row i assessed in split ((i - 1) %% 3) + 1

test_that("Full keeps the anchor in its original position; Reduced drops only that column", {
  x <- tiny_x()
  for (anchor in colnames(x)) {
    rep <- build_representations(x[1:4, ], x[5:6, ], anchor)
    expect_identical(colnames(rep$full_train), colnames(x), info = anchor)
    expect_identical(colnames(rep$full_assess), colnames(x), info = anchor)
    expect_identical(colnames(rep$reduced_train), setdiff(colnames(x), anchor), info = anchor)
    expect_identical(colnames(rep$reduced_assess), setdiff(colnames(x), anchor), info = anchor)
    expect_identical(dim(rep$full_train), c(4L, 3L))
    expect_identical(dim(rep$reduced_train), c(4L, 2L))
    # the anchor column is the raw input
    expect_equal(rep$full_train[, anchor], x[1:4, anchor])
    expect_equal(rep$full_assess[, anchor], x[5:6, anchor])
    # Reduced is exactly Full without the anchor column
    expect_equal(rep$reduced_train, rep$full_train[, colnames(x) != anchor, drop = FALSE])
    expect_equal(rep$reduced_assess, rep$full_assess[, colnames(x) != anchor, drop = FALSE])
  }
})

test_that("importance does not depend on the column order of x", {
  d <- linear_design(n = 60)
  folds <- fixed_folds(60, 3)
  r1 <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  perm <- c("x3", "x1", "x4", "x2")
  r2 <- shared_importance(d$x[, perm], d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  expect_identical(r2$importance$feature, perm)
  for (f in perm) {
    expect_equal(
      r2$importance$importance[r2$importance$feature == f],
      r1$importance$importance[r1$importance$feature == f],
      tolerance = 1e-10, info = f
    )
    expect_equal(
      fold_results(r2)$importance[fold_results(r2)$feature == f],
      fold_results(r1)$importance[fold_results(r1)$feature == f],
      tolerance = 1e-10, info = f
    )
  }
})

test_that("an exactly collinear column gets a residual of exactly zero and is kept", {
  x <- tiny_x()
  x <- cbind(x, a2 = 3 * x[, "a"] - 7) # exact affine function of the anchor
  cf <- residualize_fit(x[1:4, ], "a")
  expect_identical(unname(cf$collinear), c(FALSE, FALSE, TRUE))
  expect_equal(unname(cf$slope["a2"]), 3)
  expect_equal(unname(cf$intercept["a2"]), -7)
  r_train <- residualize_apply(x[1:4, ], cf)
  r_assess <- residualize_apply(x[5:6, ], cf)
  expect_identical(unname(r_train[, "a2"]), rep(0, 4)) # exactly zero, not just small
  expect_identical(unname(r_assess[, "a2"]), rep(0, 2))
  rep <- build_representations(x[1:4, ], x[5:6, ], "a")
  expect_identical(colnames(rep$full_train), c("a", "b", "c", "a2"))
  expect_identical(colnames(rep$reduced_train), c("b", "c", "a2"))
  expect_identical(unname(rep$reduced_assess[, "a2"]), rep(0, 2))
})

test_that("the collinearity rule is relative (SS_res <= 1e-10 SS_tot) and leaves genuine residuals alone", {
  set.seed(2)
  a <- rnorm(50)
  x <- cbind(a = a, near = 2 * a + 1e-4 * rnorm(50), far = rnorm(50), tiny = 1e-8 * (2 * a + 1))
  cf <- residualize_fit(x, "a")
  # relative SS_res / SS_tot ~ (1e-4)^2 / 4 = 2.5e-9 > 1e-10: not collinear
  expect_false(cf$collinear[["near"]])
  expect_false(cf$collinear[["far"]])
  # scale does not matter: an exact affine function with tiny scale is collinear
  expect_true(cf$collinear[["tiny"]])
  r <- residualize_apply(x, cf)
  expect_true(all(r[, "near"] != 0))
  expect_identical(unname(r[, "tiny"]), rep(0, 50))
})

test_that("a column constant on the training rows is centred, not flagged, and gives zero residuals", {
  x <- tiny_x()
  x <- cbind(x, k = 4)
  cf <- residualize_fit(x[1:4, ], "a")
  expect_false(cf$collinear[["k"]])
  expect_equal(unname(cf$slope["k"]), 0)
  expect_equal(unname(cf$intercept["k"]), 4)
  expect_equal(unname(residualize_apply(x[5:6, ], cf)[, "k"]), c(0, 0))
})

test_that("a constant anchor is detected by a zero sum of squares", {
  x <- tiny_x()
  x[, "b"] <- 0.1
  expect_warning(cf <- residualize_fit(x, "b"), class = "sharedimp_constant_anchor")
  expect_true(cf$constant_anchor)
  expect_equal(unname(cf$slope), c(0, 0))
  expect_equal(unname(cf$intercept), unname(colMeans(x[, c("a", "c")])))
  # nearly constant is not constant: ordinary OLS applies
  x[, "b"] <- 0.1 + c(0, 0, 0, 0, 0, 1e-9)
  expect_no_warning(cf2 <- residualize_fit(x, "b"))
  expect_false(cf2$constant_anchor)
})

test_that("exact duplicates: zero columns reach the learner and the pair is symmetric", {
  d <- linear_design(n = 80)
  x <- cbind(d$x, x1_copy = d$x[, "x1"])
  folds <- fixed_folds(80, 4)
  store <- new.env()
  store$fits <- list()
  store$preds <- list()
  shared_importance(x, d$y, learner = recording_learner(store), resamples = folds, features = "x1")
  # full and reduced training matrices for split 1 contain an all-zero x1_copy
  expect_identical(ncol(store$fits[[1]]$x), 5L)
  expect_identical(ncol(store$fits[[2]]$x), 4L)
  expect_identical(unname(store$fits[[1]]$x[, "x1_copy"]), rep(0, 60))
  expect_identical(unname(store$fits[[2]]$x[, "x1_copy"]), rep(0, 60))
  expect_identical(unname(store$preds[[2]][, "x1_copy"]), rep(0, 20))

  res <- shared_importance(x, d$y, learner = "linear", metric = "mse", resamples = folds)
  imp <- res$importance
  expect_equal(imp$importance[imp$feature == "x1"], imp$importance[imp$feature == "x1_copy"], tolerance = 1e-10)
  expect_true(all(is.na(fold_results(res)$error)))
  # residualizing the copy on the anchor leaves exactly zero, so the reduced
  # model loses the whole x1 signal (beta = 3): importance ~ 9, not ~ 0
  expect_gt(imp$importance[imp$feature == "x1"], 5)
})

test_that("single predictor: reduced model is the null model with hand-computed scores", {
  x <- fixture_x()[, "x1", drop = FALSE]
  y <- fixture_y()
  splits <- fixture_splits()
  res <- shared_importance(x, y, learner = ols_learner(), metric = "mse", resamples = splits)
  fr <- fold_results(res)
  expect_identical(nrow(fr), 3L)
  expect_identical(fr$n_train, rep(8L, 3))
  expect_identical(fr$n_test, rep(4L, 3))
  for (k in 1:3) {
    tr <- splits != k
    mu <- mean(y[tr])
    expect_equal(fr$score_reduced[fr$split == k], -mean((y[!tr] - mu)^2))
    fit <- stats::lm.fit(cbind(1, x[tr, ]), y[tr])
    expect_equal(fr$score_full[fr$split == k], -mean((y[!tr] - cbind(1, x[!tr, ]) %*% fit$coefficients)^2))
  }
  expect_equal(res$importance$importance, mean(fr$importance))

  # classification: null model predicts training class frequencies
  yb <- factor(rep(c("u", "v"), 6), levels = c("u", "v")) # both classes in every training fold
  resb <- shared_importance(x, yb, learner = ols_learner_cls(), metric = "log_loss", resamples = splits)
  frb <- fold_results(resb)
  for (k in 1:3) {
    tr <- splits != k
    p_u <- mean(yb[tr] == "u")
    probs <- ifelse(yb[!tr] == "u", p_u, 1 - p_u)
    expect_equal(frb$score_reduced[frb$split == k], mean(log(probs)))
  }
})

test_that("scores are oriented greater-is-better and importance is their difference", {
  d <- linear_design(n = 60)
  folds <- fixed_folds(60, 3)
  r_mse <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = folds)
  r_r2 <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "r2", resamples = folds)
  f_mse <- fold_results(r_mse)
  f_r2 <- fold_results(r_r2)
  expect_true(all(f_mse$score_full < 0)) # -mse
  expect_true(all(f_r2$score_full <= 1)) # r2 as is
  expect_equal(f_mse$importance, f_mse$score_full - f_mse$score_reduced)
  expect_equal(f_r2$importance, f_r2$score_full - f_r2$score_reduced)
  # same sign for every (split, feature): positive = anchor helps
  expect_identical(sign(f_mse$importance), sign(f_r2$importance))
  # and for an in-fold identity: r2 = 1 - mse / var_pop(y_test)
  for (k in 1:3) {
    y_te <- d$y[folds == k]
    sst <- mean((y_te - mean(y_te))^2)
    rows <- f_mse$split == k
    expect_equal(f_r2$score_full[rows], 1 - (-f_mse$score_full[rows]) / sst)
  }
  expect_identical(r_mse$metric$direction, "minimize")
  expect_identical(r_r2$metric$direction, "maximize")
})

test_that("output tables follow the common schema", {
  d <- linear_design(n = 30)
  res <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = fixed_folds(30, 3), features = c("x2", "x4"))
  imp <- res$importance
  expect_identical(names(imp), c("feature", "importance", "importance_std", "score_full", "score_reduced", "n_splits"))
  expect_type(imp$feature, "character")
  expect_type(imp$importance, "double")
  expect_type(imp$importance_std, "double")
  expect_type(imp$score_full, "double")
  expect_type(imp$score_reduced, "double")
  expect_type(imp$n_splits, "integer")
  expect_identical(imp$feature, c("x2", "x4")) # order of `features`, not sorted
  expect_identical(imp$n_splits, c(3L, 3L))

  fr <- fold_results(res)
  expect_identical(
    names(fr),
    c("feature", "split", "n_train", "n_test", "score_full", "score_reduced", "importance", "id", "id2", "error")
  )
  expect_type(fr$split, "integer")
  # rows ordered by split, then by feature (in `features` order)
  expect_identical(fr$split, rep(1:3, each = 2))
  expect_identical(fr$feature, rep(c("x2", "x4"), times = 3))
  expect_identical(fr$n_train + fr$n_test, rep(30L, 6))
  # aggregation
  for (f in c("x2", "x4")) {
    rows <- fr$feature == f
    expect_equal(imp$importance[imp$feature == f], mean(fr$importance[rows]))
    expect_equal(imp$importance_std[imp$feature == f], stats::sd(fr$importance[rows]))
    expect_equal(imp$score_full[imp$feature == f], mean(fr$score_full[rows]))
    expect_equal(imp$score_reduced[imp$feature == f], mean(fr$score_reduced[rows]))
  }
  # one split -> importance_std is NA
  sp <- rsample::make_splits(list(analysis = 1:20, assessment = 21:30), data.frame(i = 1:30))
  one <- shared_importance(d$x, d$y, learner = ols_learner(), metric = "mse", resamples = rsample::manual_rset(list(sp), "s1"))
  expect_true(all(is.na(one$importance$importance_std)))
  expect_identical(one$importance$n_splits, rep(1L, 4))
})

test_that("cross-language fixture: OLS + mse on explicit splits matches hand computation", {
  x <- fixture_x()
  y <- fixture_y()
  splits <- fixture_splits()
  res <- shared_importance(x, y, learner = "linear", metric = "mse", resamples = splits)
  fr <- fold_results(res)
  expected <- matrix(NA_real_, 3, 3, dimnames = list(NULL, colnames(x)))
  for (k in 1:3) {
    tr <- splits != k
    te <- !tr
    for (j in colnames(x)) {
      rep <- build_representations(x[tr, ], x[te, ], j)
      b_full <- stats::lm.fit(cbind(1, rep$full_train), y[tr])$coefficients
      b_red <- stats::lm.fit(cbind(1, rep$reduced_train), y[tr])$coefficients
      mse_full <- mean((y[te] - cbind(1, rep$full_assess) %*% b_full)^2)
      mse_red <- mean((y[te] - cbind(1, rep$reduced_assess) %*% b_red)^2)
      row <- fr[fr$split == k & fr$feature == j, ]
      expect_equal(row$score_full, -mse_full)
      expect_equal(row$score_reduced, -mse_red)
      expected[k, j] <- mse_red - mse_full
    }
  }
  expect_equal(res$importance$importance, unname(colMeans(expected)))
  expect_equal(res$importance$importance_std, unname(apply(expected, 2, stats::sd)))
  # Reference values recorded in docs/specification.md Section 15
  expect_equal(res$importance$importance, c(48.3228798721808, 34.4600258422062, 45.5507348310811), tolerance = 1e-12)
  expect_equal(res$importance$importance_std, c(6.17147177856012, 29.5939201894648, 22.4063405773906), tolerance = 1e-12)
  expect_equal(res$importance$score_full, rep(-0.276470061784757, 3), tolerance = 1e-12)
  expect_equal(fr$importance[fr$split == 1], c(53.5652538573545, 61.4017818839248, 36.9725234594595), tolerance = 1e-12)
})
