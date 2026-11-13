# Pin down the exact definitions of the built-in metrics so that the metric
# table in docs/specification.md (Section 9) is test-backed on the R side.
# The matching scikit-learn definitions are recorded in the specification.

test_that("r2 is 1 - SS_res / SS_tot and rsq is the squared Pearson correlation", {
  truth <- c(1, 2, 3, 4, 5, 7)
  est <- c(1.2, 1.9, 3.5, 3.6, 5.5, 6.1)
  r2 <- as_metric("r2", "regression")
  rsq <- as_metric("rsq", "regression")
  v_r2 <- sharedimp:::evaluate_metric(r2, truth, est, NULL)
  v_rsq <- sharedimp:::evaluate_metric(rsq, truth, est, NULL)
  expect_equal(v_r2, 1 - sum((truth - est)^2) / sum((truth - mean(truth))^2))
  expect_equal(v_rsq, stats::cor(truth, est)^2)
  expect_false(isTRUE(all.equal(v_r2, v_rsq))) # genuinely different quantities
  expect_identical(r2$direction, "maximize")
  # aliases
  expect_identical(as_metric("rsq_trad", "regression")$direction, "maximize")
  expect_equal(sharedimp:::evaluate_metric(as_metric("rsq_trad", "regression"), truth, est, NULL), v_r2)
})

test_that("mse, rmse and mae are the plain means", {
  truth <- c(1, 2, 3, 4)
  est <- c(1.5, 2, 2, 6)
  e <- function(name) sharedimp:::evaluate_metric(as_metric(name, "regression"), truth, est, NULL)
  expect_equal(e("mse"), mean((truth - est)^2))
  expect_equal(e("rmse"), sqrt(mean((truth - est)^2)))
  expect_equal(e("mae"), mean(abs(truth - est)))
})

test_that("log_loss is the mean negative log probability of the true class, clipped at machine epsilon", {
  lv <- c("a", "b", "c")
  truth <- factor(c("a", "b", "c", "a"), levels = lv)
  p <- rbind(
    c(0.7, 0.2, 0.1),
    c(0.3, 0.5, 0.2),
    c(0.1, 0.1, 0.8),
    c(0.0, 0.5, 0.5) # true class has probability exactly 0 -> clipped
  )
  ll <- as_metric("log_loss", "classification")
  v <- sharedimp:::evaluate_metric(ll, truth, p, lv)
  eps <- .Machine$double.eps
  p_true <- pmax(pmin(c(0.7, 0.5, 0.8, 0.0), 1 - eps), eps)
  expect_equal(v, -mean(log(p_true)))
  expect_identical(ll$direction, "minimize")
  expect_equal(sharedimp:::evaluate_metric(as_metric("mn_log_loss", "classification"), truth, p, lv), v)

  # binary: the same quantity, independent of which level is "first"
  lv2 <- c("yes", "no")
  truth2 <- factor(c("yes", "no", "no", "yes"), levels = lv2)
  p2 <- cbind(c(0.9, 0.2, 0.4, 0.6), c(0.1, 0.8, 0.6, 0.4))
  v2 <- sharedimp:::evaluate_metric(ll, truth2, p2, lv2)
  expect_equal(v2, -mean(log(c(0.9, 0.8, 0.6, 0.6))))
  v2_swapped <- sharedimp:::evaluate_metric(ll, factor(as.character(truth2), levels = rev(lv2)), p2[, 2:1], rev(lv2))
  expect_equal(v2_swapped, v2)
})

test_that("accuracy is the fraction of correct argmax labels (ties -> first level)", {
  lv <- c("a", "b")
  truth <- factor(c("a", "b", "a", "b"), levels = lv)
  p <- cbind(c(0.6, 0.4, 0.5, 0.2), c(0.4, 0.6, 0.5, 0.8)) # row 3 is a tie -> "a" (correct)
  acc <- as_metric("accuracy", "classification")
  expect_equal(sharedimp:::evaluate_metric(acc, truth, p, lv), 1)
  p[4, ] <- c(0.7, 0.3)
  expect_equal(sharedimp:::evaluate_metric(acc, truth, p, lv), 0.75)
})

test_that("binary roc_auc is the Mann-Whitney statistic and does not depend on the event level", {
  lv <- c("pos", "neg")
  truth <- factor(c("pos", "pos", "neg", "pos", "neg", "neg", "pos", "neg"), levels = lv)
  p_pos <- c(0.9, 0.8, 0.7, 0.6, 0.6, 0.3, 0.2, 0.1)
  p <- cbind(p_pos, 1 - p_pos)
  auc <- as_metric("roc_auc", "classification")
  v <- sharedimp:::evaluate_metric(auc, truth, p, lv)
  # hand computation: P(score_pos > score_neg) + 0.5 P(tie)
  pos <- p_pos[truth == "pos"]
  neg <- p_pos[truth == "neg"]
  cmp <- outer(pos, neg, function(a, b) (a > b) + 0.5 * (a == b))
  expect_equal(v, mean(cmp))
  # swapping the level order (scikit-learn uses the *second* class as positive) gives the same AUC
  v_swapped <- sharedimp:::evaluate_metric(auc, factor(as.character(truth), levels = rev(lv)), p[, 2:1], rev(lv))
  expect_equal(v_swapped, v)
})

test_that("multiclass roc_auc is Hand & Till's average of pairwise AUCs", {
  lv <- c("a", "b", "c")
  set.seed(3)
  truth <- factor(sample(lv, 30, replace = TRUE), levels = lv)
  p <- matrix(stats::runif(90), 30, 3)
  p <- p / rowSums(p)
  colnames(p) <- lv
  auc <- as_metric("roc_auc", "classification")
  v <- sharedimp:::evaluate_metric(auc, truth, p, lv)
  # Hand & Till (2001): for each pair (i, j), restrict to rows of class i or j,
  # A(i|j) uses p_i with i as event, A(j|i) uses p_j with j as event; average
  # the two, then average over the K(K-1)/2 pairs. This is also what
  # scikit-learn's roc_auc_score(multi_class = "ovo", average = "macro") computes.
  pair_auc <- function(i, j) {
    keep <- truth %in% c(i, j)
    t <- truth[keep]
    a_ij <- yardstick::roc_auc_vec(factor(t == i, levels = c(TRUE, FALSE)), p[keep, i])
    a_ji <- yardstick::roc_auc_vec(factor(t == j, levels = c(TRUE, FALSE)), p[keep, j])
    (a_ij + a_ji) / 2
  }
  pairs <- utils::combn(lv, 2)
  expected <- mean(apply(pairs, 2, function(pr) pair_auc(pr[1], pr[2])))
  expect_equal(v, expected)
})
