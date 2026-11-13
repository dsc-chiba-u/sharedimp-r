# Build-time smoke test for the Docker image (run by the Dockerfile).
# Checks that the installed package loads and that shared_importance() runs
# for regression (linear and random forest learners) and classification, and
# that the output follows the common R/Python schema. Any failure stops the
# build. Deliberately small so that it finishes in seconds.

suppressPackageStartupMessages(library(sharedimp))
cat("sharedimp", as.character(utils::packageVersion("sharedimp")), "on", R.version.string, "\n")

set.seed(1)
n <- 80
x <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
x$x4 <- 2 * x$x1 + 1 # exactly collinear with x1: residual fixed at zero
y <- 2 * x$x1 + x$x2 + rnorm(n)

summary_cols <- c("feature", "importance", "importance_std", "score_full", "score_reduced", "n_splits")
fold_cols <- c("feature", "split", "n_train", "n_test", "score_full", "score_reduced", "importance", "id", "id2", "error")

check <- function(res, label) {
  imp <- res$importance
  stopifnot(
    inherits(res, "sharedimp_result"),
    identical(names(imp), summary_cols),
    identical(names(fold_results(res)), fold_cols),
    nrow(imp) == 4L,
    all(is.finite(imp$importance)),
    all(imp$n_splits == 3L),
    all(is.na(fold_results(res)$error))
  )
  cat(sprintf("[ok] %s\n", label))
  print(imp)
}

# Regression, deterministic linear learner
r_lin <- shared_importance(x, y, learner = "linear", metric = "mse", v = 3, seed = 1)
check(r_lin, "regression / linear / mse")
imp <- r_lin$importance
stopifnot(
  imp$importance[imp$feature == "x1"] > 1,
  abs(imp$importance[imp$feature == "x1"] - imp$importance[imp$feature == "x4"]) < 1e-8,
  imp$importance[imp$feature == "x2"] > imp$importance[imp$feature == "x3"]
)

# Regression, default random forest learner (ranger)
r_rf <- shared_importance(x, y, v = 3, seed = 1)
check(r_rf, "regression / rand_forest / mse")

# Binary classification, log loss
yb <- factor(ifelse(y > stats::median(y), "high", "low"), levels = c("high", "low"))
r_cls <- suppressWarnings(shared_importance(x, yb, learner = "linear", metric = "log_loss", v = 3, seed = 1))
check(r_cls, "binary classification / linear / log_loss")
stopifnot(identical(r_cls$task, "classification"), identical(r_cls$levels, c("high", "low")))

# Multiclass classification (nnet::multinom through the "linear" learner)
ym <- cut(y, breaks = stats::quantile(y, c(0, 1 / 3, 2 / 3, 1)), labels = c("lo", "mid", "hi"), include.lowest = TRUE)
r_mc <- suppressWarnings(shared_importance(x, ym, learner = "linear", metric = "accuracy", v = 3, seed = 1))
check(r_mc, "multiclass classification / linear / accuracy")

cat("sharedimp Docker smoke test: OK\n")
