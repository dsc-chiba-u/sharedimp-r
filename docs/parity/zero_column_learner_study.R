# Does an exactly-zero predictor column change predictions or break learners?
suppressPackageStartupMessages({library(parsnip)})
set.seed(1)
n <- 120
x <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
y <- 2 * x$x1 + x$x2 + rnorm(n)
yb <- factor(ifelse(y > 0, "a", "b"), levels = c("a", "b"))
ym <- cut(y, c(-Inf, -1, 1, Inf), labels = c("lo", "mid", "hi"))
tr <- 1:90; te <- 91:120
xz <- x; xz$z <- 0                      # extra exactly-zero column
xz1 <- data.frame(z = 0, x = x$x1)      # zero column + one real column
xz_only <- data.frame(z = rep(0, n))    # only a zero column (reduced of p=2 duplicates)

run <- function(label, spec, yy, type) {
  out <- function(dat_tr, dat_te) {
    res <- tryCatch(
      withCallingHandlers({
        fit <- fit_xy(spec, x = dat_tr, y = yy[tr])
        p <- predict(fit, new_data = dat_te, type = type)
        list(pred = as.matrix(p), warn = character(0))
      }, warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
      error = function(e) list(pred = NULL, err = conditionMessage(e)))
    res
  }
  warns <- character(0)
  a <- out(x[tr, ], x[te, ])
  b <- out(xz[tr, ], xz[te, ])
  c0 <- out(xz_only[tr, , drop = FALSE], xz_only[te, , drop = FALSE])
  d <- out(xz1[tr, ], xz1[te, ])
  same <- if (!is.null(a$pred) && !is.null(b$pred)) max(abs(a$pred - b$pred)) else NA
  cat(sprintf("%-28s | +zero col: %s | maxdiff vs no-zero: %s | zero-only: %s | zero+1: %s | warnings: %d%s\n",
    label,
    if (is.null(b$pred)) paste("ERROR:", b$err) else "ok",
    format(same, digits = 3),
    if (is.null(c0$pred)) paste("ERROR:", substr(c0$err, 1, 60)) else "ok",
    if (is.null(d$pred)) paste("ERROR:", substr(d$err, 1, 60)) else "ok",
    length(warns), if (length(warns)) paste0(" [", substr(unique(warns)[1], 1, 70), "]") else ""))
}

run("lm (linear_reg)", set_engine(linear_reg(), "lm"), y, "numeric")
run("glmnet (linear_reg, pen .1)", set_engine(linear_reg(penalty = 0.1), "glmnet"), y, "numeric")
run("ranger reg", set_engine(rand_forest(mode = "regression", trees = 100, min_n = 5), "ranger"), y, "numeric")
run("xgboost reg", set_engine(boost_tree(mode = "regression", trees = 50), "xgboost"), y, "numeric")
run("rpart reg", set_engine(decision_tree(mode = "regression"), "rpart"), y, "numeric")
run("kernlab svm_rbf reg", set_engine(svm_rbf(mode = "regression"), "kernlab"), y, "numeric")
run("nnet mlp reg", set_engine(mlp(mode = "regression", hidden_units = 3, penalty = 0.01, epochs = 200), "nnet"), y, "numeric")
run("glm (logistic_reg)", set_engine(logistic_reg(), "glm"), yb, "prob")
run("glmnet logistic", set_engine(logistic_reg(penalty = 0.05), "glmnet"), yb, "prob")
run("ranger cls", set_engine(rand_forest(mode = "classification", trees = 100, min_n = 5), "ranger"), yb, "prob")
run("xgboost cls", set_engine(boost_tree(mode = "classification", trees = 50), "xgboost"), yb, "prob")
run("kernlab svm_rbf cls", set_engine(svm_rbf(mode = "classification"), "kernlab"), yb, "prob")
run("nnet multinom_reg", set_engine(multinom_reg(), "nnet"), ym, "prob")
run("ranger multiclass", set_engine(rand_forest(mode = "classification", trees = 100, min_n = 5), "ranger"), ym, "prob")

# Direct nnet::multinom with a zero column
df <- cbind(xz[tr, ], .y = ym[tr])
m <- nnet::multinom(.y ~ ., data = df, maxit = 500, trace = FALSE)
df0 <- cbind(x[tr, ], .y = ym[tr])
m0 <- nnet::multinom(.y ~ ., data = df0, maxit = 500, trace = FALSE)
cat("nnet::multinom direct, zero col max prob diff:", max(abs(predict(m, xz[te, ], type = "probs") - predict(m0, x[te, ], type = "probs"))), "\n")
# lm.fit on zero column
f <- stats::lm.fit(cbind(1, as.matrix(xz[tr, ])), y[tr]); cat("lm.fit coef on zero col:", f$coefficients["z"], "\n")
g <- suppressWarnings(stats::glm.fit(cbind(1, as.matrix(xz[tr, ])), as.integer(yb[tr] == "a"), family = binomial())); cat("glm.fit coef on zero col:", g$coefficients[5], "\n")
