devtools::load_all("/home/koki/dev/sharedimp-r", quiet = TRUE)
i <- 1:12
x <- cbind(x1 = i, x2 = (i %% 4) + 0.5 * i, x3 = c(2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11)); x <- cbind(x, x1_dup = 3 * x[, "x1"] - 2)
y <- 2 * x[, "x1"] - x[, "x2"] + 0.5 * x[, "x3"] + c(0.3, -0.2, 0.1, 0.4, -0.5, 0.2, -0.1, 0.3, -0.4, 0.1, 0.2, -0.3)
labels <- rep(1:3, 4)
for (anchor in c("x2", "x3")) {
  cat("\n=== anchor", anchor, "===\n")
  for (k in 1:3) {
    tr <- labels != k; te <- !tr
    rep <- build_representations(x[tr, ], x[te, ], anchor)
    A_tr <- cbind(1, rep$full_train); A_te <- cbind(1, rep$full_assess)
    f <- stats::lm.fit(A_tr, y[tr]); b <- f$coefficients; b[is.na(b)] <- 0
    sv <- svd(A_tr)
    keep_all <- sv$v %*% ((t(sv$u) %*% y[tr]) / sv$d)  # no truncation at all = "not detected"
    cat(sprintf("split %d: lm.fit rank %d (tol 1e-7, aliased %s): mse_test = %.6f; no-truncation SVD solve: mse_test = %.6f, max|beta| = %.3g\n",
      k, f$rank, names(f$coefficients)[is.na(f$coefficients)], mean((y[te] - A_te %*% b)^2), mean((y[te] - A_te %*% keep_all)^2), max(abs(keep_all))))
  }
}
