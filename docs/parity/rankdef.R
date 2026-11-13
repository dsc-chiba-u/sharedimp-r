# R side of the rank deficiency study (same fixture as rankdef.py).
devtools::load_all("/home/koki/dev/sharedimp-r", quiet = TRUE)
i <- 1:12
x <- cbind(x1 = i, x2 = (i %% 4) + 0.5 * i, x3 = c(2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11))
x <- cbind(x, x1_dup = 3 * x[, "x1"] - 2)
y <- 2 * x[, "x1"] - x[, "x2"] + 0.5 * x[, "x3"] + c(0.3, -0.2, 0.1, 0.4, -0.5, 0.2, -0.1, 0.3, -0.4, 0.1, 0.2, -0.3)
labels <- rep(1:3, 4)
options(digits = 15)
for (anchor in c("x2", "x3")) {
  cat("\n=== anchor", anchor, "===\n")
  for (k in 1:3) {
    tr <- labels != k; te <- !tr
    rep <- build_representations(x[tr, ], x[te, ], anchor)
    dev <- max(abs(rep$full_train[, "x1_dup"] - 3 * rep$full_train[, "x1"]))
    A_tr <- cbind(1, rep$full_train); A_te <- cbind(1, rep$full_assess)
    f1 <- stats::lm.fit(A_tr, y[tr]); b1 <- f1$coefficients; aliased <- names(b1)[is.na(b1)]; b1[is.na(b1)] <- 0
    f2 <- stats::lm(y[tr] ~ rep$full_train); b2 <- coef(f2); b2[is.na(b2)] <- 0
    # minimum-norm solution via SVD with R's tolerance analogue and with no truncation
    sv <- svd(A_tr); d <- sv$d
    minnorm <- function(tol) { keep <- d > tol * d[1]; sv$v[, keep, drop = FALSE] %*% ((t(sv$u[, keep, drop = FALSE]) %*% y[tr]) / d[keep]) }
    b_mn <- minnorm(1e-7); b_raw <- minnorm(.Machine$double.eps)
    ref <- drop(A_te %*% b1)
    cat(sprintf("split %d: |r_dup - 3 r_x1| max = %.1e; rank(lm.fit) = %d, aliased = %s; sv min/max = %.1e\n", k, dev, f1$rank, paste(aliased, collapse = ","), min(d) / max(d)))
    for (nm in c("lm.fit (pivoted QR, aliased -> 0)", "lm() formula", "min-norm SVD, tol 1e-7", "min-norm SVD, tol = machine eps")) {
      b <- switch(nm, "lm.fit (pivoted QR, aliased -> 0)" = b1, "lm() formula" = b2, "min-norm SVD, tol 1e-7" = b_mn, "min-norm SVD, tol = machine eps" = b_raw)
      pred <- drop(A_te %*% b)
      cat(sprintf("   %-36s mse_test = %.12f  max|pred - lm.fit| = %.1e  max|beta| = %.3g\n", nm, mean((y[te] - pred)^2), max(abs(pred - ref)), max(abs(b))))
    }
  }
}
