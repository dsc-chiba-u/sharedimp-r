# Numerical comparison with the reference UMFI-LR implementation

Date: 2026-11-12. Machine: Linux, R 4.4.3, ranger 0.18.0, sharedimp 0.1.0.

Reference code: https://github.com/HydroML/UMFI, commit
`54b964740ffb006d3ade759d84dc6e9af2f45998` (2024-06-24), files
`R/preprocess_lr.R` and `R/umfi.R`, sourced unchanged from a local clone. The
Python port https://github.com/guanton/umfi_python (commit
`4db1c23f2355d69fc1471ae8df89c196d7a9a347`) was inspected but not executed; it
differs from the R reference (gate on in-sample R² > 0.01 instead of slope
p < 0.01; ExtraTrees by default).

The script that produced the output below is reproduced at the end of this
file. It is a one-off run and is **not** part of the package tests (which
re-implement the reference functions instead of vendoring them; see
`tests/testthat/test-umfi-reference.R`).

## Why exact agreement is impossible

| aspect | reference `umfi(mod_meth = "lr")` | `shared_importance()` |
|---|---|---|
| residualization rows | all rows | training fold only |
| p-value gate (p < 0.01) | yes; otherwise raw column kept | no |
| performance estimate | ranger out-of-bag R², 100 trees | per-fold assessment metric (here `yardstick::rsq_trad`), ranger 500 trees |
| floors / truncation | `max(R², 0)`, `fi[fi < 0] <- 0` | none |

So the comparison asks two questions: (A) is the residualization step
identical where the reference does residualize, and (B) do the importance
values and rankings agree on the paper's synthetic designs.

## Part A: residualization on identical rows

Correlation design of the paper (n = 300): `x1, x2, x4 ~ N(0,1)`,
`x3 = x1 + N(0, 0.01)`. Where the reference's gate fires, our residuals equal
the reference residuals to machine precision. Where it does not fire the
reference keeps the raw column and we residualize anyway; the resulting
difference is the (small) fitted linear component.

```
== Part A: residualization on identical rows ==
anchor x1: gate fired for x3; max|ref - ours| where fired = 4.86e-15; where not fired (ref keeps raw X_j) = 0.104
anchor x2: gate fired for ; max|ref - ours| where fired = NA; where not fired (ref keeps raw X_j) = 0.151
anchor x3: gate fired for x1; max|ref - ours| where fired = 5.05e-15; where not fired (ref keeps raw X_j) = 0.120
anchor x4: gate fired for ; max|ref - ours| where fired = NA; where not fired (ref keeps raw X_j) = 0.119
```

## Part B: importance on the paper's synthetic designs

Five replicate data sets of n = 1000 per design; the table reports the mean
over replicates of the raw values and of each feature's share of the total.
Reference: 100-tree ranger, OOB R² difference with floors and truncation.
sharedimp: 5-fold CV, 500-tree ranger (`min_n = 5`), `rsq_trad` difference,
no truncation.

```
[independent_linear] n=1000, 5 replicate data sets (mean over replicates)
                  [,1]  [,2]  [,3]  [,4]
upstream_umfi_lr 0.665 0.318 0.082 0.013
sharedimp        0.656 0.328 0.085 0.014
upstream_share   0.617 0.295 0.076 0.012
sharedimp_share  0.606 0.303 0.078 0.013
Spearman(upstream, sharedimp) = 1.000 ; same argmin = TRUE ; same argmax = TRUE
sharedimp min over replicates: 0.624 0.277 0.054 0.011 (negatives kept)
[correlation] n=1000, 5 replicate data sets (mean over replicates)
                  [,1]  [,2]  [,3]  [,4]
upstream_umfi_lr 0.526 0.573 0.516 0.010
sharedimp        0.526 0.575 0.514 0.010
upstream_share   0.324 0.352 0.318 0.006
sharedimp_share  0.323 0.354 0.316 0.006
Spearman(upstream, sharedimp) = 1.000 ; same argmin = TRUE ; same argmax = TRUE
sharedimp min over replicates: 0.481 0.528 0.484 0.008 (negatives kept)
[nonlinear_interaction] n=1000, 5 replicate data sets (mean over replicates)
                  [,1]  [,2]  [,3]  [,4]
upstream_umfi_lr 0.372 0.399 0.189 0.180
sharedimp        0.371 0.387 0.193 0.174
upstream_share   0.327 0.350 0.166 0.158
sharedimp_share  0.330 0.344 0.172 0.155
Spearman(upstream, sharedimp) = 1.000 ; same argmin = TRUE ; same argmax = TRUE
sharedimp min over replicates: 0.349 0.366 0.163 0.153 (negatives kept)
[blood_relation] n=1000, 5 replicate data sets (mean over replicates)
                  [,1]  [,2]  [,3]  [,4]
upstream_umfi_lr 0.008 0.007 0.104 0.494
sharedimp        0.008 0.009 0.103 0.465
upstream_share   0.013 0.011 0.169 0.806
sharedimp_share  0.013 0.016 0.175 0.795
Spearman(upstream, sharedimp) = 0.800 ; same argmin = FALSE ; same argmax = TRUE
sharedimp min over replicates: 0.006 0.007 0.084 0.431 (negatives kept)
```

### Reading the results

* **independent_linear** (`y = 3x1 + 2x2 + x3 + e`): identical ranking,
  values within 0.01 of each other.
* **correlation** (paper Sec. 4.1.3): both give `x1 ≈ x3` (the shared signal
  is credited to whichever is the anchor), `x2` slightly higher, `x4 ≈ 0`.
  Values agree to within 0.002.
* **nonlinear_interaction** (paper Sec. 4.1.1): all four features positive,
  interaction partners `x1`, `x2` largest in both; values within 0.012.
* **blood_relation** (paper Sec. 4.1.4): the non-blood-related `x1`, `x2` are
  ≈ 0.01 in both; `x4` dominates and `x3` is clearly positive in both. The
  Spearman correlation of 0.8 reflects only the arbitrary ordering of the two
  near-zero values (0.008 vs 0.007 against 0.008 vs 0.009), not a
  disagreement.

In no design did the sharedimp mean over replicates become negative; the
smallest single-replicate value was 0.006 (`blood_relation`, `x1`). Negative
fold-level values do occur for noise features and are retained in
`fold_results()`.

## Script

```r
# One-off comparison of sharedimp against the upstream UMFI-LR R implementation.
# Upstream functions are sourced from a clone of https://github.com/HydroML/UMFI
# (commit as of 2026-11-12). Output is pasted into docs/umfi_comparison.md.
args <- commandArgs(trailingOnly = TRUE)
upstream <- args[1]
devtools::load_all("/home/koki/dev/sharedimp-r", quiet = TRUE)
source(file.path(upstream, "R", "preprocess_lr.R"))
source(file.path(upstream, "R", "umfi.R"))
library(ranger)

cat("ranger", as.character(packageVersion("ranger")), "| sharedimp", as.character(packageVersion("sharedimp")), "\n\n")

designs <- list(
  independent_linear = function(n) {
    x <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n))
    list(x = x, y = 3 * x$x1 + 2 * x$x2 + x$x3 + rnorm(n, sd = 0.5))
  },
  correlation = function(n) { # paper Sec. 4.1.3
    x <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
    x$x3 <- x$x1 + rnorm(n, sd = sqrt(0.01)); x$x4 <- rnorm(n)
    list(x = x, y = x$x1 + x$x2)
  },
  nonlinear_interaction = function(n) { # paper Sec. 4.1.1
    x <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n))
    list(x = x, y = x$x1 + x$x2 + sign(x$x1 * x$x2) + x$x3 + x$x4)
  },
  blood_relation = function(n) { # paper Sec. 4.1.4
    x1 <- rnorm(n); x2 <- 3 * x1 + runif(n, -1, 1); S <- rnorm(n)
    x3 <- x2 + S; y <- S + runif(n, -0.5, 0.5); x4 <- y + rexp(n)
    list(x = data.frame(x1, x2, x3, x4), y = y)
  }
)

## Part A: residualization step, same rows ----------------------------------
cat("== Part A: residualization on identical rows ==\n")
set.seed(1); d <- designs$correlation(300)
for (anchor in 1:4) {
  ref <- preprocess_lr(d$x, anchor)
  cf <- residualize_fit(d$x, anchor)
  ours <- residualize_apply(d$x, cf)
  gated <- vapply(setdiff(1:4, anchor), function(j) summary(lm(d$x[, j] ~ d$x[, anchor]))$coefficients[2, 4] < 0.01, logical(1))
  diffs <- vapply(seq_along(cf$features), function(k) max(abs(ref[, cf$features[k]] - ours[, k])), numeric(1))
  cat(sprintf("anchor %s: gate fired for %s; max|ref - ours| where fired = %.2e; where not fired (ref keeps raw X_j) = %s\n",
              names(d$x)[anchor], paste(cf$features[gated], collapse = ","),
              if (any(gated)) max(diffs[gated]) else NA,
              if (any(!gated)) sprintf("%.3f", max(diffs[!gated])) else "n/a"))
}

## Part B: importance ---------------------------------------------------------
cat("\n== Part B: importance (upstream umfi(): ranger 100 trees OOB R^2, floors, truncation; sharedimp: 5-fold CV, ranger 500 trees, rsq_trad) ==\n")
n <- 1000; reps <- 5
for (nm in names(designs)) {
  ref_mat <- matrix(NA_real_, reps, 4); our_mat <- ref_mat
  for (r in seq_len(reps)) {
    set.seed(100 + r); d <- designs[[nm]](n)
    set.seed(200 + r); ref_mat[r, ] <- umfi(d$x, d$y, mod_meth = "lr")
    our_mat[r, ] <- shared_importance(d$x, d$y, learner = "rand_forest", metric = yardstick::rsq_trad, v = 5, seed = 300 + r)$importance$importance
  }
  ref <- colMeans(ref_mat); ours <- colMeans(our_mat)
  cat(sprintf("\n[%s] n=%d, %d replicate data sets (mean over replicates)\n", nm, n, reps))
  print(round(rbind(upstream_umfi_lr = ref, sharedimp = ours, upstream_share = ref / sum(ref), sharedimp_share = ours / sum(pmax(ours, 0))), 3))
  cat(sprintf("Spearman(upstream, sharedimp) = %.3f ; same argmin = %s ; same argmax = %s\n",
              cor(ref, ours, method = "spearman"), which.min(ref) == which.min(ours), which.max(ref) == which.max(ours)))
  cat(sprintf("sharedimp min over replicates: %s (negatives kept)\n", paste(round(apply(our_mat, 2, min), 3), collapse = " ")))
}
```
