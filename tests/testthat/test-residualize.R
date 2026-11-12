test_that("residualization coefficients match lm() on the training rows", {
  x <- tiny_x()
  cf <- residualize_fit(x, anchor = "a")
  expect_s3_class(cf, "sharedimp_residualization")
  expect_identical(cf$anchor, "a")
  expect_identical(cf$features, c("b", "c"))

  for (j in c("b", "c")) {
    fit <- stats::lm(x[, j] ~ x[, "a"])
    expect_equal(unname(cf$intercept[j]), unname(coef(fit)[1]), tolerance = 1e-12)
    expect_equal(unname(cf$slope[j]), unname(coef(fit)[2]), tolerance = 1e-12)
  }

  # Hand computation for column b: slope = cov(a,b)/var(a), intercept = mean(b) - slope*mean(a)
  a <- x[, "a"]
  b <- x[, "b"]
  slope_hand <- sum((a - mean(a)) * (b - mean(b))) / sum((a - mean(a))^2)
  expect_equal(unname(cf$slope["b"]), slope_hand)
  expect_equal(unname(cf$intercept["b"]), mean(b) - slope_hand * mean(a))
})

test_that("anchor can be given by index and columns are returned in original order", {
  x <- tiny_x()
  cf_name <- residualize_fit(x, "b")
  cf_idx <- residualize_fit(x, 2)
  expect_equal(cf_name, cf_idx)
  expect_identical(cf_idx$features, c("a", "c"))
})

test_that("residualize_apply() uses the supplied coefficients, not the new rows", {
  x <- tiny_x()
  train <- x[1:4, ]
  assess <- x[5:6, ]
  cf <- residualize_fit(train, "a")
  r_assess <- residualize_apply(assess, cf)
  expect_identical(dim(r_assess), c(2L, 2L))
  expect_identical(colnames(r_assess), c("b", "c"))
  expected <- assess[, c("b", "c")] - outer(assess[, "a"], cf$slope) -
    matrix(cf$intercept, nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal(unname(r_assess), unname(expected))

  # Residuals on the training rows are exactly orthogonal to the anchor
  r_train <- residualize_apply(train, cf)
  expect_equal(unname(colSums(r_train)), c(0, 0), tolerance = 1e-12)
  expect_equal(unname(drop(crossprod(train[, "a"] - mean(train[, "a"]), r_train))), c(0, 0), tolerance = 1e-12)
  # ... but not on the assessment rows (no leakage of assessment information)
  expect_false(isTRUE(all.equal(unname(colSums(r_assess)), c(0, 0))))
})

test_that("assessment rows do not influence the coefficients", {
  x <- tiny_x()
  train <- x[1:4, ]
  assess1 <- x[5:6, ]
  assess2 <- assess1 * 10 + 7
  rep1 <- build_representations(train, assess1, "a")
  rep2 <- build_representations(train, assess2, "a")
  expect_equal(rep1$coefs, rep2$coefs)
  expect_equal(rep1$full_train, rep2$full_train)
  expect_equal(rep1$reduced_train, rep2$reduced_train)
  expect_false(isTRUE(all.equal(rep1$full_assess, rep2$full_assess)))
})

test_that("full and reduced representations follow the specification", {
  x <- tiny_x()
  train <- x[1:4, ]
  assess <- x[5:6, ]
  rep <- build_representations(train, assess, "b")
  expect_identical(colnames(rep$full_train), c("b", "a", "c"))
  expect_identical(colnames(rep$full_assess), c("b", "a", "c"))
  expect_identical(colnames(rep$reduced_train), c("a", "c"))
  expect_identical(colnames(rep$reduced_assess), c("a", "c"))
  # anchor column is raw (not centred)
  expect_equal(unname(rep$full_train[, "b"]), unname(train[, "b"]))
  expect_equal(unname(rep$full_assess[, "b"]), unname(assess[, "b"]))
  # reduced is the full representation without its first column
  expect_equal(rep$reduced_train, rep$full_train[, -1, drop = FALSE])
  expect_equal(rep$reduced_assess, rep$full_assess[, -1, drop = FALSE])
  expect_identical(rep$dropped, character(0))
})

test_that("constant anchor gives zero slopes, centred columns and a classed warning", {
  x <- tiny_x()
  x[, "a"] <- 5
  expect_warning(cf <- residualize_fit(x, "a"), class = "sharedimp_constant_anchor")
  expect_equal(unname(cf$slope), c(0, 0))
  expect_equal(unname(cf$intercept), unname(colMeans(x[, c("b", "c")])))
  expect_true(cf$constant_anchor)
  r <- suppressWarnings(residualize_apply(x, cf))
  expect_equal(unname(colMeans(r)), c(0, 0))
})

test_that("exact duplicate of the anchor yields a dropped constant residual column", {
  x <- tiny_x()
  x <- cbind(x, a2 = x[, "a"])
  rep <- build_representations(x[1:4, ], x[5:6, ], "a")
  expect_identical(rep$dropped, "a2")
  expect_identical(colnames(rep$full_train), c("a", "b", "c"))
  expect_identical(colnames(rep$reduced_assess), c("b", "c"))
  rep_keep <- build_representations(x[1:4, ], x[5:6, ], "a", drop_constant = FALSE)
  expect_identical(colnames(rep_keep$reduced_train), c("b", "c", "a2"))
  expect_equal(unname(rep_keep$reduced_train[, "a2"]), rep(0, 4))
})

test_that("single-predictor input gives empty reduced representations", {
  x <- tiny_x()[, "a", drop = FALSE]
  rep <- build_representations(x[1:4, , drop = FALSE], x[5:6, , drop = FALSE], "a")
  expect_identical(dim(rep$reduced_train), c(4L, 0L))
  expect_identical(dim(rep$reduced_assess), c(2L, 0L))
  expect_identical(colnames(rep$full_train), "a")
})

test_that("input validation errors are informative", {
  x <- tiny_x()
  expect_error(residualize_fit(x, "zzz"), "not a column")
  expect_error(residualize_fit(x, 7), "between 1 and")
  expect_error(residualize_fit(x, c("a", "b")), "single column")
  xna <- x
  xna[1, 1] <- NA
  expect_error(residualize_fit(xna, "a"), "missing")
  expect_error(residualize_apply(x, list()), "residualize_fit")
  expect_error(residualize_apply(x[, c("a", "b")], residualize_fit(x, "a")), "lacks column")
  expect_error(build_representations(x[1:4, ], x[5:6, c("b", "a", "c")], "a"), "same columns")
  df <- data.frame(a = 1:5, b = letters[1:5])
  expect_error(residualize_fit(df, "a"), "numeric")
})

test_that("data frames and unnamed matrices are accepted", {
  x <- tiny_x()
  cf_df <- residualize_fit(as.data.frame(x), "a")
  cf_m <- residualize_fit(x, "a")
  expect_equal(cf_df, cf_m)
  un <- unname(x)
  cf_un <- residualize_fit(un, 1)
  expect_identical(cf_un$features, c("V2", "V3"))
  expect_equal(unname(cf_un$slope), unname(cf_m$slope))
})
