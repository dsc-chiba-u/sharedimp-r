# syntax=docker/dockerfile:1
#
# sharedimp-r: R package `sharedimp` (UMFI-LR shared predictive importance)
# installed on a pinned, stable R release. The image is built and published by
# .github/workflows/build_test_push.yml to
#   ghcr.io/dsc-chiba-u/sharedimp-r  and  docker.io/koki/sharedimp-r
#
# Build-time checks (the build fails if any of them fails):
#   1. the package installs with its Imports,
#   2. the full testthat suite passes inside the image,
#   3. docker/smoke-test.R loads the package and runs shared_importance().
#
# sharedimp has no Bioconductor dependency, so a plain rocker/r-ver image is
# used instead of bioconductor_docker:devel. rocker/r-ver pins CRAN to the
# Posit Package Manager snapshot matching the R version and serves Linux
# binaries, which keeps the build fast and reproducible.
ARG R_VERSION=4.5.3
FROM rocker/r-ver:${R_VERSION}

LABEL org.opencontainers.image.title="sharedimp-r" \
      org.opencontainers.image.description="R package sharedimp: UMFI-LR shared predictive importance with fold-internal residualization (tidymodels)" \
      org.opencontainers.image.source="https://github.com/dsc-chiba-u/sharedimp-r" \
      org.opencontainers.image.licenses="MIT"

# Imports of DESCRIPTION, plus the Suggests needed for the test suite, the
# multiclass "linear" learner (nnet) and the VignetteBuilder (knitr/rmarkdown,
# so that R CMD build accepts the package; vignettes are not built).
# `install2.r` ships with rocker images.
RUN install2.r --error --skipinstalled --ncpus -1 \
      remotes \
      parsnip ranger rlang rsample tibble yardstick \
      nnet testthat withr knitr rmarkdown \
    && rm -rf /tmp/downloaded_packages /tmp/Rtmp*

# Install the package from the build context (see .dockerignore), with its
# tests, then run the tests and the smoke test. Everything temporary is removed
# afterwards so the image only carries the installed library.
COPY . /tmp/sharedimp
WORKDIR /tmp/sharedimp-check
RUN R -q -e "remotes::install_local('/tmp/sharedimp', dependencies = NA, upgrade = 'never', force = TRUE, build_vignettes = FALSE, INSTALL_opts = '--install-tests')" \
 && R -q -e "status <- tools::testInstalledPackage('sharedimp', types = 'tests'); fails <- Sys.glob('sharedimp-tests/*.Rout.fail'); if (length(fails)) writeLines(readLines(fails[1])); quit(status = as.integer(status != 0L))" \
 && Rscript /tmp/sharedimp/docker/smoke-test.R \
 && rm -rf /tmp/sharedimp /tmp/sharedimp-check /tmp/Rtmp*

WORKDIR /work
CMD ["R"]
