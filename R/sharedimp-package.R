#' sharedimp: Shared Predictive Importance via Fold-Internal Residualization
#'
#' A resampling-based implementation of the linear-regression variant of
#' ultra-marginal feature importance (UMFI-LR; Janssen, Guan and Robeva,
#' 2023). See [shared_importance()] for the main entry point and
#' `vignette("sharedimp")` for a worked example. The algorithm specification
#' shared with the Python implementation is in `docs/specification.md` of the
#' source repository.
#'
#' @references
#' Janssen, J., Guan, V. and Robeva, E. (2023). Ultra-marginal Feature
#' Importance: Learning from Data with Causal Guarantees. *Proceedings of the
#' 26th International Conference on Artificial Intelligence and Statistics
#' (AISTATS)*, PMLR 206:10782-10814.
#' <https://proceedings.mlr.press/v206/janssen23a.html>
#'
#' @keywords internal
#' @importFrom stats predict sd cor setNames
#' @importFrom rlang abort warn inform `%||%`
#' @importFrom ranger ranger
"_PACKAGE"
