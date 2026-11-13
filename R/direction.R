# Direction diagnostic -- EXPERIMENTAL, NOT PART OF THE v0.1 COMMON API.
#
# This function is intentionally not exported and is not reachable from
# shared_importance(). It is kept for future work only. The R draft used
#
#   dir = cor(anchor values on the assessment rows,
#             prediction with anchor - prediction without anchor)   (Pearson)
#
# whereas the Python draft used the Spearman correlation between the anchor
# and the Full model's prediction. These are different statistical quantities,
# so the diagnostic was removed from the public API of both packages until a
# single definition is agreed (specification Section 13). Whatever is adopted
# must not be described as a causal or conditional effect.
#
# For binary classification the predictions are probabilities of the first
# level. It is NA for multiclass tasks and when either input is constant.
compute_direction <- function(anchor_assess, pred_full, pred_reduced, task, levels) {
  if (task == "classification") {
    if (length(levels) != 2L) {
      return(NA_real_)
    }
    pred_full <- pred_full[, 1L]
    pred_reduced <- pred_reduced[, 1L]
  }
  delta <- pred_full - pred_reduced
  if (length(delta) < 3L || is_constant(anchor_assess) || is_constant(delta)) {
    return(NA_real_)
  }
  stats::cor(anchor_assess, delta)
}
