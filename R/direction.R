# Optional direction diagnostic (specification Section 13).
#
# dir = cor(anchor values on the assessment fold,
#           prediction with anchor - prediction without anchor)
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
