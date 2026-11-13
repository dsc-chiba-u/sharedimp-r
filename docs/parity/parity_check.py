import sys, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, root_mean_squared_error, mean_absolute_error,
    r2_score, log_loss, accuracy_score, roc_auc_score)
from sharedimp import shared_importance
sp = sys.argv[1]
# ---- fixture ----
i = np.arange(1, 13)
X = pd.DataFrame({"x1": i.astype(float), "x2": (i % 4) + 0.5 * i, "x3": [2,1,4,3,6,5,8,7,10,9,12,11]}, dtype=float)
y = 2*X.x1 - X.x2 + 0.5*X.x3 + np.array([0.3,-0.2,0.1,0.4,-0.5,0.2,-0.1,0.3,-0.4,0.1,0.2,-0.3])
labels = np.tile([1,2,3], 4)
cv = [(np.where(labels != k)[0], np.where(labels == k)[0]) for k in (1,2,3)]
res = shared_importance(LinearRegression(), X, y.to_numpy(), scoring="neg_mean_squared_error", cv=cv)
r_imp = pd.read_csv(f"{sp}/r_fixture_importance.csv"); r_f = pd.read_csv(f"{sp}/r_fixture_folds.csv")
print("== fixture: Python importances ==\n", res.importances.to_string(index=False))
for col in ["importance","importance_std","score_full","score_reduced"]:
    print(f"max |R - Py| {col:15s}: {np.max(np.abs(r_imp[col].to_numpy() - res.importances[col].to_numpy())):.2e}")
print("n_splits equal:", (r_imp.n_splits.to_numpy() == res.importances.n_splits.to_numpy()).all())
pf = res.fold_results.sort_values(["split","feature"], kind="stable")  # R: split-major, feature in x order
rf = r_f.assign(split=r_f.split - 1).sort_values(["split","feature"], kind="stable")
for col in ["score_full","score_reduced","importance"]:
    print(f"fold table max |R - Py| {col:13s}: {np.max(np.abs(rf[col].to_numpy() - pf[col].to_numpy())):.2e}")
print("fold n_train/n_test equal:", (rf.n_train.to_numpy()==pf.n_train.to_numpy()).all(), (rf.n_test.to_numpy()==pf.n_test.to_numpy()).all())
# duplicate / exact-collinearity variant
Xd = X.assign(x1_dup=3*X.x1 - 2)
resd = shared_importance(LinearRegression(), Xd, y.to_numpy(), scoring="neg_mean_squared_error", cv=cv)
r_d = pd.read_csv(f"{sp}/r_fixture_dup_importance.csv")
print("== collinear variant ==")
print(pd.DataFrame({"feature": r_d.feature, "R": r_d.importance, "Py": resd.importances.importance}).to_string(index=False))
print(f"max |R - Py| importance (collinear variant): {np.max(np.abs(r_d.importance.to_numpy() - resd.importances.importance.to_numpy())):.2e}")
# ---- metric definitions ----
reg = pd.read_csv(f"{sp}/reg.csv"); b = pd.read_csv(f"{sp}/bin.csv"); m = pd.read_csv(f"{sp}/mc.csv")
rm = pd.read_csv(f"{sp}/r_metrics.csv").set_index("metric").r_value
# binary AUC: scikit-learn's positive class is the second label in sorted order; AUC is the same
# whichever class is called positive as long as the matching probability column is used
py = {
 "mse": mean_squared_error(reg.truth, reg.est), "rmse": root_mean_squared_error(reg.truth, reg.est),
 "mae": mean_absolute_error(reg.truth, reg.est), "r2": r2_score(reg.truth, reg.est),
 "rsq": np.corrcoef(reg.truth, reg.est)[0,1]**2,
 # scikit-learn expects probability columns in lexicographic label order: ['no', 'yes']
 "log_loss_bin": log_loss(b.truth, np.column_stack([1-b.p_yes, b.p_yes]), labels=["no","yes"]),
 "accuracy_bin": accuracy_score(b.truth, np.where(b.p_yes >= 1-b.p_yes, "yes", "no")),
 "roc_auc_bin": roc_auc_score((b.truth=="yes").astype(int), b.p_yes),
 "log_loss_mc": log_loss(m.truth, m[["a","b","c"]].to_numpy(), labels=["a","b","c"]),
 "accuracy_mc": accuracy_score(m.truth, np.array(["a","b","c"])[m[["a","b","c"]].to_numpy().argmax(1)]),
 "roc_auc_mc": roc_auc_score(m.truth, m[["a","b","c"]].to_numpy(), multi_class="ovo", average="macro", labels=["a","b","c"]),
}
py["roc_auc_mc_ovr"] = roc_auc_score(m.truth, m[["a","b","c"]].to_numpy(), multi_class="ovr", average="macro", labels=["a","b","c"])
print("== metrics ==")
print(f"{'metric':14s} {'R (yardstick)':>20s} {'Python (sklearn)':>20s} {'|diff|':>10s}")
for k, v in py.items():
    r = rm.get(k, np.nan)
    print(f"{k:14s} {r:20.15f} {v:20.15f} {abs(r-v):10.2e}")
