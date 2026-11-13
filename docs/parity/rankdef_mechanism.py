# Mechanism: what happens when a solver does NOT treat the tiny singular value as zero.
import numpy as np, scipy.linalg
from sharedimp import AnchorResidualizer
i = np.arange(1, 13)
X = np.column_stack([i.astype(float), (i % 4) + 0.5*i, np.array([2,1,4,3,6,5,8,7,10,9,12,11], float)])
X = np.column_stack([X, 3*X[:,0] - 2])
y = 2*X[:,0] - X[:,1] + 0.5*X[:,2] + np.array([0.3,-0.2,0.1,0.4,-0.5,0.2,-0.1,0.3,-0.4,0.1,0.2,-0.3])
labels = np.tile([1,2,3], 4)
eps = np.finfo(float).eps
print(f"machine eps = {eps:.3e}")
for anchor in (1, 2):
    print(f"\n=== anchor x{anchor+1}: effect of the rank decision on the Full-model test MSE ===")
    for k in (1, 2, 3):
        tr, te = labels != k, labels == k
        r = AnchorResidualizer(anchor=anchor).fit(X[tr])
        A_tr = np.column_stack([np.ones(tr.sum()), r.transform(X[tr])]); A_te = np.column_stack([np.ones(te.sum()), r.transform(X[te])])
        # centred design as LinearRegression(fit_intercept=True) solves it
        C_tr = A_tr[:, 1:] - A_tr[:, 1:].mean(0)
        sv_c = np.linalg.svd(C_tr, compute_uv=False)
        ratio = sv_c.min() / sv_c.max()
        res = {}
        for name, rcond in (("detected (rcond = eps)", eps), ("NOT detected (rcond = 1e-20)", 1e-20)):
            beta = np.linalg.lstsq(A_tr, y[tr], rcond=rcond)[0]
            pred = A_te @ beta
            res[name] = (np.mean((y[te]-pred)**2), np.max(np.abs(beta)))
        # scipy lstsq as used by LinearRegression (cond=None): different LAPACK driver
        beta_s = scipy.linalg.lstsq(A_tr, y[tr], cond=None)[0]; res["scipy lstsq cond=None"] = (np.mean((y[te]-A_te@beta_s)**2), np.max(np.abs(beta_s)))
        print(f"split {k}: smallest/largest singular value of the centred design = {ratio:.2e} ({'below' if ratio < eps else 'ABOVE'} eps)")
        for name, (m, b) in res.items():
            print(f"   {name:30s} mse_test = {m:.6f}   max|beta| = {b:.3g}")
