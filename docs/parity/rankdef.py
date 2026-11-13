# Rank deficiency study: parity fixture + exactly collinear column x1_dup = 3*x1 - 2.
# For anchors x2 and x3 the residualized x1 and x1_dup are mutually collinear
# (r_dup = 3 r_x1 up to round-off). Compare OLS solvers on the Full design.
import numpy as np, scipy, sklearn
from sklearn.linear_model import LinearRegression
from sharedimp import AnchorResidualizer
np.set_printoptions(precision=15)
i = np.arange(1, 13)
X = np.column_stack([i.astype(float), (i % 4) + 0.5*i, np.array([2,1,4,3,6,5,8,7,10,9,12,11], float)])
X = np.column_stack([X, 3*X[:,0] - 2])  # x1_dup
y = 2*X[:,0] - X[:,1] + 0.5*X[:,2] + np.array([0.3,-0.2,0.1,0.4,-0.5,0.2,-0.1,0.3,-0.4,0.1,0.2,-0.3])
labels = np.tile([1,2,3], 4)
print("numpy", np.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__)
def mse(a, b): return float(np.mean((a-b)**2))
for anchor in (1, 2):  # x2, x3
    print(f"\n=== anchor x{anchor+1} ===")
    for k in (1, 2, 3):
        tr, te = labels != k, labels == k
        r = AnchorResidualizer(anchor=anchor).fit(X[tr])
        F_tr, F_te = r.transform(X[tr]), r.transform(X[te])
        # mutual collinearity of residualized x1 and x1_dup
        dev = np.max(np.abs(F_tr[:,3] - (3*F_tr[:,0])))
        A_tr = np.column_stack([np.ones(tr.sum()), F_tr]); A_te = np.column_stack([np.ones(te.sum()), F_te])
        sv = np.linalg.svd(A_tr, compute_uv=False)
        out = {}
        for name, rcond in (("lstsq rcond=None (eps*max(M,N))", None), ("lstsq rcond=-1 (old default, machine eps)", -1), ("lstsq rcond=1e-7 (R lm.fit tol)", 1e-7)):
            beta = np.linalg.lstsq(A_tr, y[tr], rcond=rcond)[0]
            out[name] = (A_te @ beta, beta)
        lr = LinearRegression().fit(F_tr, y[tr]); out["sklearn LinearRegression"] = (lr.predict(F_te), np.r_[lr.intercept_, lr.coef_])
        # explicit rank-revealing solve: drop column 4 (x1_dup) when collinear -> same column space
        beta_drop = np.linalg.lstsq(A_tr[:, :4], y[tr], rcond=None)[0]; out["drop x1_dup (rank-revealing)"] = (A_te[:, :4] @ beta_drop, beta_drop)
        ref = out["drop x1_dup (rank-revealing)"][0]
        print(f"split {k}: |r_dup - 3 r_x1| max = {dev:.1e}; singular values = {sv.round(3)}; min/max = {sv.min()/sv.max():.1e}")
        for name, (pred, beta) in out.items():
            print(f"   {name:44s} mse_test = {mse(y[te], pred):.12f}  max|pred - ref| = {np.max(np.abs(pred-ref)):.1e}  max|beta| = {np.max(np.abs(beta)):.3g}")
