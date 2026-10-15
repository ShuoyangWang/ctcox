"""End-to-end example: learn prognosis in a historical cohort, calibrate it in a smaller randomized trial,
and report prediction and treatment analyses.

Run from the repository root:  python examples/simulated_trial.py
"""
import numpy as np
import ctcox

rng = np.random.default_rng(2026)
d = 4


def base_risk(X):
    return np.sin(np.pi * X[:, 0]) + 0.6 * np.cos(np.pi * X[:, 1]) + 0.5 * X[:, 2] * X[:, 3]


# Historical cohort: background treatment only, N = 3000
N = 3000
XH = rng.uniform(-1, 1, (N, d))
TH = (rng.exponential(1, N) * np.exp(-base_risk(XH)) / 0.28) ** (1 / 1.4)
CH = rng.exponential(1 / 0.08, N)
UH = np.minimum(np.minimum(TH, CH), 2.0)
DH = ((TH <= CH) & (TH <= 2.0)).astype(float)

# Target trial: n = 600, changed linear effects (in the correction span) and a drift outside it
n, theta0 = 600, np.log(0.7)
X = rng.uniform(-1, 1, (n, d))
A = rng.binomial(1, 0.5, n).astype(float)
g0 = base_risk(X) + 0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.8 * np.sin(np.pi * X[:, 1])
T = (rng.exponential(1, n) * np.exp(-(theta0 * A + g0)) / 0.35) ** (1 / 1.2)
C = rng.exponential(1 / 0.18, n)
U = np.minimum(np.minimum(T, C), 2.0)
D = ((T <= C) & (T <= 2.0)).astype(float)

# 1. Historical learning (historical data only), then hold f_hat fixed
net = ctcox.NeuralCox(seed=1).fit(XH, UH, DH)
f_hat = net.predict(X)

# 2. Direct calibrated transfer with the linear correction z = X
ct = ctcox.CalibratedCox().fit(U, D, f_hat, z=X, treatment=A)
lo, hi = ct.conf_int()
print(f"CT  log HR {ct.theta_:.3f}  (95% CI {lo:.3f}, {hi:.3f}),  HR {np.exp(ct.theta_):.3f}")

# 3. Projected score with a generic spline dictionary fixed without target outcomes
dictionary = ctcox.SplineDictionary(n_knots=4, products=True).fit(np.vstack([XH, X]))
B = dictionary.transform(X)
pt = ctcox.projected_score(U, D, A, f_hat, X, B, folds=5, random_state=0)
lo, hi = pt.conf_int()
print(f"PT  log HR {pt.estimate:.3f}  (95% CI {lo:.3f}, {hi:.3f}),  dictionary size {B.shape[1]}")

# 4. Robustness check: standardized projected-minus-direct difference
con = ctcox.projected_minus_direct(ct, pt)
print(f"PT - CT  {con['difference']:.3f}  (SE {con['se']:.3f}, z {con['z']:.2f}, p {con['p_value']:.3f})")

# 5. Label-blind randomization test of no treatment effect
cr = ctcox.randomization_test(U, D, A, f_hat, X, n_draws=999, random_state=0)
print(f"Calibrated randomization test p = {cr['p_value']:.3f}")

# 6. Survival prediction for new control patients
Xnew = rng.uniform(-1, 1, (3, d))
S = ct.predict_survival([0.5, 1.0, 1.5], net.predict(Xnew), z=Xnew, treatment=np.zeros(3))
print("Predicted control survival at t = 0.5, 1.0, 1.5:\n", np.round(S, 3))

# 7. Comparator: adapted TransCox-type coefficient transfer toward a historical linear Cox fit
src = ctcox.fit_cox(UH, DH, XH)
tc = ctcox.AdaptedTransCox(random_state=0).fit(U, D, X, src.coef, treatment=A)
print(f"Adapted TransCox-type log HR {tc.theta_:.3f}, lambda / lambda_max = {tc.lambda_ / tc.lambda_max_:.3f}")
