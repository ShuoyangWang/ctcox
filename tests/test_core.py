import numpy as np
import pytest
import ctcox
from ctcox.risk import RiskSet


def simulate_trial(n, rng, theta=np.log(0.7), drift=0.5):
    X = rng.uniform(-1, 1, (n, 3))
    A = rng.binomial(1, 0.5, n).astype(float)
    f0 = np.sin(np.pi * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]
    g0 = f0 + 0.3 * X[:, 1] + drift * np.cos(np.pi * X[:, 2])
    T = rng.exponential(1, n) * np.exp(-(theta * A + g0)) / 0.5
    C = rng.exponential(2.0, n)
    U = np.minimum(np.minimum(T, C), 3.0)
    D = ((T <= C) & (T <= 3.0)).astype(float)
    return X, A, U, D, f0


def test_gradient_and_information_with_ties():
    rng = np.random.default_rng(1)
    n = 60
    U = np.round(rng.exponential(1, n), 1)  # many ties
    D = rng.binomial(1, 0.7, n).astype(float)
    W = rng.normal(size=(n, 3))
    rs = RiskSet(U, D)
    b = np.array([0.2, -0.1, 0.3])
    loss, g = rs.loss_grad(W @ b)
    grad = W.T @ g
    eps = 1e-6
    num = np.array([(rs.loss_grad(W @ (b + eps * e))[0] - rs.loss_grad(W @ (b - eps * e))[0]) / (2 * eps)
                    for e in np.eye(3)])
    assert np.allclose(grad, num, atol=1e-7)
    H = rs.information(W, W @ b)
    numH = np.array([(W.T @ rs.loss_grad(W @ (b + eps * e))[1] - W.T @ rs.loss_grad(W @ (b - eps * e))[1]) / (2 * eps)
                     for e in np.eye(3)])
    assert np.allclose(H, numH, atol=1e-6)


def test_cox_matches_statsmodels_breslow():
    sm = pytest.importorskip("statsmodels.duration.hazard_regression")
    rng = np.random.default_rng(2)
    n = 300
    X = rng.normal(size=(n, 2))
    U = np.round(rng.exponential(1, n) * np.exp(-X @ [0.5, -0.3]), 2)
    D = rng.binomial(1, 0.8, n).astype(float)
    ours = ctcox.fit_cox(U, D, X)
    ref = sm.PHReg(U, X, status=D, ties="breslow").fit()
    assert np.allclose(ours.coef, ref.params, atol=1e-6)


def test_calibration_invariance():
    """Proposition 4.1: f -> f + c + b'z leaves theta and survival predictions unchanged."""
    rng = np.random.default_rng(3)
    X, A, U, D, f0 = simulate_trial(400, rng)
    m1 = ctcox.CalibratedCox().fit(U, D, f0, X, A)
    f2 = f0 + 1.7 + X @ np.array([0.4, -0.8, 0.25])
    m2 = ctcox.CalibratedCox().fit(U, D, f2, X, A)
    assert abs(m1.theta_ - m2.theta_) < 1e-8
    Xn = rng.uniform(-1, 1, (20, 3)); An = rng.binomial(1, 0.5, 20)
    fn0 = np.sin(np.pi * Xn[:, 0]) + 0.5 * Xn[:, 1] * Xn[:, 2]
    s1 = m1.predict_survival([0.5, 1.0, 2.0], fn0, Xn, An)
    s2 = m2.predict_survival([0.5, 1.0, 2.0], fn0 + 1.7 + Xn @ np.array([0.4, -0.8, 0.25]), Xn, An)
    assert np.allclose(s1, s2, atol=1e-8)


def test_projected_reduces_bias_when_error_in_dictionary():
    """Drift cos(pi x3) lies outside the linear correction but inside the spline dictionary."""
    rng = np.random.default_rng(4)
    X, A, U, D, f0 = simulate_trial(20000, rng, drift=1.5)
    theta = np.log(0.7)
    ct = ctcox.CalibratedCox().fit(U, D, f0, X, A)
    B = ctcox.SplineDictionary(n_knots=5).fit_transform(X)
    pt = ctcox.projected_score(U, D, A, f0, X, B, folds=5, random_state=0)
    assert pt.success
    assert abs(pt.estimate - theta) < abs(ct.theta_ - theta)
    con = ctcox.projected_minus_direct(ct, pt)
    assert np.isfinite(con["z"]) and con["se"] > 0


def test_randomization_test_bounds():
    rng = np.random.default_rng(5)
    X, A, U, D, f0 = simulate_trial(200, rng, theta=0.0)
    out = ctcox.randomization_test(U, D, A, f0, X, n_draws=199, random_state=0)
    assert 1 / 200 <= out["p_value"] <= 1.0
    assert abs(out["residuals"].sum()) < 1e-6  # Breslow martingale residuals always sum to zero


def test_adapted_transcox_limits():
    rng = np.random.default_rng(6)
    X, A, U, D, f0 = simulate_trial(300, rng)
    W = np.column_stack([A, X])
    bH = np.array([0.5, 0.2, -0.1])
    # lambda = 0 reproduces the target-only Cox fit
    b0, kkt = ctcox.lasso_cox(U, D, W, X @ bH, np.zeros(4))
    ref = ctcox.fit_cox(U, D, W)
    assert np.allclose(b0 + np.r_[0, bH], ref.coef, atol=1e-6)
    # very large penalty keeps the source coefficients
    b1, _ = ctcox.lasso_cox(U, D, W, X @ bH, np.r_[0, np.full(3, 1e6)])
    assert np.allclose(b1[1:], 0)
    m = ctcox.AdaptedTransCox(random_state=0).fit(U, D, X, bH, A)
    assert m.kkt_ < 1e-6 and np.isfinite(m.theta_)


def test_dictionary_whitening_and_linearity():
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, (5000, 2))
    d = ctcox.SplineDictionary(n_knots=4, products=True).fit(X)
    B = d.transform(X)
    assert np.allclose(np.cov(B.T), np.eye(B.shape[1]), atol=1e-8)
    # natural splines are linear beyond the boundary knots
    x = np.linspace(1.1, 2.0, 5)
    N = ctcox.natural_spline(x, [-1, -0.3, 0.3, 1])
    assert np.allclose(np.diff(N, 2, axis=0), 0, atol=1e-10)


def test_neural_cox_bounded():
    rng = np.random.default_rng(8)
    X = rng.uniform(-1, 1, (400, 3))
    T = rng.exponential(1, 400) * np.exp(-np.sin(np.pi * X[:, 0]))
    D = np.ones(400)
    net = ctcox.NeuralCox(steps=60, seed=1).fit(X, T, D)
    f = net.predict(X)
    assert np.all(np.abs(f) <= 4.0) and np.isfinite(f).all()


def test_selection_returns_candidate():
    rng = np.random.default_rng(9)
    X, A, U, D, f0 = simulate_trial(400, rng)
    cand = {"scale": f0[:, None], "linear": X}
    sel, table = ctcox.select_correction(U, D, f0, cand, grid=np.linspace(0.3, 2.0, 8), treatment=A, random_state=0)
    assert sel in cand and all(np.isfinite(v["ibs"]) for v in table.values())
