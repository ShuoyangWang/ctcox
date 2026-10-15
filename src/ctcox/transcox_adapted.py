"""Adapted TransCox-type coefficient transfer, used as a comparator in the paper.

This is our adaptation, not the authors' TransCox software (Li, Shen and Ning, 2023, JASA 118, 2276-2287).
TransCox penalizes differences in both the regression coefficients and the baseline hazard with two lasso
penalties, solved jointly, and selects the penalties by BIC. The adaptation here keeps the lasso shrinkage
of the coefficient differences toward source coefficients, but

* estimates the target baseline hazard by Breslow's estimator without borrowing it,
* uses the partial likelihood with Breslow ties and a single penalty weighted by covariate SDs,
* chooses the penalty by five-fold cross-validated partial likelihood (Verweij and van Houwelingen, 1993),
* leaves an optional treatment coefficient unpenalized, and
* solves the problem by proximal Newton steps with coordinate descent along a warm-started path.
"""
import numpy as np
from .risk import RiskSet
from .selection import stratified_folds


def lasso_cox(time, event, design, offset, penalty, beta0=None, max_iter=200, tol=1e-9, risk=None):
    """Minimize mean negative log partial likelihood + sum_j penalty_j |beta_j| (penalty_j may be 0)."""
    rs = RiskSet(time, event) if risk is None else risk
    W = np.asarray(design, float); off = np.asarray(offset, float); lam = np.asarray(penalty, float)
    p = W.shape[1]
    b = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()

    def obj(v):
        return rs.loss_grad(off + W @ v)[0] + np.sum(lam * np.abs(v))

    cur = obj(b)
    for _ in range(max_iter):
        _, dr = rs.loss_grad(off + W @ b)
        g = W.T @ dr
        H = rs.information(W, off + W @ b) + 1e-10 * np.eye(p)
        beta = b.copy()
        for _ in range(500):
            old = beta.copy()
            for j in range(p):
                r = H[j, j] * beta[j] - (g[j] + H[j] @ (beta - b))
                beta[j] = np.sign(r) * max(abs(r) - lam[j], 0.0) / H[j, j]
            if np.max(np.abs(beta - old)) < 1e-12:
                break
        step = beta - b
        t = 1.0
        while t > 1e-8:
            nb = b + t * step
            nv = obj(nb)
            if np.isfinite(nv) and nv <= cur + 1e-15:
                break
            t /= 2
        b, cur = nb, nv
        if np.max(np.abs(t * step)) < tol:
            break
    _, dr = rs.loss_grad(off + W @ b)
    g = W.T @ dr
    nz = np.abs(b) > 1e-10
    kkt = float(np.max(np.r_[np.abs(g[nz] + lam[nz] * np.sign(b[nz])), np.maximum(np.abs(g[~nz]) - lam[~nz], 0), 0.0]))
    return b, kkt


class AdaptedTransCox:
    """Target coefficients b_source + zeta with an L1 penalty lambda * sum_j s_j |zeta_j|.

    Parameters
    ----------
    n_lambda : int
        Grid lambda_max * 10^(-3k/(n_lambda-1)), k = 0..n_lambda-1, plus lambda = 0 (which reproduces the
        target-only Cox fit).
    n_folds : int
        Folds for cross-validated partial likelihood, balanced by event status and treatment.
    """

    def __init__(self, n_lambda=12, n_folds=5, random_state=None):
        self.n_lambda, self.n_folds, self.random_state = n_lambda, n_folds, random_state

    def fit(self, time, event, X, source_coef, treatment=None):
        time = np.asarray(time, float); event = np.asarray(event, float); X = np.asarray(X, float)
        bH = np.asarray(source_coef, float)
        n = len(time)
        self.has_treatment_ = treatment is not None
        W = X if treatment is None else np.column_stack([np.asarray(treatment, float), X])
        k0 = 1 if self.has_treatment_ else 0
        off = X @ bH
        s = X.std(0); s[s < 1e-8] = 1.0

        def pen(l):
            return np.r_[np.zeros(k0), l * s]

        rs = RiskSet(time, event)
        big = np.r_[np.zeros(k0), np.full(X.shape[1], 1e12)]
        b0, _ = lasso_cox(time, event, W, off, big, risk=rs)
        _, dr = rs.loss_grad(off + W @ b0)
        g = W.T @ dr
        lmax = float(np.max(np.abs(g[k0:]) / s))
        grid = np.r_[lmax * 10 ** (-3 * np.arange(self.n_lambda) / (self.n_lambda - 1)), 0.0]
        labels = stratified_folds(event, self.n_folds, self.random_state, treatment)
        cv = np.zeros(len(grid))
        for k in range(self.n_folds):
            tr = labels != k
            rst = RiskSet(time[tr], event[tr]); bw = None
            for i, l in enumerate(grid):
                bw, _ = lasso_cox(time[tr], event[tr], W[tr], off[tr], pen(l), bw, risk=rst)
                cv[i] += rs.log_partial_likelihood(off + W @ bw) - rst.log_partial_likelihood(off[tr] + W[tr] @ bw)
        i = int(np.argmax(cv)); bw = None
        for l in grid[:i + 1]:
            bw, kkt = lasso_cox(time, event, W, off, pen(l), bw, risk=rs)
        self.lambda_, self.lambda_max_, self.grid_, self.cv_ = float(grid[i]), lmax, grid, cv
        self.deviation_ = bw[k0:]
        self.coef_ = bH + self.deviation_
        self.theta_ = float(bw[0]) if self.has_treatment_ else None
        self.kkt_ = kkt
        self.eta_ = off + W @ bw
        self.risk_ = rs
        return self

    def predict_survival(self, times, X, treatment=None):
        H = self.risk_.cumulative_hazard(self.eta_, times)
        eta = np.asarray(X, float) @ self.coef_
        if self.has_treatment_:
            if treatment is None:
                raise ValueError("treatment values are required for prediction from a trial fit")
            eta = eta + self.theta_ * np.asarray(treatment, float)
        return np.exp(-np.exp(eta)[:, None] * H[None, :])
