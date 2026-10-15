"""Cox regression with a fixed offset, by Newton steps with step halving."""
from dataclasses import dataclass, field
import numpy as np
from .risk import RiskSet


@dataclass
class CoxFit:
    """Result of :func:`fit_cox`. Information and covariance use the mean (1/n) scale for J."""
    coef: np.ndarray
    information: np.ndarray
    covariance: np.ndarray
    eta: np.ndarray
    converged: bool
    gradient_norm: float
    iterations: int
    message: str = ""
    influence: np.ndarray = field(default=None, repr=False)

    @property
    def se(self):
        return np.sqrt(np.diag(self.covariance))


def fit_cox(time, event, design, offset=None, ridge=0.0, beta0=None, max_iter=100, tol=1e-10,
            coef_bound=None, risk=None):
    """Maximize the log partial likelihood of a Cox model with linear predictor offset + design @ beta.

    Parameters
    ----------
    time, event : arrays of shape (n,)
    design : array of shape (n, p)
    offset : array of shape (n,), optional
        A fixed historical log risk f_hat(X), which receives coefficient one.
    ridge : float or array of shape (p,)
        Optional penalty 0.5 * sum(ridge * beta**2) added to the mean negative log partial likelihood.
    coef_bound : float, optional
        If given, the fit is flagged as not converged when any |beta_j| reaches this bound.

    Returns
    -------
    CoxFit
        Coefficients, mean-scale information J_n, robust sandwich covariance, linear predictor and
        per-patient influence contributions (used for contrasts).
    """
    design = np.asarray(design, dtype=float)
    n = len(time)
    design = design.reshape(n, -1)
    p = design.shape[1]
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    rs = RiskSet(time, event) if risk is None else risk
    pen = np.broadcast_to(np.asarray(ridge, dtype=float), (p,))
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float).copy()

    def objective(b):
        loss, g = rs.loss_grad(off + design @ b)
        return loss + 0.5 * np.sum(pen * b * b), design.T @ g + pen * b

    if p == 0:
        eta = off.copy()
        return CoxFit(beta, np.zeros((0, 0)), np.zeros((0, 0)), eta, True, 0.0, 0, influence=np.zeros((n, 0)))

    cur, grad = objective(beta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        H = rs.information(design, off + design @ beta) + np.diag(pen)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-10:
            nb = beta - t * step
            nv, ng = objective(nb)
            if np.isfinite(nv) and nv <= cur + 1e-15:
                break
            t /= 2
        beta, cur, grad = nb, nv, ng
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break
    eta = off + design @ beta
    info = rs.information(design, eta)
    gnorm = float(np.max(np.abs(grad)))
    msg = ""
    eig = np.linalg.eigvalsh(info + np.diag(pen))
    if eig[0] <= 1e-12:
        converged, msg = False, "singular information"
    if coef_bound is not None and np.max(np.abs(beta)) >= coef_bound:
        converged, msg = False, "coefficient bound reached"
    if gnorm > 1e-6:
        converged, msg = False, msg or "gradient not small"
    phi = rs.influence(design, eta)
    try:
        inv = np.linalg.inv(info + np.diag(pen))
        cov = inv @ (phi.T @ phi / n) @ inv / n
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    return CoxFit(beta, info, cov, eta, converged, gnorm, it, msg, influence=phi)
