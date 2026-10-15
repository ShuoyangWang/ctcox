"""Cross-fitted finite-dictionary projected score (PT-F) and the projected-minus-direct contrast."""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import norm
from .cox import fit_cox
from .risk import RiskSet

Z975 = norm.ppf(0.975)


@dataclass
class ProjectedResult:
    estimate: float
    se: float
    success: bool
    folds: list = field(default_factory=list)
    influence: np.ndarray = field(default=None, repr=False)

    def conf_int(self, level=0.95):
        z = norm.ppf(0.5 + level / 2)
        return self.estimate - z * self.se, self.estimate + z * self.se


def make_folds(n, n_folds=5, random_state=None):
    """Data-independent fold labels: a random permutation of (0, 1, ..., K-1, 0, 1, ...)."""
    rng = np.random.default_rng(random_state)
    return rng.permutation(np.arange(n) % n_folds)


def projected_score(time, event, treatment, offset, z, dictionary, folds=5, ridge=1e-3, ridge_reference=300,
                    clip=3.0, derivative_guard=1e-4, random_state=None):
    """Cross-fitted projected one-step estimator of the conditional log hazard ratio (equation 5 of the paper).

    On each training complement the direct calibrated fit (treatment and correction z, historical offset)
    gives a pilot index; the treatment indicator is projected onto the dictionary B under the
    event-weighted risk-set covariance, with ridge lambda = ridge * sqrt(ridge_reference / n_train) and
    clipping at +-clip; one Newton step of the projected score is taken on the held-out fold.

    Parameters
    ----------
    time, event, treatment, offset : arrays of shape (n,)
    z : array of shape (n, p)
        Calibration correction used for the pilot fits (the same as in the direct analysis).
    dictionary : array of shape (n, M)
        Dictionary B(X), whose span should contain z modulo constants. Generic spline dictionaries are
        available in :mod:`ctcox.dictionaries`.
    folds : int or array of shape (n,)
        Number of folds (labels drawn with ``make_folds``) or fixed fold labels 0, ..., K-1.

    Returns
    -------
    ProjectedResult
        Estimate, held-out influence standard error, per-fold diagnostics, and per-patient influence
        contributions psi_i = phi_i / (n D_k) for the contrast with the direct fit.
    """
    time = np.asarray(time, float); event = np.asarray(event, float)
    A = np.asarray(treatment, float); off = np.asarray(offset, float)
    n = len(time)
    Z = np.asarray(z, float).reshape(n, -1)
    B = np.asarray(dictionary, float).reshape(n, -1)
    labels = make_folds(n, folds, random_state) if np.isscalar(folds) else np.asarray(folds)
    W = np.column_stack([A, Z])
    est, var, diag = 0.0, 0.0, []
    psi = np.zeros(n)
    for k in np.unique(labels):
        ev = labels == k
        tr = ~ev
        pilot = fit_cox(time[tr], event[tr], W[tr], offset=off[tr])
        eta = off + W @ pilot.coef
        rs_tr = RiskSet(time[tr], event[tr])
        lam = ridge * np.sqrt(ridge_reference / tr.sum())
        S = rs_tr.covariance(eta[tr], B[tr], B[tr])
        c = rs_tr.covariance(eta[tr], B[tr], A[tr][:, None])[:, 0]
        alpha = np.linalg.solve(S + lam * np.eye(B.shape[1]), c)
        raw = B @ alpha
        q = A - np.clip(raw, -clip, clip)
        rs_ev = RiskSet(time[ev], event[ev])
        m = ev.sum()
        U = float(rs_ev.score(q[ev][:, None], eta[ev])[0])
        D = float(rs_ev.covariance(eta[ev], q[ev][:, None], A[ev][:, None])[0, 0])
        phi = rs_ev.influence(q[ev][:, None], eta[ev])[:, 0]
        V = float(np.mean(phi ** 2))
        ok = bool(pilot.converged and D > derivative_guard and np.isfinite(V) and V > 0)
        diag.append(dict(fold=int(k), n_train=int(tr.sum()), n_eval=int(m), pilot_theta=float(pilot.coef[0]),
                         score=U, derivative=D, variance=V, ridge=float(lam),
                         clipped_fraction=float(np.mean(np.abs(raw[tr]) > clip)), success=ok))
        if not ok:
            return ProjectedResult(np.nan, np.nan, False, diag, None)
        w = m / n
        est += w * (pilot.coef[0] + U / D)
        var += w ** 2 * V / (m * D ** 2)
        psi[ev] = phi / (n * D)
    return ProjectedResult(float(est), float(np.sqrt(var)), True, diag, psi)


def projected_minus_direct(direct, projected):
    """Standardized difference between the projected and direct treatment estimates (Corollary C.12).

    It checks whether the directional bias of the direct fit is negligible when the dictionary absorbs the
    remaining historical error. It is not a test of the treatment null, under which it degenerates.

    Parameters
    ----------
    direct : CalibratedCox fitted with a treatment variable on the same patients
    projected : ProjectedResult from :func:`projected_score`
    """
    if not projected.success:
        return dict(difference=np.nan, se=np.nan, z=np.nan, p_value=np.nan)
    diff = projected.estimate - direct.theta_
    se = float(np.sqrt(np.sum((projected.influence - direct.treatment_influence()) ** 2)))
    zval = diff / se if se > 0 else np.nan
    return dict(difference=float(diff), se=se, z=float(zval), p_value=float(2 * norm.sf(abs(zval))))
