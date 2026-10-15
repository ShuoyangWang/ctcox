"""Label-blind calibrated randomization (CR) test of H0: A independent of (X, U, Delta)."""
import numpy as np
from .cox import fit_cox
from .risk import RiskSet


def randomization_test(time, event, treatment, offset, z=None, n_draws=999, random_state=None):
    """Calibrated randomization test (equation 6 of the paper).

    The correction is fitted with theta = 0, using only (X, U, Delta) and the historical offset, so that no
    fitting step uses the treatment labels. The statistic |sum_i (a_i - p_hat) R_i| uses the null martingale
    residuals R_i, and its randomization distribution draws assignments uniformly with the observed number
    treated. The test is valid in finite samples under complete randomization with a fixed number treated,
    whether or not the transported prognosis is correct. Other allocation designs need their own law.

    Returns
    -------
    dict with the statistic, the p value (1 + #{T_b >= T}) / (B + 1), and the null residuals.
    """
    A = np.asarray(treatment, float)
    n = len(A)
    Z = np.zeros((n, 0)) if z is None else np.asarray(z, float).reshape(n, -1)
    rs = RiskSet(time, event)
    null = fit_cox(time, event, Z, offset=offset, risk=rs)
    R = rs.martingale_residuals(null.eta)
    p_hat = A.mean()
    stat = abs(np.sum((A - p_hat) * R))
    rng = np.random.default_rng(random_state)
    m = int(A.sum())
    count = 0
    for _ in range(n_draws):
        a = np.zeros(n)
        a[rng.choice(n, m, replace=False)] = 1.0
        count += abs(np.sum((a - p_hat) * R)) >= stat - 1e-12
    return dict(statistic=float(stat), p_value=(1 + count) / (n_draws + 1), residuals=R,
                null_fit_converged=null.converged)
