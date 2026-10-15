"""Direct calibrated transfer (CT): a fixed historical log risk enters the target partial likelihood as an
offset, with a prespecified correction z(X), a separate target baseline hazard and, in a trial, a
conditional treatment coefficient."""
import numpy as np
from scipy.stats import norm
from .cox import fit_cox
from .risk import RiskSet

Z975 = norm.ppf(0.975)


def _design(z, treatment, n):
    z = np.zeros((n, 0)) if z is None else np.asarray(z, dtype=float).reshape(n, -1)
    if treatment is None:
        return z
    return np.column_stack([np.asarray(treatment, dtype=float), z])


class CalibratedCox:
    """Direct calibrated transfer of a historical Cox risk function.

    The target model is lambda(t | a, x) = lambda_0(t) exp{theta a + f_hat(x) + gamma^T z(x)}, where
    f_hat is held fixed. Calibration removes historical error in span{1, z} exactly
    (Proposition 4.1 of the paper).

    Parameters
    ----------
    ridge : float
        Optional ridge penalty on the correction coefficients (0 gives the unpenalized fit studied in the paper;
        a positive value breaks exact calibration invariance).
    """

    def __init__(self, ridge=0.0):
        self.ridge = ridge

    def fit(self, time, event, offset, z=None, treatment=None):
        """Fit the calibrated model.

        Parameters
        ----------
        time, event : arrays of shape (n,)
        offset : array of shape (n,)
            Historical log risk f_hat(X_i), estimated from historical data only.
        z : array of shape (n, p), optional
            Correction coordinates, fixed before seeing target outcomes. None gives frozen transfer.
        treatment : array of shape (n,), optional
            Binary treatment indicator in a randomized trial. Its coefficient is unpenalized.
        """
        n = len(time)
        self.has_treatment_ = treatment is not None
        W = _design(z, treatment, n)
        pen = np.full(W.shape[1], float(self.ridge))
        if self.has_treatment_:
            pen[0] = 0.0
        self.time_, self.event_ = np.asarray(time, float), np.asarray(event, float)
        self.risk_ = RiskSet(time, event)
        self.fit_ = fit_cox(time, event, W, offset=offset, ridge=pen, risk=self.risk_)
        self.coef_ = self.fit_.coef
        self.covariance_ = self.fit_.covariance
        self.eta_ = self.fit_.eta
        self.design_ = W
        self.converged_ = self.fit_.converged
        return self

    # ------------------------------------------------------------------ treatment inference
    @property
    def theta_(self):
        self._check_treatment()
        return float(self.coef_[0])

    @property
    def se_(self):
        """Robust sandwich standard error of the treatment coefficient."""
        self._check_treatment()
        return float(np.sqrt(self.covariance_[0, 0]))

    def conf_int(self, level=0.95):
        z = norm.ppf(0.5 + level / 2)
        return self.theta_ - z * self.se_, self.theta_ + z * self.se_

    def treatment_influence(self):
        """Per-patient influence psi_i of theta_hat, with theta_hat - theta ~ sum_i psi_i (Corollary C.12)."""
        self._check_treatment()
        inv = np.linalg.inv(self.fit_.information + np.diag(np.r_[0.0, np.full(self.design_.shape[1] - 1, self.ridge)]))
        return (self.fit_.influence @ inv.T)[:, 0] / len(self.time_)

    def _check_treatment(self):
        if not self.has_treatment_:
            raise AttributeError("the model was fitted without a treatment variable")

    # ------------------------------------------------------------------ prediction
    def cumulative_baseline_hazard(self, times):
        """Breslow estimate of the target cumulative baseline hazard."""
        return self.risk_.cumulative_hazard(self.eta_, times)

    def linear_predictor(self, offset, z=None, treatment=None):
        n = len(offset)
        W = _design(z, treatment if self.has_treatment_ else None, n)
        if self.has_treatment_ and treatment is None:
            raise ValueError("treatment values are required for prediction from a trial fit")
        return np.asarray(offset, float) + W @ self.coef_

    def predict_survival(self, times, offset, z=None, treatment=None):
        """Predicted survival S(t | x) (or S_a(t | x) in a trial) at the given times, shape (n_new, n_times)."""
        H = self.cumulative_baseline_hazard(times)
        eta = self.linear_predictor(offset, z, treatment)
        return np.exp(-np.exp(eta)[:, None] * H[None, :])
