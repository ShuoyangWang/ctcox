"""Risk-set computations for the Cox partial likelihood with Breslow handling of ties.

All public methods take and return arrays in the caller's original observation order.
Losses, gradients and information matrices are on the mean scale (divided by n).
"""
import numpy as np


class RiskSet:
    """Risk sets of a right-censored sample.

    Parameters
    ----------
    time : array of shape (n,)
        Observed times U_i = min(T_i, C_i, tau).
    event : array of shape (n,)
        Event indicators Delta_i (1 = event, 0 = censored).
    """

    def __init__(self, time, event):
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=float)
        if time.ndim != 1 or time.shape != event.shape:
            raise ValueError("time and event must be one-dimensional arrays of equal length")
        self.n = len(time)
        self.order = np.argsort(time, kind="stable")
        self.inverse = np.empty_like(self.order)
        self.inverse[self.order] = np.arange(self.n)
        self.u = time[self.order]
        self.d = event[self.order]
        # first index tied with u_i (start of its risk set) and last index tied with u_i
        self.first = np.searchsorted(self.u, self.u, side="left")
        self.last = np.searchsorted(self.u, self.u, side="right") - 1

    # ------------------------------------------------------------------ internals
    def _sorted(self, v):
        return np.asarray(v, dtype=float)[self.order]

    def _unsort(self, v):
        return v[self.inverse]

    def _weights(self, r_sorted):
        shift = np.max(r_sorted)
        w = np.exp(r_sorted - shift)
        den = np.cumsum(w[::-1])[::-1][self.first]
        inc = self.d / den
        total = np.cumsum(inc)[self.last]
        return w, den, inc, total, shift

    def _means(self, design_sorted, w, den):
        return np.cumsum((w[:, None] * design_sorted)[::-1], axis=0)[::-1][self.first] / den[:, None]

    # ------------------------------------------------------------------ likelihood
    def loss_grad(self, r):
        """Mean negative log partial likelihood at linear predictor r and its gradient in r."""
        rs = self._sorted(r)
        w, den, _, total, shift = self._weights(rs)
        loss = self.d @ (np.log(den) - (rs - shift)) / self.n
        grad = (w * total - self.d) / self.n
        return float(loss), self._unsort(grad)

    def information(self, design, r):
        """Mean-scale negative Hessian J_n(r, W, W) of the log partial likelihood for design W."""
        Ws = self._sorted(design).reshape(self.n, -1)
        w, den, _, total, _ = self._weights(self._sorted(r))
        means = self._means(Ws, w, den)
        h = (Ws.T @ (Ws * (w * total)[:, None]) - means.T @ (means * self.d[:, None])) / self.n
        return (h + h.T) / 2

    def covariance(self, r, U, V):
        """Mean-scale event-weighted risk-set covariance J_n(r, U, V) of shape (p, q)."""
        Us = self._sorted(U).reshape(self.n, -1)
        Vs = self._sorted(V).reshape(self.n, -1)
        w, den, _, total, _ = self._weights(self._sorted(r))
        mu = self._means(Us, w, den)
        mv = self._means(Vs, w, den)
        return (Us.T @ (Vs * (w * total)[:, None]) - mu.T @ (mv * self.d[:, None])) / self.n

    def influence(self, design, r):
        """Per-patient influence contributions of the score in direction W, centered, shape (n, p).

        They include the correction for estimated risk-set means, so that
        (I^{-1} phi^T phi I^{-1}) / n^2 is the robust sandwich covariance of the coefficients.
        """
        Ws = self._sorted(design).reshape(self.n, -1)
        w, den, inc, total, _ = self._weights(self._sorted(r))
        means = self._means(Ws, w, den)
        comp = np.cumsum(inc[:, None] * means, axis=0)[self.last]
        phi = self.d[:, None] * (Ws - means) - w[:, None] * (Ws * total[:, None] - comp)
        phi = phi - phi.mean(axis=0)
        return phi[self.inverse]

    def score(self, design, r):
        """Mean score U_n(r, W) = -W^T grad of the mean loss."""
        _, g = self.loss_grad(r)
        return -np.asarray(design, dtype=float).reshape(self.n, -1).T @ g

    # ------------------------------------------------------------------ baseline
    def cumulative_hazard(self, r, times):
        """Breslow estimate of the cumulative baseline hazard at the given times."""
        w, den, inc, _, shift = self._weights(self._sorted(r))
        cumulative = np.cumsum(inc) * np.exp(-shift)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        ix = np.searchsorted(self.u, times, side="right") - 1
        return np.where(ix >= 0, cumulative[np.maximum(ix, 0)], 0.0)

    def martingale_residuals(self, r):
        """Delta_i - exp(r_i) * Lambda_hat(U_i) under the Breslow estimate."""
        w, _, _, total, _ = self._weights(self._sorted(r))
        return self._unsort(self.d - w * total)

    def log_partial_likelihood(self, r):
        """Summed log partial likelihood (Breslow ties)."""
        return -self.n * self.loss_grad(r)[0]
