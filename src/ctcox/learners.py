"""Bounded historical learners for f_hat. Any learner with bounded output can be used instead: the
calibrated analysis only needs the fitted values f_hat(X) on target covariates."""
import numpy as np
from .cox import fit_cox
from .dictionaries import SplineDictionary
from .risk import RiskSet


class NeuralCox:
    """ReLU network d -> h1 -> h2 -> 1 with a linear skip connection, trained on the Cox partial likelihood.

    The output is clipped to [-bound, bound] and parameters to [-5, 5], so the learned log risk is bounded.
    Training uses Adam with an L2 penalty on the weights, stops at the step with the smallest partial
    likelihood loss on a validation split, and then refits on all patients for that number of steps.
    These are the defaults of the paper's simulations.
    """

    def __init__(self, hidden=(24, 12), penalty=1e-4, steps=500, learning_rate=0.006, validation_fraction=0.2,
                 check_every=10, bound=4.0, seed=0):
        self.hidden, self.penalty, self.steps = hidden, penalty, steps
        self.learning_rate, self.validation_fraction = learning_rate, validation_fraction
        self.check_every, self.bound, self.seed = check_every, bound, seed

    def _init(self, d):
        w1, w2 = self.hidden
        r = np.random.default_rng(self.seed)
        return {"W1": r.normal(0, np.sqrt(2 / d), (d, w1)), "b1": np.zeros(w1),
                "W2": r.normal(0, np.sqrt(2 / w1), (w1, w2)), "b2": np.zeros(w2),
                "w3": r.normal(0, np.sqrt(1 / w2), w2), "wl": np.zeros(d)}

    def _forward(self, P, X):
        h1p = X @ P["W1"] + P["b1"]; h1 = np.maximum(h1p, 0)
        h2p = h1 @ P["W2"] + P["b2"]; h2 = np.maximum(h2p, 0)
        raw = h2 @ P["w3"] + X @ P["wl"]
        return np.clip(raw, -self.bound, self.bound), (X, h1p, h1, h2p, h2, raw)

    def _grads(self, P, cache, g_out):
        X, h1p, h1, h2p, h2, raw = cache
        g = g_out * ((raw > -self.bound) & (raw < self.bound))
        G = {"w3": h2.T @ g, "wl": X.T @ g}
        d2 = np.outer(g, P["w3"]) * (h2p > 0); G["W2"] = h1.T @ d2; G["b2"] = d2.sum(0)
        d1 = (d2 @ P["W2"].T) * (h1p > 0); G["W1"] = X.T @ d1; G["b1"] = d1.sum(0)
        return G

    def _train(self, X, time, event, rows, n_steps, eval_rows=None):
        P = self._init(X.shape[1])
        rs = RiskSet(time[rows], event[rows])
        rsv = RiskSet(time[eval_rows], event[eval_rows]) if eval_rows is not None else None
        mom = {k: np.zeros_like(v) for k, v in P.items()}; vel = {k: np.zeros_like(v) for k, v in P.items()}
        best = (np.inf, 0)
        Xr = X[rows]
        for t in range(1, n_steps + 1):
            out, cache = self._forward(P, Xr)
            _, g = rs.loss_grad(out)
            G = self._grads(P, cache, g)
            for k in G:
                if k in ("W1", "W2", "w3", "wl"):
                    G[k] = G[k] + 2 * self.penalty * P[k]
                mom[k] = 0.9 * mom[k] + 0.1 * G[k]; vel[k] = 0.999 * vel[k] + 0.001 * G[k] ** 2
                mh = mom[k] / (1 - 0.9 ** t); vh = vel[k] / (1 - 0.999 ** t)
                P[k] = np.clip(P[k] - self.learning_rate * mh / (np.sqrt(vh) + 1e-8), -5, 5)
            if rsv is not None and t % self.check_every == 0:
                lv, _ = rsv.loss_grad(self._forward(P, X[eval_rows])[0])
                if lv < best[0]:
                    best = (lv, t)
        return P, best

    def fit(self, X, time, event):
        X = np.asarray(X, float); time = np.asarray(time, float); event = np.asarray(event, float)
        n = len(time)
        r = np.random.default_rng(self.seed + 7919)
        idx = r.permutation(n); nv = int(round(self.validation_fraction * n))
        va, tr = idx[:nv], idx[nv:]
        _, best = self._train(X, time, event, tr, self.steps, va)
        self.n_steps_ = max(best[1], self.check_every)
        self.params_, _ = self._train(X, time, event, np.arange(n), self.n_steps_)
        return self

    def predict(self, X):
        """Fitted historical log risk f_hat(X), bounded by +-bound."""
        return self._forward(self.params_, np.asarray(X, float))[0]


class SplineCox:
    """Historical Cox model on a generic spline dictionary, optionally ridge penalized, with clipped output."""

    def __init__(self, n_knots=4, products=True, ridge=0.0, bound=4.0):
        self.n_knots, self.products, self.ridge, self.bound = n_knots, products, ridge, bound

    def fit(self, X, time, event):
        self.basis_ = SplineDictionary(self.n_knots, self.products, whiten=True).fit(X)
        B = self.basis_.transform(X)
        self.fit_ = fit_cox(time, event, B, ridge=self.ridge)
        self.coef_ = self.fit_.coef
        return self

    def predict(self, X):
        return np.clip(self.basis_.transform(X) @ self.coef_, -self.bound, self.bound)
