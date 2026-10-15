"""Generic projection dictionaries and correction bases: natural cubic splines and pairwise products.

The dictionaries are fitted on a reference sample of covariates (for example, the historical covariates or
the pooled covariates without outcomes), so that they are fixed independently of target outcomes.
"""
import numpy as np


def natural_spline(x, knots):
    """Natural cubic spline basis without intercept: x and K - 2 truncated-power terms (Hastie et al., 2009)."""
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    K = len(k)
    if K < 3:
        return x[:, None]

    def d(j):
        return (np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[-1], 0) ** 3) / (k[-1] - k[j])

    return np.column_stack([x] + [d(j) - d(K - 2) for j in range(K - 2)])


class SplineDictionary:
    """Additive natural cubic splines for continuous covariates, binary covariates unchanged, and optional
    pairwise products of the original covariates, centered and whitened on a reference sample.

    Parameters
    ----------
    n_knots : int
        Number of knots per continuous covariate, placed at equally spaced quantiles including the minimum
        and maximum (4 knots give x and two nonlinear terms).
    products : bool
        Add the products x_j x_k for all pairs j < k.
    whiten : bool
        Center and whiten under the reference covariance. Whitening changes coordinates, not the span,
        but it fixes the scale on which the projection ridge acts.
    """

    def __init__(self, n_knots=4, products=True, whiten=True):
        self.n_knots = n_knots
        self.products = products
        self.whiten = whiten

    def _raw(self, X):
        X = np.asarray(X, float)
        cols = []
        for j in range(X.shape[1]):
            if self.binary_[j]:
                cols.append(X[:, [j]])
            else:
                cols.append(natural_spline(X[:, j], self.knots_[j]))
        if self.products:
            d = X.shape[1]
            cols += [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
        return np.column_stack(cols)

    def fit(self, X_reference):
        X = np.asarray(X_reference, float)
        self.binary_ = [len(np.unique(X[:, j])) <= 2 for j in range(X.shape[1])]
        qs = np.linspace(0, 1, self.n_knots)
        self.knots_ = [None if self.binary_[j] else np.quantile(X[:, j], qs) for j in range(X.shape[1])]
        R = self._raw(X)
        self.mean_ = R.mean(0)
        if self.whiten:
            C = np.cov(R.T)
            val, vec = np.linalg.eigh(C)
            keep = val > 1e-10 * val.max()
            self.transform_ = vec[:, keep] / np.sqrt(val[keep])
        else:
            self.transform_ = np.eye(R.shape[1])
        return self

    def transform(self, X):
        return (self._raw(X) - self.mean_) @ self.transform_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def dimension(self):
        return self.transform_.shape[1]


def linear_dictionary(X):
    """The linear correction z(X) = X itself, the default calibration correction."""
    return np.asarray(X, float)
