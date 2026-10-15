"""Prediction metrics: inverse-probability-of-censoring weighted Brier score and integrated Brier score
(Graf et al., 1999), and Harrell's concordance at a horizon."""
import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def kaplan_meier(time, event):
    """Kaplan-Meier curve as (distinct times, survival just after each time)."""
    time = np.asarray(time, float); event = np.asarray(event, float)
    order = np.argsort(time, kind="stable")
    u, d = time[order], event[order]
    times, first = np.unique(u, return_index=True)
    counts = np.add.reduceat(d, first)
    at_risk = len(u) - first
    return times, np.cumprod(1 - counts / at_risk)


def _km_value(curve, t, left=False):
    times, surv = curve
    ix = np.searchsorted(times, t, side="left" if left else "right") - 1
    return np.where(ix >= 0, surv[np.maximum(ix, 0)], 1.0)


def brier_scores(time_train, event_train, time_test, event_test, survival, grid, min_censoring_survival=0.05):
    """IPCW Brier score at each time in grid for predicted survival (n_test, len(grid)).

    The censoring distribution is the Kaplan-Meier estimate from the training sample. Entries whose
    censoring survival falls below ``min_censoring_survival`` are returned as NaN.
    """
    curve = kaplan_meier(time_train, 1 - np.asarray(event_train, float))
    u = np.asarray(time_test, float); dlt = np.asarray(event_test, float)
    G_u = _km_value(curve, u, left=True)
    out = np.zeros((len(u), len(grid)))
    for j, t in enumerate(grid):
        G_t = float(_km_value(curve, np.array([t]))[0])
        if G_t < min_censoring_survival:
            out[:, j] = np.nan
            continue
        died = (u <= t) & (dlt == 1)
        alive = (u > t) | ((u == t) & (dlt == 0))
        w_d = np.where(died, 1 / np.maximum(G_u, min_censoring_survival), 0.0)
        out[:, j] = w_d * survival[:, j] ** 2 + alive / G_t * (1 - survival[:, j]) ** 2
    return out


def integrated_brier_score(time_train, event_train, time_test, event_test, survival, grid, **kw):
    """Integrated Brier score over grid (trapezoid rule divided by the grid range), averaged over patients."""
    grid = np.asarray(grid, float)
    b = brier_scores(time_train, event_train, time_test, event_test, survival, grid, **kw)
    per_patient = _trapezoid(b, grid, axis=1) / (grid[-1] - grid[0])
    return float(np.mean(per_patient))


def concordance_index(time, event, risk, horizon=np.inf):
    """Harrell's C for comparable pairs with the earlier time an event before the horizon."""
    u = np.asarray(time, float); d = np.asarray(event, float); r = np.asarray(risk, float)
    pairs = (u[:, None] < u[None, :]) & (d[:, None] == 1) & (u[:, None] <= horizon)
    diff = r[:, None] - r[None, :]
    return float(np.sum(pairs * ((diff > 0) + 0.5 * (diff == 0))) / pairs.sum())
