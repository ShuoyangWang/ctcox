"""Choice among prespecified corrections within the target training data (Corollary 4.3 of the paper)."""
import numpy as np
from .calibrate import CalibratedCox
from .metrics import integrated_brier_score


def stratified_folds(event, n_folds, random_state=None, strata=None):
    """Fold labels balanced by event status (and optional strata such as treatment arm)."""
    event = np.asarray(event).astype(int)
    s = event if strata is None else 2 * np.asarray(strata).astype(int) + event
    rng = np.random.default_rng(random_state)
    labels = np.empty(len(event), dtype=int)
    for cell in np.unique(s):
        ix = np.flatnonzero(s == cell)
        rng.shuffle(ix)
        for j, part in enumerate(np.array_split(ix, n_folds)):
            labels[part] = j
    return labels


def select_correction(time, event, offset, candidates, grid, treatment=None, n_folds=3, random_state=None):
    """Select a calibration correction by inner cross-validated integrated Brier score.

    The candidate with the fewest coefficients among those whose mean IBS is within one standard error of
    the minimum is chosen (the parsimony rule used in the paper's applications). The selection preserves the
    prediction rate (Corollary 4.3), but treatment intervals after selection are not covered by the theory,
    so the correction for treatment analysis should be fixed in advance.

    Parameters
    ----------
    candidates : dict name -> array (n, p_l) or None
        Prespecified correction coordinates. Use the offset itself as a one-column correction for free
        score rescaling.

    Returns
    -------
    (selected name, table of mean IBS, SE and dimension per candidate)
    """
    n = len(time)
    labels = stratified_folds(event, n_folds, random_state, treatment)
    rows = {}
    for name, z in candidates.items():
        losses = []
        for k in range(n_folds):
            ev = labels == k; tr = ~ev
            zt = None if z is None else np.asarray(z)[tr]
            ze = None if z is None else np.asarray(z)[ev]
            at = None if treatment is None else np.asarray(treatment)[tr]
            ae = None if treatment is None else np.asarray(treatment)[ev]
            m = CalibratedCox().fit(time[tr], event[tr], offset[tr], zt, at)
            s = m.predict_survival(grid, offset[ev], ze, ae)
            losses.append(integrated_brier_score(time[tr], event[tr], time[ev], event[ev], s, grid)
                          if m.converged_ else np.nan)
        losses = np.array(losses)
        dim = 0 if z is None else np.asarray(z).reshape(n, -1).shape[1]
        rows[name] = dict(ibs=float(np.mean(losses)), se=float(np.std(losses, ddof=1) / np.sqrt(n_folds)),
                          dimension=dim, valid=bool(np.all(np.isfinite(losses))))
    valid = {k: v for k, v in rows.items() if v["valid"]}
    if not valid:
        return None, rows
    best = min(valid, key=lambda k: valid[k]["ibs"])
    threshold = valid[best]["ibs"] + valid[best]["se"]
    eligible = [k for k, v in valid.items() if v["ibs"] <= threshold]
    selected = min(eligible, key=lambda k: (valid[k]["dimension"], valid[k]["ibs"]))
    return selected, rows
