# ctcox: calibrated transfer learning for Cox models

`ctcox` implements the methods of *Calibrated Transfer Learning for Survival Prediction and Treatment
Inference*. A historical log-risk function, learned by any bounded learner, enters the target Cox partial
likelihood as a fixed offset. The target analysis estimates a prespecified correction for covariate effects that may change, a
separate baseline hazard and, in a randomized trial, a conditional treatment coefficient. Historical
patients never enter target risk sets.

The package depends only on NumPy and SciPy.

| Paper | Function |
|---|---|
| Direct calibrated transfer (CT), Section 3.1, equation (4), sandwich SE | `CalibratedCox` |
| Cross-fitted finite-dictionary projected score (PT-F), Section 3.2, equation (5) | `projected_score` |
| Projected-minus-direct contrast, Section 4.5 and Corollary C.12 | `projected_minus_direct` |
| Calibrated randomization test, Section 3.3, equation (6) | `randomization_test` |
| Correction choice by cross-validated integrated Brier score, Corollary 4.3 | `select_correction` |
| Generic natural-spline dictionaries and corrections | `SplineDictionary` |
| Bounded historical learners (neural and spline Cox) | `NeuralCox`, `SplineCox` |
| Adapted TransCox-type comparator, Web Appendix O.4 | `AdaptedTransCox` |

## Installation

```bash
git clone <repository URL>
cd calibrated-transfer-cox
pip install -e ".[test]"
pytest
```

## Quick start

```python
import numpy as np
import ctcox

# Historical cohort (XH, UH, DH): learn and freeze the prognostic log risk
net = ctcox.NeuralCox(seed=1).fit(XH, UH, DH)
f_hat = net.predict(X)                          # evaluated on target covariates

# Target trial (X, A, U, D): direct calibrated transfer with the linear correction z = X
ct = ctcox.CalibratedCox().fit(U, D, f_hat, z=X, treatment=A)
print(ct.theta_, ct.se_, ct.conf_int())

# Robustness analysis: projected score with a generic spline dictionary fixed without target outcomes
B = ctcox.SplineDictionary(n_knots=4, products=True).fit(np.vstack([XH, X])).transform(X)
pt = ctcox.projected_score(U, D, A, f_hat, X, B, folds=5, random_state=0)
print(pt.estimate, pt.se, ctcox.projected_minus_direct(ct, pt))

# Label-blind test of no treatment effect, valid when the transported prognosis is wrong
print(ctcox.randomization_test(U, D, A, f_hat, X, n_draws=999, random_state=0)["p_value"])

# Survival prediction
S = ct.predict_survival(times=[6, 12, 18], offset=net.predict(Xnew), z=Xnew, treatment=a_new)
```

`examples/simulated_trial.py` runs the complete workflow on simulated data.

## Recommended use

* **Prediction:** use `CalibratedCox` and compare it on held-out target patients with strong target-only
  fits and conventional recalibration. A prespecified set of corrections can be compared with
  `select_correction`. Prediction gains do not imply treatment precision gains.
* **Treatment inference:** use the direct fit as the primary analysis and the projected score as a robustness
  analysis. The direct fit has a first-order bias when the remaining historical error is aligned with the
  risk-set representer (Section 4.4). The projected score removes that term when the error lies in the span
  of its dictionary. Under a fixed misspecification it then keeps only a second-order bias
  (Proposition 4.8), so it is not consistent. The standardized contrast checks whether the bias of the
  direct fit is negligible. It is not a test of the treatment null.
* **Dictionaries:** the dictionary must contain the correction `z` modulo constants. `SplineDictionary` gives
  additive natural cubic splines with knots at covariate quantiles and pairwise products, centered and
  whitened on a reference sample. Fit it on covariates only, for example historical and target covariates
  pooled, before looking at target outcomes. In the paper's simulations (Web Appendix O.7), additive splines with pairwise
  products performed like a dictionary informed by the generating law, whereas additive splines alone, which
  missed an interaction in the drift, removed only about half of the direct-fit bias.
* **Correction choice for treatment analysis:** fix it in advance. The theory does not cover treatment
  intervals after data-dependent selection.

## Conventions

* Inputs are arrays in any order. Ties are handled with Breslow's method.
* Losses, scores and information matrices are on the mean scale (divided by the sample size).
* `projected_score` uses ridge `1e-3 * sqrt(300 / n_train)`, clipping at 3 and a derivative guard of `1e-4`
  by default, as in the paper. It returns `success = False` if any fold fails.
* `AdaptedTransCox` is our adaptation of TransCox (Li, Shen and Ning, 2023), not the authors' software.
  It shrinks coefficient differences toward source coefficients with an SD-weighted lasso, chooses the
  penalty by cross-validated partial likelihood, and re-estimates the target baseline hazard. The original
  method also borrows the baseline hazard through a second penalty and selects both penalties by BIC.

## Validation against the paper's analysis code

* On replication 0 of the moderate-drift simulation (hazard ratio 0.7, target sizes 1000, 2000 and 3000),
  `NeuralCox` reproduces the historical network to 4e-14. `CalibratedCox` and `projected_score` reproduce the
  stored CT and PT-F estimates and standard errors to 2e-11, and the contrast z statistics to 2e-9.
* On OAK, with the final ReLU score of the paper as offset, `CalibratedCox` gives hazard ratio 0.703 and
  `projected_score` gives 0.723. The contrast is 0.027 with SE 0.018, matching the paper.

## Citation

Wang, S., Zhang, W., Yang, L., and Li, K. Calibrated transfer learning for survival prediction and treatment
inference. Manuscript, 2026 (full reference to be added on publication). Shuoyang Wang and Wanyu Zhang
contributed equally. Correspondence: Kexuan Li, kexuan.li.77@gmail.com.

## License

MIT. See `LICENSE`.
