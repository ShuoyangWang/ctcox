"""ctcox: calibrated transfer learning for Cox survival prediction and treatment inference.

Core components
---------------
CalibratedCox            direct calibrated transfer with a fixed historical offset (CT)
projected_score          cross-fitted finite-dictionary projected score (PT-F)
projected_minus_direct   standardized contrast between the two treatment estimates
randomization_test       label-blind calibrated randomization test
SplineDictionary         generic natural-spline dictionaries and correction bases
NeuralCox, SplineCox     bounded historical learners
select_correction        choice among prespecified corrections by cross-validated IBS
AdaptedTransCox          adapted TransCox-type comparator (coefficient deviations only)
"""
from .risk import RiskSet
from .cox import fit_cox, CoxFit
from .calibrate import CalibratedCox
from .projected import projected_score, projected_minus_direct, make_folds, ProjectedResult
from .randomization import randomization_test
from .dictionaries import SplineDictionary, natural_spline, linear_dictionary
from .learners import NeuralCox, SplineCox
from .metrics import integrated_brier_score, brier_scores, concordance_index, kaplan_meier
from .selection import select_correction, stratified_folds
from .transcox_adapted import AdaptedTransCox, lasso_cox

__version__ = "0.1.0"
__all__ = ["RiskSet", "fit_cox", "CoxFit", "CalibratedCox", "projected_score", "projected_minus_direct",
           "make_folds", "ProjectedResult", "randomization_test", "SplineDictionary", "natural_spline",
           "linear_dictionary", "NeuralCox", "SplineCox", "integrated_brier_score", "brier_scores",
           "concordance_index", "kaplan_meier", "select_correction", "stratified_folds", "AdaptedTransCox",
           "lasso_cox"]
