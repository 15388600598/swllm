"""Evaluation modules for synthetic data quality assessment."""

from .feasibility import feasibility_min, feasibility_strict
from .diversity import unique_combos, diversity_report
from .downstream import build_X_main, score_with_baseline_mnl, downstream_metrics
from .metrics import evaluate_one, evaluate_one_min
