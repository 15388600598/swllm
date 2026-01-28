"""
Swissmetro LLM-based Synthetic Data Generation Package

This package provides tools for:
- Loading and preprocessing Swissmetro SP data
- Fitting MNL (Multinomial Logit) models with bootstrap inference
- Evaluating synthetic data quality
- Generating synthetic choice data using LLMs
"""

__version__ = "0.1.0"
__author__ = "Swissmetro LLM Project"

from . import config
from . import data
from . import models
from . import evaluation
from . import generation
from . import stability
