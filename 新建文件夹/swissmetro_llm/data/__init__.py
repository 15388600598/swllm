"""Data loading and preprocessing modules."""

from .loader import load_swissmetro, build_matrices
from .preprocessing import (
    compute_scales_from_train,
    apply_scales,
    split_train_test_by_id,
    onehot,
)
