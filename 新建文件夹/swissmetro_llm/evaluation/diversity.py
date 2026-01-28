"""
Diversity evaluation for synthetic data.

This module provides functions to measure how well synthetic data
covers the demographic space of real data.
"""

from typing import Dict, List, Set, Any, Optional

import numpy as np
import pandas as pd

from ..config import COMBO_COLS


def unique_combos(
    df: pd.DataFrame,
    cols: List[str] = None
) -> pd.DataFrame:
    """Get unique demographic combinations from DataFrame.

    Args:
        df: DataFrame with demographic columns
        cols: List of column names to use (default: COMBO_COLS)

    Returns:
        DataFrame with unique combinations
    """
    if cols is None:
        cols = COMBO_COLS

    return df[cols].dropna().drop_duplicates()


def combo_to_set(df: pd.DataFrame, cols: List[str] = None) -> Set[tuple]:
    """Convert DataFrame combinations to set of tuples.

    Args:
        df: DataFrame with demographic columns
        cols: List of column names to use (default: COMBO_COLS)

    Returns:
        Set of tuples representing unique combinations
    """
    if cols is None:
        cols = COMBO_COLS

    unique_df = unique_combos(df, cols)
    return set(map(tuple, unique_df.to_numpy()))


def diversity_report(
    synth_df: pd.DataFrame,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    real_all: Optional[pd.DataFrame] = None,
    cols: List[str] = None
) -> Dict[str, Any]:
    """Generate diversity report comparing synthetic to real data.

    Metrics:
    - Coverage of training combinations
    - Coverage of test combinations
    - Coverage of "unseen" test combinations (in test but not train)
    - Precision: fraction of synthetic combos that exist in real data
    - Structural zero rate: fraction of synthetic combos not in real data

    Args:
        synth_df: Synthetic DataFrame
        real_train: Real training DataFrame
        real_test: Real test DataFrame
        real_all: Combined real data (optional, computed if not provided)
        cols: Columns to use for combinations

    Returns:
        Dictionary with diversity metrics
    """
    if cols is None:
        cols = COMBO_COLS

    if real_all is None:
        real_all = pd.concat([real_train, real_test], axis=0)

    S_set = combo_to_set(synth_df, cols)
    T_set = combo_to_set(real_train, cols)
    E_set = combo_to_set(real_test, cols)
    A_set = combo_to_set(real_all, cols)

    # Unseen in test: combinations in test but not in train
    unseen_test = E_set - T_set

    # Coverage metrics
    cover_train = len(S_set & T_set) / max(len(T_set), 1)
    cover_test = len(S_set & E_set) / max(len(E_set), 1)
    cover_unseen = len(S_set & unseen_test) / max(len(unseen_test), 1)

    # Precision: synthetic combos that are "real" (exist in real data)
    precision_real = len(S_set & A_set) / max(len(S_set), 1)

    return {
        "uniq_combos_synth": int(len(S_set)),
        "uniq_combos_train": int(len(T_set)),
        "uniq_combos_test": int(len(E_set)),
        "uniq_combos_all": int(len(A_set)),
        "cover_train_combo_rate": float(cover_train),
        "cover_test_combo_rate": float(cover_test),
        "cover_unseen_test_combo_rate": float(cover_unseen),
        "precision_combo_in_real_all": float(precision_real),
        "structural_zero_combo_rate": float(1 - precision_real),
        "n_unseen_test_combos": int(len(unseen_test)),
    }


def compute_combo_overlap(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    cols: List[str] = None
) -> Dict[str, Any]:
    """Compute overlap statistics between two DataFrames.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        cols: Columns to use for combinations

    Returns:
        Dictionary with overlap statistics
    """
    if cols is None:
        cols = COMBO_COLS

    set1 = combo_to_set(df1, cols)
    set2 = combo_to_set(df2, cols)

    intersection = set1 & set2
    union = set1 | set2
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1

    return {
        "n_df1": len(set1),
        "n_df2": len(set2),
        "n_intersection": len(intersection),
        "n_union": len(union),
        "n_only_in_df1": len(only_in_1),
        "n_only_in_df2": len(only_in_2),
        "jaccard_index": len(intersection) / max(len(union), 1),
        "overlap_rate_df1": len(intersection) / max(len(set1), 1),
        "overlap_rate_df2": len(intersection) / max(len(set2), 1),
    }


def get_unseen_combos(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    cols: List[str] = None
) -> np.ndarray:
    """Get combinations that appear in test but not in train.

    Args:
        real_train: Training DataFrame
        real_test: Test DataFrame
        cols: Columns to use for combinations

    Returns:
        Array of unseen combination tuples
    """
    if cols is None:
        cols = COMBO_COLS

    T_set = combo_to_set(real_train, cols)
    E_set = combo_to_set(real_test, cols)

    unseen = E_set - T_set
    return np.array(list(unseen))


def get_train_combos(
    real_train: pd.DataFrame,
    cols: List[str] = None
) -> np.ndarray:
    """Get unique combinations from training data.

    Args:
        real_train: Training DataFrame
        cols: Columns to use for combinations

    Returns:
        Array of training combination tuples
    """
    if cols is None:
        cols = COMBO_COLS

    T_set = combo_to_set(real_train, cols)
    return np.array(list(T_set))
