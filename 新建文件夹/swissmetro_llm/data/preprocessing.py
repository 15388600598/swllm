"""
Data preprocessing utilities for Swissmetro dataset.

This module provides functions for:
- Train/test splitting by respondent ID
- Computing and applying standardization scales
- One-hot encoding for categorical variables
"""

from typing import Dict, Any, Tuple, List, Set, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test_by_id(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int], Set[int]]:
    """Split DataFrame into train/test by respondent ID.

    This ensures that all observations from the same respondent
    are in either training or test set (not split across).

    Args:
        df: DataFrame with 'ID' column
        test_size: Fraction of respondents in test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (df_train, df_test, train_ids, test_ids)

    Example:
        >>> df_train, df_test, train_ids, test_ids = split_train_test_by_id(df)
        >>> print(f"Train: {len(df_train)}, Test: {len(df_test)}")
    """
    ids = df["ID"].unique()
    train_ids_arr, test_ids_arr = train_test_split(
        ids, test_size=test_size, random_state=seed
    )

    train_ids = set(train_ids_arr)
    test_ids = set(test_ids_arr)

    df_train = df[df["ID"].isin(train_ids)].copy()
    df_test = df[df["ID"].isin(test_ids)].copy()

    return df_train, df_test, train_ids, test_ids


def compute_scales_from_train(
    data_dict: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute standardization scales from training data.

    Computes column-wise standard deviations for TT, CO, HE matrices.
    These scales should be computed on training data only and then
    applied to both training and test data.

    Args:
        data_dict: Dictionary from build_matrices() with 'TT', 'CO', 'HE' keys

    Returns:
        Tuple of (tt_scale, co_scale, he_scale), each shape (1, 3)

    Example:
        >>> train = build_matrices(df_train)
        >>> tt_scale, co_scale, he_scale = compute_scales_from_train(train)
    """
    eps = 1e-12  # Prevent division by zero

    tt_scale = data_dict["TT"].std(axis=0, keepdims=True) + eps
    co_scale = data_dict["CO"].std(axis=0, keepdims=True) + eps
    he_scale = data_dict["HE"].std(axis=0, keepdims=True) + eps

    return tt_scale, co_scale, he_scale


def apply_scales(
    data_dict: Dict[str, Any],
    tt_scale: np.ndarray,
    co_scale: np.ndarray,
    he_scale: np.ndarray
) -> Dict[str, Any]:
    """Apply standardization scales to data dictionary.

    Creates new keys 'TTs', 'COs', 'HEs' with scaled versions
    of the original matrices.

    Args:
        data_dict: Dictionary from build_matrices()
        tt_scale: Travel time scale, shape (1, 3)
        co_scale: Cost scale, shape (1, 3)
        he_scale: Headway scale, shape (1, 3)

    Returns:
        Updated dictionary with scaled matrices added
    """
    data_dict = data_dict.copy()
    data_dict["TTs"] = data_dict["TT"] / tt_scale
    data_dict["COs"] = data_dict["CO"] / co_scale
    data_dict["HEs"] = data_dict["HE"] / he_scale
    return data_dict


def compute_scales_from_df(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Convenience function to compute scales from DataFrame.

    Splits data, builds matrices, and computes scales.
    Returns a dictionary with all relevant objects.

    Args:
        df: Full Swissmetro DataFrame
        seed: Random seed
        test_size: Test set fraction

    Returns:
        Dictionary with keys:
        - 'train_ids': Set of training respondent IDs
        - 'test_ids': Set of test respondent IDs
        - 'tt_scale', 'co_scale', 'he_scale': Scale arrays
    """
    from .loader import build_matrices

    df_train, df_test, train_ids, test_ids = split_train_test_by_id(
        df, test_size=test_size, seed=seed
    )

    train_data = build_matrices(df_train)
    tt_scale, co_scale, he_scale = compute_scales_from_train(train_data)

    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "tt_scale": tt_scale,
        "co_scale": co_scale,
        "he_scale": he_scale,
    }


def onehot(
    x: np.ndarray,
    levels: List[int],
    drop_first: bool = True
) -> np.ndarray:
    """Create one-hot encoding for categorical variable.

    Args:
        x: Array of categorical values
        levels: List of all possible levels/categories
        drop_first: If True, drop first column (reference category)

    Returns:
        One-hot encoded matrix, shape (N, len(levels)) or (N, len(levels)-1)

    Example:
        >>> age = np.array([1, 2, 3, 1, 2])
        >>> age_oh = onehot(age, levels=[1, 2, 3, 4, 5, 6], drop_first=True)
        >>> print(age_oh.shape)  # (5, 5) since level 1 is dropped
    """
    x = np.asarray(x)
    mats = [(x == lv).astype(float) for lv in levels]
    M = np.column_stack(mats)
    return M[:, 1:] if drop_first else M


def create_individual_features(
    data_dict: Dict[str, Any]
) -> np.ndarray:
    """Create individual feature matrix for Step 2 MNL.

    Constructs a feature matrix from individual characteristics:
    - FIRST (continuous)
    - MALE (binary)
    - AGE (one-hot, 5 dummies, baseline=1)
    - INCOME (one-hot, 4 dummies, baseline=0)
    - PURPOSE (one-hot, 8 dummies, baseline=1)

    Args:
        data_dict: Dictionary from build_matrices()

    Returns:
        Feature matrix X_ind, shape (N, 19)
    """
    FIRST = data_dict["FIRST"]
    MALE = data_dict["MALE"]
    AGE = data_dict["AGE"]
    INCOME = data_dict["INCOME"]
    PURPOSE = data_dict["PURPOSE"]

    X_age = onehot(AGE, levels=[1, 2, 3, 4, 5, 6], drop_first=True)     # 5 cols
    X_income = onehot(INCOME, levels=[0, 1, 2, 3, 4], drop_first=True)  # 4 cols
    X_purp = onehot(PURPOSE, levels=[1, 2, 3, 4, 5, 6, 7, 8, 9], drop_first=True)  # 8 cols

    X = np.column_stack([FIRST, MALE, X_age, X_income, X_purp])
    return X
