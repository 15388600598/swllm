"""
Feasibility checking for synthetic data.

This module provides functions to validate that synthetic data
meets basic requirements for MNL estimation and evaluation.
"""

from typing import Dict, List, Any

import numpy as np
import pandas as pd

from ..config import REQUIRED_COLS


def feasibility_min(df: pd.DataFrame) -> Dict[str, Any]:
    """Minimal feasibility check for synthetic data.

    Checks:
    - Missing required columns
    - Row-level NA rate

    Args:
        df: DataFrame to check

    Returns:
        Dictionary with:
        - missing_cols: List of missing required columns
        - row_na_rate: Fraction of rows with any NA
        - N: Number of rows
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]

    return {
        "missing_cols": missing,
        "row_na_rate": float(df.isna().any(axis=1).mean()),
        "N": int(len(df)),
    }


def feasibility_strict(df: pd.DataFrame) -> Dict[str, Any]:
    """Strict feasibility check for synthetic data.

    Checks:
    - All minimal checks
    - Negative values in time/cost/headway columns
    - Choice-availability conflicts
    - Invalid choice values

    Args:
        df: DataFrame to check

    Returns:
        Dictionary with all check results
    """
    issues = feasibility_min(df)

    # Check for negative values in level-of-service attributes
    los_cols = [
        "TRAIN_TT", "SM_TT", "CAR_TT",
        "TRAIN_CO", "SM_CO", "CAR_CO",
        "TRAIN_HE", "SM_HE"
    ]
    for c in los_cols:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            issues[f"neg_rate_{c}"] = float((v < 0).mean())

    # Check choice-availability conflicts
    if "CAR_AV" in df.columns:
        car_av = pd.to_numeric(df["CAR_AV"], errors="coerce").fillna(1)
        ch = pd.to_numeric(df["CHOICE"], errors="coerce")
        issues["choice_car_while_unavailable"] = float(((ch == 3) & (car_av <= 0)).mean())

    if "TRAIN_AV" in df.columns:
        train_av = pd.to_numeric(df["TRAIN_AV"], errors="coerce").fillna(1)
        ch = pd.to_numeric(df["CHOICE"], errors="coerce")
        issues["choice_train_while_unavailable"] = float(((ch == 1) & (train_av <= 0)).mean())

    if "SM_AV" in df.columns:
        sm_av = pd.to_numeric(df["SM_AV"], errors="coerce").fillna(1)
        ch = pd.to_numeric(df["CHOICE"], errors="coerce")
        issues["choice_sm_while_unavailable"] = float(((ch == 2) & (sm_av <= 0)).mean())

    # Check for invalid choice values
    if "CHOICE" in df.columns:
        ch = pd.to_numeric(df["CHOICE"], errors="coerce")
        issues["invalid_choice_rate"] = float((~ch.isin([1, 2, 3])).mean())

    return issues


def check_required_columns(
    df: pd.DataFrame,
    required: List[str] = None
) -> List[str]:
    """Check which required columns are missing.

    Args:
        df: DataFrame to check
        required: List of required column names (default: REQUIRED_COLS)

    Returns:
        List of missing column names
    """
    if required is None:
        required = REQUIRED_COLS
    return [c for c in required if c not in df.columns]


def validate_choice_column(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate the CHOICE column.

    Args:
        df: DataFrame with CHOICE column

    Returns:
        Dictionary with validation results
    """
    if "CHOICE" not in df.columns:
        return {"has_choice": False}

    ch = pd.to_numeric(df["CHOICE"], errors="coerce")

    return {
        "has_choice": True,
        "n_valid": int(ch.isin([1, 2, 3]).sum()),
        "n_invalid": int((~ch.isin([1, 2, 3])).sum()),
        "n_missing": int(ch.isna().sum()),
        "choice_distribution": {
            1: int((ch == 1).sum()),
            2: int((ch == 2).sum()),
            3: int((ch == 3).sum()),
        }
    }
