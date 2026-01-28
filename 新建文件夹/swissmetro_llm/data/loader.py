"""
Data loading utilities for Swissmetro dataset.

This module provides functions to load the Swissmetro stated preference
survey data and convert it to the matrix format used by MNL models.
"""

from typing import Dict, Any, Optional, Set
from pathlib import Path

import numpy as np
import pandas as pd


def load_swissmetro(
    filepath: str,
    filter_valid_choices: bool = True
) -> pd.DataFrame:
    """Load Swissmetro dataset from .dat file.

    Args:
        filepath: Path to the .dat file (whitespace-separated)
        filter_valid_choices: If True, filter to rows where CHOICE > 0

    Returns:
        DataFrame with Swissmetro survey data

    Example:
        >>> df = load_swissmetro("swissmetro.dat")
        >>> print(f"Rows: {len(df)}, Respondents: {df['ID'].nunique()}")
    """
    df = pd.read_csv(filepath, sep=r"\s+", engine="python")

    if filter_valid_choices:
        df = df.loc[df["CHOICE"] > 0].copy()

    return df


def build_matrices(df_in: pd.DataFrame) -> Dict[str, Any]:
    """Convert DataFrame to matrix format for MNL estimation.

    This function constructs the data matrices needed for multinomial logit
    estimation, including:
    - Choice outcomes (y)
    - Availability indicators (AV)
    - Level-of-service attributes (TT, CO, HE, SEATS)
    - Individual characteristics (ID, FIRST, MALE, AGE, INCOME, PURPOSE)

    Note on cost treatment:
    - TRAIN_CO is set to 0 for GA holders (GA=1)
    - SM_CO is kept as-is (no GA discount)
    - CAR_CO is the original car cost

    Args:
        df_in: DataFrame with Swissmetro data

    Returns:
        Dictionary containing:
        - N: Number of observations
        - y: Choice outcomes (0=TRAIN, 1=SM, 2=CAR)
        - AV: Availability matrix (N, 3)
        - TT: Travel time matrix (N, 3), scaled by 1/100
        - CO: Cost matrix (N, 3), scaled by 1/100, GA-adjusted for TRAIN
        - HE: Headway matrix (N, 3), scaled by 1/100
        - SEATS: Seats indicator matrix (N, 3), only SM has non-zero
        - ID, FIRST, MALE, AGE, INCOME, PURPOSE: Individual characteristics
    """
    df2 = df_in.copy()
    N = len(df2)

    # y: 1=Train, 2=SM, 3=Car -> 0/1/2 (0-indexed)
    y = df2["CHOICE"].to_numpy(int) - 1

    # Availability indicators
    AV = np.column_stack([
        df2["TRAIN_AV"].to_numpy(float),
        df2["SM_AV"].to_numpy(float),
        df2["CAR_AV"].to_numpy(float),
    ])

    # Travel time (scaled by 1/100 for numeric stability)
    TT = np.column_stack([
        df2["TRAIN_TT"].to_numpy(float),
        df2["SM_TT"].to_numpy(float),
        df2["CAR_TT"].to_numpy(float),
    ]) / 100.0

    # Cost with GA adjustment:
    # - GA holders (GA=1) pay 0 for TRAIN
    # - SM_CO is not affected by GA (kept as-is)
    # - CAR_CO is the original cost
    GA = df2["GA"].to_numpy(int)
    TRAIN_FARE = df2["TRAIN_CO"].to_numpy(float) * (GA == 0)  # GA=1 -> 0
    SM_COST = df2["SM_CO"].to_numpy(float)
    CAR_COST = df2["CAR_CO"].to_numpy(float)

    CO = np.column_stack([TRAIN_FARE, SM_COST, CAR_COST]) / 100.0

    # Headway (Car has no headway -> 0)
    HE = np.column_stack([
        df2["TRAIN_HE"].to_numpy(float),
        df2["SM_HE"].to_numpy(float),
        np.zeros(N),
    ]) / 100.0

    # Seats indicator (only SM has seats variable)
    SM_SEATS = df2["SM_SEATS"].to_numpy(float)
    SEATS = np.column_stack([
        np.zeros(N),
        SM_SEATS,
        np.zeros(N),
    ])

    # Individual characteristics (raw values)
    ID = df2["ID"].to_numpy(int)
    FIRST = df2["FIRST"].to_numpy(float)
    MALE = df2["MALE"].to_numpy(float)
    AGE = df2["AGE"].to_numpy(int)        # 1..6
    INCOME = df2["INCOME"].to_numpy(int)  # 0..4
    PURPOSE = df2["PURPOSE"].to_numpy(int)  # 1..9

    return {
        "N": N,
        "y": y,
        "AV": AV,
        "TT": TT,
        "CO": CO,
        "HE": HE,
        "SEATS": SEATS,
        "ID": ID,
        "FIRST": FIRST,
        "MALE": MALE,
        "AGE": AGE,
        "INCOME": INCOME,
        "PURPOSE": PURPOSE,
    }


def get_unique_ids(df: pd.DataFrame) -> np.ndarray:
    """Get unique respondent IDs from DataFrame.

    Args:
        df: DataFrame with 'ID' column

    Returns:
        Array of unique IDs
    """
    return df["ID"].unique()


def get_train_ids(df: pd.DataFrame, train_ids: Set[int]) -> pd.DataFrame:
    """Filter DataFrame to only include training respondents.

    Args:
        df: Full DataFrame
        train_ids: Set of training respondent IDs

    Returns:
        Filtered DataFrame
    """
    return df[df["ID"].isin(train_ids)].copy()


def get_test_ids(df: pd.DataFrame, train_ids: Set[int]) -> pd.DataFrame:
    """Filter DataFrame to only include test respondents.

    Args:
        df: Full DataFrame
        train_ids: Set of training respondent IDs

    Returns:
        Filtered DataFrame (complement of train_ids)
    """
    return df[~df["ID"].isin(train_ids)].copy()
