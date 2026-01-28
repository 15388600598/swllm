"""
Downstream evaluation for synthetic data.

This module provides functions to evaluate synthetic data quality
by scoring it with a baseline MNL model and computing fit metrics.
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_alt_attributes(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build alternative attribute matrices from DataFrame.

    Handles GA-adjusted costs for Train alternative.

    Args:
        df: DataFrame with level-of-service columns

    Returns:
        Dictionary with TT, CO, HE, SEATS, AV matrices
    """
    df = df.copy()
    N = len(df)

    # GA handling
    if "GA" in df.columns:
        GA = pd.to_numeric(df["GA"], errors="coerce").fillna(0).to_numpy(int)
    else:
        GA = np.zeros(N, dtype=int)

    # Availability (default all available if not provided)
    if "TRAIN_AV" not in df.columns:
        df["TRAIN_AV"] = 1
    if "SM_AV" not in df.columns:
        df["SM_AV"] = 1
    if "CAR_AV" not in df.columns:
        df["CAR_AV"] = 1

    AV = np.column_stack([
        pd.to_numeric(df["TRAIN_AV"], errors="coerce").fillna(1).to_numpy(float),
        pd.to_numeric(df["SM_AV"], errors="coerce").fillna(1).to_numpy(float),
        pd.to_numeric(df["CAR_AV"], errors="coerce").fillna(1).to_numpy(float),
    ])

    # Travel time (/100)
    TT = np.column_stack([
        pd.to_numeric(df["TRAIN_TT"], errors="coerce").to_numpy(float),
        pd.to_numeric(df["SM_TT"], errors="coerce").to_numpy(float),
        pd.to_numeric(df["CAR_TT"], errors="coerce").to_numpy(float),
    ]) / 100.0

    # Cost with GA adjustment (/100)
    TRAIN_FARE = pd.to_numeric(df["TRAIN_CO"], errors="coerce").to_numpy(float) * (GA == 0)
    SM_COST = pd.to_numeric(df["SM_CO"], errors="coerce").to_numpy(float)
    CAR_COST = pd.to_numeric(df["CAR_CO"], errors="coerce").to_numpy(float)
    CO = np.column_stack([TRAIN_FARE, SM_COST, CAR_COST]) / 100.0

    # Headway (/100), Car=0
    HE = np.column_stack([
        pd.to_numeric(df["TRAIN_HE"], errors="coerce").to_numpy(float),
        pd.to_numeric(df["SM_HE"], errors="coerce").to_numpy(float),
        np.zeros(N),
    ]) / 100.0

    # Seats (only SM has value)
    SEATS = np.column_stack([
        np.zeros(N),
        pd.to_numeric(df["SM_SEATS"], errors="coerce").fillna(0).to_numpy(float),
        np.zeros(N)
    ])

    return {"TT": TT, "CO": CO, "HE": HE, "SEATS": SEATS, "AV": AV}


def build_X_main(df: pd.DataFrame) -> pd.DataFrame:
    """Build individual feature matrix for Step 2 main model.

    Creates dummy variables for AGE, INCOME, PURPOSE with proper baselines.

    Args:
        df: DataFrame with individual characteristic columns

    Returns:
        DataFrame with 18 feature columns (MALE + dummies)
    """
    df = df.copy()

    for c in ["MALE", "AGE", "INCOME", "PURPOSE"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")

    X = {}
    X["MALE"] = pd.to_numeric(df["MALE"], errors="coerce").fillna(0).astype(float)

    AGE = pd.to_numeric(df["AGE"], errors="coerce")
    for k in [2, 3, 4, 5, 6]:
        X[f"AGE_{k}"] = (AGE == k).astype(float)

    INC = pd.to_numeric(df["INCOME"], errors="coerce")
    for k in [1, 2, 3, 4]:
        X[f"INCOME_{k}"] = (INC == k).astype(float)

    PUR = pd.to_numeric(df["PURPOSE"], errors="coerce")
    for k in [2, 3, 4, 5, 6, 7, 8, 9]:
        X[f"PURPOSE_{k}"] = (PUR == k).astype(float)

    return pd.DataFrame(X)


def compute_scales_from_real(
    real_df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Compute standardization scales from real data.

    Splits data by ID, computes scales on training subset.

    Args:
        real_df: Full real DataFrame
        seed: Random seed for splitting
        test_size: Fraction for test set

    Returns:
        Dictionary with tt_scale, co_scale, he_scale, train_ids
    """
    df = real_df.copy()
    df = df.loc[df["CHOICE"] > 0].copy()

    ids = df["ID"].unique()
    train_ids, _ = train_test_split(ids, test_size=test_size, random_state=seed)
    df_train = df[df["ID"].isin(train_ids)].copy()

    mats = build_alt_attributes(df_train)
    tt_scale = mats["TT"].std(axis=0, keepdims=True) + 1e-12
    co_scale = mats["CO"].std(axis=0, keepdims=True) + 1e-12
    he_scale = mats["HE"].std(axis=0, keepdims=True) + 1e-12

    return {
        "tt_scale": tt_scale,
        "co_scale": co_scale,
        "he_scale": he_scale,
        "train_ids": set(train_ids),
    }


def softmax_with_av(
    U: np.ndarray,
    AV: np.ndarray,
    large_neg: float = -1e9
) -> np.ndarray:
    """Compute softmax with availability constraints.

    Args:
        U: Utility matrix (N, 3)
        AV: Availability matrix (N, 3)
        large_neg: Value for unavailable alternatives

    Returns:
        Probability matrix (N, 3)
    """
    U2 = U.copy()
    U2[AV <= 0] = large_neg
    Umax = np.max(U2, axis=1, keepdims=True)
    expU = np.exp(U2 - Umax)
    expU[AV <= 0] = 0.0
    denom = np.sum(expU, axis=1, keepdims=True) + 1e-300
    return expU / denom


def score_with_baseline_mnl(
    df: pd.DataFrame,
    beta: Dict[str, float],
    scales: Dict[str, Any]
) -> pd.DataFrame:
    """Score DataFrame using baseline MNL model.

    Adds probability columns P_TRAIN, P_SM, P_CAR and PRED_CHOICE.

    Args:
        df: DataFrame to score
        beta: Dictionary of parameter values
        scales: Dictionary with tt_scale, co_scale, he_scale

    Returns:
        DataFrame with added probability and prediction columns
    """
    mats = build_alt_attributes(df)
    X = build_X_main(df)

    # Apply scaling
    TTs = mats["TT"] / scales["tt_scale"]
    COs = mats["CO"] / scales["co_scale"]
    HEs = mats["HE"] / scales["he_scale"]
    SEATS = mats["SEATS"]
    AV = mats["AV"]

    # Base parameters
    b_tt = float(beta.get("B_TT", 0.0))
    b_co = float(beta.get("B_CO", 0.0))
    b_he = float(beta.get("B_HE", 0.0))
    b_seats = float(beta.get("B_SEATS", 0.0))
    asc_sm = float(beta.get("ASC_SM", 0.0))
    asc_car = float(beta.get("ASC_CAR", 0.0))

    # Base utility
    U = b_tt * TTs + b_co * COs + b_he * HEs + b_seats * SEATS

    # Add interaction terms
    for p, c in beta.items():
        if p.startswith("G_SM:"):
            name = p.split("G_SM:")[1]
            if name not in X.columns:
                raise ValueError(f"beta has {p} but X missing column {name}")
            U[:, 1] += float(c) * X[name].to_numpy(float)
        elif p.startswith("G_CAR:"):
            name = p.split("G_CAR:")[1]
            if name not in X.columns:
                raise ValueError(f"beta has {p} but X missing column {name}")
            U[:, 2] += float(c) * X[name].to_numpy(float)

    # Add ASCs
    U[:, 1] += asc_sm
    U[:, 2] += asc_car

    # Compute probabilities
    P = softmax_with_av(U, AV)

    # Add to output
    out = df.copy()
    out["P_TRAIN"] = P[:, 0]
    out["P_SM"] = P[:, 1]
    out["P_CAR"] = P[:, 2]
    out["PRED_CHOICE"] = np.argmax(P, axis=1) + 1

    return out


def downstream_metrics(
    scored_df: pd.DataFrame,
    choice_col: str = "CHOICE"
) -> Dict[str, Any]:
    """Compute downstream fit metrics from scored DataFrame.

    Metrics:
    - N: Number of valid observations
    - avg_P_chosen: Average probability of chosen alternative
    - low_prob_rate(<0.01): Fraction with P(chosen) < 0.01
    - ll_per_obs: Log-likelihood per observation
    - accuracy: Prediction accuracy
    - pred_share_*: Predicted mode shares

    Args:
        scored_df: DataFrame with P_TRAIN, P_SM, P_CAR, PRED_CHOICE columns
        choice_col: Name of choice column

    Returns:
        Dictionary with fit metrics
    """
    df = scored_df.copy()
    df[choice_col] = pd.to_numeric(df[choice_col], errors="coerce")
    df = df[df[choice_col].isin([1, 2, 3])].copy()

    if len(df) == 0:
        return {"N": 0}

    # Probability of chosen alternative
    choice = df[choice_col].to_numpy()
    chosen_p = np.where(
        choice == 1, df["P_TRAIN"].to_numpy(),
        np.where(choice == 2, df["P_SM"].to_numpy(), df["P_CAR"].to_numpy())
    )

    ll = float(np.sum(np.log(np.clip(chosen_p, 1e-300, 1.0))))

    return {
        "N": int(len(df)),
        "avg_P_chosen": float(np.mean(chosen_p)),
        "low_prob_rate(<0.01)": float(np.mean(chosen_p < 0.01)),
        "ll_per_obs": float(ll / len(df)),
        "accuracy": float(np.mean(df["PRED_CHOICE"].to_numpy() == choice)),
        "pred_share_train": float(df["P_TRAIN"].mean()),
        "pred_share_sm": float(df["P_SM"].mean()),
        "pred_share_car": float(df["P_CAR"].mean()),
    }


def split_real_by_train_ids(
    real_df: pd.DataFrame,
    train_ids: set
) -> tuple:
    """Split real data by training IDs.

    Args:
        real_df: Full real DataFrame
        train_ids: Set of training respondent IDs

    Returns:
        Tuple of (real_train, real_test)
    """
    real_train = real_df[real_df["ID"].isin(train_ids)].copy()
    real_test = real_df[~real_df["ID"].isin(train_ids)].copy()
    return real_train, real_test


def load_beta(filepath: str) -> Dict[str, float]:
    """Load beta parameters from CSV file.

    Expects CSV with 'param' and 'coef' columns.

    Args:
        filepath: Path to CSV file

    Returns:
        Dictionary mapping parameter names to values
    """
    df = pd.read_csv(filepath)
    return dict(zip(df["param"], df["coef"]))
