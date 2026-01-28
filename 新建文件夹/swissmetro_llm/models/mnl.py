"""
Multinomial Logit (MNL) Model estimation.

This module provides functions for two-stage MNL estimation:
- Step 1: Basic MNL with level-of-service attributes only
- Step 2: Extended MNL with individual characteristic interactions
"""

from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .utils import softmax_with_av, neg_loglike_from_P, accuracy
from ..data.preprocessing import onehot


# =============================================================================
# Feature Matrix Construction
# =============================================================================

def get_x_names() -> List[str]:
    """Get feature names for individual characteristics.

    Returns:
        List of 19 feature names:
        - FIRST, MALE
        - AGE_2 to AGE_6 (5 dummies, baseline=1)
        - INCOME_1 to INCOME_4 (4 dummies, baseline=0)
        - PURPOSE_2 to PURPOSE_9 (8 dummies, baseline=1)
    """
    names = ["FIRST", "MALE"]
    names += [f"AGE_{k}" for k in [2, 3, 4, 5, 6]]
    names += [f"INCOME_{k}" for k in [1, 2, 3, 4]]
    names += [f"PURPOSE_{k}" for k in [2, 3, 4, 5, 6, 7, 8, 9]]
    return names


def get_step2_main_indices() -> List[int]:
    """Get indices for Step 2 main model (excluding FIRST).

    Returns:
        List of indices into the 19-dim X matrix, excluding FIRST
    """
    X_names = get_x_names()
    name_to_idx = {n: i for i, n in enumerate(X_names)}

    idx_MALE = [name_to_idx["MALE"]]
    idx_AGE = [name_to_idx[n] for n in X_names if n.startswith("AGE_")]
    idx_INCOME = [name_to_idx[n] for n in X_names if n.startswith("INCOME_")]
    idx_PURPOSE = [name_to_idx[n] for n in X_names if n.startswith("PURPOSE_")]

    # Main model: MALE + INCOME + AGE + PURPOSE (no FIRST)
    return idx_MALE + idx_INCOME + idx_AGE + idx_PURPOSE


def build_X_ind(data: Dict[str, Any]) -> np.ndarray:
    """Build individual feature matrix from data dictionary.

    Args:
        data: Dictionary from build_matrices() with individual characteristics

    Returns:
        Feature matrix X, shape (N, 19)
    """
    FIRST = data["FIRST"]
    MALE = data["MALE"]
    AGE = data["AGE"]
    INCOME = data["INCOME"]
    PURPOSE = data["PURPOSE"]

    X_age = onehot(AGE, levels=[1, 2, 3, 4, 5, 6], drop_first=True)
    X_income = onehot(INCOME, levels=[0, 1, 2, 3, 4], drop_first=True)
    X_purp = onehot(PURPOSE, levels=[1, 2, 3, 4, 5, 6, 7, 8, 9], drop_first=True)

    X = np.column_stack([FIRST, MALE, X_age, X_income, X_purp])
    return X


def build_X_ind_from_df(df) -> np.ndarray:
    """Build individual feature matrix from DataFrame.

    Args:
        df: DataFrame with FIRST, MALE, AGE, INCOME, PURPOSE columns

    Returns:
        Feature matrix X, shape (N, 19)
    """
    import pandas as pd

    FIRST = df["FIRST"].to_numpy(dtype=float)
    MALE = df["MALE"].to_numpy(dtype=float)
    AGE = df["AGE"].to_numpy(dtype=int)
    INCOME = df["INCOME"].to_numpy(dtype=int)
    PURPOSE = df["PURPOSE"].to_numpy(dtype=int)

    X_age = onehot(AGE, levels=[1, 2, 3, 4, 5, 6], drop_first=True)
    X_income = onehot(INCOME, levels=[0, 1, 2, 3, 4], drop_first=True)
    X_purp = onehot(PURPOSE, levels=[1, 2, 3, 4, 5, 6, 7, 8, 9], drop_first=True)

    X = np.column_stack([FIRST, MALE, X_age, X_income, X_purp])
    return X


# =============================================================================
# Data Preparation for MNL
# =============================================================================

def build_data_from_df(
    df,
    tt_scale: np.ndarray,
    co_scale: np.ndarray,
    he_scale: np.ndarray
) -> Dict[str, Any]:
    """Build data dictionary from DataFrame with scaling.

    Args:
        df: DataFrame with Swissmetro data
        tt_scale: Travel time scale, shape (1, 3)
        co_scale: Cost scale, shape (1, 3)
        he_scale: Headway scale, shape (1, 3)

    Returns:
        Dictionary with N, y, AV, TTs, Cos, HEs, SEATS
    """
    import pandas as pd

    y = df["CHOICE"].to_numpy(dtype=int) - 1
    N = len(df)

    # Availability
    if all(c in df.columns for c in ["TRAIN_AV", "SM_AV", "CAR_AV"]):
        AV = np.column_stack([
            df["TRAIN_AV"].to_numpy(dtype=float),
            df["SM_AV"].to_numpy(dtype=float),
            df["CAR_AV"].to_numpy(dtype=float),
        ])
    else:
        AV = np.ones((N, 3), dtype=float)

    # Travel time
    TT = np.column_stack([
        df["TRAIN_TT"].to_numpy(dtype=float),
        df["SM_TT"].to_numpy(dtype=float),
        df["CAR_TT"].to_numpy(dtype=float),
    ])

    # Cost with GA adjustment
    GA = df["GA"].to_numpy(dtype=int) if "GA" in df.columns else np.zeros(N, dtype=int)
    CO_train = df["TRAIN_CO"].to_numpy(dtype=float) * (GA == 0)
    CO_sm = df["SM_CO"].to_numpy(dtype=float)
    CO_car = df["CAR_CO"].to_numpy(dtype=float)
    CO = np.column_stack([CO_train, CO_sm, CO_car])

    # Headway
    HE_train = df["TRAIN_HE"].to_numpy(dtype=float) if "TRAIN_HE" in df.columns else np.zeros(N)
    HE_sm = df["SM_HE"].to_numpy(dtype=float) if "SM_HE" in df.columns else np.zeros(N)
    HE_car = np.zeros(N, dtype=float)
    HE = np.column_stack([HE_train, HE_sm, HE_car])

    # Seats
    seats_col = "SM_SEATS" if "SM_SEATS" in df.columns else ("SEATS" if "SEATS" in df.columns else None)
    if seats_col is None:
        SEATS_sm = np.zeros(N, dtype=float)
    else:
        SEATS_sm = df[seats_col].to_numpy(dtype=float)
    SEATS = np.column_stack([np.zeros(N), SEATS_sm, np.zeros(N)]).astype(float)

    # Apply scaling
    TTs = TT / tt_scale
    Cos = CO / co_scale
    HEs = HE / he_scale

    return {
        "N": N,
        "y": y,
        "AV": AV,
        "TTs": TTs,
        "Cos": Cos,
        "HEs": HEs,
        "SEATS": SEATS,
    }


def make_data_X_from_df(
    df_sub,
    tt_scale: np.ndarray,
    co_scale: np.ndarray,
    he_scale: np.ndarray
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Prepare data and feature matrix for Step 2 main model.

    Args:
        df_sub: DataFrame with Swissmetro data
        tt_scale, co_scale, he_scale: Scale arrays

    Returns:
        Tuple of (data_dict, X_main) where X_main has 18 columns (no FIRST)
    """
    df_sub = df_sub.loc[df_sub["CHOICE"] > 0].copy()
    data_sub = build_data_from_df(df_sub, tt_scale, co_scale, he_scale)

    X_full = build_X_ind_from_df(df_sub)
    idx_main = get_step2_main_indices()
    X_main = X_full[:, idx_main]

    return data_sub, X_main


# =============================================================================
# Step 1: Basic MNL
# =============================================================================

def fit_mnl_step1(
    data: Dict[str, Any],
    maxiter: int = 5000,
    maxfun: int = 5000
) -> OptimizeResult:
    """Fit Step 1 MNL with level-of-service attributes only.

    Estimates 6 parameters:
    - B_TT: Travel time coefficient
    - B_CO: Cost coefficient
    - B_HE: Headway coefficient
    - B_SEATS: Seats coefficient
    - ASC_SM: Alternative-specific constant for SwissMetro
    - ASC_CAR: Alternative-specific constant for Car

    Args:
        data: Dictionary with N, y, AV, TTs, COs, HEs, SEATS
        maxiter: Maximum iterations
        maxfun: Maximum function evaluations

    Returns:
        scipy OptimizeResult with estimated parameters in .x
    """
    N, y, AV = data["N"], data["y"], data["AV"]
    TT = data["TTs"]
    CO = data["COs"]
    HE = data["HEs"]
    SEATS = data["SEATS"]

    def neg_ll(theta):
        b_tt, b_co, b_he, b_seats, asc_sm, asc_car = theta
        U = b_tt * TT + b_co * CO + b_he * HE + b_seats * SEATS
        U[:, 1] += asc_sm
        U[:, 2] += asc_car
        P = softmax_with_av(U, AV)
        return neg_loglike_from_P(P, y)

    x0 = np.zeros(6)
    res = minimize(
        neg_ll, x0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": maxfun}
    )
    return res


def predict_step1(
    theta: np.ndarray,
    data: Dict[str, Any]
) -> np.ndarray:
    """Predict probabilities using Step 1 parameters.

    Args:
        theta: 6-dim parameter vector from fit_mnl_step1
        data: Dictionary with TTs, COs, HEs, SEATS, AV

    Returns:
        Probability matrix P, shape (N, 3)
    """
    TT = data["TTs"]
    CO = data["COs"]
    HE = data["HEs"]
    SEATS = data["SEATS"]
    AV = data["AV"]

    b_tt, b_co, b_he, b_seats, asc_sm, asc_car = theta

    U = b_tt * TT + b_co * CO + b_he * HE + b_seats * SEATS
    U[:, 1] += asc_sm
    U[:, 2] += asc_car

    return softmax_with_av(U, AV)


# =============================================================================
# Step 2: Extended MNL with Interactions
# =============================================================================

def fit_mnl_step2(
    data: Dict[str, Any],
    X_ind: np.ndarray,
    theta1: Optional[np.ndarray] = None,
    maxiter: int = 20000,
    maxfun: int = 20000
) -> OptimizeResult:
    """Fit Step 2 MNL with individual characteristic interactions.

    Estimates 6 + 2*K parameters:
    - 6 level-of-service coefficients
    - K gamma_sm coefficients (interactions with SM choice)
    - K gamma_car coefficients (interactions with CAR choice)

    Args:
        data: Dictionary with N, y, AV, TTs, COs, HEs, SEATS
        X_ind: Individual feature matrix, shape (N, K)
        theta1: Initial values for first 6 parameters (from Step 1)
        maxiter: Maximum iterations
        maxfun: Maximum function evaluations

    Returns:
        scipy OptimizeResult with estimated parameters in .x
    """
    N, y, AV = data["N"], data["y"], data["AV"]
    TT = data["TTs"]
    CO = data["COs"]
    HE = data["HEs"]
    SEATS = data["SEATS"]
    K = X_ind.shape[1]

    def neg_ll(theta):
        b_tt, b_co, b_he, b_seats, asc_sm, asc_car = theta[:6]
        gamma_sm = theta[6:6+K]
        gamma_car = theta[6+K:6+2*K]

        U = b_tt * TT + b_co * CO + b_he * HE + b_seats * SEATS
        U[:, 1] += asc_sm + X_ind @ gamma_sm
        U[:, 2] += asc_car + X_ind @ gamma_car

        P = softmax_with_av(U, AV)
        return neg_loglike_from_P(P, y)

    if theta1 is None:
        x0 = np.zeros(6 + 2*K)
    else:
        x0 = np.concatenate([theta1, np.zeros(2*K)])

    res = minimize(
        neg_ll, x0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": maxfun}
    )
    return res


def predict_step2(
    theta: np.ndarray,
    data: Dict[str, Any],
    X_ind: np.ndarray
) -> np.ndarray:
    """Predict probabilities using Step 2 parameters.

    Args:
        theta: (6 + 2*K)-dim parameter vector from fit_mnl_step2
        data: Dictionary with TTs, COs, HEs, SEATS, AV
        X_ind: Individual feature matrix, shape (N, K)

    Returns:
        Probability matrix P, shape (N, 3)
    """
    TT = data["TTs"]
    CO = data["COs"]
    HE = data["HEs"]
    SEATS = data["SEATS"]
    AV = data["AV"]
    K = X_ind.shape[1]

    b_tt, b_co, b_he, b_seats, asc_sm, asc_car = theta[:6]
    gamma_sm = theta[6:6+K]
    gamma_car = theta[6+K:6+2*K]

    U = b_tt * TT + b_co * CO + b_he * HE + b_seats * SEATS
    U[:, 1] += asc_sm + X_ind @ gamma_sm
    U[:, 2] += asc_car + X_ind @ gamma_car

    return softmax_with_av(U, AV)


def fit_step2_main(
    df_sub,
    tt_scale: np.ndarray,
    co_scale: np.ndarray,
    he_scale: np.ndarray,
    theta_init: Optional[np.ndarray] = None,
    maxiter: int = 20000,
    maxfun: int = 20000
) -> OptimizeResult:
    """Fit Step 2 main model directly from DataFrame.

    Convenience function that builds data and X matrix internally.

    Args:
        df_sub: DataFrame with Swissmetro data
        tt_scale, co_scale, he_scale: Scale arrays
        theta_init: Initial parameter values (optional)
        maxiter: Maximum iterations
        maxfun: Maximum function evaluations

    Returns:
        scipy OptimizeResult
    """
    data, X = make_data_X_from_df(df_sub, tt_scale, co_scale, he_scale)

    y, AV = data["y"], data["AV"]
    TT, CO, HE, SEATS = data["TTs"], data["Cos"], data["HEs"], data["SEATS"]
    K = X.shape[1]

    def neg_ll(theta):
        b_tt, b_co, b_he, b_seats, asc_sm, asc_car = theta[:6]
        gamma_sm = theta[6:6+K]
        gamma_car = theta[6+K:6+2*K]

        U = b_tt * TT + b_co * CO + b_he * HE + b_seats * SEATS
        U[:, 1] += asc_sm + X @ gamma_sm
        U[:, 2] += asc_car + X @ gamma_car

        P = softmax_with_av(U, AV)
        return neg_loglike_from_P(P, y)

    if theta_init is None:
        theta_init = np.zeros(6 + 2*K)

    res = minimize(
        neg_ll, theta_init,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": maxfun}
    )
    return res


# =============================================================================
# Parameter Names and Reporting
# =============================================================================

def get_step2_main_param_names(K: int = 18) -> List[str]:
    """Get parameter names for Step 2 main model.

    Args:
        K: Number of individual features (default 18 for main model)

    Returns:
        List of 6 + 2*K parameter names
    """
    X_names_all = get_x_names()
    X_names_main = [n for n in X_names_all if n != "FIRST"][:K]

    param_names = (
        ["B_TT", "B_CO", "B_HE", "B_SEATS", "ASC_SM", "ASC_CAR"]
        + [f"G_SM:{n}" for n in X_names_main]
        + [f"G_CAR:{n}" for n in X_names_main]
    )
    return param_names
