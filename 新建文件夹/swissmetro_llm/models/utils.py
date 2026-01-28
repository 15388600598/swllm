"""
Utility functions for MNL models.

This module provides core mathematical functions used in
multinomial logit estimation and prediction.
"""

from typing import Optional

import numpy as np


def softmax_with_av(
    U: np.ndarray,
    AV: np.ndarray,
    large_neg: float = -1e10
) -> np.ndarray:
    """Compute softmax probabilities with availability constraints.

    For alternatives that are not available (AV=0), the utility is
    set to a large negative value before applying softmax.

    Args:
        U: Utility matrix, shape (N, J) where J is number of alternatives
        AV: Availability matrix, shape (N, J), 1=available, 0=unavailable
        large_neg: Large negative value for unavailable alternatives

    Returns:
        Probability matrix P, shape (N, J), rows sum to 1

    Example:
        >>> U = np.array([[1.0, 2.0, 0.5], [0.3, 0.8, 1.2]])
        >>> AV = np.array([[1, 1, 1], [1, 1, 0]])  # CAR unavailable for 2nd obs
        >>> P = softmax_with_av(U, AV)
        >>> print(P.sum(axis=1))  # [1.0, 1.0]
    """
    # Mask unavailable alternatives
    U_masked = np.where(AV > 0, U, large_neg)

    # Numerically stable softmax
    Umax = np.max(U_masked, axis=1, keepdims=True)
    expU = np.exp(U_masked - Umax)
    denom = np.sum(expU, axis=1, keepdims=True)
    P = expU / denom

    return P


def softmax_rows(
    X: np.ndarray,
    tau: float = 1.0
) -> np.ndarray:
    """Compute row-wise softmax with temperature scaling.

    Args:
        X: Input matrix, shape (N, J)
        tau: Temperature parameter. tau=1 is standard softmax.
             Higher tau -> more uniform distribution.
             Lower tau -> more peaked distribution.

    Returns:
        Probability matrix P, shape (N, J), rows sum to 1

    Example:
        >>> X = np.array([[1.0, 2.0, 0.5]])
        >>> print(softmax_rows(X, tau=1.0))  # Standard softmax
        >>> print(softmax_rows(X, tau=0.5))  # More peaked
    """
    X_scaled = X / tau

    # Numerically stable softmax
    X_max = np.max(X_scaled, axis=1, keepdims=True)
    expX = np.exp(X_scaled - X_max)
    return expX / expX.sum(axis=1, keepdims=True)


def neg_loglike_from_P(
    P: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute negative log-likelihood from probability matrix.

    Args:
        P: Probability matrix, shape (N, J)
        y: Observed choices, shape (N,), values in {0, 1, ..., J-1}

    Returns:
        Negative log-likelihood (scalar)

    Example:
        >>> P = np.array([[0.2, 0.7, 0.1], [0.3, 0.4, 0.3]])
        >>> y = np.array([1, 2])  # Chose SM for 1st, CAR for 2nd
        >>> nll = neg_loglike_from_P(P, y)
    """
    eps = 1e-300  # Prevent log(0)
    p_chosen = P[np.arange(len(y)), y]
    return -np.sum(np.log(p_chosen + eps))


def accuracy(
    P: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute prediction accuracy.

    Args:
        P: Probability matrix, shape (N, J)
        y: Observed choices, shape (N,), values in {0, 1, ..., J-1}

    Returns:
        Accuracy as fraction in [0, 1]

    Example:
        >>> P = np.array([[0.2, 0.7, 0.1], [0.3, 0.4, 0.3]])
        >>> y = np.array([1, 0])  # Actual choices
        >>> acc = accuracy(P, y)  # P predicts 1 for 1st (correct), 1 for 2nd (wrong)
    """
    pred = np.argmax(P, axis=1)
    return float((pred == y).mean())


def loglike_from_P(
    P: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute log-likelihood from probability matrix.

    Args:
        P: Probability matrix, shape (N, J)
        y: Observed choices, shape (N,), values in {0, 1, ..., J-1}

    Returns:
        Log-likelihood (scalar, negative)
    """
    return -neg_loglike_from_P(P, y)


def loglike_per_obs(
    P: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute average log-likelihood per observation.

    Args:
        P: Probability matrix, shape (N, J)
        y: Observed choices, shape (N,)

    Returns:
        Average log-likelihood (scalar)
    """
    N = len(y)
    return loglike_from_P(P, y) / N


def avg_prob_chosen(
    P: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute average probability assigned to chosen alternative.

    A calibration metric: well-calibrated model should have
    this close to the observed choice rate.

    Args:
        P: Probability matrix, shape (N, J)
        y: Observed choices, shape (N,)

    Returns:
        Average probability of chosen alternative
    """
    p_chosen = P[np.arange(len(y)), y]
    return float(p_chosen.mean())


def low_prob_rate(
    P: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.01
) -> float:
    """Compute fraction of observations with very low predicted probability.

    Args:
        P: Probability matrix, shape (N, J)
        y: Observed choices, shape (N,)
        threshold: Probability threshold

    Returns:
        Fraction of observations where P(chosen) < threshold
    """
    p_chosen = P[np.arange(len(y)), y]
    return float((p_chosen < threshold).mean())


def predicted_shares(P: np.ndarray) -> np.ndarray:
    """Compute predicted mode shares (average predicted probabilities).

    Args:
        P: Probability matrix, shape (N, J)

    Returns:
        Array of shape (J,) with average probability for each alternative
    """
    return P.mean(axis=0)
