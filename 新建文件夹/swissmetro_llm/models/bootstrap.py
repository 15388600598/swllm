"""
Clustered Bootstrap inference for MNL models.

This module provides functions for:
- Resampling by respondent ID (cluster bootstrap)
- Computing bootstrap standard errors and confidence intervals
- Hypothesis testing with bootstrap p-values
"""

from typing import Dict, Any, Optional, List, Tuple, Set

import numpy as np
import pandas as pd
from scipy.stats import norm

from .mnl import fit_step2_main


def resample_by_id(
    df: pd.DataFrame,
    ids: Set[int],
    rng: np.random.Generator
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Resample DataFrame by respondent ID with replacement.

    This implements clustered bootstrap: entire respondents are
    resampled, keeping all their observations together.

    Args:
        df: DataFrame with 'ID' column
        ids: Set of unique respondent IDs
        rng: NumPy random generator

    Returns:
        Tuple of (resampled DataFrame, resampled IDs array)
    """
    ids_arr = np.array(list(ids))
    resampled_ids = rng.choice(ids_arr, size=len(ids_arr), replace=True)

    # Build resampled DataFrame
    dfs = []
    for rid in resampled_ids:
        dfs.append(df[df["ID"] == rid])

    df_resampled = pd.concat(dfs, ignore_index=True)
    return df_resampled, resampled_ids


def cluster_bootstrap_thetas(
    df_train: pd.DataFrame,
    train_ids: Set[int],
    tt_scale: np.ndarray,
    co_scale: np.ndarray,
    he_scale: np.ndarray,
    theta_init: Optional[np.ndarray] = None,
    B: int = 30,
    seed: int = 2025,
    maxiter: int = 20000,
    maxfun: int = 20000,
    verbose_every: int = 5
) -> np.ndarray:
    """Run clustered bootstrap to get parameter distribution.

    Resamples respondents B times, fits MNL model on each resample,
    and collects successful parameter estimates.

    Args:
        df_train: Training DataFrame
        train_ids: Set of training respondent IDs
        tt_scale, co_scale, he_scale: Scale arrays
        theta_init: Initial parameters (uses warm-start from successful fits)
        B: Number of bootstrap samples
        seed: Random seed
        maxiter: Maximum optimizer iterations
        maxfun: Maximum function evaluations
        verbose_every: Print progress every N iterations

    Returns:
        Array of shape (B_success, n_params) with successful parameter estimates
    """
    rng = np.random.default_rng(seed)
    thetas = []
    kept = 0

    # Use theta_init as starting point
    current_theta = theta_init

    for b in range(1, B + 1):
        # Resample respondents with replacement
        df_b, _ = resample_by_id(df_train, train_ids, rng)

        # Fit model with warm start
        res_b = fit_step2_main(
            df_b,
            tt_scale, co_scale, he_scale,
            theta_init=current_theta,
            maxiter=maxiter,
            maxfun=maxfun
        )

        # Retry机制：若没收敛但参数有限，用本次结果作为初值再跑一次
        if (not res_b.success) and hasattr(res_b, "x") and np.all(np.isfinite(res_b.x)):
            res_b2 = fit_step2_main(
                df_b,
                tt_scale, co_scale, he_scale,
                theta_init=res_b.x,
                maxiter=maxiter,
                maxfun=maxfun
            )
            res_b = res_b2

        if res_b.success and np.all(np.isfinite(res_b.x)):
            thetas.append(res_b.x.copy())
            kept += 1
            # Use this result as warm start for next iteration
            current_theta = res_b.x
        # If not successful, keep using previous warm start

        if (b % verbose_every) == 0:
            print(f"[bootstrap {b}/{B}] kept_success={kept}")

    if kept > 0:
        thetas = np.array(thetas)
    else:
        # Return empty array with correct shape
        n_params = len(theta_init) if theta_init is not None else 42
        thetas = np.zeros((0, n_params))

    return thetas


def compute_bootstrap_se(thetas_boot: np.ndarray) -> np.ndarray:
    """Compute bootstrap standard errors.

    Args:
        thetas_boot: Array of shape (B, n_params) from cluster_bootstrap_thetas

    Returns:
        Standard errors, shape (n_params,)
    """
    if len(thetas_boot) == 0:
        raise ValueError("No successful bootstrap samples")

    return thetas_boot.std(axis=0, ddof=1)


def compute_bootstrap_ci(
    thetas_boot: np.ndarray,
    alpha: float = 0.05,
    method: str = "percentile"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bootstrap confidence intervals.

    Args:
        thetas_boot: Array of shape (B, n_params)
        alpha: Significance level (0.05 for 95% CI)
        method: Either "percentile" or "normal"

    Returns:
        Tuple of (lower_bounds, upper_bounds), each shape (n_params,)
    """
    if len(thetas_boot) == 0:
        raise ValueError("No successful bootstrap samples")

    if method == "percentile":
        lower = np.percentile(thetas_boot, 100 * alpha / 2, axis=0)
        upper = np.percentile(thetas_boot, 100 * (1 - alpha / 2), axis=0)
    elif method == "normal":
        theta_mean = thetas_boot.mean(axis=0)
        se = compute_bootstrap_se(thetas_boot)
        z = norm.ppf(1 - alpha / 2)
        lower = theta_mean - z * se
        upper = theta_mean + z * se
    else:
        raise ValueError(f"Unknown method: {method}")

    return lower, upper


def compute_z_statistics(
    theta_point: np.ndarray,
    thetas_boot: np.ndarray
) -> np.ndarray:
    """Compute z-statistics for hypothesis testing.

    Args:
        theta_point: Point estimates, shape (n_params,)
        thetas_boot: Bootstrap samples, shape (B, n_params)

    Returns:
        z-statistics, shape (n_params,)
    """
    se = compute_bootstrap_se(thetas_boot)
    # Avoid division by zero
    se = np.where(se > 0, se, np.inf)
    return theta_point / se


def compute_bootstrap_pvalues(
    theta_point: np.ndarray,
    thetas_boot: np.ndarray,
    method: str = "normal"
) -> np.ndarray:
    """Compute p-values for hypothesis testing (H0: theta = 0).

    Args:
        theta_point: Point estimates, shape (n_params,)
        thetas_boot: Bootstrap samples, shape (B, n_params)
        method: Either "normal" (z-test) or "symmetric" (bootstrap percentile)

    Returns:
        Two-tailed p-values, shape (n_params,)
    """
    if method == "normal":
        z = compute_z_statistics(theta_point, thetas_boot)
        p = 2 * (1 - norm.cdf(np.abs(z)))
    elif method == "symmetric":
        # Symmetric two-tailed bootstrap p-value
        # Proportion of bootstrap samples with |theta_b - theta_point| >= |theta_point|
        deviations = np.abs(thetas_boot - theta_point)
        threshold = np.abs(theta_point)
        p = (deviations >= threshold).mean(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return p


def get_significance_stars(p: float) -> str:
    """Convert p-value to significance stars.

    Args:
        p: p-value

    Returns:
        Significance indicator string
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def create_bootstrap_summary_table(
    theta_point: np.ndarray,
    thetas_boot: np.ndarray,
    param_names: List[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """Create summary table with bootstrap inference results.

    Args:
        theta_point: Point estimates, shape (n_params,)
        thetas_boot: Bootstrap samples, shape (B, n_params)
        param_names: List of parameter names
        alpha: Significance level for CI

    Returns:
        DataFrame with columns: param, coef, boot_se, z, p_normal, p_boot, sig, CI_lower, CI_upper
    """
    se = compute_bootstrap_se(thetas_boot)
    z = compute_z_statistics(theta_point, thetas_boot)
    p_normal = compute_bootstrap_pvalues(theta_point, thetas_boot, method="normal")
    p_boot = compute_bootstrap_pvalues(theta_point, thetas_boot, method="symmetric")
    ci_lower, ci_upper = compute_bootstrap_ci(thetas_boot, alpha=alpha, method="percentile")

    records = []
    for i, name in enumerate(param_names):
        records.append({
            "param": name,
            "coef": theta_point[i],
            "boot_se": se[i],
            "z": z[i],
            "p_normal": p_normal[i],
            "p_boot": p_boot[i],
            "sig": get_significance_stars(p_boot[i]),
            "CI_lower": ci_lower[i],
            "CI_upper": ci_upper[i],
        })

    return pd.DataFrame(records)
