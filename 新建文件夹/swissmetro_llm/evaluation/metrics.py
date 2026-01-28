"""
Comprehensive evaluation metrics for synthetic data.

This module provides functions that combine feasibility, diversity,
and downstream metrics into unified evaluation reports.
"""

from typing import Dict, Any, Optional

import pandas as pd

from .feasibility import feasibility_min, feasibility_strict
from .diversity import diversity_report
from .downstream import score_with_baseline_mnl, downstream_metrics


def evaluate_one(
    synth_df: pd.DataFrame,
    beta: Dict[str, float],
    scales: Dict[str, Any],
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    label: str = "synth"
) -> Dict[str, Any]:
    """Comprehensive evaluation of synthetic data.

    Combines:
    - Feasibility checks (fz_*)
    - Diversity metrics (dv_*)
    - Downstream fit metrics (ds_*)

    Args:
        synth_df: Synthetic DataFrame to evaluate
        beta: Dictionary of MNL parameters
        scales: Dictionary with tt_scale, co_scale, he_scale
        real_train: Real training DataFrame
        real_test: Real test DataFrame
        label: Label for this evaluation

    Returns:
        Dictionary with all metrics, prefixed by category
    """
    out = {"label": label, "N_synth": int(len(synth_df))}

    # Feasibility
    fz = feasibility_min(synth_df)
    out["fz_row_na_rate"] = fz["row_na_rate"]
    out["fz_missing_cols"] = str(fz["missing_cols"])

    # Diversity
    dv = diversity_report(synth_df, real_train, real_test)
    for k, v in dv.items():
        out[f"dv_{k}"] = v

    # Downstream consistency
    scored = score_with_baseline_mnl(synth_df, beta, scales)
    ds = downstream_metrics(scored)
    for k, v in ds.items():
        out[f"ds_{k}"] = v

    return out


def evaluate_one_min(
    synth_df: pd.DataFrame,
    beta: Dict[str, float],
    scales: Dict[str, Any],
    label: str = "synth"
) -> Dict[str, Any]:
    """Minimal evaluation of synthetic data.

    Only includes feasibility and downstream metrics (no diversity).

    Args:
        synth_df: Synthetic DataFrame to evaluate
        beta: Dictionary of MNL parameters
        scales: Dictionary with tt_scale, co_scale, he_scale
        label: Label for this evaluation

    Returns:
        Dictionary with metrics
    """
    out = {"label": label, "N_synth": int(len(synth_df))}

    # Feasibility
    fz = feasibility_min(synth_df)
    out["fz_row_na_rate"] = fz["row_na_rate"]
    out["fz_missing_cols"] = str(fz["missing_cols"])

    # Downstream
    scored = score_with_baseline_mnl(synth_df, beta, scales)
    ds = downstream_metrics(scored)
    for k, v in ds.items():
        out[f"ds_{k}"] = v

    return out


def evaluate_multiple(
    synth_dfs: Dict[str, pd.DataFrame],
    beta: Dict[str, float],
    scales: Dict[str, Any],
    real_train: pd.DataFrame,
    real_test: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate multiple synthetic datasets.

    Args:
        synth_dfs: Dictionary mapping labels to synthetic DataFrames
        beta: Dictionary of MNL parameters
        scales: Dictionary with tt_scale, co_scale, he_scale
        real_train: Real training DataFrame
        real_test: Real test DataFrame

    Returns:
        DataFrame with one row per synthetic dataset
    """
    results = []
    for label, synth_df in synth_dfs.items():
        row = evaluate_one(synth_df, beta, scales, real_train, real_test, label)
        results.append(row)

    return pd.DataFrame(results)


def compare_to_real(
    synth_df: pd.DataFrame,
    beta: Dict[str, float],
    scales: Dict[str, Any],
    real_train: pd.DataFrame,
    real_test: pd.DataFrame
) -> pd.DataFrame:
    """Compare synthetic data to real train and test.

    Returns metrics for synth, real_train, and real_test side by side.

    Args:
        synth_df: Synthetic DataFrame
        beta: Dictionary of MNL parameters
        scales: Dictionary with tt_scale, co_scale, he_scale
        real_train: Real training DataFrame
        real_test: Real test DataFrame

    Returns:
        DataFrame with comparison
    """
    results = []

    # Evaluate real train
    scored_train = score_with_baseline_mnl(real_train, beta, scales)
    train_metrics = downstream_metrics(scored_train)
    train_metrics["label"] = "real_train"
    results.append(train_metrics)

    # Evaluate real test
    scored_test = score_with_baseline_mnl(real_test, beta, scales)
    test_metrics = downstream_metrics(scored_test)
    test_metrics["label"] = "real_test"
    results.append(test_metrics)

    # Evaluate synthetic
    scored_synth = score_with_baseline_mnl(synth_df, beta, scales)
    synth_metrics = downstream_metrics(scored_synth)
    synth_metrics["label"] = "synthetic"
    results.append(synth_metrics)

    return pd.DataFrame(results).set_index("label")


def compute_choice_shares(df: pd.DataFrame, choice_col: str = "CHOICE") -> Dict[str, float]:
    """Compute actual choice shares from DataFrame.

    Args:
        df: DataFrame with choice column
        choice_col: Name of choice column

    Returns:
        Dictionary with choice shares for each alternative
    """
    ch = pd.to_numeric(df[choice_col], errors="coerce")
    valid = ch.isin([1, 2, 3])
    ch_valid = ch[valid]

    if len(ch_valid) == 0:
        return {"train": 0.0, "sm": 0.0, "car": 0.0}

    shares = ch_valid.value_counts(normalize=True).reindex([1, 2, 3]).fillna(0.0)

    return {
        "train": float(shares.loc[1]),
        "sm": float(shares.loc[2]),
        "car": float(shares.loc[3]),
    }


def mode_share_comparison(
    synth_df: pd.DataFrame,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame
) -> pd.DataFrame:
    """Compare mode shares across datasets.

    Args:
        synth_df: Synthetic DataFrame
        real_train: Real training DataFrame
        real_test: Real test DataFrame

    Returns:
        DataFrame with mode share comparison
    """
    synth_shares = compute_choice_shares(synth_df)
    train_shares = compute_choice_shares(real_train)
    test_shares = compute_choice_shares(real_test)

    return pd.DataFrame({
        "synth": synth_shares,
        "real_train": train_shares,
        "real_test": test_shares,
    })
