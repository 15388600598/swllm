"""
Parameter stability analysis for real vs real+synthetic data.

This module provides functions for:
1. Creating augmented datasets (real + synthetic)
2. Estimating parameters on base and augmented data
3. Generating stability comparison tables with convergence checks
4. Formatting results for publication
"""

from typing import Dict, Any, Optional, Tuple, List
import os

import numpy as np
import pandas as pd

from ..models.mnl import fit_step2_main, get_step2_main_param_names
from ..data.loader import build_matrices
from ..data.preprocessing import compute_scales_from_train, apply_scales


def make_augmented(
    real_train: pd.DataFrame,
    syn_df: pd.DataFrame,
    ratio: float = 1.0,
    seed: int = 0
) -> pd.DataFrame:
    """Combine real training data with synthetic data.

    Args:
        real_train: Real training DataFrame
        syn_df: Synthetic DataFrame (must have CHOICE column)
        ratio: Ratio of synthetic to real data (1.0 = same size)
        seed: Random seed for sampling

    Returns:
        Augmented DataFrame with both real and synthetic rows
    """
    n_real = len(real_train)
    n_syn_target = int(n_real * ratio)

    if n_syn_target > len(syn_df):
        # Sample with replacement if not enough synthetic
        syn_sample = syn_df.sample(n=n_syn_target, replace=True, random_state=seed)
    elif n_syn_target < len(syn_df):
        # Sample without replacement
        syn_sample = syn_df.sample(n=n_syn_target, replace=False, random_state=seed)
    else:
        syn_sample = syn_df.copy()

    # Combine
    df_aug = pd.concat([real_train, syn_sample], ignore_index=True)

    return df_aug


def estimate_theta_step2main(
    df: pd.DataFrame,
    tt_scale: np.ndarray,
    co_scale: np.ndarray,
    he_scale: np.ndarray,
    theta_init: Optional[np.ndarray] = None,
    maxiter: int = 20000,
    maxfun: int = 20000,
    verbose: bool = True
) -> Tuple[np.ndarray, Any, bool, str]:
    """Estimate Step2(main) parameters with convergence info.

    Args:
        df: DataFrame with choice data
        tt_scale, co_scale, he_scale: Scale arrays
        theta_init: Initial parameter values
        maxiter, maxfun: Optimizer limits
        verbose: Print progress

    Returns:
        Tuple of (theta, result, converged, message)
    """
    res = fit_step2_main(
        df, tt_scale, co_scale, he_scale,
        theta_init=theta_init,
        maxiter=maxiter,
        maxfun=maxfun
    )

    converged = res.success
    message = res.message if hasattr(res, 'message') else ""

    if verbose:
        print(f"fit_step2_main success: {converged} {message}")

    return res.x, res, converged, message


def param_stability_table(
    theta_base: np.ndarray,
    theta_new: np.ndarray,
    converged_base: bool,
    converged_new: bool,
    boot_se: Optional[np.ndarray] = None,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    p_boot: Optional[np.ndarray] = None,
    param_names: Optional[List[str]] = None,
    K: int = 18
) -> pd.DataFrame:
    """Generate parameter stability comparison table.

    Args:
        theta_base: Parameters from base (real only) estimation
        theta_new: Parameters from augmented (real + synthetic) estimation
        converged_base: Whether base estimation converged
        converged_new: Whether augmented estimation converged
        boot_se: Bootstrap standard errors (optional)
        ci_lower, ci_upper: Confidence interval bounds (optional)
        p_boot: Bootstrap p-values (optional)
        param_names: List of parameter names
        K: Number of individual features (default 18)

    Returns:
        DataFrame with stability comparison
    """
    if param_names is None:
        param_names = get_step2_main_param_names(K)

    n_params = len(theta_base)

    # Build table
    records = []
    for i in range(n_params):
        name = param_names[i] if i < len(param_names) else f"param_{i}"

        coef_base = theta_base[i]
        coef_new = theta_new[i]
        diff = coef_new - coef_base
        abs_diff = abs(diff)

        record = {
            "param": name,
            "coef_base": coef_base,
            "coef_new": coef_new,
            "diff": diff,
            "abs_diff": abs_diff,
            "converged_base": converged_base,
            "converged_new": converged_new,
        }

        # Add bootstrap info if available
        if boot_se is not None and i < len(boot_se):
            se = boot_se[i]
            record["boot_se"] = se

            # Handle boot_se=0 case
            if se > 0:
                record["z_diff_vs_bootse"] = diff / se
            else:
                record["z_diff_vs_bootse"] = np.nan  # 可报告修正：se=0时z设为NaN

        if ci_lower is not None and i < len(ci_lower):
            record["ci_2.5"] = ci_lower[i]

        if ci_upper is not None and i < len(ci_upper):
            record["ci_97.5"] = ci_upper[i]

        if p_boot is not None and i < len(p_boot):
            se = boot_se[i] if boot_se is not None and i < len(boot_se) else 1
            if se > 0:
                record["p_boot"] = p_boot[i]
            else:
                record["p_boot"] = 1.0  # 可报告修正：se=0时p设为1

        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values("abs_diff", ascending=False).reset_index(drop=True)

    return df


def stability_analysis(
    real_train: pd.DataFrame,
    syn_df: pd.DataFrame,
    tt_scale: np.ndarray,
    co_scale: np.ndarray,
    he_scale: np.ndarray,
    theta_init: Optional[np.ndarray] = None,
    boot_se: Optional[np.ndarray] = None,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    p_boot: Optional[np.ndarray] = None,
    ratio: float = 1.0,
    seed: int = 0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Full stability analysis with convergence checks.

    Args:
        real_train: Real training DataFrame
        syn_df: Synthetic DataFrame
        tt_scale, co_scale, he_scale: Scale arrays
        theta_init: Initial parameters (typically from prior estimation)
        boot_se: Bootstrap standard errors from prior analysis
        ci_lower, ci_upper: Confidence interval bounds
        p_boot: Bootstrap p-values
        ratio: Synthetic to real ratio
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (stability_table DataFrame, metadata dict)
    """
    # Step 1: Estimate on base (real only)
    if verbose:
        print("=== Estimating on real_train (base) ===")

    theta_base, res_base, conv_base, msg_base = estimate_theta_step2main(
        real_train, tt_scale, co_scale, he_scale,
        theta_init=theta_init,
        verbose=verbose
    )

    # Step 2: Create augmented dataset
    df_aug = make_augmented(real_train, syn_df, ratio=ratio, seed=seed)
    if verbose:
        print(f"Augmented dataset: {len(df_aug)} rows (real: {len(real_train)}, syn: {len(df_aug) - len(real_train)})")

    # Step 3: Estimate on augmented
    if verbose:
        print("=== Estimating on augmented (real + synthetic) ===")

    theta_new, res_new, conv_new, msg_new = estimate_theta_step2main(
        df_aug, tt_scale, co_scale, he_scale,
        theta_init=theta_base,  # Use base as warm start
        verbose=verbose
    )

    # Step 4: Build stability table
    stab_df = param_stability_table(
        theta_base=theta_base,
        theta_new=theta_new,
        converged_base=conv_base,
        converged_new=conv_new,
        boot_se=boot_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_boot=p_boot
    )

    # Metadata
    metadata = {
        "n_real": len(real_train),
        "n_syn": len(df_aug) - len(real_train),
        "n_aug": len(df_aug),
        "ratio": ratio,
        "converged_base": conv_base,
        "converged_new": conv_new,
        "message_base": msg_base,
        "message_new": msg_new,
        "ll_base": -res_base.fun,
        "ll_new": -res_new.fun,
        "theta_base": theta_base,
        "theta_new": theta_new,
    }

    return stab_df, metadata


def format_stability_for_report(
    stab_df: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: Optional[str] = None,
    top_n: int = 25
) -> str:
    """Format stability table for publication/report.

    Adds notes about convergence status and boot_se=0 cases.

    Args:
        stab_df: Stability DataFrame from stability_analysis
        metadata: Metadata dict from stability_analysis
        output_path: Path to save CSV (optional)
        top_n: Number of top parameters to show in summary

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Parameter Stability Analysis: Real vs Real+Synthetic")
    lines.append("=" * 60)
    lines.append("")

    # Data info
    lines.append(f"Real training samples: {metadata['n_real']}")
    lines.append(f"Synthetic samples added: {metadata['n_syn']}")
    lines.append(f"Augmented total: {metadata['n_aug']}")
    lines.append(f"Ratio (syn/real): {metadata['ratio']:.2f}")
    lines.append("")

    # Convergence status
    lines.append("Convergence Status:")
    lines.append(f"  Base (real only): {'CONVERGED' if metadata['converged_base'] else 'NOT CONVERGED'}")
    if not metadata['converged_base']:
        lines.append(f"    Message: {metadata['message_base']}")
    lines.append(f"  Augmented (real+syn): {'CONVERGED' if metadata['converged_new'] else 'NOT CONVERGED'}")
    if not metadata['converged_new']:
        lines.append(f"    Message: {metadata['message_new']}")
    lines.append("")

    # Log-likelihood
    lines.append(f"Log-likelihood (base): {metadata['ll_base']:.2f}")
    lines.append(f"Log-likelihood (augmented): {metadata['ll_new']:.2f}")
    lines.append("")

    # Parameters with largest changes
    lines.append(f"Top {top_n} parameters by |diff|:")
    lines.append("-" * 60)

    # Check for boot_se=0 cases
    has_zero_se = False
    if "boot_se" in stab_df.columns:
        zero_se_params = stab_df[stab_df["boot_se"] == 0]["param"].tolist()
        if zero_se_params:
            has_zero_se = True

    # Format table header
    header = f"{'param':<20} {'coef_base':>10} {'coef_new':>10} {'diff':>10}"
    if "boot_se" in stab_df.columns:
        header += f" {'boot_se':>10} {'z_diff':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, row in stab_df.head(top_n).iterrows():
        line = f"{row['param']:<20} {row['coef_base']:>10.4f} {row['coef_new']:>10.4f} {row['diff']:>10.4f}"
        if "boot_se" in stab_df.columns:
            se = row.get("boot_se", np.nan)
            z = row.get("z_diff_vs_bootse", np.nan)
            se_str = f"{se:.4f}" if pd.notna(se) else "NA"
            z_str = f"{z:.2f}" if pd.notna(z) else "NA"
            line += f" {se_str:>10} {z_str:>10}"
        lines.append(line)

    # Notes
    lines.append("")
    lines.append("Notes:")
    if not metadata['converged_base'] or not metadata['converged_new']:
        lines.append("  * WARNING: One or more estimations did not converge. Interpret with caution.")
    if has_zero_se:
        lines.append(f"  * boot_se=0 for params: {', '.join(zero_se_params[:5])}{'...' if len(zero_se_params) > 5 else ''}")
        lines.append("    (z_diff and p_boot are NA/1.0 for these parameters)")

    summary = "\n".join(lines)

    # Save CSV if requested
    if output_path:
        stab_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    return summary


def load_bootstrap_results(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load bootstrap results from CSV (e.g., Appendix_A1_full_params.csv).

    Args:
        csv_path: Path to bootstrap results CSV

    Returns:
        Tuple of (boot_se, ci_lower, ci_upper, p_boot) arrays
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Bootstrap results file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    boot_se = df["boot_se"].to_numpy() if "boot_se" in df.columns else None
    ci_lower = df["ci_2.5"].to_numpy() if "ci_2.5" in df.columns else None
    ci_upper = df["ci_97.5"].to_numpy() if "ci_97.5" in df.columns else None
    p_boot = df["p_boot"].to_numpy() if "p_boot" in df.columns else None

    return boot_se, ci_lower, ci_upper, p_boot
