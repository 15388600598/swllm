"""
LLM-based synthetic data generation pipelines.

This module provides end-to-end generation pipelines:
1. Template creation from real data
2. Utility-based generation with softmax sampling
3. Residual-based generation with baseline MNL adjustment
4. Two-stage generation with repair
"""

import json
from typing import Dict, Any, Optional, List, Set, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_MODEL,
    REPAIR_MODEL,
    COMBO_COLS,
    get_format_config,
    DEFAULT_SYNTH_SIZE,
    DEFAULT_P_UNSEEN,
    DEFAULT_TAU,
    DEFAULT_LAMBDA,
)
from ..models.utils import softmax_rows
from .prompts import make_prompt, make_prompt_u, make_prompt_residual
from .batch_api import submit_batch, wait_and_download_safe
from .parsers import (
    parse_batch_output,
    parse_batch_output_util,
    parse_batch_output_residual,
    parse_batch_output_repair,
    get_failed_indices,
)


def get_unique_combos(df: pd.DataFrame, cols: List[str] = None) -> np.ndarray:
    """Get unique demographic combinations from DataFrame.

    Args:
        df: DataFrame
        cols: Columns to use (default: COMBO_COLS)

    Returns:
        Array of unique combination tuples
    """
    if cols is None:
        cols = COMBO_COLS

    unique_df = df[cols].dropna().drop_duplicates()
    return np.array([tuple(row) for row in unique_df.to_numpy()])


def create_templates(
    real_train: pd.DataFrame,
    real_test: Optional[pd.DataFrame] = None,
    N: int = DEFAULT_SYNTH_SIZE,
    p_unseen: float = DEFAULT_P_UNSEEN,
    seed: int = 123,
    cols: List[str] = None
) -> pd.DataFrame:
    """Create template DataFrame for synthetic generation.

    Templates are created by:
    1. Sampling rows from real_train (with replacement)
    2. Optionally mixing in "unseen" demographic combinations from test set

    Args:
        real_train: Real training DataFrame
        real_test: Real test DataFrame (for unseen combinations)
        N: Number of templates to create
        p_unseen: Probability of sampling unseen demographic combinations
        seed: Random seed
        cols: Columns for demographic combinations

    Returns:
        Template DataFrame with new IDs
    """
    if cols is None:
        cols = COMBO_COLS

    rng = np.random.default_rng(seed)

    # Get training combinations
    train_combos = get_unique_combos(real_train, cols)

    # Get unseen test combinations (if test data provided)
    unseen_test = np.array([])
    if real_test is not None:
        train_set = set(map(tuple, train_combos))
        test_combos = get_unique_combos(real_test, cols)
        unseen_test = np.array([c for c in test_combos if tuple(c) not in train_set])

    def draw_combo():
        if len(unseen_test) > 0 and rng.random() < p_unseen:
            return unseen_test[rng.integers(len(unseen_test))]
        return train_combos[rng.integers(len(train_combos))]

    # Sample base templates
    templates = real_train.sample(n=N, replace=True, random_state=seed).copy().reset_index(drop=True)

    # Replace demographic combinations
    for i in range(N):
        combo = draw_combo()
        for j, col in enumerate(cols):
            templates.loc[i, col] = combo[j]

    # Assign new synthetic IDs
    templates["ID"] = np.arange(1_000_000, 1_000_000 + N)

    return templates


def write_batch_jsonl_choice(
    templates: pd.DataFrame,
    jsonl_path: str,
    model: str = DEFAULT_MODEL,
    cot: bool = False
) -> None:
    """Write batch JSONL for choice prediction.

    Args:
        templates: Template DataFrame
        jsonl_path: Output path for JSONL
        model: Model name
        cot: Use chain-of-thought prompting
    """
    fmt = get_format_config("choice")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(len(templates)):
            row = templates.iloc[i]
            body = {
                "model": model,
                "input": [
                    {"role": "system", "content": "Return only strict JSON following the schema."},
                    {"role": "user", "content": make_prompt(row, cot=cot)}
                ],
                "text": {"format": fmt}
            }
            req = {"custom_id": f"req_{i}", "method": "POST", "url": "/v1/responses", "body": body}
            f.write(json.dumps(req) + "\n")


def write_batch_jsonl_utility(
    templates: pd.DataFrame,
    jsonl_path: str,
    model: str = DEFAULT_MODEL,
    cot: bool = False
) -> None:
    """Write batch JSONL for utility estimation.

    Args:
        templates: Template DataFrame
        jsonl_path: Output path for JSONL
        model: Model name
        cot: Use chain-of-thought prompting
    """
    fmt = get_format_config("utility")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(len(templates)):
            row = templates.iloc[i]
            body = {
                "model": model,
                "input": [
                    {"role": "system", "content": "Return only strict JSON following the schema."},
                    {"role": "user", "content": make_prompt_u(row, cot=cot)}
                ],
                "text": {"format": fmt}
            }
            req = {"custom_id": f"req_{i}", "method": "POST", "url": "/v1/responses", "body": body}
            f.write(json.dumps(req) + "\n")


def write_batch_jsonl_residual(
    templates: pd.DataFrame,
    jsonl_path: str,
    model: str = DEFAULT_MODEL,
    cot: bool = False
) -> None:
    """Write batch JSONL for residual estimation.

    Args:
        templates: Template DataFrame
        jsonl_path: Output path for JSONL
        model: Model name
        cot: Use chain-of-thought prompting
    """
    fmt = get_format_config("residual")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(len(templates)):
            row = templates.iloc[i]
            body = {
                "model": model,
                "input": [
                    {"role": "system", "content": "Return only strict JSON following the schema."},
                    {"role": "user", "content": make_prompt_residual(row, cot=cot)}
                ],
                "text": {"format": fmt}
            }
            req = {"custom_id": f"req_{i}", "method": "POST", "url": "/v1/responses", "body": body}
            f.write(json.dumps(req) + "\n")


def generate_from_utilities_batch(
    templates: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    tau: float = DEFAULT_TAU,
    jsonl_path: str = "./tmp_util_batch.jsonl",
    out_path: str = "./tmp_util_out.jsonl",
    seed: int = 123,
    cot: bool = False,
) -> pd.DataFrame:
    """Generate synthetic data using utility estimation.

    Pipeline:
    1. Write batch JSONL for utility estimation
    2. Submit batch and download results
    3. Parse utilities and apply softmax with temperature
    4. Sample choices from probability distribution

    Args:
        templates: Template DataFrame
        model: Model name
        tau: Softmax temperature (higher = more uniform)
        jsonl_path: Path for batch JSONL
        out_path: Path for output JSONL
        seed: Random seed
        cot: Use chain-of-thought prompting

    Returns:
        Synthetic DataFrame with CHOICE column
    """
    N = len(templates)

    # Write batch JSONL
    write_batch_jsonl_utility(templates, jsonl_path, model, cot)

    # Submit and download
    batch_id = submit_batch(jsonl_path)
    out_jsonl, _ = wait_and_download_safe(batch_id, out_path, poll_s=20)

    # Parse utilities
    U = parse_batch_output_util(out_jsonl, N)
    ok = np.isfinite(U).all(axis=1)
    print(f"parsed utilities: {ok.sum()} / {N}")

    # Handle partial failures
    if ok.sum() < N:
        templates_ok = templates.loc[np.where(ok)[0]].copy().reset_index(drop=True)
        U_ok = U[ok]
    else:
        templates_ok = templates.copy().reset_index(drop=True)
        U_ok = U

    # Sample choices from softmax
    P = softmax_rows(U_ok, tau=tau)
    rng = np.random.default_rng(seed)
    choice = 1 + np.array([rng.choice(3, p=p) for p in P])

    syn_u = templates_ok.copy()
    syn_u["CHOICE"] = choice.astype(int)

    return syn_u


def generate_from_residual_batch(
    templates: pd.DataFrame,
    score_fn,  # Function: df -> df with P_TRAIN, P_SM, P_CAR columns
    model: str = DEFAULT_MODEL,
    lam: float = DEFAULT_LAMBDA,
    jsonl_path: str = "./tmp_resid_batch.jsonl",
    out_path: str = "./tmp_resid_out.jsonl",
    seed: int = 123,
    cot: bool = False,
) -> pd.DataFrame:
    """Generate synthetic data using residual adjustment.

    Pipeline:
    1. Compute baseline MNL probabilities
    2. Get LLM residual adjustments
    3. Combine: logits' = log(P_baseline) + lambda * dU
    4. Sample choices from adjusted probabilities

    Args:
        templates: Template DataFrame
        score_fn: Function to compute baseline probabilities
        model: Model name
        lam: Residual perturbation strength (0.1-0.5 typical)
        jsonl_path: Path for batch JSONL
        out_path: Path for output JSONL
        seed: Random seed
        cot: Use chain-of-thought prompting

    Returns:
        Synthetic DataFrame with CHOICE column
    """
    N = len(templates)

    # Get baseline probabilities
    scored = score_fn(templates.copy())
    P = scored[["P_TRAIN", "P_SM", "P_CAR"]].to_numpy()

    # Convert to logits
    eps = 1e-12
    base_logits = np.log(np.clip(P, eps, 1.0))

    # Write batch JSONL
    write_batch_jsonl_residual(templates, jsonl_path, model, cot)

    # Submit and download
    batch_id = submit_batch(jsonl_path)
    out_jsonl, _ = wait_and_download_safe(batch_id, out_path, poll_s=20)

    # Parse residuals
    dU = parse_batch_output_residual(out_jsonl, N)
    ok = np.isfinite(dU).all(axis=1)
    print(f"parsed residuals: {ok.sum()} / {N}")

    # Fill missing residuals with 0 (no adjustment)
    dU[~ok] = 0.0

    # Combine logits with residual adjustment
    logits = base_logits + lam * dU
    P_new = softmax_rows(logits)

    # Sample choices
    rng = np.random.default_rng(seed)
    choice = 1 + np.array([rng.choice(3, p=p) for p in P_new])

    syn_r = templates.copy()
    syn_r["CHOICE"] = choice.astype(int)

    return syn_r


def generate_two_stage(
    templates: pd.DataFrame,
    score_fn,  # Function for baseline scoring
    model_stage1: str = DEFAULT_MODEL,
    model_stage2: str = REPAIR_MODEL,
    low_prob_threshold: float = 0.01,
    jsonl_dir: str = "./",
    seed: int = 123,
    cot_stage1: bool = False,
    cot_stage2: bool = True,
) -> pd.DataFrame:
    """Two-stage generation with repair.

    Stage 1: Generate choices with faster model
    Stage 2: Repair failed/low-probability choices with stronger model

    Args:
        templates: Template DataFrame
        score_fn: Function to compute baseline probabilities
        model_stage1: Model for stage 1
        model_stage2: Model for stage 2 (repair)
        low_prob_threshold: Threshold for identifying bad choices
        jsonl_dir: Directory for JSONL files
        seed: Random seed
        cot_stage1: Use CoT in stage 1
        cot_stage2: Use CoT in stage 2

    Returns:
        Synthetic DataFrame with CHOICE column
    """
    N = len(templates)
    jsonl_dir = Path(jsonl_dir)

    # Stage 1: Initial generation
    print("=== Stage 1: Initial generation ===")
    jsonl1 = str(jsonl_dir / "batch_stage1.jsonl")
    out1 = str(jsonl_dir / "batch_stage1_out.jsonl")

    write_batch_jsonl_choice(templates, jsonl1, model_stage1, cot_stage1)
    batch1_id = submit_batch(jsonl1)
    out1_path, _ = wait_and_download_safe(batch1_id, out1, poll_s=20)

    choices1 = parse_batch_output(out1_path, N)
    print(f"Stage 1 parsed: {sum(c is not None for c in choices1)} / {N}")

    # Create intermediate DataFrame
    syn1 = templates.copy()
    syn1["CHOICE"] = choices1

    # Identify bad indices (None or low probability)
    syn1_ok = syn1.dropna(subset=["CHOICE"]).copy()
    syn1_ok["CHOICE"] = syn1_ok["CHOICE"].astype(int)

    scored1 = score_fn(syn1_ok)

    # Compute chosen probability
    ch = scored1["CHOICE"].to_numpy()
    chosen_p = np.where(
        ch == 1, scored1["P_TRAIN"].to_numpy(),
        np.where(ch == 2, scored1["P_SM"].to_numpy(), scored1["P_CAR"].to_numpy())
    )

    bad_idx = set(syn1.index[syn1["CHOICE"].isna()].tolist())
    bad_idx |= set(scored1.index[chosen_p < low_prob_threshold].tolist())

    print(f"Stage 1 bad indices: {len(bad_idx)}")

    if len(bad_idx) == 0:
        return syn1_ok

    # Stage 2: Repair
    print("=== Stage 2: Repair ===")
    bad_list = sorted(list(bad_idx))
    M = len(bad_list)

    jsonl2 = str(jsonl_dir / "batch_stage2.jsonl")
    out2 = str(jsonl_dir / "batch_stage2_out.jsonl")

    fmt = get_format_config("choice")

    with open(jsonl2, "w", encoding="utf-8") as f:
        for i in bad_list:
            row = templates.loc[i]
            body = {
                "model": model_stage2,
                "input": [
                    {"role": "system", "content": "Return only strict JSON following the schema."},
                    {"role": "user", "content": make_prompt(row, cot=cot_stage2)}
                ],
                "text": {"format": fmt}
            }
            req = {"custom_id": f"bad_{i}", "method": "POST", "url": "/v1/responses", "body": body}
            f.write(json.dumps(req) + "\n")

    batch2_id = submit_batch(jsonl2)
    out2_path, _ = wait_and_download_safe(batch2_id, out2, poll_s=20)

    fixes = parse_batch_output_repair(out2_path, prefix="bad_")
    print(f"Stage 2 repaired: {len(fixes)} / {M}")

    # Apply fixes
    syn_final = syn1.copy()
    for i, ch in fixes.items():
        syn_final.loc[i, "CHOICE"] = ch

    # Clean up
    syn_final = syn_final.dropna(subset=["CHOICE"]).copy()
    syn_final["CHOICE"] = syn_final["CHOICE"].astype(int)

    print(f"Final synthetic N: {len(syn_final)}")

    return syn_final
