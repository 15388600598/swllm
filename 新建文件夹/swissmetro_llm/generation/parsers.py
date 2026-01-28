"""
Output parsing utilities for LLM batch responses.

This module provides functions to parse different types of
LLM outputs (choice, utility, residual) from batch API responses.
"""

import json
from typing import Optional, List, Dict, Any

import numpy as np


def extract_output_text(resp_body: Dict[str, Any]) -> Optional[str]:
    """Extract output text from response body.

    Handles different response structures from OpenAI API.

    Args:
        resp_body: Response body dictionary

    Returns:
        Extracted text or None if not found
    """
    if not isinstance(resp_body, dict):
        return None

    # Try direct output_text field first
    if resp_body.get("output_text") is not None:
        return resp_body["output_text"]

    # Try nested structure
    for item in resp_body.get("output", []):
        for c in item.get("content", []):
            if c.get("type") in ["output_text", "text"] and "text" in c:
                return c["text"]

    return None


def parse_batch_output(
    out_jsonl_path: str,
    N: int
) -> List[Optional[int]]:
    """Parse batch output for choice prediction.

    Args:
        out_jsonl_path: Path to output JSONL file
        N: Expected number of responses

    Returns:
        List of choices (1, 2, 3) or None for failed/missing responses
    """
    choices: List[Optional[int]] = [None] * N

    with open(out_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id", "")

            # Parse index from custom_id (e.g., "req_0", "req_1")
            if not cid.startswith("req_"):
                continue
            i = int(cid.replace("req_", ""))

            # Skip error responses
            if obj.get("error") is not None:
                continue

            resp = obj.get("response", {})
            body = resp.get("body", {})
            txt = extract_output_text(body)

            if not txt:
                continue

            try:
                d = json.loads(txt)
                choice = int(d["CHOICE"])
                if choice in [1, 2, 3]:
                    choices[i] = choice
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

    return choices


def parse_batch_output_util(
    out_jsonl_path: str,
    N: int
) -> np.ndarray:
    """Parse batch output for utility estimation.

    Args:
        out_jsonl_path: Path to output JSONL file
        N: Expected number of responses

    Returns:
        Utility matrix (N, 3) with columns [U_TRAIN, U_SM, U_CAR].
        Missing values are NaN.
    """
    U = np.full((N, 3), np.nan, dtype=float)

    with open(out_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id", "")

            if not cid.startswith("req_"):
                continue
            i = int(cid.replace("req_", ""))

            if obj.get("error") is not None:
                continue

            body = obj.get("response", {}).get("body", {})
            txt = extract_output_text(body)

            if not txt:
                continue

            try:
                d = json.loads(txt)
                U[i, 0] = float(d["U_TRAIN"])
                U[i, 1] = float(d["U_SM"])
                U[i, 2] = float(d["U_CAR"])
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass

    return U


def parse_batch_output_residual(
    out_jsonl_path: str,
    N: int
) -> np.ndarray:
    """Parse batch output for residual/preference adjustments.

    Args:
        out_jsonl_path: Path to output JSONL file
        N: Expected number of responses

    Returns:
        Residual matrix (N, 3) with columns [dU_TRAIN, dU_SM, dU_CAR].
        Missing values are NaN.
    """
    dU = np.full((N, 3), np.nan, dtype=float)

    with open(out_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id", "")

            if not cid.startswith("req_"):
                continue
            i = int(cid.replace("req_", ""))

            if obj.get("error") is not None:
                continue

            body = obj.get("response", {}).get("body", {})
            txt = extract_output_text(body)

            if not txt:
                continue

            try:
                d = json.loads(txt)
                dU[i, 0] = float(d["dU_TRAIN"])
                dU[i, 1] = float(d["dU_SM"])
                dU[i, 2] = float(d["dU_CAR"])
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass

    return dU


def parse_batch_output_repair(
    out_jsonl_path: str,
    prefix: str = "bad_"
) -> Dict[int, int]:
    """Parse batch output for repair stage.

    Args:
        out_jsonl_path: Path to output JSONL file
        prefix: Custom ID prefix (e.g., "bad_" for repair requests)

    Returns:
        Dictionary mapping index to choice value
    """
    fixes: Dict[int, int] = {}

    with open(out_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id", "")

            if not cid.startswith(prefix):
                continue
            i = int(cid.replace(prefix, ""))

            if obj.get("error") is not None:
                continue

            body = obj.get("response", {}).get("body", {})
            txt = extract_output_text(body)

            if not txt:
                continue

            try:
                d = json.loads(txt)
                choice = int(d["CHOICE"])
                if choice in [1, 2, 3]:
                    fixes[i] = choice
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

    return fixes


def count_successful_parses(
    choices: List[Optional[int]]
) -> int:
    """Count number of successfully parsed responses.

    Args:
        choices: List of parsed choices (may contain None)

    Returns:
        Count of non-None values
    """
    return sum(c is not None for c in choices)


def get_failed_indices(
    choices: List[Optional[int]]
) -> List[int]:
    """Get indices of failed/missing responses.

    Args:
        choices: List of parsed choices

    Returns:
        List of indices where choice is None
    """
    return [i for i, c in enumerate(choices) if c is None]


def validate_utilities(U: np.ndarray) -> Dict[str, Any]:
    """Validate utility matrix.

    Args:
        U: Utility matrix (N, 3)

    Returns:
        Dictionary with validation statistics
    """
    ok = np.isfinite(U).all(axis=1)

    return {
        "n_total": len(U),
        "n_valid": int(ok.sum()),
        "n_invalid": int((~ok).sum()),
        "valid_rate": float(ok.mean()),
        "has_inf": bool(np.isinf(U).any()),
        "has_nan": bool(np.isnan(U).any()),
    }


def validate_residuals(dU: np.ndarray) -> Dict[str, Any]:
    """Validate residual matrix.

    Args:
        dU: Residual matrix (N, 3)

    Returns:
        Dictionary with validation statistics
    """
    ok = np.isfinite(dU).all(axis=1)

    # Check if values are within expected range [-1, 1]
    in_range = (dU >= -1) & (dU <= 1)
    all_in_range = in_range.all(axis=1)

    return {
        "n_total": len(dU),
        "n_valid": int(ok.sum()),
        "n_invalid": int((~ok).sum()),
        "valid_rate": float(ok.mean()),
        "n_out_of_range": int((ok & ~all_in_range).sum()),
        "out_of_range_rate": float((ok & ~all_in_range).mean()),
    }
