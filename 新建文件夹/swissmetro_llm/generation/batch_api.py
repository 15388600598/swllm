"""
OpenAI Batch API utilities.

This module provides functions for submitting, monitoring, and
downloading results from OpenAI's Batch API.
"""

import time
import json
from typing import Optional, Tuple, Dict, Any, Set
from collections import Counter
from pathlib import Path

from openai import OpenAI

from ..config import BATCH_ENDPOINT, BATCH_COMPLETION_WINDOW, DEFAULT_POLL_INTERVAL


# Terminal batch statuses
TERMINAL_STATUSES: Set[str] = {"completed", "failed", "expired", "cancelled"}


def get_client() -> OpenAI:
    """Get OpenAI client.

    Uses OPENAI_API_KEY environment variable.

    Returns:
        OpenAI client instance
    """
    return OpenAI()


def _fmt_elapsed(sec: float) -> str:
    """Format elapsed seconds as HH:MM:SS.

    Args:
        sec: Elapsed seconds

    Returns:
        Formatted time string
    """
    sec = int(sec)
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def submit_batch(
    jsonl_path: str,
    endpoint: str = BATCH_ENDPOINT,
    client: Optional[OpenAI] = None
) -> str:
    """Submit a batch job to OpenAI.

    Args:
        jsonl_path: Path to JSONL file with batch requests
        endpoint: API endpoint for the batch
        client: OpenAI client (created if not provided)

    Returns:
        Batch ID string
    """
    if client is None:
        client = get_client()

    # Upload the file
    with open(jsonl_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")

    # Create the batch
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint=endpoint,
        completion_window=BATCH_COMPLETION_WINDOW
    )

    return batch.id


def wait_batch_with_progress(
    batch_id: str,
    poll_s: int = DEFAULT_POLL_INTERVAL,
    client: Optional[OpenAI] = None
) -> Any:
    """Wait for batch completion with progress updates.

    Args:
        batch_id: Batch ID to monitor
        poll_s: Polling interval in seconds
        client: OpenAI client

    Returns:
        Final batch object
    """
    if client is None:
        client = get_client()

    start = time.time()
    last_line = None

    while True:
        batch = client.batches.retrieve(batch_id)
        rc = getattr(batch, "request_counts", None)

        prog = ""
        if rc is not None:
            prog = f" done {rc.completed}/{rc.total} | failed {rc.failed}"

        line = f"[{_fmt_elapsed(time.time() - start)}] {batch_id} -> {batch.status}{prog}"

        if line != last_line:
            print(line)
            last_line = line

        if batch.status in TERMINAL_STATUSES:
            return batch

        time.sleep(poll_s)


def download_file_if_any(
    file_id: Optional[str],
    save_path: str,
    client: Optional[OpenAI] = None
) -> Optional[str]:
    """Download file from OpenAI if file_id exists.

    Args:
        file_id: OpenAI file ID (or None)
        save_path: Local path to save file
        client: OpenAI client

    Returns:
        Save path if downloaded, None otherwise
    """
    if not file_id:
        return None

    if client is None:
        client = get_client()

    content = client.files.content(file_id)
    with open(save_path, "wb") as f:
        f.write(content.read())

    return save_path


def summarize_error_file(
    err_path: str,
    topk: int = 5
) -> None:
    """Print summary of error file.

    Args:
        err_path: Path to error JSONL file
        topk: Number of top errors to show
    """
    ctr: Counter = Counter()

    with open(err_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            err = obj.get("error") or {}
            code = err.get("code") or err.get("type") or "unknown"
            msg = (err.get("message") or "")[:120]
            ctr[(code, msg)] += 1

    print("\n--- Error summary (top) ---")
    for (code, msg), cnt in ctr.most_common(topk):
        print(f"{cnt}x  {code} | {msg}")


def wait_and_download_safe(
    batch_id: str,
    out_path: str,
    err_path: Optional[str] = None,
    poll_s: int = DEFAULT_POLL_INTERVAL,
    client: Optional[OpenAI] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Wait for batch and download results safely.

    Args:
        batch_id: Batch ID
        out_path: Path for output JSONL
        err_path: Path for error JSONL (default: out_path with _errors suffix)
        poll_s: Polling interval
        client: OpenAI client

    Returns:
        Tuple of (output_path, error_path)

    Raises:
        RuntimeError: If no output file is produced
    """
    if client is None:
        client = get_client()

    if err_path is None:
        err_path = out_path.replace(".jsonl", "_errors.jsonl")

    batch = wait_batch_with_progress(batch_id, poll_s=poll_s, client=client)

    out_id = getattr(batch, "output_file_id", None)
    err_id = getattr(batch, "error_file_id", None)

    print("\nBatch finished.")
    print("status:", batch.status)
    print("output_file_id:", out_id)
    print("error_file_id:", err_id)

    out_saved = download_file_if_any(out_id, out_path, client)
    err_saved = download_file_if_any(err_id, err_path, client)

    if out_saved is None:
        if err_saved:
            summarize_error_file(err_saved, topk=8)
        raise RuntimeError("No output_file_id (no successful outputs). Check *_errors.jsonl")

    if err_saved:
        summarize_error_file(err_saved, topk=5)

    print("Saved output:", out_saved)
    if err_saved:
        print("Saved errors:", err_saved)

    return out_saved, err_saved


def debug_batch(
    batch_id: str,
    client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """Get debug information about a batch.

    Args:
        batch_id: Batch ID
        client: OpenAI client

    Returns:
        Dictionary with batch metadata
    """
    if client is None:
        client = get_client()

    batch = client.batches.retrieve(batch_id)

    return {
        "id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "completed_at": getattr(batch, "completed_at", None),
        "expired_at": getattr(batch, "expired_at", None),
        "request_counts": {
            "total": batch.request_counts.total if batch.request_counts else None,
            "completed": batch.request_counts.completed if batch.request_counts else None,
            "failed": batch.request_counts.failed if batch.request_counts else None,
        },
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }


def cancel_batch(
    batch_id: str,
    client: Optional[OpenAI] = None
) -> str:
    """Cancel a running batch.

    Args:
        batch_id: Batch ID
        client: OpenAI client

    Returns:
        New batch status
    """
    if client is None:
        client = get_client()

    batch = client.batches.cancel(batch_id)
    return batch.status


def list_batches(
    limit: int = 20,
    client: Optional[OpenAI] = None
) -> list:
    """List recent batches.

    Args:
        limit: Maximum number to return
        client: OpenAI client

    Returns:
        List of batch objects
    """
    if client is None:
        client = get_client()

    batches = client.batches.list(limit=limit)
    return list(batches)
