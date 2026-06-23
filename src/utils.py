"""
utils.py
--------
Display helpers and small utility functions used across the project.
"""

from __future__ import annotations

import json
from datetime import datetime

from constants import COLOUR_RESET, PRIORITY_COLOURS
from schema import TriageResult


# ---------------------------------------------------------------------------
# Pretty-print a TriageResult to the terminal
# ---------------------------------------------------------------------------

def print_result(
    result: TriageResult,
    message: str = "",
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
) -> None:
    """
    Print a colour-coded, human-readable summary of a TriageResult.

    Parameters
    ----------
    result     : TriageResult
    message    : str   — original customer message (for context preview)
    latency_ms : float — LLM round-trip time in ms
    cost_usd   : float — estimated request cost in USD
    """

    colour = PRIORITY_COLOURS.get(result.priority.value, "")
    reset  = COLOUR_RESET
    sep    = "─" * 56

    print(f"\n{sep}")

    if message:
        preview = message[:120] + ("…" if len(message) > 120 else "")
        print(f"  📩  {preview}")
        print(sep)

    print(f"  Category  : {result.category.value}")
    print(f"  Priority  : {colour}{result.priority.value}{reset}")
    print(f"  Summary   : {result.summary}")
    print(f"  Human?    : {'✅ Yes' if result.needs_human else '❌ No'}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"\n  Suggested actions:")
    for i, action in enumerate(result.suggested_actions, 1):
        print(f"    {i}. {action}")

    # --- Metrics footer -------------------------------------------------
    print(sep)
    latency_str = f"{latency_ms:,.0f} ms" if latency_ms else "n/a (blocked)"
    cost_str    = f"${cost_usd:.6f}" if cost_usd else "$0.000000 (blocked)"
    print(f"  ⏱  Latency        : {latency_str}")
    print(f"  💰  Estimated Cost : {cost_str}")
    print(sep)


# ---------------------------------------------------------------------------
# Save a result to a JSON file
# ---------------------------------------------------------------------------

def save_result(
    result: TriageResult,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    output_dir: str = "data",
) -> str:
    """Serialise and save a TriageResult (with metrics) as a timestamped JSON file."""

    import os
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath  = os.path.join(output_dir, f"triage_{timestamp}.json")

    payload = result.to_display_dict()
    payload["_metrics"] = {
        "latency_ms": latency_ms,
        "cost_usd":   round(cost_usd, 8),
    }

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    return filepath


# ---------------------------------------------------------------------------
# Load a multi-line message from a plain-text file
# ---------------------------------------------------------------------------

def load_message_from_file(filepath: str) -> str:
    """Read a customer message stored in a .txt file."""
    with open(filepath, encoding="utf-8") as fh:
        return fh.read().strip()