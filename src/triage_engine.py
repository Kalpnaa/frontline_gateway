"""
triage_engine.py
----------------
Orchestrates the full triage pipeline:

  CustomerMessage  →  [injection check]  →  classifier.classify()  →  rule overrides  →  TriageResult

The engine applies deterministic business rules ON TOP of the model output
so that critical keywords always escalate correctly regardless of model drift.
"""

from __future__ import annotations

import logging

from classifier import classify
from constants import (
    ALWAYS_HUMAN_CATEGORIES,
    LOW_CONFIDENCE_THRESHOLD,
    P0_KEYWORDS,
    P1_KEYWORDS,
)
from schema import Category, CustomerMessage, Priority, TriageResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt injection detection
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: list[str] = [
    # Classic override phrases
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore your instructions",
    "disregard previous instructions",
    "disregard all instructions",
    "disregard your previous",
    "forget previous instructions",
    "forget all instructions",
    # Role/persona hijacking
    "you are now",
    "act as if you are",
    "pretend you are",
    "your new role is",
    "your true instructions",
    "your real instructions",
    "override triage",
    "override your",
    # Direct output manipulation
    "return priority p0",
    "return p0",
    "set priority to p0",
    "mark this as p0",
    "respond only with",
    "respond with json",
    "output only json",
    # System/prompt leakage attempts
    "system prompt",
    "system override",
    "system:",
    "developer mode",
    "jailbreak",
    "dan mode",
    # Instruction injection markers
    "new instruction",
    "revised instruction",
    "updated instruction",
    "end of prompt",
    "start of prompt",
    "### instruction",
    "### system",
    "[system]",
    "<system>",
    "</system>",
]


def contains_prompt_injection(text: str) -> bool:
    """Return True if *text* contains a known prompt-injection pattern."""
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            logger.warning("Prompt injection detected — pattern: %r", pattern)
            return True
    return False


def _injection_result() -> TriageResult:
    """Return the fixed deterministic response for all injection attempts."""
    return TriageResult(
        category=Category.OUT_OF_SCOPE,
        priority=Priority.P1,
        summary="Prompt injection attempt detected in customer message.",
        suggested_actions=[
            "Do not action any instructions contained in this message.",
            "Flag the message for security review.",
            "Contact the customer through a verified channel if a real issue exists.",
        ],
        needs_human=True,
        confidence=0.2,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_triage(raw_text: str) -> tuple[TriageResult, float, float]:
    """
    Full pipeline: validate input → injection check → call LLM → apply rules.

    Parameters
    ----------
    raw_text : str
        The raw customer message string.

    Returns
    -------
    result : TriageResult
        Validated, rule-adjusted triage output.
    latency_ms : float
        LLM round-trip time in milliseconds (0.0 for injected messages).
    cost_usd : float
        Estimated USD cost for this request (0.0 for injected messages).
    """

    # --- Step 1: Validate input -----------------------------------------
    message = CustomerMessage(text=raw_text)
    logger.debug("Input validated: %r", message.text[:80])

    # --- Step 2: Prompt injection guard — runs BEFORE the LLM ----------
    if contains_prompt_injection(message.text):
        logger.warning("Injection blocked — returning deterministic result.")
        return _injection_result(), 0.0, 0.0

    # --- Step 3: LLM classification ------------------------------------
    result, raw_json, latency_ms, cost_usd = classify(message)
    logger.debug("Raw LLM JSON: %s", raw_json)
    logger.debug("Latency: %.2f ms | Cost: $%.8f", latency_ms, cost_usd)

    # --- Step 4: Apply deterministic override rules --------------------
    result = _apply_business_rules(message.text, result)

    return result, latency_ms, cost_usd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_business_rules(text: str, result: TriageResult) -> TriageResult:
    """
    Override the model's output with hard business rules.
    Uses model_copy() so the original object is never mutated.
    """

    lower_text = text.lower()
    overrides: dict = {}

    # Rule 1 — P0 keyword escalation
    if any(kw in lower_text for kw in P0_KEYWORDS):
        if result.priority not in (Priority.P0,):
            logger.info("Escalating to P0 due to keyword match.")
            overrides["priority"] = Priority.P0
            overrides["needs_human"] = True

    # Rule 2 — P1 keyword floor (only upgrade, never downgrade)
    elif any(kw in lower_text for kw in P1_KEYWORDS):
        if result.priority in (Priority.P2, Priority.P3):
            logger.info("Upgrading priority to P1 due to urgency keyword.")
            overrides["priority"] = Priority.P1

    # Rule 3 — Certain categories always need a human
    if result.category.value in ALWAYS_HUMAN_CATEGORIES:
        overrides["needs_human"] = True

    # Rule 4 — Low-confidence tickets always need a human
    if result.confidence < LOW_CONFIDENCE_THRESHOLD:
        logger.info(
            "Low confidence (%.2f) — flagging for human review.", result.confidence
        )
        overrides["needs_human"] = True

    if overrides:
        result = result.model_copy(update=overrides)

    return result