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
import re

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
    # Account / security threat language
    "hijack",
    "hijacking",
    "takeover",
    "take over",
    "steal account",
    "hack the",
    "bypass security",
    "bypass login",
    "brute force",
    "credential stuffing",
    "phishing",
    "social engineering",
]


def contains_prompt_injection(text: str) -> bool:
    """Return True if *text* contains any known prompt-injection pattern."""
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            return True
    return False


def _injection_result() -> TriageResult:
    return TriageResult(
        category=Category.OUT_OF_SCOPE,
        priority=Priority.P1,
        summary="Prompt injection attempt detected.",
        suggested_actions=[
            "Do not execute instructions in message",
            "Flag for security review",
            "Escalate if needed"
        ],
        needs_human=True,
        confidence=0.2
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
        logger.info("Prompt injection detected — blocking LLM call.")
        return _injection_result(), 0.0, 0.0

    # --- Step 3: LLM classification (with rule-based fallback) ----------
    try:
        result, raw_json, latency_ms, cost_usd = classify(message)
        logger.debug("Raw LLM JSON: %s", raw_json)
        logger.debug("Latency: %.2f ms | Cost: $%.8f", latency_ms, cost_usd)

        # --- Step 4: Apply deterministic override rules ----------------
        result = _apply_business_rules(message.text, result)

        return result, latency_ms, cost_usd

    except (RuntimeError, ValueError) as exc:
        # RuntimeError  → rate limit / connection failure / Groq API error
        # ValueError    → malformed / empty response from the model
        print(f"\u26a0  Groq API unavailable: {exc}")
        print("\u26a0  Using fallback rule-based triage.")
        logger.warning("LLM call failed (%s) — switching to rule-based fallback.", exc)

        result = _rule_based_triage(message.text)
        result = _apply_business_rules(message.text, result)   # escalate P0/P1 keywords
        return result, 0.0, 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _matches_any(text: str, keywords: list[str]) -> bool:
    """Word-boundary keyword match — avoids false positives on substrings."""
    pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


def _apply_business_rules(text: str, result: TriageResult) -> TriageResult:
    """
    Override the model's output with hard business rules.
    Uses model_copy() so the original object is never mutated.
    """

    overrides: dict = {}

    # Rule 1 — P0 keyword escalation
    if _matches_any(text, P0_KEYWORDS):
        if result.priority not in (Priority.P0,):
            logger.info("Escalating to P0 due to keyword match.")
            overrides["priority"] = Priority.P0
            overrides["needs_human"] = True

    # Rule 2 — P1 keyword floor (only upgrade, never downgrade)
    elif _matches_any(text, P1_KEYWORDS):
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


def _rule_based_triage(text: str) -> TriageResult:
    """
    Pure keyword-based fallback classifier — used when the Groq API is
    unavailable (rate limit, timeout, connection failure).

    No external calls. Always returns a valid TriageResult.
    Covers English + Spanish + Hindi/Devanagari + Hinglish patterns.
    Business rule overrides (_apply_business_rules) are applied on top
    by the caller, so this function focuses on category + base priority.
    """

    lower = text.lower()

    # ------------------------------------------------------------------
    # Urgency signal — upgrades delivery/technical base priority P2 → P1
    # ------------------------------------------------------------------
    is_urgent = any(kw in lower for kw in (
        "asap", "urgent", "immediately", "right now", "emergency",
        "critical", "as soon as possible",
    ))

    # ------------------------------------------------------------------
    # Hedging signal — softens billing/refund base priority P1 → P2
    # Catches: "i think maybe there was an extra charge? not sure tbh"
    # ------------------------------------------------------------------
    is_hedging = any(kw in lower for kw in (
        "not sure", "maybe", "i think", "feels like", "feels off",
        "not certain", "possibly", "might be", "could be", "rly", "tbh",
        "kind of", "sort of", "something feels", "i guess",
    ))

    # ------------------------------------------------------------------
    # Security / account compromise — always P0, skip scoring
    # English + Spanish + Hindi transliteration
    # ------------------------------------------------------------------
    if any(kw in lower for kw in (
        "without my permission", "unauthorized", "compromised",
        "someone logged", "hacked", "stolen", "security breach",
        "sin mi permiso", "acceso no autorizado", "cuenta comprometida",
        "account hack", "mere account mein", "bina permission",
    )):
        return TriageResult(
            category          = Category.ACCOUNT_ISSUE,
            priority          = Priority.P0,
            summary           = "Fallback: potential account security incident detected.",
            suggested_actions = ["Escalate to security team immediately", "Manual review recommended"],
            needs_human       = True,
            confidence        = 0.6,
        )

    # ------------------------------------------------------------------
    # Multi-issue scoring — count keyword hits per category, pick winner
    # Fixes: "billed twice AND dashboard wrong AND can't export" → technical
    # ------------------------------------------------------------------
    scores: dict = {
        "billing":   0,
        "account":   0,
        "technical": 0,
        "delivery":  0,
        "refund":    0,
    }

    # Billing signals
    for kw in ("billing", "charge", "charged", "payment", "invoice",
               "overcharged", "billed", "double charge", "duplicate charge",
               "bill", "extra charge", "cobrado", "factura", "pago", "cargo"):
        if kw in lower:
            scores["billing"] += 1

    # Refund signals (weighted higher — explicit intent)
    for kw in ("refund", "money back", "reimburse", "reembolso", "devolver dinero"):
        if kw in lower:
            scores["refund"] += 2

    # Account signals
    # English + Spanish + Hindi/Devanagari script + Hinglish
    for kw in ("login", "log in", "password", "locked", "account", "sign in",
               "access", "reset", "two factor", "2fa", "authenticate",
               "iniciar sesión", "contraseña", "bloqueado", "restablecer",
               "sesión", "cuenta",
               "\u0905\u0915\u093e\u0909\u0902\u0924", "\u092c\u0902\u0926",
               "\u092a\u093e\u0938\u0935\u0930\u094d\u0921", "\u0932\u0949\u0917\u093f\u0928",
               "\u0916\u093e\u0924\u093e",
               "account band", "password reset", "login nahi"):
        if kw in lower:
            scores["account"] += 1

    # Technical signals — removed generic "cant/cannot" (too noisy)
    for kw in ("crash", "error", "api down", "not working", "broken", "bug",
               "slow", "timeout", "500", "404", "blank", "loading",
               "dashboard", "export", "import", "sync", "integration",
               "feature", "reports", "wrong data", "display issue",
               "doesn't work", "doesnt work", "glitch", "freeze",
               "aplicación", "falla", "no funciona"):
        if kw in lower:
            scores["technical"] += 1

    # Delivery signals
    for kw in ("package", "delivery", "tracking", "shipped", "order",
               "arrived", "arrive", "shipping", "courier", "dispatch",
               "paquete", "entrega", "pedido", "envío"):
        if kw in lower:
            scores["delivery"] += 1

    # ------------------------------------------------------------------
    # Pick category by highest score; ties broken by dict ordering above
    # ------------------------------------------------------------------
    best = max(scores, key=lambda k: scores[k])

    if scores[best] == 0:
        # No signals — check out-of-scope then default
        if any(kw in lower for kw in (
            "joke", "weather", "essay", "homework", "poem", "recipe",
            "who is", "what is", "capital of", "translate",
        )):
            return TriageResult(
                category          = Category.OUT_OF_SCOPE,
                priority          = Priority.P3,
                summary           = "Fallback rule-based classification.",
                suggested_actions = ["Manual review recommended"],
                needs_human       = False,
                confidence        = 0.5,
            )
        return TriageResult(
            category          = Category.GENERAL_QUERY,
            priority          = Priority.P3,
            summary           = "Fallback rule-based classification.",
            suggested_actions = ["Manual review recommended"],
            needs_human       = False,
            confidence        = 0.5,
        )

    # Map winner to category / priority / needs_human
    if best == "refund":
        category    = Category.REFUND_REQUEST
        priority    = Priority.P2 if is_hedging else Priority.P1
        needs_human = True

    elif best == "billing":
        category    = Category.BILLING_ISSUE
        priority    = Priority.P2 if is_hedging else Priority.P1
        needs_human = True

    elif best == "account":
        category    = Category.ACCOUNT_ISSUE
        priority    = Priority.P1
        needs_human = True

    elif best == "technical":
        category    = Category.TECHNICAL_ISSUE
        priority    = Priority.P1 if is_urgent else Priority.P2
        needs_human = False

    else:  # delivery
        category    = Category.DELIVERY_ISSUE
        priority    = Priority.P1 if is_urgent else Priority.P2
        needs_human = False

    return TriageResult(
        category          = category,
        priority          = priority,
        summary           = "Fallback rule-based classification.",
        suggested_actions = ["Manual review recommended"],
        needs_human       = needs_human,
        confidence        = 0.5,
    )