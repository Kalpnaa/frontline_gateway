"""
classifier.py
-------------
Thin wrapper around the Groq SDK (groq >= 1.0).

Sends the customer message to a Groq-hosted LLM with the triage system
prompt, receives the raw JSON response, and returns a validated TriageResult
together with latency and estimated cost metrics.
"""

from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv
from groq import Groq, APIError, APIConnectionError, RateLimitError

from constants import (
    GROQ_MODEL,
    INPUT_TOKEN_COST,
    MAX_TOKENS,
    OUTPUT_TOKEN_COST,
    TRIAGE_SYSTEM_PROMPT,
)
from schema import CustomerMessage, TriageResult


# ---------------------------------------------------------------------------
# Custom exception — lets callers detect rate-limit failures specifically
# ---------------------------------------------------------------------------

class GroqRateLimitError(RuntimeError):
    """Raised when Groq returns HTTP 429 (tokens-per-minute exceeded)."""

# ---------------------------------------------------------------------------
# Load .env and initialise the Groq client once at import time.
# ---------------------------------------------------------------------------

load_dotenv()

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Get a free key at https://console.groq.com/keys and add it to .env: "
        "GROQ_API_KEY=gsk_..."
    )

_client = Groq(api_key=_api_key)


# ---------------------------------------------------------------------------
# Internal: token & cost estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> float:
    """Rough token count: word count × 1.3 (accounts for subword splitting)."""
    return len(text.split()) * 1.3


def _estimate_cost(input_tokens: float, output_tokens: float) -> float:
    """Return estimated USD cost given token counts and per-million rates."""
    return (
        (input_tokens  / 1_000_000) * INPUT_TOKEN_COST
      + (output_tokens / 1_000_000) * OUTPUT_TOKEN_COST
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def classify(message: CustomerMessage) -> tuple[TriageResult, str, float, float]:
    """
    Send *message* to Groq and return a validated TriageResult with metrics.

    Parameters
    ----------
    message : CustomerMessage
        The validated customer message wrapper.

    Returns
    -------
    result : TriageResult
        Fully validated triage output.
    raw_json : str
        The raw JSON string returned by the model (useful for debugging).
    latency_ms : float
        Round-trip API call duration in milliseconds.
    cost_usd : float
        Rough estimated cost in USD for this request.

    Raises
    ------
    ValueError
        If the model returns an empty response, non-JSON text, or JSON
        that does not conform to the TriageResult schema.
    RuntimeError
        On Groq API errors (rate limit, connection failure, etc.).
    """

    # --- 1. Call the Groq API (timed) -----------------------------------
    _start = time.perf_counter()

    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {"role": "user",   "content": message.text},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
    except RateLimitError as exc:
        # Re-raise as a named exception so the evaluator retry wrapper
        # can distinguish rate-limit failures from other errors.
        raise GroqRateLimitError(
            "Groq rate limit reached. Wait a moment and try again."
        ) from exc
    except APIConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Groq API. Check your internet connection."
        ) from exc
    except APIError as exc:
        raise RuntimeError(f"Groq API error: {exc}") from exc
    finally:
        # Capture elapsed time even if the call raises
        _end = time.perf_counter()

    latency_ms: float = round((_end - _start) * 1000, 2)

    # --- 2. Extract text ------------------------------------------------
    choices = response.choices
    if not choices or not choices[0].message.content:
        raise ValueError("Groq returned an empty response.")

    raw_text: str = choices[0].message.content.strip()

    # Safety net: strip markdown fences in case response_format is ignored
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.rsplit("```", 1)[0].strip()

    # --- 3. Parse JSON --------------------------------------------------
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Groq returned non-JSON output:\n{raw_text}") from exc

    required_fields = {
        "category",
        "priority",
        "summary",
        "suggested_actions",
        "needs_human",
        "confidence",
    }

    missing = required_fields - set(data.keys())

    if missing:
        print(f"Error: Missing fields from model output: {missing}")
        data = {
            "category": "out_of_scope",
            "priority": "P2",
            "summary": "Malformed model output detected.",
            "suggested_actions": [
                "Escalate for manual review",
                "Reprocess request if needed"
            ],
            "needs_human": True,
            "confidence": 0.1,
        }

    # --- 4. Validate with Pydantic --------------------------------------
    try:
        # If the original message is too short/meaningless, force a safe summary
        # regardless of what the model invented.
        if len(message.text.strip()) < 10 or not any(c.isalpha() for c in message.text):
            data["summary"] = "No meaningful message provided."
            data["category"] = "out_of_scope"
            data["priority"] = "P3"
            data["needs_human"] = False
            data["confidence"] = 0.1
            data["suggested_actions"] = ["Ask customer to describe their issue"]

        # Repair malformed fields before validation
        if not data.get("summary") or len(data["summary"].strip()) < 5:
            data["summary"] = "Insufficient issue details provided."

        if not data.get("suggested_actions"):
            data["suggested_actions"] = ["Escalate to human support"]
        result = TriageResult(**data)
    except Exception as exc:
        raise ValueError(
            f"Groq JSON does not match TriageResult schema:\n{data}\n\n{exc}"
        ) from exc

    # --- 5. Estimate tokens & cost --------------------------------------
    # Use actual usage counts from the response if available (Groq returns them),
    # otherwise fall back to the word-count estimate.
    usage = getattr(response, "usage", None)
    if usage and getattr(usage, "prompt_tokens", None):
        input_tokens  = float(usage.prompt_tokens)
        output_tokens = float(usage.completion_tokens)
    else:
        input_tokens  = _estimate_tokens(TRIAGE_SYSTEM_PROMPT + " " + message.text)
        output_tokens = _estimate_tokens(raw_text)

    cost_usd: float = _estimate_cost(input_tokens, output_tokens)

    return result, raw_text, latency_ms, cost_usd