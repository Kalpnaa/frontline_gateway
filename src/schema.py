"""
schema.py
---------
Pydantic data models for the AI Triage System.

Defines the exact shape of a triage result — used for validation,
serialisation (JSON output), and type-safety throughout the project.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations  (single source of truth for valid values)
# ---------------------------------------------------------------------------

class Category(str, Enum):
    """All recognised support-ticket categories."""
    BILLING_ISSUE    = "billing_issue"
    TECHNICAL_ISSUE  = "technical_issue"
    ACCOUNT_ISSUE    = "account_issue"
    DELIVERY_ISSUE   = "delivery_issue"
    REFUND_REQUEST   = "refund_request"
    COMPLAINT        = "complaint"
    GENERAL_QUERY    = "general_query"
    OUT_OF_SCOPE     = "out_of_scope"


class Priority(str, Enum):
    """
    Priority levels — matches standard incident-management conventions.
      P0 = Critical  (system down / data loss / legal risk)
      P1 = High      (major feature broken, revenue impact)
      P2 = Medium    (degraded experience, workaround exists)
      P3 = Low       (question, cosmetic issue, nice-to-have)
    """
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


# ---------------------------------------------------------------------------
# Core triage result model
# ---------------------------------------------------------------------------

class TriageResult(BaseModel):
    """
    Structured output produced by the triage engine for a single
    customer message.

    All fields map directly to the required output schema.
    """

    category: Category = Field(
        ...,
        description="Broad category that best describes the customer's issue.",
    )

    priority: Priority = Field(
        ...,
        description="Urgency level from P0 (critical) to P3 (low).",
    )

    summary: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="One-sentence human-readable summary of the issue.",
    )

    suggested_actions: List[str] = Field(
        default_factory=list,
        description="Ordered list of recommended next steps for the support agent.",
    )

    needs_human: bool = Field(
        ...,
        description="True when the ticket requires a human agent to intervene.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence score (0.0 – 1.0).",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("suggested_actions")
    @classmethod
    def at_least_one_action(cls, v: List[str]) -> List[str]:
        """Ensure there is always at least one suggested action."""
        if not v:
            raise ValueError("suggested_actions must contain at least one item.")
        return v

    @field_validator("summary")
    @classmethod
    def summary_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("summary cannot be blank or whitespace.")
        return v.strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_display_dict(self) -> dict:
        """
        Return a plain dict with enum values serialised as strings —
        handy for JSON output or pretty-printing.
        """
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Input wrapper (optional — useful when the engine needs metadata later)
# ---------------------------------------------------------------------------

class CustomerMessage(BaseModel):
    """
    Thin wrapper around the raw customer text.
    Extend with fields like `channel`, `customer_id`, `timestamp` as needed.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The raw, unprocessed customer message.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Customer message text cannot be blank.")
        return v.strip()