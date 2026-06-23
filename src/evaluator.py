"""
evaluator.py
------------
Batch evaluation scaffold for the triage system.

Feed it a list of test cases with expected outputs and it will report
accuracy, category confusion, and priority error rates.

Usage (from project root):
    python src/evaluator.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
from constants import DATASET_PATH

from triage_engine import run_triage
from schema import Category, Priority

import time

from classifier import GroqRateLimitError

# ---------------------------------------------------------------------------
# Retry / rate-limit config
# ---------------------------------------------------------------------------

MAX_RETRIES         = 3           # attempts per row before giving up
BACKOFF_SECONDS     = [2, 5, 10]  # wait times between consecutive retries
INTER_REQUEST_DELAY = 0.8         # seconds to sleep between every row call


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single labelled test case."""
    message:           str
    expected_category: Optional[Category] = None   # None = don't check
    expected_priority: Optional[Priority] = None   # None = don't check
    label:             str = ""                    # Optional human label


# ---------------------------------------------------------------------------
# Built-in sample test cases  (expand as you add real data)
# ---------------------------------------------------------------------------

def load_dataset():
    return pd.read_csv(DATASET_PATH)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    total:            int = 0
    category_correct: int = 0
    priority_correct: int = 0
    errors:           List[dict] = field(default_factory=list)

    @property
    def category_accuracy(self) -> float:
        checked = sum(1 for e in self.errors if "category" in e.get("checks", []))
        total_checked = self.total - (self.total - checked)
        return self.category_correct / max(total_checked, 1)

    def print_summary(self) -> None:
        print(f"\n{'='*50}")
        print(f"  Eval Results  ({self.total} test cases)")
        print(f"{'='*50}")
        print(f"  Category accuracy : {self.category_correct}/{self.total}")
        print(f"  Priority accuracy : {self.priority_correct}/{self.total}")
        if self.errors:
            print(f"\n  Failures:")
            for err in self.errors:
                print(f"    [{err['label']}]  {err['reason']}")
        print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _triage_with_retry(message: str):
    """
    Call run_triage with exponential backoff on Groq 429 errors.

    Retries up to MAX_RETRIES times. Only rate-limit errors trigger a retry;
    all other exceptions propagate immediately so the evaluator can count them.
    """
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            return run_triage(message)
        except GroqRateLimitError as exc:
            last_exc = exc
            wait = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            print(
                f"  ⚠  Rate limit hit (attempt {attempt + 1}/{MAX_RETRIES}). "
                f"Retrying in {wait}s…"
            )
            time.sleep(wait)
        # Non-rate-limit errors bubble up immediately
    raise last_exc  # all retries exhausted


def run_eval():
    df = pd.read_csv(DATASET_PATH)

    total = len(df)
    category_correct = 0
    priority_correct = 0
    human_correct = 0
    total_latency = 0
    total_cost = 0
    failures = []
    errors = 0
    llm_calls = 0

    try:
        for idx, (_, row) in enumerate(df.iterrows(), start=1):   # sequential
            print(f"  [{idx:>2}/{total}] Processing…", end="\r", flush=True)
            try:
                result, latency_ms, cost_usd = _triage_with_retry(row["message"])

                if latency_ms > 0:
                    total_latency += latency_ms
                    llm_calls += 1
                total_cost += cost_usd

                predicted_category = result.category.value
                predicted_priority = result.priority.value
                predicted_human = result.needs_human

                expected_category = row["category"]
                expected_priority = row["priority"]
                expected_human = str(row["expected_needs_human"]).lower() == "true"

                if predicted_category == expected_category:
                    category_correct += 1

                if predicted_priority == expected_priority:
                    priority_correct += 1

                if predicted_human == expected_human:
                    human_correct += 1

                if (
                    predicted_category != expected_category
                    or predicted_priority != expected_priority
                ):
                    failures.append({
                        "message": row["message"],
                        "expected_category": expected_category,
                        "predicted_category": predicted_category,
                        "expected_priority": expected_priority,
                        "predicted_priority": predicted_priority,
                    })

            except Exception as e:
                errors += 1
                print(f"\n  ✗  Row {idx} failed after retries: {e}")

            # Small delay between every request to avoid TPM bursts
            time.sleep(INTER_REQUEST_DELAY)

    finally:
        # Always print the report — even if the loop is interrupted
        processed = category_correct + priority_correct + errors  # noqa: rough
        safe_total = max(total, 1)
        safe_llm_calls = max(llm_calls, 1)

        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"Total Samples         : {total}")
        print(f"Category Accuracy     : {(category_correct/safe_total)*100:.2f}%")
        print(f"Priority Accuracy     : {(priority_correct/safe_total)*100:.2f}%")
        print(f"Human Escalation Acc  : {(human_correct/safe_total)*100:.2f}%")
        print(f"Average Latency       : {total_latency/safe_llm_calls:.2f} ms  (LLM calls only)")
        print(f"Estimated Total Cost  : ${total_cost/safe_llm_calls:.6f}")
        print(f"Errors                : {errors}")

        if failures:
            print("\nFAILURE ANALYSIS")
            print("=" * 60)
            for i, fail in enumerate(failures[:5], start=1):
                print(f"\nCase #{i}")
                print(f"Message            : {fail['message']}")
                print(f"Expected Category  : {fail['expected_category']}")
                print(f"Predicted Category : {fail['predicted_category']}")
                print(f"Expected Priority  : {fail['expected_priority']}")
                print(f"Predicted Priority : {fail['predicted_priority']}")

# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")   # ensure src/ imports work when run from root

    eval_results = run_eval()