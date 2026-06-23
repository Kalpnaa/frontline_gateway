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

    for _, row in df.iterrows():
        try:
            result, latency_ms, cost_usd = run_triage(row["message"])

            total_latency += latency_ms
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
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Samples         : {total}")
    print(f"Category Accuracy     : {(category_correct/total)*100:.2f}%")
    print(f"Priority Accuracy     : {(priority_correct/total)*100:.2f}%")
    print(f"Human Escalation Acc  : {(human_correct/total)*100:.2f}%")
    print(f"Average Latency       : {total_latency/total:.2f} ms")
    print(f"Estimated Total Cost  : ${total_cost:.6f}")
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
