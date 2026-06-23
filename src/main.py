"""
main.py
-------
CLI entry point for the AI Customer Support Triage System.

Usage examples:
    python src/main.py
    python src/main.py --message "I was charged twice this month!"
    python src/main.py --file data/sample_message.txt
    python src/main.py --message "Refund me now" --save
    python src/main.py --message "App keeps crashing" --json-only
    python src/main.py --eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from constants import BANNER
from triage_engine import run_triage
from utils import load_message_from_file, print_result, save_result

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING").upper(),
    format="%(levelname)s | %(name)s | %(message)s",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-triage",
        description="AI Customer Support Triage System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group()
    parser.add_argument("--eval", action="store_true")
    source.add_argument("--message", "-m", type=str, help="Customer message as a string.")
    source.add_argument("--file",    "-f", type=str, help="Path to a .txt file containing the message.")
    source.add_argument("--eval",    action="store_true", help="Run the built-in evaluator.")
    parser.add_argument("--save",    "-s", action="store_true", help="Save result as JSON in data/.")
    parser.add_argument("--json-only", action="store_true", help="Print raw JSON only (pipe-friendly).")
    return parser


def triage_and_display(message: str, save: bool, json_only: bool) -> int:
    """Run the triage pipeline and display the result. Returns 0 on success, 1 on error."""

    try:
        result, latency_ms, cost_usd = run_triage(message)
    except ValueError as exc:
        print(f"\n[ERROR] Triage failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}", file=sys.stderr)
        return 1

    if json_only:
        payload = result.to_display_dict()
        payload["_metrics"] = {"latency_ms": latency_ms, "cost_usd": round(cost_usd, 8)}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_result(result, message=message, latency_ms=latency_ms, cost_usd=cost_usd)

    if eval:
        from evaluator import run_eval
        run_eval()
        return

    if save:
        path = save_result(result, latency_ms=latency_ms, cost_usd=cost_usd)
        if not json_only:
            print(f"\n  💾  Saved → {path}\n")

    return 0


def interactive_loop(save: bool, json_only: bool) -> None:
    if not json_only:
        print(BANNER)
        print("  Type a customer message and press Enter.")
        print("  Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            raw = input("Customer message> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        triage_and_display(raw, save=save, json_only=json_only)


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    if args.eval:
        from evaluator import run_eval
        results = run_eval()
        results.print_summary()
        return 0

    if args.message:
        return triage_and_display(args.message, args.save, args.json_only)

    if args.file:
        try:
            message = load_message_from_file(args.file)
        except FileNotFoundError:
            print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
            return 1
        return triage_and_display(message, args.save, args.json_only)

    if not sys.stdin.isatty():
        message = sys.stdin.read().strip()
        if message:
            return triage_and_display(message, args.save, args.json_only)

    interactive_loop(save=args.save, json_only=args.json_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())