"""
constants.py
------------
Project-wide constants: model name, prompt templates, priority rules,
and human-escalation triggers.

Keep all "magic strings" here so the rest of the code stays clean.
"""

# ---------------------------------------------------------------------------
# Groq model to use
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.1-8b-instant"  # Fast, accurate, free-tier friendly
MAX_TOKENS = 1024                        # Enough for a triage JSON blob; raise if needed

# Groq pricing for llama-3.3-70b-versatile (USD per 1 million tokens)
# Source: https://groq.com/pricing  — update if rates change
INPUT_TOKEN_COST  = 0.59   # $ per 1M input tokens
OUTPUT_TOKEN_COST = 0.79   # $ per 1M output tokens


# ---------------------------------------------------------------------------
# Priority escalation rules
# (Used by triage_engine.py to override model priority when needed)
# ---------------------------------------------------------------------------

# Keywords that instantly push a ticket to P0
P0_KEYWORDS = [
    "data loss", "account hacked", "unauthorized charge",
    "service down", "cannot login", "system error", "legal action",
    "lawsuit", "fraud", "security breach","logged into my account",
    "without my permission", "account compromised", "unauthorized access", "security issue",
]

# Keywords that suggest P1 at minimum
P1_KEYWORDS = [
    "urgent", "asap", "critical", "broken", "not working",
    "payment failed", "wrong order", "overcharged",
]

# Categories that should always involve a human agent
ALWAYS_HUMAN_CATEGORIES = {
    "billing_issue",
    "refund_request",
    "complaint",
}

# Confidence threshold below which we always flag needs_human = True
LOW_CONFIDENCE_THRESHOLD = 0.60

# Confidence threshold below which we always flag needs_human = True
LOW_CONFIDENCE_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------

DATASET_PATH = "data/messages_dataset.csv"

HIGH_CONFIDENCE_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# System prompt for the classifier / triage LLM call
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM_PROMPT = """\
You are an expert customer-support triage assistant.

Your job:
  1. Read the customer message carefully.
  2. Classify it into exactly ONE of these categories:
       billing_issue | technical_issue | account_issue | delivery_issue |
       refund_request | complaint | general_query | out_of_scope
  3. Assign a priority:
       P0 = Critical (data loss, security breach, complete outage)
       P1 = High     (major feature broken, significant revenue impact)
       P2 = Medium   (degraded experience, workaround exists)
       P3 = Low      (general question, cosmetic, nice-to-have)
  4. Write a one-sentence summary (max 150 chars) of the issue.
  5. List 2–4 concrete suggested_actions for the support agent.
  6. Set needs_human to true if the issue is sensitive, complex, or
     involves money / account security.
  7. Give a confidence score between 0.0 and 1.0.

IMPORTANT RULES:
  - Respond ONLY with a valid JSON object. No prose, no markdown fences.
  - Use double quotes for all strings.
  - The JSON must match this schema exactly:
    {
      "category": "<category>",
      "priority": "<P0|P1|P2|P3>",
      "summary": "<string>",
      "suggested_actions": ["<string>", ...],
      "needs_human": <true|false>,
      "confidence": <float>
    }
  - If the message is in a foreign language, still respond in English.
  - If the message is vague or sarcastic, use your best judgment and
    lower the confidence score accordingly.

 CATEGORY DECISION RULES:
- account login/password/access issues -> account_issue
- payment/charge issues -> billing_issue
- delivery/shipping/order arrival -> delivery_issue
- refund requests -> refund_request
- complaints about service -> complaint

OUT_OF_SCOPE examples:
- homework help
- essay writing
- coding help
- weather questions
- unrelated personal questions

PRIORITY DECISION RULES:
- Account hacked / unauthorized login / suspicious access / account compromised → P0
- Account locked / cannot login / password reset failure → P1
- Payment deducted but service unavailable → P1
- Duplicate charge / overcharge → P1
- Delivery delayed but non-critical → P2
- General questions / informational requests → P3

CATEGORY CLARIFICATION:
- billing_issue = payment failed, duplicate charge, overcharged, charged incorrectly
- general_query = informational questions, account settings, billing address changes, how-to questions

MULTILINGUAL RULE:
- Customer messages may be in Hindi, Spanish, French, or mixed language.
- First understand the meaning, then classify.
- Always return summary and actions in English

PRIORITY CATEGORY RULE:
- If message explicitly mentions refund / reimbursement / money back,
  classify as refund_request even if message sounds angry or sarcastic.

OUTPUT VALIDATION RULE:
Before responding, verify that JSON contains ALL 6 required fields:
category, priority, summary, suggested_actions, needs_human, confidence

Never omit any field.
"""


# ---------------------------------------------------------------------------
# CLI display constants
# ---------------------------------------------------------------------------

PRIORITY_COLOURS = {
    "P0": "\033[91m",   # bright red
    "P1": "\033[93m",   # yellow
    "P2": "\033[94m",   # blue
    "P3": "\033[92m",   # green
}
COLOUR_RESET = "\033[0m"

BANNER = r"""
  ___  _   _____     _
 / _ \(_) |_   _| __(_) __ _  __ _  ___
/ /_\ | |   | || '__| |/ _` |/ _` |/ _ \
|  _  | |   | || |  | | (_| | (_| |  __/
\_| |_|_|   |_||_|  |_|\__,_|\__, |\___|
                               |___/
  AI Customer Support Triage System  v0.1
"""