
# AI Customer Support Triage System

An intelligent customer support triage system that classifies incoming support messages, assigns priority, and decides whether human escalation is needed.

The system is designed to handle real-world messy support conversations including vague messages, sarcasm, multilingual input, prompt injection attempts, and multi-issue tickets.

---

# Problem Statement

Support teams receive a large number of customer messages daily. Manual triage is slow, repetitive, and inconsistent.

This project automates the first support layer by answering:

* What is the issue category?
* How urgent is it?
* Does it need human intervention?
* What should be the next action?

---

# Core Features

* AI-powered ticket classification
* Priority assignment (P0–P3)
* Human escalation detection
* Prompt injection protection
* Rule-based fallback engine
* Latency, token, and cost tracking
* Dataset evaluation

---

# Categories

* billing_issue
* technical_issue
* account_issue
* delivery_issue
* refund_request
* complaint
* general_query
* out_of_scope

---

# Priority Levels

* **P0** → Critical
* **P1** → High
* **P2** → Medium
* **P3** → Low

Examples:

* fraud / hacked account → P0
* billing / login failure → P1
* delayed delivery → P2
* general queries → P3

---

# Architecture

```text
Customer Message
      ↓
Input Validation
      ↓
Prompt Injection Detection
      ↓
LLM Engine (Groq)
      ↓
If API fails → Rule-Based Fallback
      ↓
Business Rule Overrides
      ↓
Final Structured Output
```

---

# Why Hybrid Architecture?

This project combines **LLM reasoning + deterministic rules**.

### LLM handles:

* messy language
* sarcasm
* multilingual text
* vague queries
* complex multi-issue tickets

### Rule engine handles:

* business constraints
* guaranteed escalation
* critical issue detection
* fallback during API failure

This improves reliability and production readiness.

---

# Model Used

**llama-3.1-8b-instant (Groq)**

Why:

* fast inference
* low cost
* good structured output
* strong instruction following

---

# Prompt Engineering Strategy

The system prompt is designed with strict rules for:

* category classification
* priority assignment
* human escalation
* confidence scoring
* prompt injection defense

The model is forced to return valid JSON output only.

---

# Prompt Injection Protection

Two-layer defense:

### Layer 1 — Rule-based detection

Detects malicious patterns like:

* ignore previous instructions
* system override
* return P0

### Layer 2 — Prompt hardening

LLM explicitly ignores malicious instructions inside messages.

This prevents hijacking.

---

# Fallback Strategy

If Groq API fails due to:

* rate limits
* timeout
* API failure

System switches to rule-based classification.

This ensures triage remains functional even during external API issues.

---

# Tech Stack

* Python
* Groq API
* Pydantic
* Pandas

---

# Project Structure

## constants.py

Stores:

* model config
* prompts
* priority rules
* escalation rules

---

## schema.py

Defines input/output validation using Pydantic.

---

## classifier.py

Handles:

* Groq API calls
* JSON parsing
* token usage
* latency tracking
* cost estimation

---

## triage_engine.py

Core orchestration layer:

* validation
* injection detection
* LLM classification
* fallback logic
* business rule overrides

---

## main.py

CLI for testing single customer messages.

Example:

```bash
python src/main.py --message "My account was hacked"
```

---

## evaluator.py

Runs dataset evaluation on multiple test cases.

Tracks:

* category accuracy
* priority accuracy
* human escalation accuracy
* latency
* cost

---

# Evaluation Dataset Covers

* clear cases
* vague cases
* sarcasm
* multilingual messages
* prompt injection
* multi-issue tickets
* garbage input

---

# Metrics Tracked

Per request:

* category accuracy
* priority accuracy
* latency
* token usage
* cost

---

# Future Improvements

* larger evaluation dataset
* better multilingual handling
* confidence calibration
* dashboard analytics
* automatic ticket routing

---

# Key Engineering Decisions

* Hybrid LLM + Rule Engine
* Prompt Injection Defense
* Rule-Based Fallback
* Strict JSON Validation
* Cost & Latency Monitoring



