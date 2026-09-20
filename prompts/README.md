# Prompt register

Prompts are design artefacts. They live in version-controlled files, are loaded by
[../src/prompts.py](../src/prompts.py), and the version travels with every decision
into the `prompt_version` column of the decision log — so any logged decision can be
traced back to the exact prompt that produced it.

Each file carries YAML front matter:

```yaml
---
name: classify_ticket
version: 1.2
requirements: REQ-F02
---
```

## Register

| Prompt | Version | Purpose | Requirement | Used by |
|---|---|---|---|---|
| [build/classify_ticket.md](build/classify_ticket.md) | 1.2 | Intent, urgency and calibrated confidence | REQ-F02 | `src/classify.py` |
| [build/generate_response.md](build/generate_response.md) | 1.3 | Grounded reply with citations | REQ-F05, REQ-F06 | `src/generate.py` |
| [evaluation/judge_groundedness.md](evaluation/judge_groundedness.md) | 1.0 | Judge a reply for unsupported claims | REQ-E02 | Manual hallucination review |

`build/` holds prompts that run inside the system. `evaluation/` holds prompts used to
judge output; they must never be used to produce it.

## Version history

### classify_ticket

| Version | Change | Why |
|---|---|---|
| 1.0 | Intent and urgency only | First draft |
| 1.1 | Added the confidence field and its rubric | Routing needs a number it can threshold on; without a rubric the model returned 0.9 for everything |
| 1.2 | Added "classified on meaning, never on fluency"; urgency judged on stated impact, not tone | Non-fluent speakers show CSAT 1.95 against 2.80 and FCR 21% against 54%. Tone-based urgency would encode that gap into the routing |

### generate_response

| Version | Change | Why |
|---|---|---|
| 1.0 | Answer from the passages, cite documents | First draft |
| 1.1 | Added `INSUFFICIENT_CONTEXT` | The model was padding thin retrieval into a confident answer. It needs an explicit way to decline |
| 1.2 | Added the forbidden-claim rules | Taken from `must_not_claim`, present on all 200 reference responses. The guardrail catches these anyway, but a blocked reply is a wasted call and an escalation that did not need to happen |
| 1.3 | Added the 80–220 word bound and the British English instruction | Replies were drifting to 400+ words and to American spelling, neither of which matches the reference responses |

### judge_groundedness

| Version | Change | Why |
|---|---|---|
| 1.0 | Initial | Supports the manual hallucination review the Evaluation Framework requires |

## What makes a prompt worth keeping

- **It states the failure mode it is preventing.** Every rule in these prompts exists
  because output was observed doing the opposite.
- **It gives the model a way to refuse.** `INSUFFICIENT_CONTEXT` is not a fallback, it
  is a first-class outcome that routes to a human.
- **It does not rely on politeness to enforce a constraint.** Anything that genuinely
  must not happen is also checked in code. The prompt reduces how often the guardrail
  has to fire; it is not the guardrail.
- **It is versioned before it is edited.** A prompt change that cannot be tied to a
  logged decision cannot be evaluated.

## Traceability check

| Requirement | Specification | Prompt | Enforced in code |
|---|---|---|---|
| REQ-F02 | Classify intent from 22 classes with honest confidence | `classify_ticket` 1.2 | Intent coerced against `INTENTS`; confidence clamped and calibrated |
| REQ-F05 | Draft only from retrieved passages | `generate_response` 1.3 | Generation refused when no passages retrieved |
| REQ-F06 | Every citation resolves to a retrieved passage | `generate_response` 1.3 | `validate_citations`, then the `citation_resolves` guardrail |
| REQ-F07 | Never make a forbidden claim | `generate_response` 1.3 | `FORBIDDEN_CLAIMS` in `src/guardrails.py` |
| REQ-E02 | Measure groundedness | `judge_groundedness` 1.0 | Automated lower bound in `evaluation/metrics.py` |
