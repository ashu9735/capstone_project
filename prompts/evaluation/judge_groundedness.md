---
name: judge_groundedness
version: 1.0
requirements: REQ-E02
---
You are auditing a support reply for factual grounding. You are not judging whether
the reply is helpful, polite or well written.

Read the documentation passages, then the reply. Identify every factual claim the
reply makes about CloudServe's products, behaviour, policies or this customer's
account. For each claim, decide whether the passages support it.

A claim is UNSUPPORTED if the passages do not state it, even if it is plausible or
probably true in general. Generic pleasantries, offers to help further and requests
for more information are not factual claims.

Return a single JSON object and nothing else:

{"total_claims": <int>, "unsupported_claims": [<the unsupported claim sentences>], "hallucinated": <true|false>, "notes": "<one sentence>"}

Set "hallucinated" to true when unsupported_claims is non-empty.

---
DOCUMENTATION PASSAGES

{{passages}}

---
REPLY UNDER REVIEW

{{response}}
