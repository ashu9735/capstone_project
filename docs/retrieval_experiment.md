# Retrieval experiment

Chunk size and the relevance floor are design decisions. This records how they were
chosen, including the selection rule that was tried first and rejected.

Reproduce with:

```bash
python -m scripts.tune_chunking --tickets data/development_tickets.json --limit 300
```

Machine-readable output: [retrieval_experiment.json](retrieval_experiment.json).

## Method

300 development tickets, five chunk configurations, fifteen relevance floors from 0.30
to 1.00. Each configuration is indexed once and every hit recorded with its distance,
so floors are swept without re-embedding. Scored against `labels.expected_doc_ids`:

- **Hit rate** — share of tickets with at least one expected document retrieved.
- **Recall** — mean share of a ticket's expected documents retrieved.
- **Precision** — mean share of retrieved documents that were expected.
- **F1** — harmonic mean of recall and precision.
- **Correct abstention** — share of tickets with *no* expected documents for which the
  retriever correctly returned nothing.

## The first selection rule, and why it was rejected

The obvious rule is to maximise the mean of hit rate and correct abstention: reward
finding the right document, and equally reward staying silent when there is none. It
produced this:

| chunk | floor | hit rate | correct abstention | precision | mean |
|---|---|---|---|---|---|
| 400 / 60 | 0.60 | 92.2% | 11.7% | 76.1% | 52.0% |
| 1200 / 200 | **0.30** | **12.6%** | **92.6%** | 12.6% | **52.6%** |

The winner retrieves nothing for 92.6% of tickets. Because the sample is imbalanced —
roughly two thirds of tickets have an expected document — an average of the two rates
is maximised by collapsing onto whichever class is easier, and abstention is trivially
easy: set the floor low enough and the system abstains on everything. The rule rewarded
a retriever that does not retrieve.

## The rule used instead

**Highest retrieval F1, hit rate breaking ties.** F1 cannot be gamed by abstention,
because recall collapses with it. Abstention is still reported, but it is not
optimised, for a specific reason: the router escalates independently whenever retrieval
returns nothing, *and* independently on the four always-escalate intents — which are
exactly the intents with no documentation coverage. Abstention is already protected one
layer down, so buying it at the cost of recall would be paying twice.

## Results

Best floor per configuration, 300 tickets, top-k 5:

| Chunk size | Overlap | Passages | Best floor | F1 | Hit rate | Recall | Precision | Correct abstention |
|---|---|---|---|---|---|---|---|---|
| **400** | **60** | **170** | **0.55** | **79.5%** | **90.8%** | **78.4%** | **80.6%** | 11.7% |
| 600 | 100 | 89 | 0.55 | 78.1% | 89.8% | 77.2% | 79.0% | 11.7% |
| 800 | 120 | 59 | 0.55 | 76.8% | 87.9% | 76.0% | 77.6% | 11.7% |
| 1200 | 200 | 33 | 0.60 | 78.1% | 92.2% | 81.8% | 74.6% | 11.7% |
| 1600 | 240 | 29 | 0.60 | 77.8% | 91.8% | 81.3% | 74.6% | 11.7% |

## Selected

```
CHUNK_SIZE=400
CHUNK_OVERLAP=60
RETRIEVAL_MAX_DISTANCE=0.55
RETRIEVAL_TOP_K=5
```

## Reading the result

The spread is genuinely narrow — every configuration lands between 76.8% and 79.5% F1 —
so this is a preference, not a discovery. What the grid does show clearly is the
trade-off direction: **larger chunks buy recall and lose precision.** 1200-character
chunks find an expected document more often (92.2% against 90.8%) but six percentage
points more of what they return is irrelevant.

400/60 was chosen because precision is the more expensive failure here. Irrelevant
passages reach the generation prompt and become the raw material for a plausible,
well-cited, wrong answer. A missed passage costs one escalation of a ticket that could
have been answered — measured at a median 48 minutes of handling. A false passage risks
a confident wrong answer to a customer, which is the failure mode that damages the
client relationship and the one the guardrails exist to catch.

The corpus is also small — 29 documents, 31,733 characters total — so 400-character
chunks still produce only 170 passages. There is no retrieval-scale argument for larger
chunks at this size.

**The floor matters more than the chunk size.** Moving it from the 0.85 default to the
measured 0.55 changed precision by roughly 25 points. The 0.85 default was retrieving
five passages for essentially every ticket, including the 170 of 580 that have no
expected document at all.

## Limitation worth stating

Correct abstention tops out at 11.7% across every configuration tested. The embedding
model does not separate "no relevant documentation exists" from "relevant documentation
exists" at any floor that preserves usable recall: irrelevant queries still score around
0.6 cosine distance. This is a real weakness of the retrieval layer, and it is mitigated
downstream rather than solved here — by the always-escalate intent list, by the
confidence threshold, and by the `grounding_required` guardrail. It should be treated as
a known gap rather than a solved problem.
