# CloudServe Support Automation

Automated triage, retrieval and grounded response drafting for CloudServe Solutions'
support queue. Tickets arrive from four channels, are classified, matched against the
knowledge base, and either answered with citations or escalated to a human with a
handoff summary. Every decision is logged and reconcilable.

## What this system is for

CloudServe asked for a chatbot. The discovery evidence pointed somewhere else: **71.4%
of tickets are already answerable from CloudServe's own documentation, yet 38.7% of
those were escalated anyway.** That is a findability problem, not an answer shortage.
The system therefore optimises for *routing the answerable tickets to an answer* and
*escalating everything else fast and well*, rather than for conversational ability.

---

## Quick start

Requires Python 3.10 or later (3.11 recommended) and about 1 GB of free disk space.

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2. Configuration
cp .env.example .env                 # Windows: copy .env.example .env
# Edit .env and set OPENROUTER_API_KEY (and optionally GROQ_API_KEY for failover)

# 3. Build the vector index (downloads the embedding model on first run, ~90 MB)
python -m scripts.build_index

# 4. Process a ticket file end to end
python -m evaluation.harness --input data/validation_tickets.json --output results/
```

`results/` will contain `metrics.md`, `metrics.json`, `results.jsonl`,
`responses.json`, `calibration.json` and `decisions.db`.

### Running the tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

The suite needs no API key and makes no network calls: the model provider is stubbed.

### Running the service

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

`/health`, `/ticket`, `/decisions/{ticket_id}`, `/admin/kill-switch`, `/metrics`.

---

## The harness contract

```
python -m evaluation.harness --input <path to tickets json> --output <directory>
```

The input path is an argument. Nothing is hardcoded, because this will be pointed at a
file that did not exist when the code was written. The harness accepts a JSON array or
newline-delimited JSON, processes tickets concurrently, never prompts for input, and
always writes a report — including when tickets fail.

| Option | Default | Purpose |
|---|---|---|
| `--input` | required | Ticket file to process |
| `--output` | required | Directory for results and metrics |
| `--limit N` | all | Process only the first N tickets |
| `--workers N` | `MAX_CONCURRENCY` (4) | Concurrent tickets |
| `--rebuild-index` | off | Rebuild the vector store first |
| `--hidden-set-runs N` | 0 | Recorded in the report for provenance |
| `--metrics-port N` | off | Expose Prometheus metrics during the run |

**If no API key is configured the harness still completes.** Retrieval and the
deterministic rules run, and every ticket that would need generation escalates instead.
The run is honest about this: the report records `model: none (ENABLE_LLM=false)`.

---

## Architecture

```
ticket ─► ingest ─► input guardrails ─► classify ─► retrieve ─► route ─┬─► generate ─► output guardrails ─► decision log
                                                                       └─► escalation summary ───────────┘
```

| Module | Responsibility |
|---|---|
| [src/ingest.py](src/ingest.py) | Normalises email, chat, forum and docs-comment tickets into one shape |
| [src/classify.py](src/classify.py) | Intent (22 classes), urgency (3 levels), calibrated confidence |
| [src/retrieve.py](src/retrieve.py) | Chroma vector search over the 29 knowledge base articles |
| [src/route.py](src/route.py) | The escalation decision — a pure function, no model call |
| [src/generate.py](src/generate.py) | Grounded drafting, citation extraction and validation |
| [src/guardrails.py](src/guardrails.py) | Checks that can block a reply outright |
| [src/logging_store.py](src/logging_store.py) | The decision log and its reconciliation |
| [src/pipeline.py](src/pipeline.py) | The LangGraph state machine wiring the above |
| [src/api.py](src/api.py) | FastAPI service, kill switch, Prometheus endpoint |
| [evaluation/harness.py](evaluation/harness.py) | Unattended batch processing |
| [evaluation/metrics.py](evaluation/metrics.py) | The three tiers of measurement |

Full detail in [docs/architecture.md](docs/architecture.md).

### How the system decides to answer or escalate

A ticket is answered automatically **only** when all of these hold:

1. The kill switch is released.
2. The ticket is not flagged `must_not_auto_respond`.
3. The intent is not on the always-escalate list — `security_incident`,
   `compliance_request`, `feature_request`, `unclear_request`.
4. At least one documentation passage cleared the relevance floor.
5. Classification confidence is at or above `CONFIDENCE_THRESHOLD` (0.80).

Anything else escalates, with a machine-readable reason code and a handoff summary.
After drafting, the output guardrails can still block the reply and force escalation.

---

## Design decisions worth knowing about

**Retrieval is allowed to return nothing.** 170 of the 580 supplied tickets have no
expected document at all. A retriever that always returns its top 5 would manufacture a
citation for every one of them. The relevance floor is set at cosine distance 0.55.

**Chunk size was measured, not assumed.** `python -m scripts.tune_chunking` sweeps five
chunk configurations against fifteen relevance floors over 300 development tickets. 400
characters with 60 overlap won on retrieval F1 (79.5%, precision 80.6%). Evidence and
the rejected first selection rule are in [docs/retrieval_experiment.md](docs/retrieval_experiment.md).

**Confidence is calibrated, not trusted.** The routing threshold is meaningless if the
score behind it is not honest. `evaluation/calibration.json` maps stated confidence
onto observed accuracy per band, and the classifier applies that correction.

**Two providers, because one provider is a single point of failure.** OpenRouter is
primary, Groq is the automatic fallback, and total failure degrades to escalation
rather than to an exception.

**Deviations from the supplied `06_Configuration/requirements.txt`**, all deliberate:

| Change | Reason |
|---|---|
| LangChain 0.3.x, Chroma 0.5.x, Pydantic 2.10 | The supplied pins (`langchain==0.1.0` + `chromadb==0.3.21` + `pydantic==2.0.0`) cannot co-resolve; a clean `pip install` aborts |
| `sentence-transformers`/`torch` moved to `requirements-optional.txt` | The same all-MiniLM-L6-v2 model is served by Chroma's bundled ONNX runtime: ~90 MB instead of ~900 MB, and no CUDA wheel resolution on the grading machine. Set `EMBEDDING_BACKEND=sentence-transformers` to use the torch path |
| `psycopg2-binary` dropped | SQLite only, as the Setup Guide recommends |
| `jupyter`, `ipython`, test and lint tools moved to `requirements-dev.txt` | Not needed to run the system |

---

## Configuration

Every value lives in `.env`; see [.env.example](.env.example) for the full list with
placeholders. The ones that change behaviour most:

| Variable | Default | Effect |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.80` | Below this, the ticket escalates |
| `RETRIEVAL_MAX_DISTANCE` | `0.55` | Above this, a passage is discarded |
| `RETRIEVAL_TOP_K` | `5` | Passages considered per ticket |
| `ENABLE_LLM` | `true` | `false` runs the whole pipeline with no provider |
| `MAX_CONCURRENCY` | `4` | Tickets processed in parallel |
| `KILL_SWITCH` | `false` | `true` escalates everything immediately |

**Never commit `.env`.** It is in `.gitignore`, and CI fails the build if it appears or
if anything resembling a key is found in the tree.

---

## Monitoring

The harness and the API both expose Prometheus metrics:
`tickets_processed_total{channel,outcome}`, `response_seconds`,
`guardrail_blocks_total{guardrail}`, `classification_confidence`,
`llm_provider_failures_total{provider}`, `tickets_degraded_total{reason}`,
`kill_switch_engaged`.

```bash
prometheus --config.file=monitoring/prometheus.yml
```

Import [monitoring/grafana_dashboard.json](monitoring/grafana_dashboard.json) into
Grafana for throughput by channel, the automatic/escalated split, latency percentiles,
guardrail activations and the confidence distribution.

---

## What happens when things fail

| Failure | Behaviour |
|---|---|
| No documentation matches | Escalates with reason `no_supporting_documentation` |
| Model provider times out | Retries with backoff, then fails over to Groq |
| Both providers down | Escalates with reason `generation_unavailable`; the run completes |
| Rate limited | Exponential backoff, then failover, then escalation |
| Vector store unreachable | Escalates with reason `no_supporting_documentation`, flagged `retrieval_unavailable` |
| Malformed ticket | Logged, escalated, the batch continues |
| Guardrail blocks the reply | Escalates with reason `guardrail_blocked_response` |

Every one of these is covered by a test in [tests/test_pipeline.py](tests/test_pipeline.py).

---

## Repository layout

```
README.md                    this file
requirements.txt             pinned runtime dependencies
requirements-dev.txt         test and lint tooling
requirements-optional.txt    the torch embedding path
.env.example                 every variable, placeholder values only
src/                         ingest, classify, retrieve, route, generate,
                             guardrails, logging_store, api, pipeline, llm, config
prompts/build/               prompts that run inside the system
prompts/evaluation/          prompts used to judge output
prompts/README.md            the prompt register, with versions
tests/                       75 tests, no API key required
evaluation/harness.py        unattended batch processing
evaluation/metrics.py        the three tiers of measurement
evaluation/results/          dated output from each run
scripts/build_index.py       build the vector store
scripts/tune_chunking.py     the chunking experiment
monitoring/                  prometheus.yml, grafana_dashboard.json
docs/architecture.md         design detail and traceability
data/                        the supplied datasets
storage/                     generated at runtime, never committed
.github/workflows/ci.yml     tests and a harness smoke run on every push
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on an installed package | The virtual environment is not active in this terminal | Activate it; check `python -c "import sys; print(sys.prefix)"` points inside `.venv` |
| `FileNotFoundError: No Chroma store at ...` | The index was never built | `python -m scripts.build_index` |
| Every ticket escalates | No API key, so the system is in its degraded mode | Set `OPENROUTER_API_KEY` in `.env`; check `GET /health` |
| `401` from the provider | Key has a stray newline from being pasted | Print `len(key)`; if longer than expected there is whitespace in it |
| First run is very slow | The embedding model is downloading (~90 MB) | Expected once; do it before a demonstration, not during one |
| Retrieval returns nothing for everything | `RETRIEVAL_MAX_DISTANCE` set too low | 0.55 is the measured value; raise it to loosen |
