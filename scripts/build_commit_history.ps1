# Builds the commit history in logical units of work.
# One-off: run from Project_build after `git init`.
$ErrorActionPreference = 'Stop'

$commits = @(
    @{ m = "chore: project scaffold, pinned dependencies and environment template";
       f = @('.gitignore', '.flake8', 'pyproject.toml', 'requirements.txt', 'requirements-dev.txt', 'requirements-optional.txt', '.env.example') },
    @{ m = "feat(config): settings loaded once, paths resolved against the project root";
       f = @('src/__init__.py', 'src/config.py', 'src/schemas.py') },
    @{ m = "feat(data): vendor the supplied datasets so the repo is self-contained";
       f = @('data/') },
    @{ m = "feat(ingest): normalise tickets from all four channels (A2)";
       f = @('src/ingest.py') },
    @{ m = "feat(retrieve): chroma index with a relevance floor so retrieval may return nothing (A4)";
       f = @('src/retrieve.py', 'scripts/__init__.py', 'scripts/build_index.py') },
    @{ m = "feat(llm): openrouter primary with automatic groq failover (A11)";
       f = @('src/llm.py') },
    @{ m = "feat(prompts): versioned prompt files loaded from disk";
       f = @('src/prompts.py', 'prompts/build/classify_ticket.md', 'prompts/build/generate_response.md', 'prompts/evaluation/judge_groundedness.md') },
    @{ m = "feat(classify): intent, urgency and calibrated confidence with a keyword fallback (A3)";
       f = @('src/classify.py') },
    @{ m = "feat(route): deterministic escalation decision, no model call (A5)";
       f = @('src/route.py') },
    @{ m = "feat(generate): grounded drafting with citation validation (A6)";
       f = @('src/generate.py') },
    @{ m = "feat(guardrails): blocking checks for pii, forbidden claims and grounding (A7)";
       f = @('src/guardrails.py') },
    @{ m = "feat(logging): decision log with exact reconciliation to ticket count (A8)";
       f = @('src/logging_store.py') },
    @{ m = "feat(observability): prometheus metrics and structured logging";
       f = @('src/observability.py', 'monitoring/prometheus.yml', 'monitoring/grafana_dashboard.json') },
    @{ m = "feat(pipeline): langgraph state machine wiring the stages together";
       f = @('src/pipeline.py') },
    @{ m = "feat(eval): metrics across the three tiers of the evaluation framework (A10)";
       f = @('evaluation/__init__.py', 'evaluation/metrics.py') },
    @{ m = "feat(eval): unattended harness taking --input and --output (A1, A9)";
       f = @('evaluation/harness.py') },
    @{ m = "feat(api): service, kill switch and metrics endpoint";
       f = @('src/api.py') },
    @{ m = "test: suite covering the acceptance criteria, no api key required (A12)";
       f = @('tests/') },
    @{ m = "ci: tests, credential scan and a harness smoke run on every push";
       f = @('.github/workflows/ci.yml') },
    @{ m = "experiment(retrieval): sweep chunk size against relevance floor";
       f = @('scripts/tune_chunking.py', 'docs/retrieval_experiment.json') },
    @{ m = "perf(retrieval): adopt measured chunk 400/60 and floor 0.55, F1 79.5%";
       f = @('src/config.py', '.env.example', 'docs/retrieval_experiment.md') },
    @{ m = "docs: readme, architecture and prompt register with traceability";
       f = @('README.md', 'docs/architecture.md', 'prompts/README.md') },
    @{ m = "chore(eval): baseline run with no provider configured, proving the degraded path";
       f = @('evaluation/results/') }
)

foreach ($c in $commits) {
    $existing = $c.f | Where-Object { Test-Path $_ }
    if (-not $existing) { Write-Host "skip (no files): $($c.m)"; continue }
    git add -- $existing
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { Write-Host "skip (nothing staged): $($c.m)"; continue }
    git commit -q -m $c.m
    Write-Host "committed: $($c.m)"
}

git add -A
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -q -m "chore: remaining project files"
    Write-Host "committed: remaining project files"
}
