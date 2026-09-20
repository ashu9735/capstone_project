"""Choose the chunk configuration and the relevance floor together.

    python -m scripts.tune_chunking --tickets data/development_tickets.json

Chunk size is a design decision, and so is the distance above which a passage is
discarded. They interact: smaller chunks score closer, so a floor tuned for one
configuration is wrong for another. Both are selected here on the same evidence and
recorded in docs/retrieval_experiment.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.ingest import load_tickets, normalise
from src.retrieve import Retriever, build_index, load_documents

CONFIGURATIONS = [(400, 60), (600, 100), (800, 120), (1200, 200), (1600, 240)]
DISTANCE_FLOORS = [round(0.30 + 0.05 * i, 2) for i in range(15)]  # 0.30 .. 1.00


def _collect(chunk_size: int, chunk_overlap: int, tickets, store_dir: Path, top_k: int) -> dict[str, Any]:
    """Index once, then record every hit so floors can be swept without re-embedding."""
    settings = get_settings().model_copy(
        update={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chroma_path": str(store_dir / f"c{chunk_size}_{chunk_overlap}"),
            "retrieval_top_k": top_k,
        }
    )
    passages = build_index(settings=settings, documents=load_documents(settings=settings))
    retriever = Retriever(settings)

    observations = []
    for raw in tickets:
        try:
            ticket = normalise(raw)
        except ValueError:
            continue
        hits = retriever.search(ticket.text, max_distance=99.0)
        observations.append(
            {
                "expected": set(raw.get("labels", {}).get("expected_doc_ids") or []),
                "hits": [(h.doc_id, h.distance) for h in hits],
            }
        )
    return {"passages": passages, "observations": observations}


def _score(observations, floor: float) -> dict[str, float]:
    hits = scored = abstain_total = abstain_correct = 0
    recalls, precisions = [], []
    for obs in observations:
        retrieved = {doc_id for doc_id, distance in obs["hits"] if distance <= floor}
        expected = obs["expected"]
        if expected:
            scored += 1
            overlap = retrieved & expected
            hits += 1 if overlap else 0
            recalls.append(len(overlap) / len(expected))
            precisions.append(len(overlap) / len(retrieved) if retrieved else 0.0)
        else:
            abstain_total += 1
            abstain_correct += 1 if not retrieved else 0

    hit_rate = 100 * hits / scored if scored else 0.0
    abstention = 100 * abstain_correct / abstain_total if abstain_total else 0.0
    recall = 100 * statistics.mean(recalls) if recalls else 0.0
    precision = 100 * statistics.mean(precisions) if precisions else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "floor": floor,
        "hit_rate_pct": round(hit_rate, 2),
        "mean_recall_pct": round(recall, 2),
        "precision_pct": round(precision, 2),
        "f1_pct": round(f1, 2),
        "correct_abstention_pct": round(abstention, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select chunk size and relevance floor.")
    parser.add_argument("--tickets", default="data/development_tickets.json")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="docs/retrieval_experiment.json")
    args = parser.parse_args()

    tickets = load_tickets(args.tickets)[: args.limit]
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp:
        for chunk_size, overlap in CONFIGURATIONS:
            collected = _collect(chunk_size, overlap, tickets, Path(tmp), args.top_k)
            best_here: dict[str, Any] | None = None
            for floor in DISTANCE_FLOORS:
                row = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                    "passages": collected["passages"],
                    **_score(collected["observations"], floor),
                }
                rows.append(row)
                if best_here is None or row["f1_pct"] > best_here["f1_pct"]:
                    best_here = row
            assert best_here is not None
            print(
                f"chunk_size={chunk_size:<5} overlap={overlap:<4} passages={collected['passages']:<4} "
                f"best floor={best_here['floor']:<5} f1={best_here['f1_pct']:>6}% "
                f"hit={best_here['hit_rate_pct']:>6}% recall={best_here['mean_recall_pct']:>6}% "
                f"precision={best_here['precision_pct']:>6}% abstain={best_here['correct_abstention_pct']:>6}%"
            )

    best = max(rows, key=lambda r: (r["f1_pct"], r["hit_rate_pct"]))
    payload = {
        "tickets_evaluated": len(tickets),
        "top_k": args.top_k,
        "selection_rule": (
            "highest retrieval F1 (mean recall against expected documents versus precision), "
            "hit rate breaks ties. A hit-rate/abstention average was tried first and rejected: "
            "it is maximised by a configuration that abstains on 92% of tickets. Abstention is "
            "reported but not optimised here, because the router escalates independently "
            "whenever retrieval returns nothing."
        ),
        "selected": {
            "chunk_size": best["chunk_size"],
            "chunk_overlap": best["chunk_overlap"],
            "retrieval_max_distance": best["floor"],
        },
        "selected_scores": best,
        "grid": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"\nselected CHUNK_SIZE={best['chunk_size']} CHUNK_OVERLAP={best['chunk_overlap']} "
        f"RETRIEVAL_MAX_DISTANCE={best['floor']}\n"
        f"  F1 {best['f1_pct']}% | hit rate {best['hit_rate_pct']}% | recall {best['mean_recall_pct']}% "
        f"| precision {best['precision_pct']}% | correct abstention {best['correct_abstention_pct']}%\n"
        f"  detail written to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
