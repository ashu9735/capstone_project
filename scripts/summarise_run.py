"""Print a one-screen summary of a harness run.

    python -m scripts.summarise_run evaluation/results/validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise a harness run directory.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    report = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    run = report["run"]
    t1 = report["tier_one_business_outcomes"]
    t2 = report["tier_two_technical_performance"]
    t3 = report["tier_three_governance_conditions"]
    cls = t2["intent_classification"]

    print(f"RUN        {run['input_file']}")
    print(f"           {run['tickets']} tickets in {run.get('wall_clock_seconds')}s "
          f"| model {run['model']} | threshold {run['confidence_threshold']}")
    print()
    print("TIER 1  BUSINESS")
    print(f"  first contact resolution   {t1['first_contact_resolution_pct']:>7}%   (baseline 42, target 60)")
    print(f"  escalation rate            {t1['escalation_rate_pct']:>7}%   (baseline 58, target <=30)")
    print(f"  mean reply                 {t1['time_to_first_reply']['mean_minutes']:>7} min (target <5)")
    print(f"  satisfaction proxy         {t1['satisfaction_proxy']['mean_score_out_of_5']:>7} /5  (baseline 3.2, target 4.0)")
    print()
    print("TIER 2  TECHNICAL")
    print(f"  intent accuracy            {cls.get('accuracy_pct', 'n/a'):>7}%")
    print(f"  intent weighted precision  {cls.get('weighted_precision_pct', 'n/a'):>7}%   (target 85)")
    print(f"  urgency accuracy           {t2['urgency']['accuracy_pct']:>7}%")
    print(f"  routing agreement          {t2['routing']['agreement_pct']:>7}%   "
          f"(false auto-respond: {t2['routing']['false_auto_respond']})")
    print(f"  retrieval hit rate         {t2['retrieval']['hit_rate_at_k_pct']:>7}%")
    print(f"  retrieval recall           {t2['retrieval']['mean_recall_at_k_pct']:>7}%")
    print(f"  correct abstention         {t2['retrieval']['correctly_returned_nothing_pct']:>7}%")
    print(f"  citation resolvable        {t2['citation_accuracy']['resolvable_pct']:>7}%   (target 95)")
    print(f"  hallucination (automated)  {t2['hallucination']['automated_hallucination_rate_pct']:>7}%   (target <=5)")
    print(f"  latency p95                {t2['latency_seconds']['p95']:>7} s   (target <3)")
    print(f"  handled without error      {t2['availability']['handled_without_error_pct']:>7}%")
    print(f"  degraded                   {t2['availability']['degraded_pct']:>7}%")
    for reason, count in sorted(t2["availability"]["degradation_reasons"].items()):
        print(f"      {reason:<28} {count}")
    print()
    print("TIER 3  GOVERNANCE (binary)")
    for key in ("private_data_in_outbound_text", "quality_across_customer_groups",
                "decision_logging", "confidence_calibration"):
        holds = t3[key].get("condition_holds")
        print(f"  {key:<32} {'HOLDS' if holds else 'FAILS'}")
    print(f"  {'ALL CONDITIONS':<32} {'HOLD' if t3['all_conditions_hold'] else 'DO NOT HOLD'}")
    print()
    print("ROUTING REASONS")
    for reason, count in sorted(t2["routing"]["reason_breakdown"].items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<36} {count}")

    fairness = t3["quality_across_customer_groups"]
    print()
    print(f"FAIRNESS  gated on {', '.join(fairness['gated_on'])}"
          f" | worst {fairness['worst_measure']} = {fairness['max_variation_points']} pts")
    for dimension, payload in fairness.items():
        if not isinstance(payload, dict) or "groups" not in payload:
            continue
        print(f"  {dimension}  (variation {payload['variation_points']} pts)")
        print(f"    {'group':<16} {'n':>5} {'intent acc':>11} {'false auto':>11} {'auto %':>8} {'conf':>6}")
        for name, row in payload["groups"].items():
            print(f"    {name:<16} {row['n']:>5} {row['intent_accuracy_pct']:>10}% "
                  f"{row['false_auto_respond_pct']:>10}% {row['auto_respond_rate_pct']:>7}% "
                  f"{row['mean_confidence']:>6}")

    calib = t3["confidence_calibration"]
    print()
    print(f"CALIBRATION  max gap {calib['max_gap_points']} pts")
    for band in calib["bands"]:
        print(f"  {band['low']:.1f}-{band['high']:.1f}  n={band['count']:<5} "
              f"stated {band['stated']:.3f}  observed {band['observed']:.3f}  "
              f"gap {band['gap_points']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
