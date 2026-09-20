# Evaluation report — CloudServe support automation

- Run generated: 2026-09-11T09:31:25+00:00
- Input file: `data\development_tickets.json`
- Tickets processed: 500
- System version: 1.0.0
- Model: none (no provider configured)
- Confidence threshold: 0.8
- Runs against the hidden evaluation set: 0

## Headline results

| Measure | Baseline | Target | Achieved |
|---|---|---|---|
| First contact resolution | 42% | 60% | 0.0% |
| Mean time to first reply | 8 to 12 hrs | < 5 min | 0.0009 min |
| Satisfaction proxy | 3.2 / 5 | 4.0 | 3.692 / 5 |
| Escalation rate | 58% | <= 30% | 100.0% |
| Classification precision | — | 85% | 75.03% |
| Hallucination rate | — | <= 5% | 0.0% |
| Citation accuracy | — | 95% | 0.0% |
| Latency p95 | — | < 3 s | 0.074 s |
| Private data occurrences | — | 0 | 0 |
| Cross-group variation | — | < 5 pts | 11.42 pts |

## Tier one — business outcomes

- First contact resolution: **0.0%** against a 42% baseline and a 60% target.
- Escalation rate: **100.0%** against a 58% baseline and a 30% target.
- Time to first reply: mean 0.0009 min, median 0.0007 min, p95 0.0012 min.
- Satisfaction proxy: 3.692 / 5 over 500 tickets. proxy only, not collected from customers.
- Repeat-contact risk: 0 of 0 automatic replies carried no citation.

## Tier two — technical performance

- Intent classification: accuracy 59.8%, weighted precision 75.03%, weighted recall 59.8%.
- Urgency accuracy: 32.8%.
- Routing agreement with the expected route: 37.8% (0 tickets answered automatically that should have escalated).
- Retrieval hit rate at k: 92.16%; correctly returned nothing on 14.69% of tickets with no expected document.
- Citation accuracy: 0.0% of citations resolve to a retrieved passage.
- Automated hallucination lower bound: 0.0%.
- Latency: mean 0.054s, p50 0.043s, p95 0.074s.
- Handled without error: 100.0%; 100.0% completed on a degraded path.

## Tier three — governance conditions

| Condition | Requirement | Result | Holds |
|---|---|---|---|
| Private data in outbound text | Zero occurrences | 0 | yes |
| Quality across customer groups | Under 5 points | 11.42 pts | NO |
| Decision logging | Complete coverage | 500 of 500 | yes |
| Confidence calibration | Within 5 points | 0.03 pts | yes |

**All governance conditions hold: NO**

## Confidence calibration detail

| Band | n | Stated | Observed | Gap (pts) |
|---|---|---|---|---|
| 0.0–0.2 | 125 | 0.120 | 0.120 | +0.00 |
| 0.6–0.8 | 375 | 0.757 | 0.757 | -0.03 |

## Fairness detail

Gated on intent_accuracy_pct, false_auto_respond_pct. Auto-respond rate and routing agreement are shown for context but not gated: both move with a group's underlying answerability.

### language fluency (variation 7.41 pts)

| Group | n | Intent accuracy % | False auto-respond % | Auto-respond % | Routing agreement % | Mean confidence |
|---|---|---|---|---|---|---|
| fluent | 380 | 61.58 | 0.0 | 0.0 | 39.21 | 0.611 |
| non_fluent | 120 | 54.17 | 0.0 | 0.0 | 33.33 | 0.555 |

### customer tier (variation 5.36 pts)

| Group | n | Intent accuracy % | False auto-respond % | Auto-respond % | Routing agreement % | Mean confidence |
|---|---|---|---|---|---|---|
| business | 164 | 59.76 | 0.0 | 0.0 | 35.37 | 0.609 |
| enterprise | 83 | 63.86 | 0.0 | 0.0 | 38.55 | 0.573 |
| standard | 253 | 58.5 | 0.0 | 0.0 | 39.13 | 0.598 |

### customer region (variation 4.81 pts)

| Group | n | Intent accuracy % | False auto-respond % | Auto-respond % | Routing agreement % | Mean confidence |
|---|---|---|---|---|---|---|
| asia_pacific | 119 | 61.34 | 0.0 | 0.0 | 46.22 | 0.58 |
| europe | 151 | 56.95 | 0.0 | 0.0 | 31.79 | 0.571 |
| latin_america | 60 | 58.33 | 0.0 | 0.0 | 35.0 | 0.64 |
| north_america | 170 | 61.76 | 0.0 | 0.0 | 38.24 | 0.618 |

### channel (variation 11.42 pts)

| Group | n | Intent accuracy % | False auto-respond % | Auto-respond % | Routing agreement % | Mean confidence |
|---|---|---|---|---|---|---|
| chat | 155 | 52.26 | 0.0 | 0.0 | 32.26 | 0.539 |
| docs_comment | 78 | 62.82 | 0.0 | 0.0 | 47.44 | 0.602 |
| email | 212 | 63.68 | 0.0 | 0.0 | 37.26 | 0.628 |
| forum | 55 | 61.82 | 0.0 | 0.0 | 41.82 | 0.641 |

## Limitations

The figures above should be treated with caution because the satisfaction score is a
rubric proxy rather than customer feedback, the hallucination rate is an automated
lower bound that counts only machine-checkable failures, and the evaluation set is
drawn from a single quarterly extract of roughly 39 tickets a week rather than the
500 a week the client reports, so the intent mix may not match live traffic.

## Appendix — per-class classification report

```
                         precision    recall  f1-score   support

         account_access      0.792     0.864     0.826        22
          api_key_issue      0.333     0.056     0.095        18
     api_usage_question      0.182     0.167     0.174        24
 authentication_failure      1.000     0.650     0.788        20
          billing_query      0.857     1.000     0.923        24
     compliance_request      0.525     0.808     0.636        26
     configuration_help      0.172     0.294     0.217        17
            data_export      1.000     1.000     1.000        29
         data_residency      0.000     0.000     0.000        29
         database_issue      1.000     0.962     0.980        26
     deployment_failure      0.659     1.000     0.794        27
        feature_request      1.000     0.500     0.667        20
       integration_help      1.000     0.286     0.444        21
             onboarding      1.000     0.182     0.308        22
performance_degradation      0.947     0.783     0.857        23
       quota_or_overage      0.733     0.478     0.579        23
             rate_limit      1.000     0.692     0.818        13
       rollback_request      1.000     1.000     1.000        28
      security_incident      1.000     0.231     0.375        26
      sso_configuration      1.000     0.654     0.791        26
        unclear_request      0.120     1.000     0.214        15
          webhook_issue      1.000     0.333     0.500        21

               accuracy                          0.598       500
              macro avg      0.742     0.588     0.590       500
           weighted avg      0.750     0.598     0.607       500
```
