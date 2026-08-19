# STEP 6 — Second-Stage Answerability Reorder Experiment (OFFLINE)

Determines whether an answerability/usefulness signal can improve the ordering of the
frozen reranked Top-10 results, evaluated ONLY against the frozen binary relevance labels.

## Constraints

- Reorders only the frozen Top-10 per question (permutation of the same 10 chunks).
- Never modifies frozen files, labels, baselines, datasets, or the production pipeline.
- All artifacts are new, under `evaluation/experiments/step6/`.

## Reproduce

```bash
# 1. Run the full experiment (baseline + oracle + lqo + crossset + suppression, alpha sweep)
.venv/bin/python evaluation/experiments/step6/run_step6.py

# 2. Affected-subset + per-question + regression analysis
.venv/bin/python evaluation/experiments/step6/analyze_step6.py
```

Outputs (all under `results/`):
- `<variant>_<set>_order.jsonl` — new top-10 order per question for every variant/alpha.
- `summary.json` — full-set metrics + deltas for all variants.
- `analysis.json` — affected-subset metrics, per-question deltas, regression counts.

## Variants

| variant | type | leakage |
|---|---|---|
| baseline | frozen | — |
| oracle_usefulness (α sweep) | ORACLE | uses target question's LLM usefulness |
| oracle_class (α sweep) | ORACLE | uses target question's LLM class |
| lqo (α sweep) | NON-ORACLE | leave-question-out classifier |
| crossset (α sweep) | NON-ORACLE | train bench→apply hold & vice versa |
| suppress_rationale | ORACLE (leaky class) | hard-demotes RATIONALE/INDEX_TOC |

## Scoring

```
norm_ce = per-question min-max of reranker_score within the top-10 pool
combined = (1 - alpha) * norm_ce + alpha * norm(answerability_score)
```

Answerability scores: LLM usefulness (high=1/med=.5/low=0), LLM class map, or predicted
usefulness from a LogisticRegression over all-MiniLM-L6-v2 (question, chunk) embeddings
(lqo/crossset). alpha ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}.

## Key result

- **ORACLE**: usefulness weighting (α=0.5) → bench P@3 0.533→0.650 (+.117), hold 0.580→0.679
  (+.099), zero per-question regressions. Upper bound.
- **NON-ORACLE** (leave-question-out classifier): bench P@3 +.033, hold +.025 at α=0.3, with
  3-4 per-question regressions. Classifier agreement with the oracle is weak (Pearson ≈0.26);
  254 diagnostic examples are insufficient for a reliable model.
- **Suppressing RATIONALE hard is unsafe** (regresses relevant RATIONALE chunks H08/H17).

Full analysis: `STEP6_REPORT.md`.