# Evaluation Results

[← Evaluation Framework](README.md) | [Home](../../README.md)

**Status: Work in progress.** Evaluation infrastructure is complete (see [Evaluation Framework](README.md)); results will be added after running the comprehensive grid search across all 46 strategy configurations.

## Planned Evaluation

### Configurations to Test

**Chunking Strategies:**
- `section` — 800-token baseline
- `semantic_std2` — Semantic boundaries (k=2.0 std coefficient)
- `semantic_std3` — Semantic boundaries (k=3.0 std coefficient)
- `contextual` — LLM-enriched chunks
- `raptor` — Hierarchical summaries

**Preprocessing Strategies:**
- `none` — Original query
- `hyde` — Hypothetical document
- `decomposition` — Sub-queries + union merge
- `graphrag` — Entity extraction + communities

**Alpha Values (BM25/Vector Balance):**
- `0.0` — Pure BM25
- `0.5` — Balanced hybrid
- `1.0` — Pure vector

### Expected Output Format

```markdown
## Leaderboard (Top 10)

| Rank | Chunking | Preprocessing | Alpha | Faithfulness | Relevancy | Precision | Correctness |
|------|----------|---------------|-------|--------------|-----------|-----------|-------------|
| 1 | contextual | decomposition | 0.7 | 0.89 | 0.85 | 0.82 | 0.71 |
| 2 | raptor | hyde | 0.7 | 0.87 | 0.84 | 0.80 | 0.69 |
| ... | | | | | | | |

## Best Configuration by Question Type

| Type | Best Config | Why |
|------|-------------|-----|
| Factual | section + none | Direct matching, fast |
| Conceptual | contextual + hyde | Semantic enrichment helps |
| Comparative | any + decomposition | Multi-aspect coverage |
| Synthesis | raptor + graphrag | Cross-document reasoning |
```

## Running the Evaluation

```bash
# Full grid search (~2-3 hours)
python -m src.stages.run_stage_7_evaluation --comprehensive

# Results appended to: memory-bank/evaluation-history.md
```

## Interpretation Notes

- **Faithfulness > 0.8** = minimal hallucination
- **Context Precision < 0.5** = retrieval needs tuning
- **Large alpha gap** = BM25 keywords matter for this corpus


---

## Navigation

**Related:**
- [Getting Started](../getting-started.md) — Start over
- [Evaluation Framework](README.md) — RAGAS metrics and testing
