# Metrics report

_Generated 2026-09-30 12:22:36 by `src/build_metrics_report.py`._

Rows marked _not yet measured_ have no result file yet; the command that produces each one is given beside it.

## Official test-split result

Model **codefuse-ai/F2LLM-v2-1.7B** at revision `3766d46e7a68`, config `official_f2llm17b_noreranker.json`, 948.4 s on cuda.

| k | NDCG@k | MRR@k | Recall@k | Precision@k | MAP@k |
|---|---|---|---|---|---|
| 1 | 0.8884 | 0.8884 | 0.8884 | 0.8884 | 0.8884 |
| 3 | 0.9283 | 0.9190 | 0.9548 | 0.3183 | 0.9190 |
| 5 | 0.9345 | 0.9225 | 0.9697 | 0.1939 | 0.9225 |
| 10 | 0.9376 | 0.9238 | 0.9796 | 0.0979 | 0.9238 |
| 20 | 0.9397 | 0.9243 | 0.9880 | 0.0494 | 0.9243 |
| 100 | 0.9413 | 0.9246 | 0.9966 | 0.0100 | 0.9245 |
| 1000 | 0.9416 | 0.9246 | 0.9989 | 0.0010 | 0.9246 |

## Latency and serving cost

| Setting | p50 | p95 | Peak RSS | Threads |
|---|---|---|---|---|
| CPU latency | _not yet measured_ | | | |

CPU precision (fp32 / bf16 / int8): _not yet measured_ — `python src/check_cpu_precision.py --index full --modes fp32 int8`

## Indexing cost

| Corpus | Documents | Encode time | Index size | Notes |
|---|---|---|---|---|
| APPS (full, 1.7B) | 8,765 | 477 s | 41.3 MB | exported by the official run |
| click history | 8275 rows / 224 lineages | 75 s | — | 40 commits, 95.4% embeddings reused |

## P1 (incremental rebuild) and Bonus (evolutionary retrieval)

**Real history (click, 40 commits)**

| Measure | Value |
|---|---|
| embeddings, full rebuild | 8275 |
| embeddings, incremental | 379 |
| saved | 95.4% |
| speed-up | 7.38x |
| P1: correct lineage at rank 1 when targeting a version | 0.795 |
| Bonus: top-10 duplicate slots, all versions | 89.8% |
| Bonus: top-10 duplicate slots, collapsed | 0.0% |
| Bonus: lineage recall@10, all versions -> collapsed | 0.807 -> 0.879 |

**Synthetic fixture** (known-by-construction mutations): 1200 -> 978 embeddings, 18.5% saved.

## Agent vs plain dense search

27 labelled questions ({'structural': 10, 'usage': 5, 'semantic': 12}), ground truth by {'grep': 15, 'docstring': 12} — never by the parser under test.

| Group | P@k dense | P@k agent | Recall dense | Recall agent | ms dense | ms agent |
|---|---|---|---|---|---|---|
| all (n=27) | 0.000 | 0.585 | 0.000 | 0.758 | 56 | 54 |
| semantic (n=12) | 0.000 | 0.278 | 0.000 | 0.667 | 56 | 56 |
| structural (n=10) | 0.000 | 0.805 | 0.000 | 0.950 | 57 | 0 |
| usage (n=5) | 0.000 | 0.880 | 0.000 | 0.593 | 56 | 2 |

## Dev experiments (adoption rule: >= +0.005 NDCG@10, CI excludes zero, worsened <= half improved)

| Experiment | Best setting | NDCG@10 | Delta | Verdict |
|---|---|---|---|---|
| A. Confidence-gated rerank | _not yet measured_ | | | |
| F. Code-to-description fusion | _not yet measured_ | | | |
| B. Fine-tuned 0.6B (holdout) | _not yet measured_ | | | |
| E. Category tiebreaker | _not yet measured_ | | | |

- gated: `python src/eval/dev_gated_rerank.py --preset f2llm-v2-1.7b --device cuda`
- descriptions: `python src/eval/dev_descriptions.py --preset f2llm-v2-1.7b --device cuda`
- finetune: `python src/train/finetune_lite.py --stage eval --device cuda`
- categories: `python src/eval/dev_categories.py --preset f2llm-v2-1.7b --device cuda`

## Coverage

- measured: 6/13 — official, history, history_ingest, synthetic_p1, bonus, agent
- not yet measured: cpu_full, cpu_lite, precision, gated, descriptions, finetune, categories
