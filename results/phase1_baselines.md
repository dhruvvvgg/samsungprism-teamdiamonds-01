# Phase 1 baselines: MTEB `AppsRetrieval` (test split, 8,765 docs / 3,765 queries)

All "ours" numbers come from Kaggle runs of `src/eval/run_baseline.py` (`mteb.evaluate`, dataset revision `f22508f9`).
"Published" numbers are from the public MTEB results repo (`embeddings-benchmark/results`), same dataset revision.
A dash means not measured / not reported yet.

| Model | Params | Ours NDCG@10 | Ours MRR@10 | Published NDCG@10 | Run config | Eval time | Peak RSS | Notes |
|---|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 22M | 0.06596 | 0.0558 | 0.0660 | CPU, max_seq 256, batch 64 | 902 s | 2,472 MB | Matches the published score, so harness and dataset are validated. |
| `nomic-ai/CodeRankEmbed` | 137M | 0.23523 | 0.20577 | none published | cuda fp32, max_seq 1024, batch 32, prefix "Represent this query for searching relevant code: " | 1,148 s | 2,988 MB | Unsorted batches, eager attention, which explains the slow GPU run. |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 0.731 (reported by user) | - | 0.7534 | config not recorded here | - | - | Fill MRR / config / time from the run's `_run_info`. |
| `Alibaba-NLP/gte-modernbert-base` | 149M | - | - | 0.5641 | - | - | - | Not run yet. |
| `codefuse-ai/F2LLM-v2-0.6B` | 0.6B | **TBD** | **TBD** | 0.9045 (MRR@10 0.8840) | cuda fp16, max_seq 8192, batch 16, length-sorted, preset `f2llm-v2-0.6b` | TBD | TBD | Placeholder. Fill from `outputs/baseline_f2llm_v2_0.6b.json` (`scores.test[0]` and `_run_info`). Published run: MTEB 2.6.7, bf16, 75 s. |

## F2LLM-v2-0.6B run checklist (fill after the Kaggle run)
- [ ] `[dense]` line shows `model_device=cuda:0`, `dtype=torch.float16` (or `float32` if it fell back; `_run_info.fell_back_to_fp32`)
- [ ] `[dense] EOS check ok`
- [ ] NDCG@10 within about 0.01 of 0.9045? If not, compare `_run_info.versions` and dtype against the published run
- [ ] Copy NDCG@10, MRR@10, `total_eval_seconds`, `peak_rss_mb` into the table above
