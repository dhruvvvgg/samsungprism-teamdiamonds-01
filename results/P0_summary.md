# P0 — the retrieval search, and how the final configuration was chosen

Task: MTEB `AppsRetrieval` (CoIR-Retrieval/apps) — 3,765 test queries against the full 8,765-document
corpus. Headline metrics NDCG@10 and MRR@10.

**Final configuration: F2LLM-v2-1.7B alone. No reranker, no fusion, no BM25, no LLM re-judge.**
Config file: [`configs/official_f2llm17b_noreranker.json`](../configs/official_f2llm17b_noreranker.json).

---

## 1. Official test result

Run on a Kaggle T4 at commit `2060b88`, 3,765 queries × 8,765 documents, via `mteb.evaluate`:

| | NDCG@10 | MRR@10 |
|---|---|---|
| **F2LLM-v2-1.7B alone (submitted)** | **0.9376** | **0.9238** |
| published MTEB score for the same model | 0.93692 | 0.92288 |
| F2LLM-v2-0.6B alone (our run) | 0.90443 | 0.88397 |
| published MTEB score for 0.6B | 0.90446 | 0.884035 |

Reproducing the published numbers to within 0.0007 (1.7B) and 0.00003 (0.6B) is the harness check: it
says the evaluation path, the prompt and the pooling are right, so the remaining differences between
candidates are real differences between candidates.

Timing for the submitted run: **996 s total** — 476 s document encoding, 503 s query encoding. No fp32
fallback was triggered (the fp16 run stayed numerically stable throughout).

## 2. How candidates were judged

Every choice — model, variant, weights, top-k, thresholds — was made on a **dev protocol built on the
train split**, never on test:

* 4,000-query tune set drawn from the CoIR-APPS train partition, with a fixed 1,000-query slice used for
  the sweeps, against the **full 8,765-document corpus**. Only the query count was ever reduced; the
  document pool was never subsampled.
* A separate 1,000-query holdout (seed 13) was carved out and reserved.
* The test split was touched exactly once, by `src/eval/run_official.py`, for the one chosen candidate.

**Adoption rule**, applied to every candidate:

1. NDCG@10 gain ≥ **0.005** over the base, **and**
2. the 95% paired-bootstrap CI on the gain **excludes zero**, **and**
3. it does not worsen lopsidedly: worsened queries ≤ **½ × improved** queries.

All three had to hold. A point estimate on its own was never enough — two candidates below died on
criterion 2 alone (see §4).

## 3. Dev-slice results (1,000-query tune slice)

| Candidate | NDCG@10 | MRR@10 | Verdict |
|---|---|---|---|
| F2LLM-v2-0.6B alone | 0.8959 | 0.8757 | baseline |
| EmbeddingGemma-300M alone | 0.8260 | 0.7973 | weaker than 0.6B |
| 0.6B + EmbeddingGemma fusion (w=0.4) | 0.9100 | 0.8928 | beats 0.6B, still below 1.7B |
| **F2LLM-v2-1.7B alone** | **0.9299** | **0.9167** | **the chosen system** |
| 1.7B + 0.6B fusion (w=0.25) | 0.9353 | 0.9218 | passed adoption |
| 1.7B + EmbeddingGemma fusion (w=0.3) | 0.9356 | 0.9227 | **failed** adoption |
| 1.7B + Qwen3-Reranker-0.6B (apps, k=30, α=0.5) | 0.9371 | 0.9228 | passed adoption |
| fusion + reranker, stacked | 0.9404 | 0.9274 | **rejected** — CI included zero |

Two candidates passed the adoption rule (the 1.7B+0.6B fusion and the reranker) and were still not
submitted; §5 says why.

## 4. What was tried and rejected

| Direction | Outcome |
|---|---|
| BM25 hybrid (RRF over dense + sparse) | Hurt at **every** setting, −0.004 to −0.21. BM25 alone scored 0.19 on the tune set |
| Query variants — instruction wording, example/spec stripping | **All** below the baseline; worst was narrative-only at −0.127 |
| BGE-reranker-v2-m3 | Hurt at every setting |
| Ettin-Reranker (68M and 150M) | Best case **+0.0011** — not significant, and far below the 0.005 bar. Its general-MTEB advantage over Qwen3-Reranker did not transfer to code retrieval |
| LLM re-judge stage | Trigger almost never fired: recall@100 is 0.991, so there is nothing for a judge to rescue |
| F2LLM-v2-4B | Does not fit a T4 — registry footprint 15.3 GB vs 14.56 GB total; OOM even at batch size 1 |
| Stacked fusion + reranker | Best point estimate 0.9404, but the 95% CI included zero — rejected on criterion 2 |

The negative results carried real weight here: they are why the final system is a single model with no
second stage, rather than a stack of components each justified by a point estimate.

## 5. Why 1.7B alone, when two candidates scored higher on dev

Both adopted candidates were deliberately not submitted, on cost:

* **1.7B + Qwen3-Reranker-0.6B** — +0.0072 NDCG@10 on dev, but several hours of GPU per full run
  (~113k cross-encoder pairs) and per-query cross-encoder latency at serving time.
* **1.7B + 0.6B fusion** — +0.0055 NDCG@10 on dev, but it doubles per-query encoding cost, since every
  query has to go through both models.

1.7B alone gives the best score-to-resource ratio: 996 s for a full test-split run, one model in memory,
one forward pass per query. The two more expensive configurations are kept, working and reproducible, on
their own branches (`qwen3-reranker-official`, `embeddinggemma-fusion-experiment`) rather than deleted.

## 6. Contamination check

The dev protocol uses train-split queries, so a model that had memorised the training data would score
higher on dev than on test. Measured gap, dev tune vs published test:

| Model | Gap (dev − published test) |
|---|---|
| F2LLM-v2-0.6B | +0.0035 |
| F2LLM-v2-1.7B | −0.0034 |

Both within ±0.004, and in opposite directions — no sign of train-split memorisation, and it establishes
the ±0.004 band that the official run's discrepancy check uses as its alarm threshold.

## 7. Earlier baselines on test

The path to the final model, all on the test split via the same harness:

| Model | NDCG@10 | MRR@10 |
|---|---|---|
| all-MiniLM-L6-v2 | 0.066 | 0.056 |
| CodeRankEmbed | 0.235 | 0.206 |
| gte-modernbert-base | 0.576 | 0.528 |
| Qwen3-Embedding-0.6B | 0.731 | 0.684 |
| F2LLM-v2-0.6B | 0.90443 | 0.88397 |
| **F2LLM-v2-1.7B** | **0.9376** | **0.9238** |

The 0.066 from all-MiniLM was initially read as a bug in the harness. It was not: the published MTEB
score for that model on this task is 0.0660, which our run reproduced as 0.06596. The lesson shaped the
rest of the work — on this task, model choice dominates everything else by an order of magnitude more
than any pipeline stage.

## 8. Reproducing it

```bash
python src/eval/run_official.py --config configs/official_f2llm17b_noreranker.json \
    --device cuda --confirm-test
python src/verify_submission.py --confirm-test
```

The run is guarded (`--confirm-test`, plus a lock file so the test split is not silently re-consumed)
and writes `outputs/appsretrieval_results.json`, the top-100 rankings, a checksum file and the served
`runtime_index/`. `verify_submission.py` recomputes NDCG@10 and MRR@10 from the stored rankings with an
independent metric implementation and re-checks every hash.

## Not measured

Recorded as TBD rather than estimated:

* MRR@10 for the individual dev rows not listed in §3 (the variant, BM25 and reranker sweeps were
  selected on NDCG@10; their MRR@10 values were not tabulated): **TBD**
* F2LLM-v2-4B dev baseline — the model does not fit a T4, so there is no number: **TBD (blocked)**
* The n=4,000 re-run of the stacked fusion + reranker comparison, to tighten the CI that rejected it on
  the 1,000-query slice: **TBD**
* Per-stage wall-clock for the reranked configuration on the full test split: **TBD** (never run on test)
* CPU serving latency: **now measured** — see the resource table in the README (1.7B ~7.4 s p50 on two
  cores, 11.0 GB RSS; lite 0.6B ~2.9 s, 4.5 GB).
* P1 / Bonus benchmark figures: **not yet measured** with a real encoder. The machinery is implemented
  and unit-tested, but only mock-encoder plumbing runs have been done, so no number is claimed.
