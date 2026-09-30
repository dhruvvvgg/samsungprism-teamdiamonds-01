# Dev runbook (Kaggle, T4). Order = cheapest first, respecting dependencies

Rules: sweeps use the **train-split** dev protocol only (4,000 "tune" queries; the 1,000 holdout is never
touched unless you pass `--use-holdout`). The test split is touched only by step 9 (`run_official.py`).
Every script appends rows to `outputs/dev/results.jsonl` and rewrites `results/dev_table.md`.
`dev_select.py` turns adopted rows into `outputs/dev/chosen.json`; later scripts build on that file.

```
# 0. setup (fresh kernel, GPU on)
!git clone <repo-url>
%cd <repo-dir>
!pip install -q -r requirements.txt

# 1. [GPU ~10-15 min] dev protocol + fp16 embedding cache + baseline + dev-vs-test contamination check
!python src/eval/dev_split.py --device cuda

# 2. [CPU ~5 min, no model load] hybrid sweep on the cached baseline dense scores
!python src/eval/dev_hybrid.py
!python src/eval/dev_select.py --group hybrid

# 3. [GPU ~15-25 min] query-side variants (7 variants x 5,000 queries, cached; averages are free)
!python src/eval/dev_variants.py --device cuda
!python src/eval/dev_select.py --group variants
#    if variants were adopted, redo step 2 on the new dense scores (CPU only):
!python src/eval/dev_hybrid.py
!python src/eval/dev_select.py --group hybrid

# 4. [GPU ~15-30 min] rerankers on a fixed 1,000-query subsample of tune (2 rerankers, top-30 cached)
!python src/eval/dev_rerank.py --device cuda --n-queries 1000
!python src/eval/dev_select.py --group rerank

# 5. [API cost] LLM re-judge. Plumbing check first (no key, mock judge, numbers are NOT results):
!python src/eval/dev_rejudge.py --device cuda --n-queries 100 --mock
#    real run: set LLM_PROVIDER (+ LLM_MODEL) and the matching *_API_KEY in the notebook environment
%env LLM_PROVIDER=openai
%env OPENAI_API_KEY=...        # your key, set in the notebook only
!python src/eval/dev_rejudge.py --device cuda --n-queries 300
!python src/eval/dev_select.py --group rejudge

# 9. ONE official run of the chosen candidate (touches the test split; refuses to run twice)
!python src/eval/run_official.py --config configs/official_f2llm17b_noreranker.json --device cuda --confirm-test
```

Read `results/dev_table.md` after each step. ADOPT = dNDCG@10 >= +0.005, 95% paired-bootstrap CI > 0 and
worsened <= 0.5 x improved. If nothing is adopted for a stage, `dev_select.py` leaves it OFF.

Step 1 also prints a dev-vs-test check: baseline tune NDCG@10 vs the published TEST NDCG@10 for that
preset (`dev_lib.TEST_NDCG_REFERENCE`). A gap above +0.03 means the model has probably seen the train
pairs and dev gains may not transfer; this is flagged loudly but never blocks the run.

## Comparing a different first-stage model (e.g. F2LLM-v2-1.7B vs the 0.6B default)

Every script above takes `--preset NAME` (default `f2llm-v2-0.6b`; see `src/retrieval/model_presets.py`
for what's registered). Rows are tagged with the preset they came from, so a second preset's sweep never
collides with or gets mixed into the first's -- `dev_select.py` also needs `--preset NAME` to select from
and write to that preset's own rows / `outputs/dev/chosen_<preset>.json` (the default preset still uses
the plain `outputs/dev/chosen.json`, unchanged). One cell, same session (each `!python` line is its own
process, so GPU memory from step (a) is fully gone before step (b) starts regardless):

```
!python src/eval/dev_split.py --preset f2llm-v2-1.7b --device cuda --batch-size 8
!python src/eval/dev_rerank.py --preset f2llm-v2-1.7b --device cuda --n-queries 1000
```
`--batch-size 8` on the first line is a precaution: 1.7B is ~2.9x 0.6B's parameter count, so its encoding
activations are proportionally bigger on a T4. `dev_rerank.py`'s own reranker-scoring memory is unaffected
by the first-stage model size (`--rerank-batch-size` default 8 still applies); watch its `[dense] GPU mem
before/after release` log lines -- if step (a) already cached every embedding `dev_rerank.py` needs (same
`--config`'s `dense_variants` as what step (a) cached, i.e. `registry+full` unless a variants sweep was
separately adopted for this preset), step (b) never reloads the model at all and those lines will show
near-zero either side; if it does need to reload (e.g. a different `dense_variants` from `chosen.json`),
they'll show the real before/after numbers.

## Available rerankers

`dev_rerank.py --rerankers ...` (default: `qwen3-reranker-0.6b bge-reranker-v2-m3`):

| name | HF repo | loads via | instructions? |
|---|---|---|---|
| `qwen3-reranker-0.6b` | `Qwen/Qwen3-Reranker-0.6B` | `AutoModelForCausalLM`, yes/no logits | yes (`--instructions apps card`) |
| `bge-reranker-v2-m3` | `BAAI/bge-reranker-v2-m3` | `AutoModelForSequenceClassification` | no |
| `ettin-reranker-150m` | `cross-encoder/ettin-reranker-150m-v1` | `sentence_transformers.CrossEncoder` | no |
| `ettin-reranker-68m` | `cross-encoder/ettin-reranker-68m-v1` | `sentence_transformers.CrossEncoder` | no |

The Ettin checkpoints are Sentence-Transformers CrossEncoder repos (ModernBERT base + a
Dense/LayerNorm/Dense head in `2_Dense/`, `3_LayerNorm/`, `4_Dense/`), **not** plain
`*ForSequenceClassification` models -- loading them the way BGE loads would silently attach a
randomly-initialised head. They are by far the cheapest rerankers here (68M/150M vs 0.6B), and they
reuse whatever first-stage embeddings are already cached, so they need no new encoding pass.

## Dense+dense fusion (two F2LLM sizes)

`dev_fuse.py` fuses two presets' **cached** embeddings -- same `rrf_fuse`/`top_n` machinery as the
BM25+dense hybrid, fed two dense score lists instead of one dense + one sparse, plus a
normalise-then-weighted-average fusion for comparison. CPU only, no model loads:

```
!python src/eval/dev_fuse.py --preset f2llm-v2-1.7b --preset-b f2llm-v2-0.6b --n-queries 1000
```

`--preset` is model A (the stronger one; the adoption rule is judged against the better of the two).
It reads both presets' embeddings from `./data/cache/embeddings` and **exits immediately** naming the
missing half if either is absent, rather than silently starting a GPU encode. That cache is gitignored,
so a fresh clone or a new Kaggle session starts empty and both halves must be rebuilt with
`dev_split.py --preset <name> --device cuda` first.

### Stacking fusion under reranking

`dev_rerank.py --fuse-with` uses the score-averaged fusion as the **first stage**, then reranks its
top-k. Because the fused score is no longer a raw cosine, the interpolation weight must be re-swept:
pass an explicit `--alphas` grid rather than trusting a value tuned on cosines. Sweeping alpha is free
(the reranker scores each pair once and every k/alpha combination is computed offline from that), so
use a fine grid.

`--compare-to-rerank` makes the adoption base the **standing-best reranked pipeline** (unfused first
stage + `--base-reranker`/`--base-k`/`--base-alpha`) rather than the raw first stage, so the question
being answered is "does this beat the best we already have". That base is recomputed in the same run on
the same queries, which is what makes the paired bootstrap CI valid -- comparing against a number
copied from an earlier run would not be a paired comparison.

```
!python src/eval/dev_rerank.py --preset f2llm-v2-1.7b --n-queries 1000 \
    --rerankers qwen3-reranker-0.6b --instructions apps \
    --fuse-with f2llm-v2-0.6b --fuse-w-b 0.25 --fuse-norm zscore \
    --compare-to-rerank --base-k 30 --base-alpha 0.5 \
    --alphas 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

This needs a GPU: the fused first stage produces different candidates, so the reranker's cached pair
scores do not apply and it scores them afresh.

Fusion rows are recorded with an empty `config_patch` and are **not** adoptable into `chosen.json`:
wiring dense+dense into the official pipeline would mean `HybridSearchModel` loading and encoding with
two models, which is a separate change with its own GPU-memory implications. The sweep measures whether
that is worth doing.

## Memory warning

`dev_split.py` prints a loud (non-blocking) warning before loading when the preset's registered MTEB
footprint is at/over `--memory-warn-mb` (default 13000, leaving headroom under a T4's 14.56 GB).
`f2llm-v2-4b` trips this: registry `memory_usage_mb=15344` (~15.3 GB) exceeds a T4's total, and ~8.0 GB
of that is fp16 weights that `--batch-size` cannot reduce.
