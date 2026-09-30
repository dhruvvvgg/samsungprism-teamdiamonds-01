# Agentic Code Intelligence: code retrieval on CoIR-APPS

Samsung PRISM GenAI Hackathon, Theme 01.

A code retrieval system for the MTEB AppsRetrieval task (CoIR-Retrieval/apps): 3,765 test queries against the full 8,765-document corpus. It ships as a working tool, not just an evaluation script: a CPU-only CLI, HTTP API and web UI that serve a prebuilt index, plus code-structure search, a retrieval agent, and incremental re-indexing across a repository's git history.

## Headline result

NDCG@10 0.9376 · MRR@10 0.9238 on the official test split.

| Metric | Value |
|---|---|
| NDCG@10 | 0.9376 |
| MRR@10 | 0.9238 |
| Recall@10 | 0.97955 |
| Recall@100 | 0.99655 |
| Published reference for the same model | 0.93692 / 0.92288 |

System: F2LLM-v2-1.7B alone, with no reranker, no fusion and no query cap. The 3,765 queries and 8,765 documents ran on a Kaggle T4 in 948 s (476 s document encoding, 462 s query encoding).

Matching the published reference to within 0.0007 is the harness check. It shows the evaluation path, the prompt and the pooling are right, so differences measured between candidates are real differences between candidates.

A lite variant (F2LLM-v2-0.6B) scores 0.9044 / 0.8840 on the same test split and is ~2.5× faster on CPU. It is offered for interactive use; the submitted result is the 1.7B.

## Quick start (CPU)

Needs Python 3.11+ and a prebuilt index directory (`runtime_index/`, see Building an index).

```bash
git clone <repo-url>
cd <repo-dir>
pip install -r requirements-serve.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

CLI, one question:

```bash
python src/cli.py "count the number of primes below n" -k 5
```

CLI, interactive (the model stays warm). This is the mode to demo, since the model load is paid once per process, not per query.

```bash
python src/cli.py --interactive
> count the primes below n
> :k 5
> :quit
```

API and web UI:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
# then open http://localhost:8000/
curl localhost:8000/health
curl -s localhost:8000/search -H 'Content-Type: application/json' \
     -d '{"query": "binary search over a sorted array", "k": 5}'
```

The query box is a multiline textarea (newlines are preserved; Ctrl/Cmd+Enter searches) with an example selector holding four APPS-style problem statements and one code-style query. Implemented and unit-tested; not benchmarked.

On a versioned or git-history index, results are grouped by lineage: one entry per function with its best version on top, and the other versions that made the list expandable underneath. Each entry has a View history button that lists every version oldest first with its commit. Grouping is a view over the ranking, not a re-rank. Implemented and unit-tested; not benchmarked.

A compact performance panel under the results shows query-encode, search and total time, device, model, query cap, whether the query was truncated, and the index's document count and size. All figures come from the response; nothing is estimated. Implemented and unit-tested; not benchmarked.

The status line describes the selected index (`GET /health?index=NAME` reports on an index without loading it). Agent mode disables the version and category controls it ignores, and the page has explicit loading, empty, unavailable-index and error states. Scores are labelled similarity score (cosine similarity, useful only for ordering) and are never shown as a percentage or a confidence.

Each result has an expandable Why this result line: the query's route and reason, the similarity gap to the next result, and the matched terms (query words that also appear among the snippet's identifiers, read from its AST; plural and snake/camel-case parts ignored). It costs no model call and is lexical only. It does not explain why the embedding model scored a snippet as it did, and the UI says so. Opt in with `"explain": true` on `POST /search`. Implemented and unit-tested; not benchmarked.

Exact-query embedding cache (serving only). An LRU keyed by (query text, model, revision, query prompt, query token cap, dtype) lets a repeated identical query skip the encoder; timings report `cached: true/false`. It is on in the API/UI and the CLI's `--interactive` mode, and off with `QUERY_CACHE=0` (`QUERY_CACHE_SIZE` sets the size). It is absent by construction from `run_official` and every benchmark: `SearchService` defaults to no cache, and a test fails if any other script opts in. There is no fuzzy matching, and no latency figure is claimed for it. Implemented and unit-tested; not benchmarked.

Clicking a result opens the full snippet (`GET /doc?id=...`) with line numbers, a copy button and expand/collapse for long snippets. APPS documents show their document ID; a `file:start-end` (and real file line numbers) appears only for indexes that carry source-file metadata. Implemented and unit-tested; not benchmarked.

The UI is a single static page with no build step and no external dependency of any kind. The Python syntax highlighter is inline rather than a CDN library, so the page works fully offline. A test asserts the served HTML contains no external URL.

## Docker (CPU-only)

```bash
docker build -t apps-retrieval .
docker run --rm -p 8000:8000 -v "$PWD/runtime_index:/app/runtime_index:ro" apps-retrieval
# open http://localhost:8000/
```

### One command: docker compose up

```bash
docker compose up --build        # then open http://localhost:8000/
```

`docker-compose.yml` builds the same CPU image, mounts `./runtime_index` read-only, serves the API and the web UI on port 8000, and caches the downloaded model in a named volume so a restart does not fetch it again. Choose another index or port with environment variables:

```bash
INDEX_HOST_DIR=./runtime_index_lite docker compose up --build   # the 0.6B lite index, ~4.5 GB RAM
PORT=9000 SEARCH_THREADS=2 docker compose up --build
```

If the mounted directory has no index, the container still starts and `GET /health` reports `degraded` with the reason. The first search downloads the model (network access required, once); everything after that is offline. Implemented; not run in this environment. Tests that read the compose file and a CI build of the same image check it, but `docker compose up` itself has not been executed.

One-off CLI query instead of the server:

```bash
docker run --rm -v "$PWD/runtime_index:/app/runtime_index:ro" apps-retrieval \
    python src/cli.py "count the primes below n" -k 5
```

The index is mounted, not baked in. It is model output and changes independently of the code, and the image carries no CUDA libraries, no mteb and no datasets. Environment variables: `INDEX_DIR`, `SEARCH_DEVICE`, `SEARCH_THREADS`, `CPU_DTYPE`, `MAX_QUERY_TOKENS`, `ALLOWED_INDEXES`.

`GET /health` reports what is loaded. The model loads at startup, so no user request pays for it. A load failure is recorded rather than raised, so the container reports `degraded` with the reason instead of dying with a traceback nobody sees.

## Reproducing the official result

Needs a GPU (a T4 is enough) and about 20 minutes.

```bash
pip install -r requirements.txt
python src/eval/run_official.py --config configs/official_f2llm17b_noreranker.json \
    --device cuda --confirm-test
```

One run produces everything, from the single encoding pass it already paid for:

| Output | What it is |
|---|---|
| `outputs/appsretrieval_results.json` | MTEB's own result dict (`task_result.to_dict()`) plus a `_run_info` block |
| `outputs/appsretrieval_rankings.json` | ordered top-100 document ids per test query |
| `outputs/submission_checksums.json` | SHA-256 of the files above and of every index file |
| `runtime_index/` | the served index: fp16 corpus embeddings, ids, texts, manifest |

The run is guarded. `--confirm-test` is required, a completed run writes a lock so the test split cannot be silently re-consumed, every write is atomic, and the lock is written last so a killed run leaves no state that looks finished. Each config states the score it expects, and the run warns if the gap exceeds ±0.004.

## Building an index

```bash
python src/build_index.py --preset f2llm-v2-1.7b --device cuda             # 572 s, 35.9 MB
python src/build_index.py --preset f2llm-v2-0.6b --device cuda --out lite  # 293 s, 18.0 MB
```

This reads the corpus through the dev loader (train qrels only), so the test split is untouched. An official run exports the same index for free.

## Resources

| | Value |
|---|---|
| Model | codefuse-ai/F2LLM-v2-1.7B, revision 3766d46e |
| Parameters | 1.7 B (~3.4 GB in fp16) |
| Full test-split run, T4 | 948 s (476 s docs + 462 s queries) |
| Index build, T4 | 572 s → 35.9 MB (lite: 293 s → 18.0 MB) |

CPU serving, measured on 2 physical cores, fp32, uncapped:

| | 1.7B (full) | 0.6B (lite) |
|---|---|---|
| Official test NDCG@10 / MRR@10 | 0.9376 / 0.9238 | 0.9044 / 0.8840 |
| p50 latency | ~7.4 s | ~2.9 s |
| p95 latency | ~19 s | ~8.4 s |
| 1 thread, p50 | 13.6 s | 5.3 s |
| Peak RSS | 11.0 GB | 4.5 GB |
| Model load (once per process) | ~15 s | not yet measured |
| Vector search itself | 4–6 ms | 4–6 ms |
| Warm short query (~22 tokens) | ~1.05 s | not yet measured |

Nearly all of the time is query encoding; retrieval is 4–6 ms. APPS queries are whole problem statements and attention cost grows faster than linearly in sequence length, so latency scales with query length, not with corpus size. Two cores against one is roughly 1.8×, so more cores is the one lever with no quality cost. The warm short-query figure is the one that matters for a live demo.

Serving query cap: 1024 tokens for the CLI, API and UI. It is the largest cap tested that was not significantly worse on dev (−0.0010 NDCG@10, 2.6% of queries truncated), and it bounds a very long tail. The official run is uncapped, and a test asserts it.

CPU precision. fp32 is the default. bf16 is kept as a flag but is 4× slower on the tested CPU, which has no native bf16, and the loader warns when that is the case. int8 dynamic quantisation first had a bug: identical output, unchanged speed and +5.4 GB RAM, which is what building a second model while the original keeps encoding looks like. The fix quantises in place, with a census that raises if any `nn.Linear` survived, so a quantisation that does nothing now fails. With the fix, int8 ran about 1.3× faster on the full model (p50 5.8 s against 7.5 s for fp32) but wrecked the embeddings. On 20 queries the cosine similarity to the fp32 embeddings was 0.068 and the top-10 overlap was 0.01, and the top result changed on 19 of 20. int8 is therefore rejected. It fails closed unless an explicit lossy-int8 flag is passed, and it is never recommended.

## P1: retrieval across versions

CoIR-APPS is a flat corpus with no history, so version-aware retrieval is built and tested two ways.

Every version of a snippet is keyed by the SHA-256 of its normalised code (comments, blank lines, trailing whitespace and line endings removed; indentation, identifiers, literals and operators kept). A rebuild re-embeds only the hashes it has not seen. This is exact reuse, not an approximation, since an identical hash means identical normalised text.

```bash
python src/build_version_index.py --device cuda   # synthetic fixture, ~500 snippets x 4 versions
python src/bench_versions.py --device cuda        # full vs incremental rebuild, per version

python src/build_history_index.py --device cuda --max-commits 40   # a real repo, commit by commit
python src/bench_real_history.py --device cuda --max-commits 40
```

The synthetic fixture has mutations known by construction (unchanged, rename, logic, add_lines, remove_lines), so its reuse rate has an exact right answer to check against. The benchmark fails if measured reuse falls below the fixture's own count of byte-identical versions.

The real-history path ingests a Python git repository commit by commit. The default is pallets/click, chosen because it is pure Python so every file parses with `ast`, its functions genuinely evolve, it is small enough to index a few hundred commits on a T4, and it is BSD-3-Clause licensed. A lineage is `file::qualname`, not a line range. Line numbers move on every edit above a function, so a line-based identity would report an unrelated edit as a full rebuild.

Version-targeted queries are generated with answers that do not come from the retriever: the text is the function's own docstring, and the discriminating token is an identifier the commit introduced, found by diffing identifier sets.

Status: not yet measured. The synthetic fixture has only been run with the mock hashing encoder (a plumbing check, not a result), and the real-history benchmark has not been run on a GPU.

## Rename and move tracking

Implemented and unit-tested; not benchmarked. `python src/build_history_index.py --track-renames` (off by default; without it the output is unchanged) keeps a lineage across a renamed file or a renamed or moved function. Evidence is read within a single commit: git's own rename detection for files, and for functions the Jaccard similarity (default >= 0.8, `--rename-threshold`) of their normalised tokens, with the function's own name masked. The row after a link records `link_kind` (rename or move), `link_similarity` and `link_from`, and the ingest stats report how many links were made. Bodies under 8 tokens are never matched, and a copy that leaves the original in place is not a move. The threshold was chosen by reasoning, not tuned on data.

## Version-delta vectors (experiment)

Implemented and unit-tested; not benchmarked; off by default and not used by any served path. `src/versioning/delta_index.py` stores one base vector per lineage plus an int8 residual per distinct version, and searches in two stages (lineages by base vector, then their versions by reconstructed full vector). `python src/bench_real_history.py --delta-vectors` reports its index size and top-1 version accuracy against the current per-row vectors on the same queries. No size or accuracy figure is claimed. The storage saving depends on how many versions are unchanged, and the cost (int8 noise, and a lineage prefilter that can drop a lineage) has not been measured.

## Incremental update of a served index

Implemented and unit-tested; not benchmarked. A custom-folder or git-history index remembers its source, so it can be brought up to date without a rebuild:

```bash
python src/reindex.py --index my_code_index             # CLI
python src/reindex.py --index my_code_index --dry-run   # what would change; writes nothing
curl -s localhost:8000/reindex -H 'Content-Type: application/json' -d '{"index": "my_code_index"}'
```

The UI has an Update index button, shown only for indexes that have a source. The operation re-scans the source recorded in the index manifest and reports chunks added, modified, unchanged and removed, embeddings reused against recomputed, and the elapsed time. It encodes only chunks whose text has no stored vector; a test with a spy encoder proves unchanged chunks are never re-encoded. The next search sees the edit. The official APPS indexes are refused, the scan path comes from the manifest and never from a request, and API callers can only name indexes on an allowlist (`ALLOWED_INDEXES`, the named indexes and the server default). No timing is claimed, since it has not been measured with a real encoder.

## Bonus: evolutionary retrieval

The versioned index holds every version at once, and which versions compete is decided per query:

```bash
python src/cli.py "sum the scores" --index versions                 # collapsed (default)
python src/cli.py "sum the scores" --index versions --all-versions  # every version competes
python src/cli.py "sum the scores" --index versions --version 2     # only version 2
python src/cli.py --history "src/click/core.py::Command.invoke" --index history
python src/bench_evolution.py --device cpu
```

Compare two versions. Implemented and unit-tested; not benchmarked. `GET /compare?q=...&a=1&b=3` runs one query against two versions (encoding it once) and marks each result of B as appeared, moved (with the rank change) or same, and each result of A as disappeared or kept. "Appeared" means in B's top-k but not A's, not that it did not exist in A. `GET /diff?snippet_id=...&a=1&b=3` returns the unified diff (difflib) of one lineage between two versions. The UI has a Compare versions panel with side-by-side results and a per-result diff.

By default each lineage collapses to its best-scoring version, so near-identical versions of one snippet stop sharing the top 10 between them and crowding other snippets out.

Status: not yet measured on a real encoder, for the same reason as P1.

## Features, and how far each has been taken

"Benchmarked" means measured with a real model against a held-out set. "Implemented and unit-tested" means it works and is covered by tests, but its quality has not been measured at scale.

| Feature | Status |
|---|---|
| Dense retrieval on CoIR-APPS (the submitted system) | Benchmarked — official test split, 0.9376 / 0.9238 |
| Lite 0.6B variant | Benchmarked — official test split, 0.9044 / 0.8840 |
| CPU serving: CLI, interactive mode, API, web UI | Benchmarked for latency and memory; functionally unit-tested |
| Serving query cap (1024 tokens) | Benchmarked on the dev slice (−0.0010 NDCG@10) |
| Custom-folder indexing (any Python folder, function/class chunks with file:line) | Implemented and unit-tested; not benchmarked at scale |
| Query router (problem statement / intent / code / structural) | Implemented and unit-tested; not benchmarked at scale |
| Structural index, cross-file call graph, usage search | Implemented and unit-tested; not benchmarked at scale |
| Retrieval agent (plan → search → read → refine, 6-step cap, full trace) | Implemented and unit-tested; not benchmarked at scale |
| Snippet categories (AST family + labelled embedding clusters) | Implemented and unit-tested; not benchmarked at scale |
| Optimisation notes on surfaced code (an extra) | Implemented and unit-tested; not benchmarked at scale |
| Incremental "Update index" for folder and history indexes (`src/reindex.py`, `POST /reindex`, UI button) | Implemented and unit-tested; not benchmarked |
| P1 incremental re-indexing (content hash) | Implemented and unit-tested; not yet measured on a real encoder |
| Rename / move tracking in the git-history ingester (`--track-renames`, off by default) | Implemented and unit-tested; not benchmarked |
| Version-delta vectors: base vector per lineage + int8 residual per version (experiment, off by default) | Implemented and unit-tested; not benchmarked |
| Version comparison and lineage diff (`GET /compare`, `GET /diff`, UI panel) | Implemented and unit-tested; not benchmarked |
| Bonus evolutionary retrieval (lineage collapsing) | Implemented and unit-tested; not yet measured on a real encoder |
| int8 CPU quantisation | Measured and rejected: quality collapses (cosine 0.068 to fp32 on 20 queries); fails closed unless an explicit lossy flag is passed |

The agent runs with no LLM at all. The planner is deterministic, and the optional LLM planner goes through one provider-agnostic wrapper whose default is a keyless mock, so a missing API key can't be the reason a demo fails.

## Limitations

- Latency is dominated by query encoding, not retrieval. On two CPU cores a full APPS problem statement takes ~7.4 s with the 1.7B. A short question with a warm model is ~1.05 s, and the lite model is ~2.5× faster throughout. This comes from running a 1.7B encoder on a CPU.
- Memory: the 1.7B needs ~11 GB RSS on CPU. A 12 GB machine should serve the lite index instead.
- The reranker and dense-dense fusion are implemented and measured but excluded from the submitted path for resource reasons (see the table below).
- P1, Bonus and the four dev experiments are not yet measured with a real encoder. They are implemented, unit-tested and off by default, and no number from them is claimed anywhere in this README.
- F2LLM-v2-4B does not fit a T4 (registry footprint 15.3 GB; out of memory even at batch size 1), so the accuracy available above 1.7B was not reachable on the hardware at hand.
- A function that moves between files starts a new lineage in the version index by default. `--track-renames` joins renamed files and renamed or moved functions (implemented and unit-tested; not benchmarked; see above).
- Call-order queries ("which files call X before Y") are lexical, not an execution trace. Anything stronger would need control-flow analysis.

## Experiments, and what was rejected

Every candidate was judged on a dev protocol built on the train split: a 4,000-query tune partition and a reserved 1,000-query holdout, against the same full 8,765-document corpus. No candidate below was judged on the test split. Adoption required all three of: NDCG@10 gain ≥ 0.005, a 95% paired-bootstrap CI excluding zero, and worsened queries ≤ half of improved.

| Direction | Dev-slice NDCG@10 | Outcome |
|---|---|---|
| F2LLM-v2-0.6B alone | 0.8959 | baseline |
| F2LLM-v2-1.7B alone | 0.9299 | submitted |
| 1.7B + 0.6B dense fusion | 0.9353 | passed adoption; excluded — doubles per-query encoding cost |
| 1.7B + Qwen3-Reranker-0.6B | 0.9371 | passed adoption; excluded — hours of GPU per run, plus per-query cross-encoder latency |
| Fusion + reranker stacked | 0.9404 | rejected — the 95% CI includes zero |
| BM25 hybrid | −0.004 to −0.21 | rejected — hurt at every setting |
| Query rewrites / instruction variants | below baseline | rejected — worst was −0.127 |
| BGE-reranker-v2-m3 | below baseline | rejected — hurt at every setting |
| Ettin rerankers (68M, 150M) | +0.0011 | rejected — not significant |
| LLM re-judge stage | — | rejected — recall@100 is 0.991, so the trigger almost never fires |
| F2LLM-v2-4B | — | rejected — out of memory on a T4 |

The submitted system is a single model with no second stage. Fusion and reranking each passed adoption on dev but cost too much to run, and stacking them gave no significant gain.

Four further experiments are implemented, off by default and not yet evaluated: confidence-gated reranking (A), code-to-description fusion (F), LoRA fine-tuning of the lite model (B) and a category tiebreaker (E). Each has a dev script and would go through the same adoption rule. No result is claimed for any of them.

Experiment B's evaluation stage is paired. The base 0.6B and the tuned model are scored on the same 1,000 holdout queries in one run, per-query NDCG@10 / MRR@10 are saved, and the verdict comes from a paired bootstrap under the adoption rule above (optionally also against the 1.7B, whose per-query ranks come from `--stage eval --base-only`). Implemented and unit-tested with fake encoders; not run. The holdout is read only by the eval stage.

The full write-up of the search is in `results/P0_summary.md`.

## Verifying this submission

```bash
# 1. recompute the headline metrics from the stored ranking, with an independent implementation
python src/verify_submission.py --confirm-test

# 2. see what the headline number hides: which queries miss the top 10, and why
python src/analyze_failures.py --confirm-test

# 3. collect every result file into one report
python src/build_metrics_report.py        # -> results/metrics.md

# 4. the test suite (fully mocked: no model, no dataset, no network)
pytest tests -q
flake8 src tests --max-line-length=110 \
    --extend-ignore=E501,W503,E226,E741,E402,E305,E306,E127,E128
```

`verify_submission.py` recomputes NDCG@10 and MRR@10 from the exported rankings with its own metric implementation and its own read of the qrels, then re-hashes every artefact against the recorded SHA-256. It loads no model. A disagreement points at exactly one of three things: a wrong ranking export, a metric misunderstanding, or a file that changed after the run.

`build_metrics_report.py` tolerates missing inputs. Each one becomes a "not yet measured" row naming the command that would fill it.

The test suite runs with no model download, no dataset and no network access.

## Repository layout

```
src/retrieval      dense encoder, BM25, RRF + score-average fusion, rerankers, pipeline, query router
src/indexing       code chunking, structural index + call graph, categories, optimisation notes
src/versioning     content hashing, version fixture, git-history ingestion, versioned index
src/agent          provider-agnostic LLM wrapper, the optional re-judge stage, the retrieval agent
src/eval           dev protocol (dev_*.py), the official run, metrics, data loading
src/train          optional LoRA fine-tuning of the lite model
src/static         the single-page web UI
src/cli.py  src/api.py  src/build_index.py  src/search_service.py  src/runtime_index.py
src/verify_submission.py  src/analyze_failures.py  src/build_metrics_report.py
configs            version-controlled official run configurations
results            the P0 write-up, the dev runbook, benchmark outputs
tests              mocked unit tests and end-to-end smoke tests
```

Dependencies are split so a package nobody uses cannot break an install that does not need it:

| File | For |
|---|---|
| `requirements.txt` | evaluation, the dev protocol, the tests, CI |
| `requirements-serve.txt` | CPU serving and the Docker image: no mteb, no datasets, no CUDA |
| `requirements-experiments.txt` | optional: LoRA fine-tuning and local description generation (peft, accelerate) |
| `requirements-llm.txt` | optional: provider SDKs for the disabled LLM re-judge stage |

Every import of an optional package is function-local, so the code loads and the whole test suite passes without any of them installed.
