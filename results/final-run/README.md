# Final Run Artifacts & Benchmarks

This directory contains the result files and evaluation artifacts from the final submission runs.

## Artifact Inventory

| File | Generating Command / Stage | Description & Key Metrics | Caveats & Notes |
|---|---|---|---|
| `metrics.md` | `src/build_metrics_report.py` | Official test-split retrieval evaluation metrics across NDCG, MRR, Recall, Precision, and MAP at k=1..1000. | Official submitted run: NDCG@10 0.9376, MRR@10 0.9238, R@10 0.9796. |
| `submission_checksums.json` | `src/verify_submission.py` | SHA-256 hashes and byte lengths for output files and runtime index files. | Validates integrity of official artifacts against the evaluated commit. |
| `failure_analysis.json` | `src/analyze_failures.py` | Distribution of the 77 failing queries out of 3,765 test queries (2.0% failure rate) that miss the top 10 (covers queries that miss top-10, not all non-rank-1 queries). | Group labels (near-duplicate corpus, generic wording, other) are heuristic categorizations, not proved causal factors. |
| `failure_examples.md` | `src/analyze_failures.py` | Qualitative walkthrough of specific pathological failure queries. | Case studies include base64-exec solution, image-only prompt, and magic-number hardcoded solution. |
| `real_history_benchmark.json` | `src/bench_real_history.py` | 40-commit history benchmark on `pallets/click` (0.6B lite model, T4 GPU). | 95.4% embeddings reused (7.38x speedup); duplicates collapsed from 89.8% to 0%; exact version retrieval (9.6%) is ill-posed when code is identical across commits. |
| `delta_k10.json` | `src/versioning/delta_index.py` | Experimental delta vector index evaluation. | Experiment, off by default. Delta index is 0.88 MB vs 16.9 MB (0.052x size). Part of exact version gain (9.6% -> 51.8%) comes from query labels using commit-introduced identifiers. |
| `history_ingest_renames.json` | `src/versioning/renames.py` | Per-commit ingest statistics with rename tracking on (`--track-renames`). | Experiment, off by default. Found 7 rename links across 1 commit; pairs are heuristic and not hand-verified. |
| `evolution_benchmark.json` | `src/bench_evolution.py` | Synthetic evolution benchmark on a SYNTHETIC fixture (300 snippets x 4 versions). | **Correctness check only**, not a performance claim. Duplicates collapsed from 64.9% to 0%. |
| `version_rebuild_benchmark.json` | `src/bench_versions.py` | Incremental rebuild benchmark on a SYNTHETIC fixture (300 snippets x 4 versions). | **Correctness check only**, not a performance claim. Saved 18.5% embeddings (1.27x speedup), matching fixture unchanged rate. |
| `agent_benchmark_first_run.json` | `src/bench_agent.py` | First evaluation run of agentic vs. dense retrieval on 27 repository questions. | **CAVEAT**: The `dense` baseline columns in this file are **INVALID** (scoring bug fixed, not rerun). Only the `agent` columns are valid (hit 23/27 overall; structural 10/10 hit, P@5 0.805; usage 5/5 hit, P@5 0.88; semantic 8/12 hit, P@5 0.28). Dense-vs-agent comparative metrics are not measured. |
| `int8_console_output.md` | `src/check_cpu_precision.py` | Console transcript of CPU int8 dynamic quantization test on 20 APPS queries. | Dynamic int8 was tested and **rejected** due to severe retrieval quality degradation (cosine 0.068 to fp32, top-10 overlap 0.01, rank-1 changed on 19 of 20 queries). The original JSON was not retained. |

## Caveats and Not-Measured Items

1. **Agent Benchmark Dense Baseline**:
   - In `agent_benchmark_first_run.json`, dense baseline scored 0.000 across all metrics due to relative file path mismatches between question labels and index chunk paths.
   - The bug is fixed in PR 1, but this first-run artifact is preserved unedited for auditability.
   - Comparative dense vs. agent performance is classified as **not measured**.

2. **CPU Dynamic int8 Quantization**:
   - Tested on CPU with full 1.7B model and 20 APPS queries.
   - Although latency dropped from 7,994 ms to 5,976 ms, quality was destroyed (cosine 0.068, top-10 overlap 1%).
   - int8 is rejected and fails closed; it is never enabled in API, CLI, or Docker serving.

3. **Synthetic Fixture Benchmarks**:
   - `evolution_benchmark.json` and `version_rebuild_benchmark.json` were run on generated synthetic code snippets. They verify indexing and collapsing correctness, not empirical retrieval superiority.

4. **150-Commit Large Repository Benchmark**:
   - A 150-commit history benchmark was attempted but terminated due to resource limits (killed by OOM, exit code -9). It is not measured and not included.
