# Consolidated test plan

One pass over everything that needs a real model, cheapest first. Written to be run **after every
feature branch is merged into `develop`** — every cell clones `develop`, never `main`.

**Scope.** This plan covers everything in `develop`: the CI fix and CPU serving, custom-folder indexing
and the web UI, real-history P1/Bonus, code intelligence (router, structural search, categories, agent),
and the three dev experiments (A confidence-gated reranking, F code-to-description fusion, B fine-tuned
lite model). Nothing is left unbuilt.

Every step is independent: a failure in one does not block the others. Run what you have time for.

| # | Step | Where | Time |
|---|---|---|---|
| 0 | Setup: clone + install | any | 4 min |
| 1 | Index load check | Kaggle CPU | 1 min |
| 2 | int8 re-test | Kaggle CPU | 15 min |
| 3 | Capped vs uncapped latency + thread sweep | Kaggle CPU | 20 min |
| 4 | Custom-folder index on CPU | Kaggle CPU | 5 min |
| 8 | Router, structural search, agent trace | Kaggle CPU | 10 min |
| 5 | Real-history P1/Bonus | Kaggle GPU (T4) | 25 min |
| 8b | Agent vs dense on 30 labelled questions | Kaggle GPU (T4) | 15 min |
| 8c | Category tiebreaker (experiment E) | Kaggle GPU (T4) | 20 min |
| 7a | A. Confidence-gated reranking | Kaggle GPU (T4) | 70 min |
| 7b | F. Code-to-description fusion | Kaggle GPU (T4) | 60–130 min |
| 7c | B. Fine-tuned lite model | Kaggle GPU (T4) | 100–140 min |
| 6 | Colab web UI | Colab | 15 min |
| 9 | Final submission run | Kaggle GPU + CPU | 45 min |
| 10 | Demo-polish features with a real encoder (reindex, compare, cache) | Kaggle CPU | 20 min |

**Budget note.** Steps 7a–7c are the expensive ones and they are independent of each other, so they fit
three separate Kaggle GPU sessions (each ≤ 12 h) without any of them blocking the rest of the plan. If
time is short, run **7a first** — it is the cheapest and the most likely to change the submitted system.

---

## 00. The one-notebook run (supersedes the per-step cells below)

For a single end-to-end pass in a **brand-new** Kaggle notebook (GPU T4, Internet on, secret `GITHUB_TOKEN`,
no input datasets) use `scripts/run_all_checks.py`. It builds every index itself, runs each phase in its
own time-boxed try/except with a log, keeps `results/run_state.json` so a re-run skips finished phases,
re-zips `results/` to `/kaggle/working/results_partial.zip` after every phase, and writes
`results/FINAL_SUMMARY.md` (PASS/FAIL/SKIPPED plus the exact lines to paste back). The notebook is five cells,
stored in `scripts/kaggle_cells/`: 1 setup, 2 Tier 1 + 2, 3 optional Tier 3, 4 collect (+ optional push of
small results to `results/final-run`), 5 diagnostics. Tier 1 is the official run and its verification,
both indexes, failure analysis and the CPU checks; Tier 2 is the API smoke test over HTTP and the real
history; the sections below remain the reference for what each phase runs and what a correct result is.

---

## 0. Setup (every session starts with this)

Works on Kaggle and Colab, private repo, token masked, deletes any previous checkout.

```python
import os, shutil, subprocess, sys
from pathlib import Path

REPO_HOST = "github.com/dhruvvvgg/samsungprism-teamdiamonds-01.git"
BRANCH    = "main"
ROOT      = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/content")
REPO      = ROOT / "prism-test"

try:                                       # Kaggle
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:                          # Colab
    from google.colab import userdata
    TOKEN = userdata.get("GITHUB_TOKEN")

def run(cmd, cwd=None, env=None, check=True, quiet=False):
    """Run a command with list args. The token never reaches stdout."""
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    out = (p.stdout + p.stderr).replace(TOKEN, "***") if TOKEN else (p.stdout + p.stderr)
    if not quiet:
        print(out[-4000:])
    if check and p.returncode != 0:
        raise SystemExit(f"step failed with exit {p.returncode}")
    return p.returncode

if REPO.exists():
    shutil.rmtree(REPO)                    # a stale checkout is the bug that cost us a Kaggle run
clone_url = f"https://x-access-token:{TOKEN}@{REPO_HOST}" if TOKEN else f"https://{REPO_HOST}"
run(["git", "clone", "--branch", BRANCH, "--single-branch", clone_url, str(REPO)])
run(["git", "-C", str(REPO), "remote", "set-url", "origin", f"https://{REPO_HOST}"])
run(["git", "-C", str(REPO), "log", "-1", "--oneline"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")])
ENV = dict(os.environ, PYTHONPATH=str(REPO))
print("ready:", REPO)
```

**Correct result:** one commit line from `develop`, then a quiet pip install.
**Paste back:** the commit line.

### Locating your indexes (steps 1–4, 6)

Handles `indexes.zip` **or** already-extracted folders under `/kaggle/input`.

```python
import glob, shutil
from pathlib import Path

def install_indexes(repo):
    """Put runtime_index/ and runtime_index_lite/ at the repo root, however they were attached."""
    placed = []
    for name in ("runtime_index", "runtime_index_lite"):
        target = Path(repo) / name
        if (target / "manifest.json").exists():
            placed.append(f"{name}: already present"); continue
        hits = glob.glob(f"/kaggle/input/**/{name}/manifest.json", recursive=True)
        if hits:
            shutil.copytree(Path(hits[0]).parent, target, dirs_exist_ok=True)
            placed.append(f"{name}: copied from {Path(hits[0]).parent}"); continue
        zips = glob.glob("/kaggle/input/**/indexes.zip", recursive=True)
        if zips:
            subprocess.run(["unzip", "-q", "-o", zips[0], "-d", str(repo)], check=True)
            placed.append(f"{name}: extracted from {zips[0]}" if (target / "manifest.json").exists()
                          else f"{name}: NOT in {zips[0]}")
            continue
        placed.append(f"{name}: NOT FOUND")
    return placed

for line in install_indexes(REPO):
    print(line)
```

**If nothing is found**, rebuild on a **GPU** session (T4, ~15 min total) and save the notebook version:

```python
run([sys.executable, "src/build_index.py", "--preset", "f2llm-v2-1.7b",
     "--device", "cuda", "--out", "full"], cwd=REPO, env=ENV)     # ~572 s -> 35.9 MB
run([sys.executable, "src/build_index.py", "--preset", "f2llm-v2-0.6b",
     "--device", "cuda", "--out", "lite"], cwd=REPO, env=ENV)     # ~293 s -> 18.0 MB
run(["zip", "-qr", "/kaggle/working/indexes.zip", "runtime_index", "runtime_index_lite"], cwd=REPO)
```

These read the corpus through the dev loader (train qrels only), so the test split stays untouched.

---

## 1. Index load check — Kaggle CPU, ~1 min

Loads no model. Your indexes were built at `12ef00e`, before the index format gained `chunks.json`;
that sidecar is optional, so they load unchanged — this confirms it on your actual files.

```python
CHECK = r'''
from src.runtime_index import INDEX_NAMES, RuntimeIndex
for name in ("full", "lite"):
    try:
        ix = RuntimeIndex.load(INDEX_NAMES[name])
    except Exception as e:
        print(f"{name:5s} FAIL: {type(e).__name__}: {str(e).splitlines()[0]}"); continue
    bad = ix.verify_files()
    m = ix.manifest
    print(f"{name:5s} rows={m['n_docs']:6d} dim={m['dim']:5d} kind={m.get('kind')} "
          f"model={m.get('model')} chunks={'yes' if ix.chunks else 'no'} "
          f"hashes={'OK' if not bad else bad}")
'''
run([sys.executable, "-c", CHECK], cwd=REPO, env=ENV)
```

**Correct result:**

```
full  rows=  8765 dim= 2048 kind=flat model=codefuse-ai/F2LLM-v2-1.7B chunks=no hashes=OK
lite  rows=  8765 dim= 1024 kind=flat model=codefuse-ai/F2LLM-v2-0.6B chunks=no hashes=OK
```

`chunks=no` is right for APPS. A hash mismatch means a corrupt zip — rebuild (setup section).
**Paste back:** both lines.

## 2. int8 re-test — Kaggle CPU, ~15 min

The first run also downloads the APPS dataset for real query texts (~2 min, one-off).

```python
for idx in ("full", "lite"):
    print(f"\n===== {idx} =====")
    run([sys.executable, "src/check_cpu_precision.py", "--index", idx,
         "--modes", "fp32", "int8", "--n-queries", "20"], cwd=REPO, env=ENV, check=False)
```

**Correct result — either of these is a pass:**

- **Quantisation applied:** `int8 census: Linear layers fp32 N -> 0`, then **cosine below 1.000000**
  with some top-10 movement, and int8 peak RSS **at or below** fp32.
- **Or a loud failure:** `RuntimeError: int8 dynamic quantisation did not take: K of N Linear layers
  are still fp32 (e.g. [...])`. Naming the layers is a correct outcome.

**The bug is still there** only if you see all three of: cosine exactly `1.000000`, 20/20 identical
top-10 lists, **and** int8 RSS well above fp32.
**Paste back:** the `int8 census:` line and the `CPU PRECISION vs fp32` table for each index.

## 3. Latency, capped vs uncapped, with thread sweep — Kaggle CPU, ~20 min

```python
for idx in ("full", "lite"):
    run([sys.executable, "src/bench_cpu.py", "--index", idx, "--thread-sweep",
         "--n-queries", "20", "--out", f"results/cpu_{idx}_capped.json"],
        cwd=REPO, env=ENV, check=False)
    run([sys.executable, "src/bench_cpu.py", "--index", idx, "--thread-sweep",
         "--n-queries", "20", "--max-query-tokens", "0",
         "--out", f"results/cpu_{idx}_uncapped.json"], cwd=REPO, env=ENV, check=False)
```

Run both arms: the earlier 7,291 ms p50 was uncapped, on a different query sample, over 50 queries
rather than 20 — so only the capped/uncapped pair **from this same session** is a fair comparison.

**Correct result:** capped p50 ≤ uncapped p50 with a small gap (the cap bites on ~2.6% of queries);
the tokens line shows `(capped at 1024)` in the capped arm only; 2 threads ≈ 1.8–1.9× faster than 1;
`vector search` stays ~5 ms in every run.
**Paste back:** the `CPU RESULTS` and `thread scaling` blocks from all four runs, or the four JSON files.

## 4. Custom-folder index on CPU — Kaggle CPU, ~5 min

```python
run([sys.executable, "src/build_index.py", "--source", "examples/textkit",
     "--preset", "f2llm-v2-0.6b", "--device", "cpu", "--out", "examples/_demo_index"],
    cwd=REPO, env=ENV, check=False)
run([sys.executable, "examples/demo_queries.py", "--index", "examples/_demo_index"],
    cwd=REPO, env=ENV, check=False)
run([sys.executable, "src/cli.py", "remove accents from text", "-k", "3",
     "--index", "examples/_demo_index"], cwd=REPO, env=ENV, check=False)
```

**Correct result:** 19 chunks indexed; every hit prints `file.py:start-end` with the function name;
`remove accents from text` should surface `tokenizing.py` → `strip_accents` at or near rank 1 (this is
the real model, so semantics should work — unlike the mock-encoder smoke runs).
**Paste back:** the demo output block and the three CLI hits.

## 5. Real-history P1 / Bonus — Kaggle **GPU (T4)**, ~25 min

Needs network for the first clone of the demo repo.

```python
run([sys.executable, "src/build_history_index.py", "--device", "cuda",
     "--max-commits", "40"], cwd=REPO, env=ENV, check=False)          # ~12 min
run([sys.executable, "src/bench_real_history.py", "--device", "cuda",
     "--max-commits", "40", "--index", "history"], cwd=REPO, env=ENV, check=False)   # ~10 min
run([sys.executable, "src/cli.py", "--history", "src/click/core.py::Command.invoke",
     "--index", "history", "--device", "cuda"], cwd=REPO, env=ENV, check=False)
```

Start at `--max-commits 40`. Raise it to 150–300 once the shape looks right; ingest time grows roughly
linearly, embedding time much less (that is the result being measured).

**Correct result:** the ingest line reports far fewer distinct contents than rows; the per-commit table
shows incremental recomputing only what changed; **`warnings` is empty** (a warning means measured reuse
fell below the ingester's own unchanged count, i.e. the content hash is missing identical text); the
Bonus block shows collapsed duplicate slots at **0.0%** and all-versions well above it. `--history` lists
that function's versions oldest first with their commits.
**Paste back:** `results/real_history_benchmark.json`, plus the `REAL HISTORY` summary block.

### 5b. Version-delta vectors (experiment) — same session, ~5 min extra

Off by default and not part of any served path. It reuses the `history` index built above and asks one
question: does storing one base vector per lineage plus an int8 residual per distinct version cost
retrieval accuracy, and how much smaller is it?

The benchmark repeats its cheap rebuild section; the delta comparison is the new part.

```python
run([sys.executable, "src/bench_real_history.py", "--device", "cuda", "--max-commits", "40",
     "--index", "history", "--delta-vectors", "--out", "results/delta_default.json"],
    cwd=REPO, env=ENV, check=False)
run([sys.executable, "src/bench_real_history.py", "--device", "cuda", "--max-commits", "40",
     "--index", "history", "--delta-vectors", "--delta-lineage-k", "10",
     "--out", "results/delta_k10.json"], cwd=REPO, env=ENV, check=False)
```

**Correct result:** a `Delta vectors (experiment)` line giving the index size against the current
per-row vectors (`x` ratio, expected well under 1 when most versions are unchanged) and the top-1 exact
version accuracy for `delta` against `current`, plus how often the two agree. Adopt nothing unless
`delta` matches `current` on lineage and exact-version accuracy to within noise; a large drop with the
default first stage, recovered by `--delta-lineage-k 10`, means the lineage prefilter is the cost.
**Paste back:** the `Delta vectors` line from both runs, or `results/delta_default.json` and `results/delta_k10.json`.

### 5c. Rename and move tracking — same session, ~2 min (no model needed for the ingest)

```python
run([sys.executable, "src/build_history_index.py", "--device", "cuda", "--max-commits", "150",
     "--track-renames", "--out", "history_renames", "--queries-out", "results/history_queries_renames.json",
     "--stats-out", "results/history_ingest_renames.json"], cwd=REPO, env=ENV, check=False)
```

**Correct result:** a `rename tracking: N link(s) {'rename': a, 'move': b}` line, and `lineages` lower than in
the same run without `--track-renames`. Then read a few links in `results/history_ingest_renames.json` /
the index's `versions.json` (`link_kind`, `link_similarity`, `link_from`) and check by hand that each is a
real rename or move in the click history (`git log -M --follow` on the file is a good cross-check). The
0.8 threshold was chosen by reasoning, not tuned: report how many links look wrong before trusting it.
**Paste back:** the `rename tracking` line, the two `lineages` counts, and any link you judge wrong.

## 6. Colab web UI — ~15 min

Use the **lite** index: the 1.7B is ~11 GB resident and a free Colab runtime has ~12.7 GB. Do not extract
`runtime_index/` here; if it is present it appears in the dropdown and selecting it will OOM the runtime.

```python
# after step 0 (Colab branch of the token code), upload indexes.zip, then:
from google.colab import files, output
import time, json, urllib.request
up = files.upload()                                   # pick indexes.zip
run(["unzip", "-q", "-o", next(iter(up)), "runtime_index_lite/*", "-d", str(REPO)])

run([sys.executable, "src/build_index.py", "--source", "examples/textkit",
     "--preset", "f2llm-v2-0.6b", "--device", "cpu", "--out", "examples/_demo_index"],
    cwd=REPO, env=ENV)                                # CPU is fine: 19 chunks, no GPU needed

srv = dict(ENV, INDEX_DIR="examples/_demo_index",
           ALLOWED_INDEXES="examples/_demo_index,lite", SEARCH_DEVICE="cpu")
log = open("/content/api.log", "w")
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app",
                         "--host", "0.0.0.0", "--port", "8000"],
                        cwd=str(REPO), env=srv, stdout=log, stderr=subprocess.STDOUT)
for i in range(120):                                  # the model loads during startup
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as r:
            h = json.loads(r.read())
            if h.get("loaded"):
                print("ready:", h["index"]["model"], h["index"]["n_docs"], "docs"); break
    except Exception:
        if proc.poll() is not None:
            print(open("/content/api.log").read()[-3000:]); raise SystemExit("server exited")
    time.sleep(2)
output.serve_kernel_port_as_window(8000)
```

To see **versions** in the UI instead, build the history index on a GPU Colab runtime and start the
server with `INDEX_DIR=history`.

While the UI is up, also check the code-intelligence surfaces: the **agent** checkbox turns a query into
a step trace, the **category** dropdown filters by algorithm family (after `build_categories.py`), and
**extras** shows rule-based performance notes under a result. A structural question typed into the box
(`who calls tokenize`) shows its route in the banner above the results.

**Correct result:** the header shows the model, doc count and query cap; a code query returns
syntax-highlighted hits titled `file.py:start-end`; switching to `lite` and asking an APPS-style question
returns numeric document ids. On a history index the hits also show the commit, and **History…** lists a
lineage's versions.
**Paste back:** the `ready:` line and one screenshot.

**Note:** the index dropdown is populated from `GET /indexes`, which lists the named indexes that exist
plus the server's own `INDEX_DIR`. There is no free-text field, so a custom path is only selectable when
it is the server default — which is why the command above sets it that way.

### 6b. `docker compose up` — a local machine with Docker (not Kaggle)

```bash
git clone https://github.com/dhruvvvgg/samsungprism-teamdiamonds-01.git && cd samsungprism-teamdiamonds-01
# put runtime_index/ (or runtime_index_lite/) in the repo root first
INDEX_HOST_DIR=./runtime_index_lite docker compose up --build
# in another terminal:
curl localhost:8000/health
curl -s localhost:8000/search -H 'Content-Type: application/json' -d '{"query":"count the primes below n","k":3}'
docker compose down
```

**Correct result:** the image builds, `/health` answers `ok` immediately, the first search downloads the
model into the `hf-cache` volume and returns 3 hits, the UI loads at `http://localhost:8000/`, and a
second `docker compose up` does not download the model again. With no index directory mounted the
container still starts and `/health` says `degraded`. **This file has never been run**; a failure here is
expected information, not a regression.
**Paste back:** the `/health` output and the first-search timing.

## 7a. A — confidence-gated reranking — Kaggle **GPU (T4)**, ~70 min

Rerank only where the first stage is unsure. One reranking pass is paid; every threshold is then
evaluated offline from the cached pair scores, so the sweep itself is free.

```python
run([sys.executable, "src/eval/dev_gated_rerank.py", "--preset", "f2llm-v2-1.7b",
     "--device", "cuda"], cwd=REPO, env=ENV, check=False)
```

**Time:** the cost is one pass of 1,000 queries × 30 candidates = 30,000 cross-encoder pairs. At the
rate the earlier reranker sweeps measured that is **~55–65 min**, plus ~10 min for the first-stage
embeddings if their cache is cold. The sweep afterwards is seconds.

**Correct result:** the `t=0.0` row reproduces never-rerank (0.9299) and `t=99.0` reproduces
always-rerank (0.9371) — if those endpoints do not match, the wiring is wrong and nothing between them
is trustworthy. The useful outcome is a middle threshold that keeps most of always-rerank's gain while
reranking a minority of queries: the summary prints it as `best gate`.

Then confirm the chosen threshold **once** on the holdout:

```python
run([sys.executable, "src/eval/dev_gated_rerank.py", "--preset", "f2llm-v2-1.7b", "--device", "cuda",
     "--use-holdout", "--thresholds", "<the chosen one>"], cwd=REPO, env=ENV, check=False)
```

**Paste back:** `results/gated_rerank.json` and the summary block.

## 7b. F — code-to-description fusion — Kaggle **GPU (T4)**, ~60–130 min

```python
# sample 200 documents first and read the descriptions before committing to the full corpus
run([sys.executable, "src/build_descriptions.py", "--device", "cuda", "--limit", "200"],
    cwd=REPO, env=ENV, check=False)
```

**Look at the output before going further.** The script prints three examples. If they are generic
("code that processes input"), the fusion cannot help and the full run is not worth the GPU hour.

```python
run([sys.executable, "src/build_descriptions.py", "--device", "cuda"], cwd=REPO, env=ENV, check=False)
run([sys.executable, "src/eval/dev_descriptions.py", "--preset", "f2llm-v2-1.7b", "--device", "cuda"],
    cwd=REPO, env=ENV, check=False)
```

**Time:** generation is the cost — 8,765 documents at batch 16 and 40 new tokens is **~45–110 min**
depending on how long the snippets are, plus ~10 min to embed the descriptions and seconds to sweep.
Descriptions are cached by content hash and flushed every 200 items, so a killed session resumes by
rerunning the same command; it prints `N already described, M to do`.

**Correct result:** `description-only` should land well below code-only (it is a lossy view) while the
fusion sweep shows whether adding it helps. Any weight with `ADOPT` is worth confirming on the holdout.
A coverage warning means generation did not finish — finish it before believing the numbers.
**Paste back:** `results/description_fusion.json`, plus the three sample descriptions.

## 7c. B — fine-tuned lite model — Kaggle **GPU (T4)**, ~100–140 min

Three stages, each resumable:

```python
run([sys.executable, "src/train/finetune_lite.py", "--stage", "mine", "--device", "cuda"],
    cwd=REPO, env=ENV, check=False)                                   # ~10 min
run([sys.executable, "src/train/finetune_lite.py", "--stage", "train", "--device", "cuda"],
    cwd=REPO, env=ENV, check=False)                                   # ~60-90 min
run([sys.executable, "src/train/finetune_lite.py", "--stage", "eval", "--device", "cuda"],
    cwd=REPO, env=ENV, check=False)                                   # ~15 min
```

**Memory:** LoRA (r=16) on the 0.6B at sequence length 512 and batch 8 with gradient checkpointing uses
roughly **6–9 GB** of the T4's 14.56 GB — the fp16 base is ~1.2 GB, the adapter and its optimiser state
are tens of MB, and activations dominate. Full fine-tuning was rejected for exactly this reason: Adam
moments for every parameter would be ~7 GB before any activations, leaving no room for a batch large
enough for in-batch negatives to teach anything. If it still OOMs, halve `--batch-size` and raise
`--hard-per-batch` to keep the negatives.

**Time:** mining encodes the corpus once plus 4,000 tune queries; training is one epoch over 4,000 pairs;
evaluation encodes the corpus once more plus the 1,000 holdout queries. Training checkpoints every 100
steps to `outputs/finetune_lite/adapter`, so a killed session resumes from the same command.

**Paired evaluation.** The eval stage now scores the **base 0.6B and the tuned model on the same 1,000
holdout queries in one run** (the corpus is encoded once per model, so it takes roughly twice as long as
before, ~25 min on a T4). It saves per-query NDCG@10 and MRR@10 for each model to
`outputs/finetune_lite/holdout_per_query.json`, and reports the paired-bootstrap 95% CI, the improved /
worsened counts and the adoption verdict (NDCG@10 gain ≥ 0.005, CI excludes zero, worsened ≤ half of
improved). To also pair against the **1.7B**, run this once first — it is still the eval stage, so the
holdout stays touched by nothing else:

```python
run([sys.executable, "src/train/finetune_lite.py", "--stage", "eval", "--base-only",
     "--preset", "f2llm-v2-1.7b", "--device", "cuda"], cwd=REPO, env=ENV, check=False)   # ~10 min
```

**Correct result:** a `PAIRED HOLDOUT EVALUATION` block with one line per model, then one verdict line per
comparison (`tuned vs base 0.6B`, and `tuned vs 1.7B` when its ranks exist), each with
`dNDCG@10 [95% CI]`, `improved / worsened / same` and `ADOPT` or `do not adopt`. The measured base 0.6B
should land near the recorded 0.9054; a `WARNING` line means it did not. The realistic outcome is that the
tuned model becomes the lite model and is not a P0 candidate.
**Not yet run:** no number from this experiment is claimed anywhere.
**Paste back:** `results/finetune_lite.json`, `outputs/finetune_lite/holdout_per_query.json` and `outputs/finetune_lite/adapter_revision.json`.

## 8. Code intelligence: router, structural search, agent — Kaggle CPU, ~10 min

Everything here is CPU-only: the structural index is `ast` parsing, the router is rules, and the agent's
dense steps reuse the already-loaded model. Run it in the **same CPU session** as steps 1–4.

```python
# tag an index with offline categories (no model, no re-encoding, ~seconds)
run([sys.executable, "src/build_categories.py", "--index", "examples/_demo_index", "--clusters", "8"],
    cwd=REPO, env=ENV, check=False)

# the router and the exact structural answers
for q in ["who calls tokenize",
          "where is stats imported",
          'where is the string "not a docstring" used',
          "which files call tokenize before cosine_similarity"]:
    run([sys.executable, "src/cli.py", q, "--index", "examples/_demo_index",
         "--structural-root", "examples/textkit"], cwd=REPO, env=ENV, check=False)

# the agent, with its full trace
run([sys.executable, "src/cli.py", "who calls tokenize", "--agent",
     "--index", "examples/_demo_index", "--structural-root", "examples/textkit"],
    cwd=REPO, env=ENV, check=False)

# performance extras on real results
run([sys.executable, "src/cli.py", "rank documents against a query", "-k", "3",
     "--index", "examples/_demo_index", "--suggestions"], cwd=REPO, env=ENV, check=False)
```

**Correct result:** each structural query prints `route : structural (<intent>)` and then **exact**
`file:line` answers, not ranked guesses — `who calls tokenize` should list call sites in `search.py`
and `stats.py`. The agent prints `planner : deterministic (no LLM)`, a numbered trace with a `why:` on
every step, and `stopped : evidence sufficient…` well inside the 6-step cap. `build_categories.py` ends
with `index still verifies: yes`.
**Paste back:** the four route lines, the agent trace, and the categories summary.

### 8b. Agent vs dense on labelled questions — Kaggle **GPU (T4)**, ~15 min

Run on the click history index from step 5, where the repo is big enough for the comparison to mean
something (on `examples/textkit` both score ~1.0 because there are only 19 chunks).

```python
run([sys.executable, "src/bench_agent.py", "--index", "history",
     "--source", "data/repos/click/src/click", "--device", "cuda", "--max-questions", "30"],
    cwd=REPO, env=ENV, check=False)
```

Ground truth is built by grep/regex and docstrings, never by the `ast` walk the agent uses — the report
breaks results down by label method so you can see that.

**Correct result:** ~30 questions across structural/usage/semantic; the agent should beat plain dense on
the **structural and usage** rows (dense cannot answer "who calls X" at all) and be roughly level on
**semantic** ones, at higher latency and a median of 2–3 steps. If the agent loses on structural rows,
something is wrong with the call graph, not with the scoring.
**Paste back:** `results/agent_benchmark.json`, or the `AGENT vs DENSE` table.

### 8c. Category tiebreaker (dev experiment E) — Kaggle **GPU (T4)**, ~20 min

```python
run([sys.executable, "src/eval/dev_categories.py", "--preset", "f2llm-v2-1.7b", "--device", "cuda"],
    cwd=REPO, env=ENV, check=False)
```

Selected on the 1,000-query dev slice against the full corpus, never on test.

**Correct result:** a sweep of bonus weights, each with its delta, CI and adopt/reject. **The expected
outcome is FAILS** — the query side has no code to parse, so its family is inferred from keywords, which
is a weak signal. A clean reject is a real result and keeps the feature off; an ADOPT would be a
pleasant surprise worth re-checking on the holdout before believing.
**Paste back:** `results/category_tiebreak.json`.

---

## 9. Final submission run — Kaggle GPU then CPU, ~45 min

Run this **last**, from a clean clone of the final repository, once every experiment above has been
decided. It regenerates the submitted result and everything that describes it.

### 9a. The official run — GPU (T4), ~20 min

```python
run([sys.executable, "src/eval/run_official.py",
     "--config", "configs/official_f2llm17b_noreranker.json",
     "--device", "cuda", "--confirm-test"], cwd=REPO, env=ENV, check=False)
```

One run writes the results file, the top-100 rankings, the checksums and `runtime_index/`. If an earlier
run's lock is present it refuses; add `--allow-rerun` only if you mean to spend another test-split
evaluation.

**Correct result:** NDCG@10 ≈ **0.9376**, MRR@10 ≈ **0.9238**, and the gap line reads `OK: within the
dev-vs-test gap`. A LOUD WARNING means the config or the model changed — stop and investigate before
using the number.

### 9b. Verify it — CPU, ~2 min

```python
run([sys.executable, "src/verify_submission.py", "--confirm-test"], cwd=REPO, env=ENV, check=False)
```

**Correct result:** `PASSED` — the metrics recomputed from the stored ranking match the reported ones
and every hash checks out.

### 9c. Failure analysis — CPU, ~3 min

```python
run([sys.executable, "src/analyze_failures.py", "--confirm-test"], cwd=REPO, env=ENV, check=False)
```

Reads the rankings file only; no model, no encoding.

**Correct result:** ~6% of test queries miss the top 10, grouped into very-long-query, near-duplicate
corpus, generic wording and other, plus `results/failure_examples.md` with 3–5 worked examples ready for
a slide.
**Paste back:** `results/failure_analysis.json` and `results/failure_examples.md`.

### 9d. Metrics report — CPU, seconds

```python
run([sys.executable, "src/build_metrics_report.py"], cwd=REPO, env=ENV, check=False)
```

Collects every result file written by steps 1–8 into `results/metrics.md`. **Missing inputs are not an
error**: each becomes a "not yet measured" row naming the command that fills it, so this is worth running
at any point to see what is still outstanding.

**Correct result:** the final line reads `measured: N/13 sources present`, and the coverage section lists
anything still missing.
**Paste back:** `results/metrics.md`.

### 9e. Fill the README placeholders

`README.md` carries `{{PLACEHOLDER}}` markers for every number that only a real run produces; they are
listed at the bottom of the README. Once steps 1–9d are done, the values come from:

| Placeholders | Source file |
|---|---|
| `{{CPU_*}}`, `{{THREAD_*}}` | `results/cpu_*_capped.json`, `results/cpu_*_uncapped.json` |
| `{{FP32_*}}`, `{{BF16_*}}`, `{{INT8_*}}` | `results/cpu_precision.json` |
| `{{CAP_*}}`, `{{QUERY_TOKENS_*}}` | `results/query_cap_sweep.json` |
| `{{HIST_*}}` | `results/real_history_benchmark.json`, `results/history_ingest.json` |
| `{{P1_*}}`, `{{BONUS_*}}` | `results/version_rebuild_benchmark.json`, `results/evolution_benchmark.json` |
| `{{AGENT_*}}` | `results/agent_benchmark.json` |
| `{{GATE_*}}`, `{{DESC_*}}`, `{{FT_*}}`, `{{CATEGORY_VERDICT}}` | `results/gated_rerank.json`, `description_fusion.json`, `finetune_lite.json`, `category_tiebreak.json` |
| `{{INDEX_SIZE_MB}}` | the `build_index.py` output, or the official run's index manifest |

Paste the JSON files back and I will fill them in; they are all in `results/`, so a single zip of that
directory carries everything.

## 10. Demo-polish features with a real encoder — Kaggle CPU, ~20 min

The features from `feat/demo-polish` are unit-tested with the hashing encoder only. This step is what
turns "implemented and unit-tested" into a measured statement. It uses the **lite** model on CPU, so no
GPU is needed; run it after the branch is merged into `develop`. Start from step 0 (clone + install).

### 10a. Incremental update (`src/reindex.py`)

```python
import numpy as np, json, shutil
from pathlib import Path

IDX = REPO / "examples" / "_reindex_demo"
run([sys.executable, "src/build_index.py", "--source", "examples/textkit", "--preset", "f2llm-v2-0.6b",
     "--device", "cpu", "--out", str(IDX)], cwd=REPO, env=ENV, check=False)

def snapshot():
    texts = json.loads((IDX / "corpus_texts.json").read_text())
    return dict(zip(texts, np.load(IDX / "embeddings.npy")))
before = snapshot()

target = REPO / "examples" / "textkit" / "tokenizing.py"
original = target.read_text()
target.write_text(original + '\n\ndef shout(text):\n    """Upper-case the text and end it with an exclamation mark."""\n    return text.upper() + "!"\n')

run([sys.executable, "src/reindex.py", "--index", str(IDX), "--device", "cpu", "--dry-run"], cwd=REPO, env=ENV, check=False)
run([sys.executable, "src/reindex.py", "--index", str(IDX), "--device", "cpu"], cwd=REPO, env=ENV, check=False)
run([sys.executable, "src/cli.py", "make text upper case and add an exclamation mark", "-k", "3",
     "--index", str(IDX), "--device", "cpu"], cwd=REPO, env=ENV, check=False)
run([sys.executable, "src/reindex.py", "--index", str(IDX), "--device", "cpu"], cwd=REPO, env=ENV, check=False)  # again: nothing to do

after = snapshot()
kept = [t for t in before if t in after]
print("unchanged chunks whose stored vector is byte-identical:",
      sum(np.array_equal(before[t], after[t]) for t in kept), "of", len(kept))
target.write_text(original)                 # restore the example folder
shutil.rmtree(IDX, ignore_errors=True)
```

**Correct result:** the dry run says `+1 added`, `19 unchanged`, `1 to recompute`, and writes nothing; the
real run recomputes exactly **1** embedding (`embeddings: 19 reused, 1 recomputed`) and `shout` is the
top hit for the query; the second real run says `index already up to date` and finishes in well under a
second (it never loads the model); the byte-identical count equals the number of unchanged chunks.
**Paste back:** the four `[reindex]` blocks and the byte-identical line. The `elapsed`/`encode` seconds
are the figures the README does not yet quote.

### 10b. Version comparison, diff, viewer and "why" on a real history index

Needs the `history` index from step 5 (lite model, CPU is enough for serving).

```python
code = r'''
import os, json
os.environ.update(INDEX_DIR="history", SEARCH_DEVICE="cpu", MAX_QUERY_TOKENS="1024")
from fastapi.testclient import TestClient
import src.api as api
c = TestClient(api.app)
q = "run the command and handle the context"
cmp = c.get("/compare", params={"q": q, "a": 1, "b": 40, "k": 5}).json()
print("compare summary:", cmp["summary"], cmp["timings_ms"])
changed = [h for h in cmp["b"] if h["content_changed"]]
if changed:
    d = c.get("/diff", params={"snippet_id": changed[0]["snippet_id"], "a": 1, "b": 40}).json()
    print(d["snippet_id"], "+%d -%d" % (d["added_lines"], d["removed_lines"]))
    print(d["diff"][:1200])
s = c.post("/search", json={"query": q, "k": 5, "all_versions": True, "explain": True}).json()
print("groups:", [(g["snippet_id"], g["n_in_results"]) for g in s["groups"]])
print("why:", s["hits"][0]["why"])
doc = c.get("/doc", params={"id": s["hits"][0]["doc_id"]}).json()
print(doc["location"], doc["n_lines"], "lines, first line", doc["first_line"])
'''
run([sys.executable, "-c", code], cwd=REPO, env=ENV, check=False)
```

**Correct result:** `compare` returns 5 results per side with a `summary`; when a result has
`content_changed`, `/diff` shows a plausible unified diff between those two commits; `groups` collapses
repeated lineages under `all_versions`; `why.matched_terms` lists identifiers that really occur in the
snippet; `/doc` returns a `file:start-end` location and real file line numbers. Adjust the query if the
history index does not contain a `Command`-style function.
**Paste back:** the printed block.

### 10c. Exact-query cache latency

```python
code = r'''
import time
from src.search_service import SearchService
svc = SearchService("lite", device="cpu", query_cache_size=8)
q = "count the number of primes below n"
for label in ("cold", "warm (cached)", "warm (cached)"):
    t = time.time(); r = svc.search(q, k=3); ms = 1000 * (time.time() - t)
    print(f"{label:14s} {ms:8.1f} ms  encode {r['timings_ms']['encode_query_ms']:.1f} ms  cached={r['timings_ms']['cached']}  top={r['hits'][0]['doc_id']}")
print(svc.describe()["query_cache"])
'''
run([sys.executable, "-c", code], cwd=REPO, env=ENV, check=False)
```

**Correct result:** the first search pays the encode; the repeats report `cached=True` with an encode time
of a fraction of a millisecond and the **same top document**. The cache never applies to benchmarks; this
is the only place its effect is measured.
**Paste back:** the four printed lines.

### 10d. The page, by eye (Colab, with step 6)

Open the UI from step 6 and check: the status line changes when you pick another index; the textarea keeps
newlines and Ctrl/Cmd+Enter searches; an example fills it; clicking a hit opens the full snippet with line
numbers and a working Copy; **Update index** appears only for a folder/history index; on the history
index, **Compare versions** and **View history** work; scores read "similarity score"; agent mode greys
out the version/category controls.

---

## What to send back

Fastest for me to act on, in order of preference:

1. the JSON files: `results/cpu_precision.json`, `results/cpu_*_capped.json`,
   `results/cpu_*_uncapped.json`, `results/real_history_benchmark.json`, `results/history_ingest.json`;
2. otherwise the summary blocks quoted above;
3. for anything that failed: the last ~40 lines, including the traceback.

Each step writes its JSON before printing its summary, so a file exists even if a later step fails.
