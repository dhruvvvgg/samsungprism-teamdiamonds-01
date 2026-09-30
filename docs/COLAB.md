# Google Colab & Kaggle Execution Guide

This guide provides two complete workflows for running the Samsung PRISM GenAI Hackathon (Theme 01) submission:

1. **Judge Quick-Start Serving Demo Path (CPU, ~2 minutes)**: Download precomputed release indexes, start the serving API and web UI, and test queries immediately (serves prebuilt indexes without re-running heavy model encoding or evaluation on CPU).
2. **Maintainer / Full Reproduction Path (T4 GPU, ~16 minutes)**: Re-run the complete evaluation pipeline from scratch on a GPU instance (Cells 1 to 4).

---

## Path 1: Judge Quick-Start Serving Demo (CPU, ~2 minutes)

Judges can explore and demo the serving system on a standard CPU runtime using precomputed indexes without re-encoding the corpus or downloading gigabytes of model weights.

### Step 1: Clone Repository & Install Serving Dependencies

```bash
git clone https://github.com/dhruvvvgg/samsungprism-teamdiamonds-01.git
cd samsungprism-teamdiamonds-01
pip install -r requirements-serve.txt
```

### Step 2: Fetch Precomputed Release Assets

Download the verified prebuilt runtime indexes attached to GitHub Release `PRISM_GENAI_HACKATHON_Y2026`:

```bash
# Downloads runtime_index.zip, runtime_index_lite.zip, and history_index.zip
python scripts/fetch_release_assets.py --assets indexes
```

*(Optional: To also download official test split rankings, results, and execution logs, pass `--assets all`.)*

### Step 3: Start the Serving Gateway

```bash
# Start with --mock for immediate UI testing without downloading neural weights:
python scripts/serve.py --host 0.0.0.0 --port 8000 --mock
```

`serve.py` automatically detects the downloaded indexes (`runtime_index`, `runtime_index_lite`, `history_index`) and configures `ALLOWED_INDEXES`. The `--mock` flag enables the deterministic mock query encoder for UI development and fast inspection. (To serve with neural embeddings, ensure model weights are cached locally or run on a GPU instance).

### Step 4: Access the Web UI or Query the API

In Google Colab, you can expose the local port directly via the Colab proxy:

```python
from google.colab.output import eval_js
print("Web UI Link:", eval_js("google.colab.kernel.proxyPort(8000)"))
```

Alternatively, use `localtunnel` or `ngrok` in a separate terminal/cell:

```bash
# Localtunnel
npx localtunnel --port 8000
```

Or query the REST API directly:

```bash
# Health check
curl -s http://localhost:8000/health

# Dense search
curl -s http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "count the primes below n", "k": 5}'

# Code intelligence agent
curl -s http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "How does context management work?", "k": 5}'
```

---

## Path 2: Maintainer & Reproduction Path (T4 GPU, ~16 minutes)

To reproduce the official evaluation metrics (NDCG@10 0.9376, MRR@10 0.9238) on a Kaggle or Google Colab T4 GPU notebook, run `scripts/run_all_checks.py`:

### Step 1: Environment Setup & Hardware Health Check

```bash
python scripts/run_all_checks.py --only P0
```
- Sets up environment and verifies GPU availability, disk space, and Python dependencies.
- Runs Phase 0 (P0) diagnostics.

### Step 2: Tier 1 & Tier 2 Evaluation Pipeline

```bash
python scripts/run_all_checks.py --tier 2 --resume --max-hours 9
```
- Executes official MTEB APPS retrieval evaluation (`configs/official_f2llm17b_noreranker.json`) on 3,765 test queries and 8,765 documents.
- Exports dense `runtime_index/` and lite `runtime_index_lite/`.
- Executes failure analysis (77 failure query analysis) and CPU precision benchmarks.
- Runs Tier 2 Click 40-commit real history benchmark.

### Step 3: Collect & Package Release Artifacts

```bash
python scripts/run_all_checks.py --only C
```
- Validates output checksums against official baselines.
- Archives release packages (`runtime_index.zip`, `runtime_index_lite.zip`, `results_final.zip`, etc.).
- Compiles final metrics summary.
