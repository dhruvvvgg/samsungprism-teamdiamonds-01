# Google Colab & Kaggle Execution Guide

This guide provides two complete workflows for running the Samsung PRISM GenAI Hackathon (Theme 01) submission:

1. **Judge Quick-Start Path (CPU, ~2 minutes)**: Download precomputed release indexes, start the serving API and web UI, and test queries immediately.
2. **Maintainer / Full Reproduction Path (T4 GPU, ~16 minutes)**: Re-run the complete evaluation pipeline from scratch on a GPU instance (Cells 1 to 4).

---

## Path 1: Judge Fast Evaluation (CPU, ~2 minutes)

Judges can evaluate the system on a standard CPU runtime without re-encoding the corpus or downloading gigabytes of model weights.

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
python scripts/serve.py --host 0.0.0.0 --port 8000
```

`serve.py` automatically detects the downloaded indexes (`runtime_index`, `runtime_index_lite`, `history_index`), configures `ALLOWED_INDEXES`, and selects the appropriate query encoder. If neural weights are not cached locally, it safely defaults to `MOCK_ENCODER=1` to allow zero-weight testing of all API and UI flows.

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

To reproduce the official evaluation metrics (NDCG@10 0.9376, MRR@10 0.9238) on a Kaggle or Google Colab T4 GPU notebook, run the sequential runner cells located in `scripts/kaggle_cells/`:

### Cell 1: Environment Setup & Hardware Health Check

```python
# In notebook cell 1:
%run scripts/kaggle_cells/cell1_setup.py
```
- Sets up environment and verifies GPU availability, disk space, and Python dependencies.
- Runs Phase 0 (P0) diagnostics.

### Cell 2: Tier 1 & Tier 2 Evaluation Pipeline

```python
# In notebook cell 2:
%run scripts/kaggle_cells/cell2_tier1_2.py
```
- Executes official MTEB APPS retrieval evaluation (`configs/official_f2llm17b_noreranker.json`) on 3,765 test queries and 8,765 documents.
- Exports dense `runtime_index/` and lite `runtime_index_lite/`.
- Executes failure analysis (77 failure query analysis) and CPU precision benchmarks.
- Runs Tier 2 Click 40-commit real history benchmark.

### Cell 3: Tier 3 Benchmarks (Optional)

```python
# In notebook cell 3:
%run scripts/kaggle_cells/cell3_tier3_optional.py
```
- Evaluates code evolution benchmark, multi-step code intelligence agent benchmark, and incremental index rebuild benchmarks.

### Cell 4: Collect & Package Release Artifacts

```python
# In notebook cell 4:
%run scripts/kaggle_cells/cell4_collect.py
```
- Validates output checksums against official baselines.
- Archives release packages (`runtime_index.zip`, `runtime_index_lite.zip`, `results_final.zip`, etc.).
- Compiles final metrics summary.
