# Development Experiments & Empirical Evidence

This document compiles the empirical evaluation results, candidate sweeps, and architectural decisions conducted during development for the Samsung PRISM GenAI Hackathon (Theme 01, Code Retrieval on CoIR-APPS).

To keep the submission repository lean and focused on judge verification, experimental sweep drivers and un-shipped candidate code reside in the development repository and are not shipped in this submission.

---

## 1. Development Protocol & Adoption Rule

### Protocol Facts
- **Corpus**: Full 8,765-document CoIR-APPS corpus.
- **Dataset Partition**: 5,000 CoIR-APPS *train* queries evaluated against the full corpus using *train* qrels only.
- **Split Construction**: A fixed random seed (`seed=13`) splits the 5,000 train queries into:
  - **Tune Slice**: 4,000 queries used for hyperparameter sweeps, candidate exploration, and weight tuning.
  - **Holdout Slice**: 1,000 reserved queries used strictly for paired validation of candidate setups.
- **Test Split Isolation**: The 3,765 official test queries were touched **exactly once** for the final official submission run (`configs/official_f2llm17b_noreranker.json`). No parameter tuning, model selection, or ablation was performed on the test split.

### The Adoption Rule
To prevent overfitting to point estimates and retain only robust improvements, a candidate feature was adopted if and only if it met **all three** criteria on the dev holdout:
1. **NDCG@10 Gain**: An improvement of at least **+0.005** NDCG@10 over the baseline.
2. **Statistical Significance**: A 95% paired-bootstrap confidence interval (1,000 resamples) strictly excluding zero (`lower_ci > 0`).
3. **Safety Ratio**: The count of worsened queries must be at most **half** the count of improved queries (`worsened <= 0.5 * improved`).

Nothing was selected or tuned on the test split.

---

## 2. Dev Slice Candidate Results (Holdout Slice)

Evaluated on the 1,000-query dev holdout against the full 8,765-document corpus:

| Configuration | NDCG@10 | MRR@10 | Outcome & Decision |
| :--- | :---: | :---: | :--- |
| **F2LLM-v2-0.6B** (single model) | 0.8959 | 0.8757 | Baseline lite model |
| **EmbeddingGemma-300M** | 0.8260 | 0.7973 | Sub-baseline; rejected |
| **0.6B + EmbeddingGemma-300M** (weight 0.4) | 0.9100 | 0.8928 | Stronger than 0.6B alone (+0.0141) |
| **F2LLM-v2-1.7B** (single model) | **0.9299** | **0.9167** | **Selected submission model** |
| **1.7B + 0.6B dense fusion** (z-score, weight 0.25) | 0.9353 | 0.9218 | **Adopted on dev** (+0.0054, passed all criteria). *Excluded from final submission for latency/cost (doubles per-query encoding cost).* |
| **1.7B + EmbeddingGemma-300M** (weight 0.3) | 0.9356 | 0.9227 | **Failed adoption rule**: 49 improved vs. 26 worsened queries (worsened ratio 0.53 > 0.50). |
| **1.7B + Qwen3-Reranker-0.6B** ($k=30, \alpha=0.5$) | 0.9371 | 0.9228 | **Adopted on dev** (+0.0073, 95% CI [+0.0042, +0.0108]). *Excluded from final submission for latency/cost.* |
| **Fusion + Reranker stacked** | 0.9404 | 0.9274 | **Rejected**: 95% paired bootstrap interval includes zero against the best single component. |

### Why Reranking and Fusion Were Dropped from Final Submission
While cross-encoder reranking and dual-model dense fusion demonstrated positive gains on the dev slice, both were deliberately excluded from the official submission path due to operational and evaluation cost:
- **Reranker Cost**: Reranking the top-30 candidates across 3,765 test queries requires scoring **112,950 query-document pairs**. At ~4.4 pairs/second on a single Kaggle T4 GPU, a full test evaluation would take **~7.1 hours** (exceeding standard 9-hour session budgets and drastically increasing submission fragility).
- **Fusion Cost**: Dense fusion requires running two distinct LLM forward passes for every query, doubling per-query latency and GPU RAM requirements during CPU/GPU serving.
- **Submission Simplicity**: The standalone F2LLM-v2-1.7B achieved **0.9376 NDCG@10** on the test split without any second-stage complexity.

---

## 3. Rejected Architectures & Negative Findings

Extensive experiments showed that several widely used retrieval enhancements actively harmed code retrieval performance on CoIR-APPS:

1. **BM25 Hybrid Combination**:
   - BM25 alone scored only **0.19** NDCG@10 (code tokens and problem statement terminology share minimal vocabulary).
   - Reciprocal Rank Fusion (RRF) and linear score interpolation with BM25 degraded NDCG@10 by **-0.004 to -0.21** across all tested weights.
2. **Query Rewrites & Instruction Variants**:
   - Prompt rewrites, docstring summarizations, and hypothetical document embeddings (HyDE) reduced accuracy across the board (worst variant showed **-0.127** NDCG@10 drop).
3. **RRF Fusion**:
   - Rank-based reciprocal rank fusion produced no statistically significant gain over calibrated z-score normalization.
4. **LLM Re-Judge Stage**:
   - Because initial dense retrieval achieves **Recall@100 = 0.99655** (and Recall@10 = 0.97955), the trigger condition for an expensive generative LLM re-judge almost never fires on real queries.
5. **BGE-reranker-v2-m3**:
   - Cross-encoder reranking with multilingual BGE severely degraded rankings across code snippets compared to code-specialized models.
6. **Ettin Rerankers (68M / 150M)**:
   - Modest gain of **+0.0011** NDCG@10 was not statistically significant (95% CI spans zero).
7. **F2LLM-v2-4B**:
   - The 4B model has a memory footprint of **15.3 GB** in fp16, causing immediate Out-Of-Memory (OOM) failures on standard 14.56 GB T4 GPUs even at batch size 1.

---

## 4. Serving Query Cap Sweep (F2LLM-v2-1.7B)

APPS problem statements have a long sequence length tail. Because attention computation scales quadratically with length, query token caps were systematically swept on the dev slice:

| Max Query Tokens | NDCG@10 | Delta vs. Uncapped | 95% Confidence Interval | Queries Truncated |
| :---: | :---: | :---: | :---: | :---: |
| **Uncapped** | **0.9299** | Baseline | — | 0.0% |
| **1024** | **0.9289** | **-0.0010** | [-0.0040, 0.0000] | **2.6%** |
| **512** | 0.9222 | -0.0077 | [-0.0125, -0.0031] | 18.4% |
| **256** | 0.8515 | -0.0784 | [-0.0910, -0.0650] | 46.2% |

### Policy Decision:
- **Serving Path**: The serving gateway (`src/cli.py`, `src/api.py`, `scripts/serve.py`) defaults to **1024 tokens**, capping long-tail latency on CPU while losing less than 0.001 NDCG@10.
- **Official Submission Run**: The official benchmark run (`run_official.py`) stays completely **uncapped** (verified by regression tests).

---

## 5. Official Test Split Performance Across Models

For benchmark context, multiple model families were evaluated under identical conditions on the full CoIR-APPS test split (3,765 queries × 8,765 documents):

| Model | Parameters | Official Test NDCG@10 | Official Test MRR@10 |
| :--- | :---: | :---: | :---: |
| **F2LLM-v2-1.7B** (Official Submission) | 1.7 B | **0.9376** | **0.9238** |
| **F2LLM-v2-0.6B** (Interactive Lite Variant) | 0.6 B | 0.9044 | 0.8840 |
| **Qwen3-Embedding-0.6B** | 0.6 B | 0.7310 | — |
| **gte-modernbert-base** | 149 M | 0.5760 | 0.5280 |
| **CodeRankEmbed** | 137 M | 0.2350 | 0.2060 |
| **all-MiniLM-L6-v2** | 22 M | 0.0660 | 0.0560 |

---

## 6. Designed But Not Run (Future Work)

Four additional research directions were formally designed, scaffolded, and unit-tested in the development repository, but were **not run at scale on GPU** and their code is not shipped:
1. **Confidence-Gated Reranking (Experiment A)**: Dynamically activating cross-encoder reranking only when the top-1 to top-2 cosine similarity margin falls below a confidence threshold.
2. **Code-to-Description Fusion (Experiment F)**: Extracting natural language descriptions from solution code via lightweight generative models and indexing dual views.
3. **LoRA Fine-Tuning of the 0.6B Model (Experiment B)**: Contrastive fine-tuning of the lite 0.6B encoder on synthetic hard negatives from the train split.
4. **Category Tiebreaker (Experiment E)**: Utilizing AST-derived algorithmic categories as rank-preserving tiebreakers for near-identical embedding scores.
