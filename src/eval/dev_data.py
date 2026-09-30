"""Dev protocol data: CoIR-APPS *train* queries against the FULL corpus, using *train* qrels only.

Guarantees:
  * only `load_dataset("CoIR-Retrieval/apps", "default", split="train")` qrels are read; the test qrels
    are never loaded here. (The test split is touched only by src/eval/run_official.py -> mteb.evaluate.)
  * corpus order == the HF corpus order == MTEB's corpus order, so doc indices line up everywhere.
  * a fixed-seed 1,000-query HOLDOUT is carved out of the 5,000 train queries. All sweeps use the
    remaining 4,000 ("tune"); the holdout is reserved (fine-tuning / one final confirmation).

Note: the corpus rows include the test-partition solutions (no labels used) because at test time the
model also ranks against the full 8,765-doc corpus. Train queries therefore face the same distractors.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "apps" / "dev_train.json"
SPLIT_PATH = ROOT / "outputs" / "dev" / "split.json"
HOLDOUT_SEED, HOLDOUT_SIZE = 13, 1000


def make_split(query_ids, seed=HOLDOUT_SEED, holdout_size=HOLDOUT_SIZE):
    """Deterministic (tune_ids, holdout_ids) partition of `query_ids` (order-independent)."""
    ids = sorted(query_ids)
    rnd = random.Random(seed)
    holdout = set(rnd.sample(ids, min(holdout_size, len(ids))))
    return [q for q in ids if q not in holdout], sorted(holdout)


def load_dev(force=False):
    """Return dict: doc_ids, doc_texts, query_ids, query_texts, rel_idx (doc index per query),
    tune_idx / holdout_idx (indices into query_ids). Cached under ./data; rebuilt if missing."""
    if CACHE.exists() and not force:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    from datasets import load_dataset
    corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")
    queries = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
    qrels = load_dataset("CoIR-Retrieval/apps", "default", split="train")   # TRAIN qrels only
    doc_ids = [str(r["_id"]) for r in corpus]
    doc_texts = [r.get("text") or "" for r in corpus]
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    qtext = {str(r["_id"]): (r.get("text") or "").strip() for r in queries}
    rel = {}
    for r in qrels:
        if int(r["score"]) > 0 and str(r["corpus-id"]) in doc_index and qtext.get(str(r["query-id"])):
            rel.setdefault(str(r["query-id"]), str(r["corpus-id"]))   # exactly one relevant doc/query
    query_ids = sorted(rel)
    tune, holdout = make_split(query_ids)
    pos = {q: i for i, q in enumerate(query_ids)}
    out = {
        "doc_ids": doc_ids, "doc_texts": doc_texts, "query_ids": query_ids,
        "query_texts": [qtext[q] for q in query_ids],
        "rel_idx": [doc_index[rel[q]] for q in query_ids],
        "tune_idx": [pos[q] for q in tune], "holdout_idx": [pos[q] for q in holdout],
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out), encoding="utf-8")
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps({"seed": HOLDOUT_SEED, "tune": tune, "holdout": holdout}))
    return out
