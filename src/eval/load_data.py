"""Download CoIR-Retrieval/apps from HuggingFace, cache under ./data, print sanity stats.

STANDING RULE: the TEST split is touched only by the official run (src/eval/run_official.py /
run_baseline.py via mteb.evaluate). `load_apps` therefore loads TEST qrels and refuses to run unless
allow_test=True is passed explicitly. Everything else uses the train-split dev protocol:
src/eval/dev_data.py::load_dev.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "apps"
REPO_ID = "CoIR-Retrieval/apps"


def _rows(ds):
    return [dict(r) for r in ds]


def load_apps(force=False, allow_test=False):
    """TEST-split loader (see module docstring). Return (corpus{id:text}, queries{id:text}, qrels{qid:{did:score}}). Cached; rebuilt if missing."""
    if not allow_test:
        raise PermissionError(
            "load_apps() reads the TEST split, which is reserved for the official run. Use "
            "src.eval.dev_data.load_dev() (train split) for development, or pass allow_test=True.")
    cache = DATA_DIR / "apps.json"
    if cache.exists() and not force:
        return tuple(json.loads(cache.read_text(encoding="utf-8")).values())
    from datasets import load_dataset
    corpus_ds = load_dataset(REPO_ID, "corpus", split="corpus")
    query_ds = load_dataset(REPO_ID, "queries", split="queries")
    qrels_ds = load_dataset(REPO_ID, "default", split="test")
    corpus = {str(r["_id"]): (r.get("text") or "") for r in corpus_ds}
    queries = {str(r["_id"]): (r.get("text") or "").strip() for r in query_ds}
    qrels = {}
    for r in qrels_ds:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    # keep only test queries that have judgments and non-empty text; drop dangling doc ids
    qrels = {q: {d: s for d, s in ds.items() if d in corpus and s > 0} for q, ds in qrels.items()}
    qrels = {q: ds for q, ds in qrels.items() if ds and queries.get(q)}
    queries = {q: queries[q] for q in qrels}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"corpus": corpus, "queries": queries, "qrels": qrels}),
                     encoding="utf-8")
    return corpus, queries, qrels


def main():
    """Sanity check on the TRAIN-split dev data (never the test split)."""
    from src.eval.dev_data import load_dev
    d = load_dev()
    print(f"corpus size:         {len(d['doc_ids'])}")
    print(f"train dev queries:   {len(d['query_ids'])} (tune {len(d['tune_idx'])}, holdout {len(d['holdout_idx'])})")
    q, r = 0, d["rel_idx"][0]
    print(f"\n--- example query ({d['query_ids'][q]}) ---\n{d['query_texts'][q][:600]}")
    print(f"\n--- relevant doc ({d['doc_ids'][r]}) ---\n{d['doc_texts'][r][:600]}")


if __name__ == "__main__":
    sys.exit(main())
