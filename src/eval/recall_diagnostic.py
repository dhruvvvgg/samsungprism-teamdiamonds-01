"""Is the relevant document in the candidate pool at all? recall@10/100/1000 (+ NDCG/MRR).

Each query has exactly one relevant doc, so recall@k == hit-rate@k.
Runs on the TRAIN-split dev data (never the test split). Default: 150-query / ~1k-doc subset (fast);
--full uses the whole 8,765-doc corpus with all 5,000 train queries.
NOTE: on the ~1k subset recall@1000 is 1.0 by construction (pool <= 1000); use --full for that.

    !python src/eval/recall_diagnostic.py --model nomic-ai/CodeRankEmbed --trust-remote-code \
        --device cuda --max-seq-length 1024 --query-prefix "Represent this query for searching relevant code: " --full
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
KS = (10, 100, 1000)


def metrics(S, qids, docs, rel):
    doc_index = {d: i for i, d in enumerate(docs)}
    ranks = []
    for i, q in enumerate(qids):
        target = S[i, doc_index[rel[q]]]
        ranks.append(int((S[i] > target).sum()) + 1)  # 1-based rank of the relevant doc
    r = np.array(ranks)
    out = {f"recall@{k}": float((r <= k).mean()) for k in KS}
    out["ndcg@10"] = float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean())
    out["mrr@10"] = float(np.where(r <= 10, 1 / r, 0).mean())
    out["median_rank"] = float(np.median(r))
    out["pool_size"] = len(docs)
    return out, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--doc-prefix", default="")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-queries", type=int, default=150)
    ap.add_argument("--n-docs", type=int, default=986)
    ap.add_argument("--also-no-prefix", action="store_true",
                    help="also score queries encoded WITHOUT the prefix, to measure its effect")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "recall_diagnostic.json"))
    a = ap.parse_args()
    a.query_prefix = a.query_prefix.replace("\\n", "\n")  # allow a literal \n in notebook args
    a.doc_prefix = a.doc_prefix.replace("\\n", "\n")

    from src.eval.dev_data import load_dev
    from src.retrieval.dense_encoder import DenseEncoder

    d = load_dev()   # TRAIN-split dev data; the test split is reserved for the official run
    corpus = dict(zip(d["doc_ids"], d["doc_texts"]))
    queries = dict(zip(d["query_ids"], d["query_texts"]))
    rel = {q: d["doc_ids"][i] for q, i in zip(d["query_ids"], d["rel_idx"])}
    if a.full:
        qids, docs = sorted(queries), sorted(corpus)
    else:
        random.seed(0)
        qids = random.sample(sorted(queries), a.n_queries)
        docs = sorted(set(rel[q] for q in qids) |
                      set(random.sample(sorted(corpus), a.n_docs - a.n_queries)))
    print(f"queries={len(qids)} docs={len(docs)}", flush=True)

    enc = DenseEncoder(a.model, a.device, a.max_seq_length, query_prefix=a.query_prefix,
                       doc_prefix=a.doc_prefix, trust_remote_code=a.trust_remote_code, fp16=a.fp16)
    t = time.time()
    D = enc.embed([a.doc_prefix + corpus[d] for d in docs], batch_size=a.batch_size,
                  show_progress_bar=True)
    print(f"docs encoded in {time.time() - t:.0f}s", flush=True)
    results = {}
    variants = [("with_prefix" if a.query_prefix else "no_prefix", a.query_prefix)]
    if a.also_no_prefix and a.query_prefix:
        variants.append(("no_prefix", ""))
    for name, pre in variants:
        t = time.time()
        Q = enc.embed([pre + queries[q] for q in qids], batch_size=a.batch_size,
                      show_progress_bar=True)
        m, ranks = metrics(Q @ D.T, qids, docs, rel)
        m["query_encode_s"] = round(time.time() - t, 1)
        results[name] = m
        print(f"\n[{name}] " + json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
        if len(docs) <= max(KS):
            print(f"  note: pool ({len(docs)}) <= {max(KS)}, so recall@{max(KS)} is 1.0 by construction")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "results": results}, indent=2))


if __name__ == "__main__":
    main()
