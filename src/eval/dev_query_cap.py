"""What does capping the query length cost in quality? (dev protocol, train split, never test)

    !python src/eval/dev_query_cap.py --preset f2llm-v2-1.7b --device cuda --caps 256 512 1024

CPU serving is dominated by query encoding -- measured on Kaggle, p50 9.3 s per query for the 1.7B,
with the vector search itself at ~5 ms. Attention cost grows faster than linearly in sequence length, so
capping the query is the biggest available lever. The question is what it costs in retrieval quality,
and that has to be measured on dev before it is offered as a serving option.

What this does:
  * takes the SAME 1,000-query dev slice every other experiment used (seed 7, tune partition);
  * reuses the existing document embeddings -- documents are NEVER truncated, and nothing about them
    changes when a query cap is applied, so the doc side is a cache hit;
  * re-encodes ONLY those 1,000 queries at each cap, plus once uncapped as the in-session baseline;
  * reports NDCG@10 / MRR@10 per cap, the delta against uncapped with a 95% paired-bootstrap CI, and
    how many queries each cap actually truncates.

Reference for the 1.7B on this slice, uncapped: NDCG@10 0.9299 / MRR@10 0.9167. The in-session uncapped
row should reproduce it; if it does not, something else changed and the cap deltas are not trustworthy.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval import dev_metrics as M  # noqa: E402
from src.eval.dev_lib import BASELINE_VARIANT, DevContext, add_common_args, print_code_version  # noqa: E402
from src.retrieval.query_variants import format_query  # noqa: E402
from src.utils_io import write_json_atomic  # noqa: E402

UNCAPPED_REFERENCE = {"f2llm-v2-1.7b": {"ndcg@10": 0.9299, "mrr@10": 0.9167},
                      "f2llm-v2-0.6b": {"ndcg@10": 0.8959, "mrr@10": 0.8757}}


def encode_at_cap(ctx, texts, cap, batch_size):
    """Embed `texts` with the query cap applied (None = the preset's full length). Returns
    (embeddings, seconds). Truncation is a tokenizer setting, so this is the same model either way."""
    from src.retrieval.dense_encoder import encode_length_sorted
    full = ctx.preset["max_seq_length"]
    ctx.encoder.set_max_seq_length(cap or full)
    t0 = time.time()
    emb = encode_length_sorted(list(texts), lambda t: ctx.encoder.embed(t, batch_size=batch_size),
                               batch_size)
    return emb, time.time() - t0


def eos_survives_truncation(ctx, text, cap):
    """True if a query longer than `cap` still ends in EOS after truncation.

    This matters more than it looks: F2LLM pools the LAST token, so if truncation cut the EOS off, the
    embedding would be read from an arbitrary interior token and the cap would look catastrophic for
    reasons that have nothing to do with losing text. Checked rather than assumed."""
    tok = ctx.encoder.model.tokenizer
    ids = tok(text, truncation=True, max_length=cap)["input_ids"]
    return len(ids) <= cap and ids[-1] == tok.eos_token_id, len(ids)


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--caps", nargs="+", type=int, default=[256, 512, 1024],
                    help="query token caps to evaluate")
    ap.add_argument("--out", default=str(ROOT / "results" / "query_cap_sweep.json"))
    ap.set_defaults(n_queries=1000, preset="f2llm-v2-1.7b")
    args = ap.parse_args()
    print_code_version("query-cap")
    ctx = DevContext(args)
    ref = UNCAPPED_REFERENCE.get(args.preset)
    print(f"[cap] preset {args.preset}, {len(ctx.idx)} queries from the {ctx.split_name} partition, "
          f"full corpus ({len(ctx.doc_texts)} docs)")

    t0 = time.time()
    D, dmeta = ctx.doc_emb()                    # documents: cache hit if dev_split.py already ran
    print(f"[cap] document embeddings: {D.shape} "
          f"({'cache hit' if dmeta.get('cache_hit') else 'freshly encoded'}, "
          f"{time.time() - t0:.1f}s). Documents are never truncated by a query cap.", flush=True)

    texts = [format_query(ctx.query_texts[i], BASELINE_VARIANT) for i in ctx.idx]
    rel = ctx.rel_of()
    tokens = np.array([ctx.encoder.token_count(t) for t in texts])
    print(f"[cap] query prompt length in tokens: mean {tokens.mean():.0f}, p50 {np.percentile(tokens, 50):.0f}, "
          f"p90 {np.percentile(tokens, 90):.0f}, p99 {np.percentile(tokens, 99):.0f}, max {tokens.max()}",
          flush=True)

    rows, base_ranks = [], None
    for cap in [None] + list(args.caps):
        label = "uncapped" if cap is None else f"cap {cap}"
        affected = int((tokens > cap).sum()) if cap else 0
        if cap:
            longest = texts[int(np.argmax(tokens))]
            ok, n_ids = eos_survives_truncation(ctx, longest, cap)
            verdict = ("OK, still ends in EOS" if ok else
                       "FAILED -- last-token pooling would read the wrong position")
            print(f"[cap] {label}: EOS check on the longest query ({tokens.max()} tokens -> {n_ids}): "
                  f"{verdict}", flush=True)
        emb, secs = encode_at_cap(ctx, texts, cap, args.batch_size)
        ranks = M.ranks_from_scores(emb @ D.T, rel)
        s = M.summarize(ranks)
        row = {"cap": cap, "label": label, "ndcg@10": s["ndcg@10"], "mrr@10": s["mrr@10"],
               "recall@10": s["recall@10"], "recall@100": s["recall@100"],
               "queries_truncated": affected,
               "queries_truncated_pct": round(100.0 * affected / len(texts), 1),
               "encode_seconds": round(secs, 1),
               "encode_ms_per_query": round(1000 * secs / len(texts), 1)}
        if base_ranks is None:
            base_ranks = ranks
            if ref:
                gap = s["ndcg@10"] - ref["ndcg@10"]
                row["reference_ndcg@10"] = ref["ndcg@10"]
                row["reference_gap"] = round(gap, 4)
                print(f"[cap] uncapped in-session NDCG@10 {s['ndcg@10']:.4f} vs the recorded "
                      f"{ref['ndcg@10']:.4f} -> {gap:+.4f}"
                      + ("  OK" if abs(gap) < 0.002 else
                         "  <-- DOES NOT REPRODUCE; treat the cap deltas below with suspicion"),
                      flush=True)
        else:
            cmp = M.compare(ranks, base_ranks)
            row.update({"delta_ndcg@10": round(cmp["delta_ndcg@10"], 4),
                        "ci_low": round(cmp["ci_low"], 4), "ci_high": round(cmp["ci_high"], 4),
                        "improved": cmp["improved"], "worsened": cmp["worsened"],
                        "significant_loss": cmp["ci_high"] < 0})
        rows.append(row)
        print(f"[cap] {label:10s} NDCG@10 {s['ndcg@10']:.4f}  MRR@10 {s['mrr@10']:.4f}  "
              f"truncated {affected}/{len(texts)}  encode {secs:.0f}s "
              f"({row['encode_ms_per_query']:.0f} ms/query on {args.device})", flush=True)

    print("\n" + "=" * 96)
    print(f"[cap] QUERY LENGTH CAP vs QUALITY  ({args.preset}, {len(texts)}-query dev slice)")
    print("=" * 96)
    print(f"  {'cap':>9} | {'NDCG@10':>8} | {'MRR@10':>8} | {'dNDCG':>8} | {'95% CI':>18} | "
          f"{'truncated':>10} | {'ms/query':>9}")
    for r in rows:
        d = "" if r["cap"] is None else f"{r['delta_ndcg@10']:+.4f}"
        ci = "" if r["cap"] is None else f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"  {r['label']:>9} | {r['ndcg@10']:8.4f} | {r['mrr@10']:8.4f} | {d:>8} | {ci:>18} | "
              f"{r['queries_truncated_pct']:9.1f}% | {r['encode_ms_per_query']:9.0f}")
    print("\n  A cap is only worth taking if the quality it costs is smaller than the latency it buys;")
    print("  'significant_loss' is true when the whole 95% CI sits below zero.")
    write_json_atomic(args.out, {"preset": args.preset, "n_queries": len(texts),
                                 "split": ctx.split_name, "device": args.device,
                                 "token_stats": {"mean": float(tokens.mean()),
                                                 "p50": float(np.percentile(tokens, 50)),
                                                 "p90": float(np.percentile(tokens, 90)),
                                                 "p99": float(np.percentile(tokens, 99)),
                                                 "max": int(tokens.max())},
                                 "rows": rows, "when": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2)
    print(f"\n  written to {args.out}")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
