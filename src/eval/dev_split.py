"""Step 0: build the dev protocol and cache F2LLM embeddings, then score the baseline.

  * 5,000 CoIR-APPS *train* queries vs the FULL 8,765-doc corpus, *train* qrels only.
  * fixed-seed (13) 1,000-query holdout, reserved; sweeps use the other 4,000 ("tune").
  * caches fp16 doc + query embeddings under ./data/cache/embeddings (rebuilt if missing).
  * reports NDCG@10, MRR@10, recall@1/10/100 (+@1000) for all 5,000 / tune / holdout, and checks the dev
    baseline against the published TEST NDCG@10 for this preset: a large gap would mean the dev protocol
    is contaminated (the model may have seen train pairs) and dev sweeps cannot be trusted.
  * on CUDA, reports PEAK GPU memory (torch.cuda.max_memory_allocated/reserved, reset before the model
    is even constructed so weight-loading memory counts too) and total encoding time for 8,765 docs +
    5,000 queries -- a real measured number, not an estimate; a cache hit skips real encoding and the
    peak/time will read near-zero (the printed line says so explicitly).

    !python src/eval/dev_split.py --preset f2llm-v2-4b --device cuda --batch-size 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval import dev_metrics as M  # noqa: E402
from src.eval.dev_lib import (BASELINE_VARIANT, TEST_NDCG_REFERENCE, DevContext, add_common_args,  # noqa: E402
                              print_code_version, record, render_table)
from src.retrieval.dense_encoder import cuda_peak_report, cuda_peak_reset  # noqa: E402
from src.retrieval.model_presets import memory_warning  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--memory-warn-mb", type=int, default=13000,
                    help="warn (never fail) before loading if the preset's registered memory footprint "
                         "is at/over this many MB; default leaves headroom under a T4's 14.56 GB")
    args = ap.parse_args()
    print_code_version("dev")
    args.n_queries = 0

    warning = memory_warning(args.preset, args.memory_warn_mb)
    if warning:
        print("\n" + "!" * 78)
        print("MEMORY WARNING (run continues anyway):\n  " + warning.replace("\n", "\n  "))
        print("!" * 78 + "\n", flush=True)

    ctx = DevContext(args)
    d = ctx.data
    print(f"[dev] corpus={len(d['doc_ids'])} docs | train queries={len(d['query_ids'])} | "
          f"tune={len(d['tune_idx'])} holdout={len(d['holdout_idx'])} (seed 13)", flush=True)

    cuda_peak_reset()   # before model construction, so weight-load memory is part of the reported peak
    D, dmeta = ctx.doc_emb()
    Q, qmeta = ctx.query_emb(BASELINE_VARIANT)
    total_encode_s = dmeta["encode_seconds"] + qmeta["encode_seconds"]
    cache_hit = dmeta["cache_hit"] and qmeta["cache_hit"]
    peak = cuda_peak_report("after encoding 8,765 docs + 5,000 queries"
                            + (" [CACHE HIT: no real encoding ran, this reading is not meaningful]"
                               if cache_hit else ""))
    print(f"[dev] doc embeddings {D.shape} (encode {dmeta['encode_seconds']}s, cache_hit={dmeta['cache_hit']}) | "
          f"query embeddings {Q.shape} (encode {qmeta['encode_seconds']}s, cache_hit={qmeta['cache_hit']}, "
          f"{1000 * qmeta['encode_seconds'] / len(Q):.1f} ms/query amortised) | "
          f"TOTAL encoding time = {total_encode_s:.1f}s", flush=True)

    S = Q @ D.T
    rel = np.array(d["rel_idx"])
    ranks = M.ranks_from_scores(S, rel)
    print("\nBaseline F2LLM (registry instruction, full statement):")
    for name, idx in (("all 5000", np.arange(len(ranks))), ("tune", np.array(d["tune_idx"])),
                      ("holdout", np.array(d["holdout_idx"]))):
        s = M.summarize(ranks[idx])
        print(f"  {name:9s} n={s['n']:5d} NDCG@10={s['ndcg@10']:.4f} MRR@10={s['mrr@10']:.4f} "
              f"R@1={s['recall@1']:.3f} R@10={s['recall@10']:.3f} R@100={s['recall@100']:.3f} "
              f"R@1000={(ranks[idx] <= 1000).mean():.3f}")
    tune = M.summarize(ranks[np.array(d["tune_idx"])])
    ref = TEST_NDCG_REFERENCE.get(args.preset)
    if ref is None:
        print(f"\n[dev-vs-test check] SKIPPED: no published test NDCG@10 on file for preset {args.preset!r} "
              f"(known: {sorted(TEST_NDCG_REFERENCE)}). Add it to TEST_NDCG_REFERENCE in dev_lib.py.")
        gap = None
    else:
        gap = tune["ndcg@10"] - ref
        print(f"\n[dev-vs-test check] tune NDCG@10 {tune['ndcg@10']:.4f} vs published TEST {ref:.4f} "
              f"(gap {gap:+.4f})")
        if gap > 0.03:
            print("  " + "!" * 70)
            print("  WARNING WARNING WARNING: dev is much easier than test (gap > 0.03). The model has "
                  "probably seen train pairs, so gains measured on dev may not transfer to test. This does "
                  "NOT block the run -- treat every later dev-sweep number as directional only.")
            print("  " + "!" * 70)
        elif gap < -0.03:
            print("  NOTE: dev is harder than test (train/test distribution difference); gains should transfer.")
        else:
            print("  OK: dev difficulty is close to test.")

    ctx.idx = np.array(d["tune_idx"])
    gap_note = "dev-vs-test gap unknown (no reference)" if gap is None else f"dev-vs-test gap {gap:+.4f}"
    if cache_hit:
        gap_note += " [encoding was a cache hit; peak_gpu/encode_seconds below are NOT meaningful]"
    record(ctx, f"F2LLM {args.preset} registry+full (baseline)", "baseline", ranks[ctx.idx], None, {},
           latency_ms=0.0, notes=gap_note,
           extra={"total_encode_seconds": round(total_encode_s, 1), "encode_cache_hit": cache_hit,
                  "peak_gpu_max_allocated_gb": None if peak is None else round(peak["max_allocated_gb"], 2),
                  "peak_gpu_max_reserved_gb": None if peak is None else round(peak["max_reserved_gb"], 2)})
    render_table()


if __name__ == "__main__":
    main()
