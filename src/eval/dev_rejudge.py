"""Step 4 (Phase 6, LLM re-judge only): re-order the top few candidates by an LLM's "does this code solve
this problem?" score, only when the top-1 / top-2 margin is small.

Runs on the CURRENT chosen pipeline (outputs/dev/chosen.json: query variants / hybrid / rerank). All LLM
traffic goes through src.agent.llm_client.call_llm (set LLM_PROVIDER + its API key in the notebook env),
and every call is cached on disk, so the sweep re-uses judged pairs: the first pass runs the most
permissive setting (largest margin threshold and judge_top), which fills the cache for every later row.

    !python src/eval/dev_rejudge.py --device cuda --n-queries 300            # real LLM (needs a key)
    !python src/eval/dev_rejudge.py --device cuda --n-queries 300 --mock     # plumbing check, no key

Do not read --mock numbers as results: the mock judge is a word-overlap heuristic.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    from src.eval.dev_lib import add_common_args
    add_common_args(ap)
    ap.add_argument("--mock", action="store_true", help="LLM_PROVIDER=mock (no key, no network)")
    ap.add_argument("--margins", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--judge-tops", nargs="+", type=int, default=[2, 3, 5])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.3, 0.5, 0.7, 1.0])
    ap.set_defaults(n_queries=300)    # LLM calls cost money; 0 still means the whole tune set
    args = ap.parse_args()
    if args.mock:
        os.environ["LLM_PROVIDER"] = "mock"

    import numpy as np

    from src.agent.llm_rejudge import rejudge
    from src.eval import dev_metrics as M
    from src.eval.dev_lib import BASELINE_VARIANT, DevContext, load_base_config, record, render_table
    from src.retrieval.pipeline import run_stages

    ctx = DevContext(args)
    cfg = load_base_config(args.config)
    cfg["rejudge"]["enabled"] = False
    qtexts = [ctx.query_texts[i] for i in ctx.idx]
    S_ref, _ = ctx.query_scores([BASELINE_VARIANT])
    ref = M.ranks_from_scores(S_ref, ctx.rel_of())
    S, _ = ctx.query_scores(cfg["dense_variants"])
    Sb = None
    if cfg["bm25"]["enabled"]:
        from src.retrieval.bm25_index import build_bm25, tokenize_code
        bm = build_bm25(ctx.doc_texts, cfg["bm25"]["keep_whole"], cfg["bm25"]["k1"], cfg["bm25"]["b"])
        Sb = bm.scores([tokenize_code(t, cfg["bm25"]["keep_whole"]) for t in qtexts])
    scorer = None
    if cfg["rerank"]["enabled"]:
        from src.retrieval.reranker import load_reranker
        rc = cfg["rerank"]
        ctx.release_encoder()   # F2LLM's job is done; free its GPU memory before the reranker loads
        scorer = load_reranker(rc["name"], args.device, rc["dtype"], rc["max_length"], rc["instruction"])
    lists, scores, _ = run_stages(S, Sb, qtexts, ctx.doc_texts, cfg, scorer)
    base = ctx.ranks_from_lists(lists)
    print(f"[rejudge] base pipeline NDCG@10 {M.summarize(base)['ndcg@10']:.4f} on {len(base)} queries "
          f"(provider={os.environ.get('LLM_PROVIDER', 'openai')})", flush=True)

    combos = sorted(((m, j, b) for m in args.margins for j in args.judge_tops for b in args.betas),
                    key=lambda c: (-c[0], -c[1]))           # most permissive first: fills the disk cache
    call_seconds = None
    for margin, judge_top, beta in combos:
        t0 = time.time()
        new, infos = [], []
        for i, q in enumerate(qtexts):
            idx, _, info = rejudge(q, lists[i], scores[i], ctx.doc_texts, judge_top=judge_top,
                                   margin_z_threshold=margin, beta=beta)
            new.append(idx)          # full reordered list (judged block re-sorted, tail untouched)
            infos.append(info)
        secs = time.time() - t0
        calls = sum(x["n_calls"] for x in infos)
        if call_seconds is None and calls:
            call_seconds = secs / calls           # first pass = uncached: real per-call latency
        trig = np.mean([x["triggered"] for x in infos])
        est_ms = 1000 * (call_seconds or 0.0) * calls / len(qtexts)
        patch = {"rejudge": {"enabled": True, "judge_top": judge_top, "margin_z": margin, "beta": beta}}
        record(ctx, f"rejudge margin_z<{margin} top={judge_top} beta={beta}", "rejudge",
               ctx.ranks_from_lists(new), base, patch, latency_ms=est_ms, ref_ranks=ref,
               extra={"triggered_frac": float(trig), "calls_per_query": calls / len(qtexts)},
               notes=f"triggered {trig:.0%}, {calls / len(qtexts):.2f} calls/query; latency = measured "
                     f"s/call x calls/query" + (" [MOCK]" if args.mock else ""))
    render_table()


if __name__ == "__main__":
    main()
