"""Step 3 (Phase 3): rerank the top-k (10/20/30) of the current first stage with a cross-encoder / LLM
reranker, and interpolate the reranker score with the first-stage score.

The reranker scores the top-KMAX candidates of every query ONCE (cached to ./data/cache/rerank); k in
{10,20,30} then just uses the first k candidates (a pair's score does not depend on the other candidates),
and the interpolation weight alpha is swept offline:  final = alpha*z(first stage) + (1-alpha)*z(reranker).
Expensive, so it runs on a fixed seeded subsample of the tune set (--n-queries, default 1000).

GPU memory on a T4 (14.56 GB): the dense F2LLM model is released (weights freed, CUDA cache emptied)
right after the first-stage dense/hybrid scores are computed and BEFORE any reranker loads -- the two
models were both resident at once before this fix, which is what caused the OOM. Pair-scoring batch
size is `--rerank-batch-size` (default 8; was 16, halved as a further safety margin on top of the
release). Qwen3-Reranker also now asks the model for only the LAST position's logits
(`logits_to_keep=1`) instead of materializing logits for the whole sequence over its 151,936-token
vocab and discarding all but the last row -- with 2048-token sequences that discarded tensor alone
was themselves gigabytes per batch.

    !python src/eval/dev_rerank.py --device cuda --rerankers qwen3-reranker-0.6b bge-reranker-v2-m3

--preset (inherited from add_common_args, default f2llm-v2-0.6b) selects the FIRST-STAGE dense model,
e.g. --preset f2llm-v2-1.7b. Every row this script writes is tagged with that preset (dev_lib.record's
row_preset), so different presets' sweeps never collide or get mixed together by dev_select.py.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval import dev_metrics as M  # noqa: E402
from src.eval.dev_lib import (BASELINE_VARIANT, CacheMissError, DevContext, add_common_args,  # noqa: E402
                              cached_embeddings, load_base_config, print_code_version, record, render_table)
from src.retrieval.hybrid_encoder import score_average_fuse  # noqa: E402
from src.retrieval.pipeline import first_stage  # noqa: E402
from src.retrieval.reranker import RERANKERS, load_reranker, rerank_order, rerank_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "rerank"
K_LIST = [10, 20, 30]
ALPHAS = [0.0, 0.25, 0.5, 0.75]


def cached_rerank(name, instruction, max_length, dtype, args, ctx, cand_lists, kmax):
    """Reranker scores for the top-kmax candidates of each query: (n, kmax) array, total seconds."""
    # cast every numeric piece to plain Python types: ctx.idx and doc indices inside cand_lists can be
    # numpy int64 (from np.argpartition/np.array), which json.dumps refuses outright.
    key_src = json.dumps([name, instruction, max_length, dtype,
                          [ctx.data["query_ids"][int(i)] for i in ctx.idx],
                          [[int(d) for d in l[:kmax]] for l in cand_lists]])
    path = CACHE / f"{name}__{instruction}__{hashlib.sha256(key_src.encode()).hexdigest()[:16]}.npz"
    if path.exists():
        z = np.load(path)
        print(f"[rerank] cache hit {path.name}", flush=True)
        return z["scores"], float(z["seconds"])
    scorer = load_reranker(name, args.device, dtype, max_length, instruction)
    qtexts = [ctx.query_texts[i] for i in ctx.idx]
    print(f"[rerank] scoring {len(qtexts) * kmax} pairs with {name} ({instruction}) ...", flush=True)
    rr, secs = rerank_scores(scorer, qtexts, [l[:kmax] for l in cand_lists], ctx.doc_texts,
                             batch_size=args.rerank_batch_size)
    scores = np.stack([np.pad(r, (0, kmax - len(r)), constant_values=-1e9) for r in rr])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, scores=scores, seconds=secs)
    del scorer
    return scores, secs


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--rerankers", nargs="+", default=["qwen3-reranker-0.6b", "bge-reranker-v2-m3"],
                    choices=sorted(RERANKERS))
    ap.add_argument("--instructions", nargs="+", default=["apps", "card"],
                    help="Qwen3 reranker instruction wording(s) to try (ignored by BGE)")
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--rerank-batch-size", type=int, default=8,
                    help="pairs per reranker forward pass; lower if you still see CUDA OOM "
                         "(30,000 pairs at max_length tokens through a 0.6B model adds up fast)")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--ks", nargs="+", type=int, default=K_LIST,
                    help="rerank depths to sweep (each must be <= --kmax, which is what actually gets "
                         "scored); narrow this to the promising region to avoid paying for cells you "
                         "have already ruled out")
    ap.add_argument("--alphas", nargs="+", type=float, default=ALPHAS,
                    help="interpolation weights to sweep: alpha*z(first stage) + (1-alpha)*z(reranker). "
                         "Sweeping is free (the reranker scores each pair once), so pass a fine grid "
                         "whenever the first stage changes and the old best alpha may no longer hold")
    ap.add_argument("--fuse-with", default=None,
                    help="second preset whose CACHED embeddings are score-averaged into the first stage "
                         "(dense+dense fusion), e.g. f2llm-v2-0.6b")
    ap.add_argument("--fuse-w-b", type=float, default=0.25, help="weight on --fuse-with's model")
    ap.add_argument("--fuse-norm", default="zscore", choices=["zscore", "minmax"])
    ap.add_argument("--compare-to-rerank", action="store_true",
                    help="judge the adoption rule against the ALREADY-RERANKED pipeline on the unfused "
                         "first stage (--base-reranker/--base-k/--base-alpha) instead of against the raw "
                         "first stage -- i.e. 'does this beat the standing best', computed here on the "
                         "same queries so the paired comparison is valid")
    ap.add_argument("--base-reranker", default="qwen3-reranker-0.6b", choices=sorted(RERANKERS))
    ap.add_argument("--base-instruction", default="apps")
    ap.add_argument("--base-k", type=int, default=30)
    ap.add_argument("--base-alpha", type=float, default=0.5)
    # default to the 1000-query slice (reranking is expensive) while leaving --n-queries 0 meaning
    # "the whole tune set", as add_common_args documents. Overriding a falsy 0 back to 1000 here would
    # silently ignore an explicit request for the full set and hand back a 1000-query answer.
    ap.set_defaults(n_queries=1000)
    args = ap.parse_args()
    print_code_version("rerank")
    bad_ks = [k for k in args.ks if k > args.kmax]
    if bad_ks:
        raise SystemExit(f"--ks {bad_ks} exceed --kmax {args.kmax}; only the top {args.kmax} candidates "
                         f"are scored, so those cells cannot be evaluated. Raise --kmax or drop them.")
    ctx = DevContext(args)
    cfg = load_base_config(args.config)
    cfg["rerank"]["enabled"] = False

    S_ref, _ = ctx.query_scores([BASELINE_VARIANT])
    ref = M.ranks_from_scores(S_ref, ctx.rel_of())
    S_a, _ = ctx.query_scores(cfg["dense_variants"])
    Sb = None
    if cfg["bm25"]["enabled"]:
        from src.retrieval.bm25_index import build_bm25, tokenize_code
        bm = build_bm25(ctx.doc_texts, cfg["bm25"]["keep_whole"], cfg["bm25"]["k1"], cfg["bm25"]["b"])
        Sb = bm.scores([tokenize_code(ctx.query_texts[i], cfg["bm25"]["keep_whole"]) for i in ctx.idx])

    # the unfused first stage is always built: it is what --compare-to-rerank's base is computed from
    fs_a = [first_stage(S_a[i], None if Sb is None else Sb[i], cfg) for i in range(len(S_a))]
    lists_a, scores_a = [f[0] for f in fs_a], [f[1] for f in fs_a]

    S = S_a
    if args.fuse_with:
        try:
            doc_emb, query_emb = cached_embeddings(args.fuse_with, ctx.doc_texts, ctx.query_texts,
                                                   BASELINE_VARIANT)
        except CacheMissError as exc:
            raise SystemExit(f"\n[rerank] CACHE MISS for --fuse-with {args.fuse_with!r} -- nothing was "
                             f"encoded, no GPU time spent.\n  {exc}\n") from exc
        S_b = query_emb[ctx.idx] @ doc_emb.T
        S = score_average_fuse(S_a, S_b, w_b=args.fuse_w_b, how=args.fuse_norm)
        print(f"[rerank] first stage = score-avg fusion of {args.preset} + {args.fuse_with} "
              f"(w_B={args.fuse_w_b}, {args.fuse_norm}); the fused score is NOT a raw cosine, so alpha is "
              f"swept fresh over {args.alphas}", flush=True)
        fs = [first_stage(S[i], None if Sb is None else Sb[i], cfg) for i in range(len(S))]
        lists, scores = [f[0] for f in fs], [f[1] for f in fs]
    else:
        lists, scores = lists_a, scores_a

    first_ranks = ctx.ranks_from_lists(lists)
    print(f"[rerank] first stage NDCG@10 {M.summarize(first_ranks)['ndcg@10']:.4f} on {len(first_ranks)} "
          f"queries; recall@{args.kmax} of the candidate pool = {(first_ranks <= args.kmax).mean():.3f} "
          f"(ceiling for any reranker at k={args.kmax})", flush=True)

    ctx.release_encoder()   # F2LLM's job is done; free its GPU memory before a reranker loads

    base, base_desc = first_ranks, "first stage (unreranked)"
    if args.compare_to_rerank:
        rr_b, _ = cached_rerank(args.base_reranker, args.base_instruction, args.max_length, args.dtype,
                                args, ctx, lists_a, args.kmax)
        base_lists = []
        for l, s, r in zip(lists_a, scores_a, rr_b):
            head, _ = rerank_order(l[:args.base_k], s[:args.base_k], r[:args.base_k], args.base_alpha)
            base_lists.append(head + l[args.base_k:])
        base = ctx.ranks_from_lists(base_lists)
        base_desc = (f"standing best: unfused first stage + {args.base_reranker}[{args.base_instruction}] "
                     f"k={args.base_k} alpha={args.base_alpha}")
        print(f"[rerank] ADOPTION BASE = {base_desc}: NDCG@10 {M.summarize(base)['ndcg@10']:.4f} "
              f"(every row below is judged against THIS, not against the raw first stage)", flush=True)
        record(ctx, f"BASE {base_desc}", "rerank", base, first_ranks, {}, latency_ms=None, ref_ranks=ref,
               notes="reference row, not a candidate (the base every other row is judged against)")

    for name in args.rerankers:
        instrs = args.instructions if RERANKERS[name]["kind"] == "qwen3" else ["-"]
        for ins in instrs:
            rr, secs = cached_rerank(name, ins, args.max_length, args.dtype, args, ctx, lists, args.kmax)
            for k in [k for k in args.ks if k <= args.kmax]:
                ms = 1000 * secs * (k / args.kmax) / len(lists)   # pairs scale linearly with k
                for a in args.alphas:
                    new = []
                    for l, s, r in zip(lists, scores, rr):
                        head, _ = rerank_order(l[:k], s[:k], r[:k], a)
                        new.append(head + l[k:])
                    patch = {"rerank": {"enabled": True, "name": name, "k": k, "alpha": a,
                                        "instruction": ins if ins != "-" else "apps",
                                        "max_length": args.max_length, "dtype": args.dtype,
                                        "batch_size": args.rerank_batch_size}}
                    tag = (f"fused({args.preset}+{args.fuse_with} w_B={args.fuse_w_b}) + "
                           if args.fuse_with else "")
                    note = "latency pro-rated from the k=%d pair-scoring pass" % args.kmax
                    if args.fuse_with:
                        note += ("; first stage is score-avg fusion, so this row is NOT adoptable into "
                                 "chosen.json (fusion is not wired into the official pipeline)")
                    extra = {"first_stage": "fused" if args.fuse_with else "dense",
                             "adoption_base": base_desc}
                    if args.fuse_with:
                        extra.update({"fuse_with": args.fuse_with, "fuse_w_b": args.fuse_w_b,
                                      "fuse_norm": args.fuse_norm})
                    record(ctx, f"{tag}{name}[{ins}] k={k} alpha={a}", "rerank",
                           ctx.ranks_from_lists(new), base,
                           {} if args.fuse_with else patch, latency_ms=ms, ref_ranks=ref,
                           extra=extra, notes=note)
    render_table()


if __name__ == "__main__":
    main()
