"""Step 1: query-side variants (instruction wording, stripping examples / I-O spec, embedding averages).

Each single variant is encoded once for all 5,000 train queries and cached (fp16); averages are computed
from the cached embeddings, so they cost nothing extra. Rows are compared against the baseline
(registry instruction + full statement) on the tune set.

    !python src/eval/dev_variants.py --device cuda
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval import dev_metrics as M  # noqa: E402
from src.eval.dev_lib import BASELINE_VARIANT, DevContext, add_common_args, record, render_table  # noqa: E402

SINGLES = ["registry+full", "contest+full", "code_contest+full", "solution+full",
           "registry+no_examples", "registry+narrative", "solution+no_examples"]
AVERAGES = {
    "avg(instructions, full)": ["registry+full", "contest+full", "code_contest+full", "solution+full"],
    "avg(texts, registry)": ["registry+full", "registry+no_examples", "registry+narrative"],
    "avg(registry+full, solution+no_examples)": ["registry+full", "solution+no_examples"],
    "avg(all singles)": SINGLES,
}


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--only", nargs="*", help="restrict to these experiment names")
    args = ap.parse_args()
    ctx = DevContext(args)
    base_S, _ = ctx.query_scores([BASELINE_VARIANT])
    base = M.ranks_from_scores(base_S, ctx.rel_of())
    print("| group | experiment | NDCG@10 | MRR@10 | R@1 | R@10 | R@100 | improved/worsened | dNDCG@10 [95% CI] "
          "| +ms/query | adopt | set |", flush=True)
    exps = [(v, [v]) for v in SINGLES if v != BASELINE_VARIANT] + list(AVERAGES.items())
    for name, variants in exps:
        if args.only and name not in args.only:
            continue
        S, added_ms = ctx.query_scores(variants)
        ranks = M.ranks_from_scores(S, ctx.rel_of())
        record(ctx, name, "variants", ranks, base, {"dense_variants": variants}, latency_ms=added_ms,
               notes=("averaged embeddings of " + ", ".join(variants)) if len(variants) > 1 else "")
    render_table()


if __name__ == "__main__":
    main()
