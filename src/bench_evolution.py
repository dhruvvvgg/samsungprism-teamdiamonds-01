"""Bonus benchmark: evolutionary retrieval across all versions, with lineage tracking.

Run after src/build_version_index.py (it needs that index and its fixture):

    python src/build_version_index.py --mock-encoder --n-snippets 100
    python src/bench_evolution.py --mock-encoder

Two questions, measured rather than asserted:

1. Does collapsing lineages actually clear the top-10? Several near-identical versions of one snippet
   otherwise share the slots between them and push different snippets out. Reported as duplicate slots
   in the top-10 with collapsing off vs on, and as distinct lineages surfaced.

2. For a query that names something a particular version introduced, does that version rank first?
   Reported twice, because a renamed identifier persists into later versions too: `exact` (top-1 is the
   version that introduced it) and `token_consistent` (top-1 is any version that still contains it).
   The second is the fair number; the first is reported alongside so the gap stays visible.

It goes through SearchService, so it measures the serving path the CLI and the API use.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", dest="index_dir", default="versions",
                    help="index name (versions) or a path")
    ap.add_argument("--index-dir", dest="index_dir", default=argparse.SUPPRESS,
                    help=argparse.SUPPRESS)
    ap.add_argument("--fixture", default=str(ROOT / "results" / "version_fixture.json"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mock-encoder", action="store_true")
    ap.add_argument("--limit-queries", type=int, default=0, help="0 = all fixture queries")
    ap.add_argument("--out", default=str(ROOT / "results" / "evolution_benchmark.json"))
    a = ap.parse_args()

    from src.search_service import SearchService
    from src.utils_io import write_json_atomic
    from src.versioning.version_index import duplication_at_k, parse_doc_id

    fpath = Path(a.fixture)
    if not fpath.exists():
        raise SystemExit(f"No fixture at {fpath}. Build the versioned index first:\n"
                         f"  python src/build_version_index.py --device cuda")
    fx = json.loads(fpath.read_text(encoding="utf-8"))
    svc = SearchService(a.index_dir, device=a.device, mock=a.mock_encoder)
    if not svc.versioned:
        raise SystemExit(f"{a.index_dir} is not a versioned index (no versions.json). Build it with "
                         f"src/build_version_index.py.")
    queries = fx["queries"][:a.limit_queries] if a.limit_queries else fx["queries"]
    print(f"[bonus] index {a.index_dir}: {svc.index.manifest['n_docs']} rows "
          f"({len({v['snippet_id'] for v in svc.index.versions})} lineages)")
    print(f"[bonus] {len(queries)} queries, k={a.k}", flush=True)

    dup_off, dup_on, distinct_off, distinct_on = [], [], [], []
    lineage_hit_off, lineage_hit_on, n_generic = 0, 0, 0
    exact, token_consistent, targeted_hit, n_specific = 0, 0, 0, 0
    t0 = time.time()
    for i, q in enumerate(queries):
        raw = svc.search(q["text"], k=a.k, all_versions=True)
        col = svc.search(q["text"], k=a.k)
        d_off = duplication_at_k(
            [(h["doc_id"], h["score"], svc.index.row_of(h["doc_id"])) for h in raw["hits"]],
            svc.index, k=a.k)
        d_on = duplication_at_k(
            [(h["doc_id"], h["score"], svc.index.row_of(h["doc_id"])) for h in col["hits"]],
            svc.index, k=a.k)
        dup_off.append(d_off["duplicate_pct"])
        dup_on.append(d_on["duplicate_pct"])
        distinct_off.append(d_off["distinct_lineages"])
        distinct_on.append(d_on["distinct_lineages"])
        if q["kind"] == "generic":
            n_generic += 1
            lineage_hit_off += int(any(h.get("snippet_id") == q["target_snippet"] for h in raw["hits"]))
            lineage_hit_on += int(any(h.get("snippet_id") == q["target_snippet"] for h in col["hits"]))
        else:
            n_specific += 1
            top = raw["hits"][0] if raw["hits"] else None
            if top:
                sid, ver = parse_doc_id(top["doc_id"])
                if sid == q["target_snippet"]:
                    exact += int(ver == q["target_version"])
                    token_consistent += int(ver in q.get("acceptable_versions", [q["target_version"]]))
            # and with the version filter applied, which is the version-targeted retrieval path
            tgt = svc.search(q["text"], k=a.k, version=q["target_version"])
            targeted_hit += int(bool(tgt["hits"])
                                and tgt["hits"][0].get("snippet_id") == q["target_snippet"])
        if (i + 1) % 50 == 0:
            print(f"[bonus]   {i + 1}/{len(queries)} queries", flush=True)
    elapsed = time.time() - t0

    def mean(xs):
        return round(statistics.fmean(xs), 2) if xs else None

    out = {
        "index_dir": str(a.index_dir), "fixture": fx["meta"], "k": a.k,
        "model": svc.index.manifest.get("model"), "mock_encoder": bool(a.mock_encoder),
        "n_queries": len(queries), "seconds": round(elapsed, 2),
        "duplication_at_k": {
            "all_versions_duplicate_pct_mean": mean(dup_off),
            "collapsed_duplicate_pct_mean": mean(dup_on),
            "all_versions_distinct_lineages_mean": mean(distinct_off),
            "collapsed_distinct_lineages_mean": mean(distinct_on),
        },
        "generic_queries": {
            "n": n_generic,
            f"lineage_recall_at_{a.k}_all_versions": round(lineage_hit_off / max(n_generic, 1), 4),
            f"lineage_recall_at_{a.k}_collapsed": round(lineage_hit_on / max(n_generic, 1), 4),
        },
        "version_specific_queries": {
            "n": n_specific,
            "top1_exact_version": round(exact / max(n_specific, 1), 4),
            "top1_token_consistent_version": round(token_consistent / max(n_specific, 1), 4),
            "version_filtered_top1_correct_snippet": round(targeted_hit / max(n_specific, 1), 4),
        },
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("\n" + "=" * 70)
    print("[bonus] EVOLUTIONARY RETRIEVAL")
    print("=" * 70)
    d = out["duplication_at_k"]
    print(f"  top-{a.k} duplicate slots : {d['all_versions_duplicate_pct_mean']}% with all versions  "
          f"->  {d['collapsed_duplicate_pct_mean']}% collapsed")
    print(f"  distinct lineages shown  : {d['all_versions_distinct_lineages_mean']}  ->  "
          f"{d['collapsed_distinct_lineages_mean']} (out of {a.k} slots)")
    g = out["generic_queries"]
    print(f"  generic queries ({g['n']}): lineage recall@{a.k} "
          f"{g[f'lineage_recall_at_{a.k}_all_versions']}  ->  collapsed "
          f"{g[f'lineage_recall_at_{a.k}_collapsed']}")
    v = out["version_specific_queries"]
    print(f"  version-specific ({v['n']}): top-1 is the exact version "
          f"{v['top1_exact_version']}, a token-consistent version {v['top1_token_consistent_version']}")
    print(f"  with --version filter    : correct snippet at rank 1 for "
          f"{v['version_filtered_top1_correct_snippet']}")
    if a.mock_encoder:
        print("  NOTE: mock encoder -- structural numbers (duplication) are real, retrieval quality "
              "is not.")
    write_json_atomic(a.out, out, indent=2)
    print(f"\n  written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
