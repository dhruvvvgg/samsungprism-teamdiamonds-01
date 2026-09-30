"""Ingest a real repository's history and build a served index over every version of every function.

    # the demo repo (clones pallets/click on first use), last 40 mainline commits
    python src/build_history_index.py --device cuda --max-commits 40

    # any local repository, no network
    python src/build_history_index.py --repo /path/to/checkout --subdir src --mock-encoder

Writes `history_index/` (a versioned runtime index: embeddings + versions.json + chunks.json) and
`results/history_queries.json` (the auto-generated version-targeted queries with their known answers),
so the benchmark and the CLI/API/UI all work off the same artefacts.

Embedding is content-hash deduplicated across the whole history: a function that did not change between
two commits is embedded once, which is the entire point of the content-hash index.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None,
                    help="path to a local git checkout; default clones the demo repo")
    ap.add_argument("--repo-url", default=None, help="clone this URL instead of the demo repo")
    ap.add_argument("--subdir", default=None,
                    help="limit to a subdirectory (default: the demo repo's package dir)")
    ap.add_argument("--max-commits", type=int, default=40,
                    help="most recent N mainline commits (0 = the whole history)")
    ap.add_argument("--track-renames", action="store_true",
                    help="join a renamed file or a renamed / moved function to its earlier lineage "
                         "(git rename detection + AST body similarity). OFF by default")
    ap.add_argument("--rename-threshold", type=float, default=0.8,
                    help="minimum body similarity (Jaccard of normalised tokens) for a function link")
    ap.add_argument("--out", default="history")
    ap.add_argument("--queries-out", default=str(ROOT / "results" / "history_queries.json"))
    ap.add_argument("--stats-out", default=str(ROOT / "results" / "history_ingest.json"))
    ap.add_argument("--preset", default="f2llm-v2-0.6b",
                    help="the lite model by default: this index is for demos and P1/Bonus, not for P0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mock-encoder", action="store_true")
    ap.add_argument("--clone-dir", default=str(ROOT / "data" / "repos"))
    a = ap.parse_args()

    from src.runtime_index import make_doc_encoder, resolve_index_dir
    from src.utils_io import write_json_atomic
    from src.versioning.git_history import DEFAULT_REPO, DEFAULT_SUBDIR, ensure_repo, ingest
    from src.versioning.history_queries import build_queries, query_stats
    from src.versioning.version_index import build_versioned_index, embeddings_for

    if a.repo:
        repo, subdir = Path(a.repo), a.subdir
    else:
        url = a.repo_url or DEFAULT_REPO
        name = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        print(f"[history] cloning/reusing {url} -> {Path(a.clone_dir) / name}", flush=True)
        repo = ensure_repo(url, Path(a.clone_dir) / name)
        subdir = a.subdir if a.subdir is not None else (DEFAULT_SUBDIR if url == DEFAULT_REPO else None)
    print(f"[history] repo {repo}" + (f" (subdir {subdir})" if subdir else ""), flush=True)

    rows, per_commit, stats = ingest(repo, max_commits=a.max_commits or None, subdir=subdir,
                                     track_renames=a.track_renames,
                                     rename_threshold=a.rename_threshold)
    print(f"[history] {stats['commits']} commits -> {stats['rows']} rows over "
          f"{stats['lineages']} lineages, {stats['distinct_hashes']} distinct contents "
          f"({stats['walk_seconds']}s to walk)")
    print(f"[history] changes: {stats['changed_totals']}", flush=True)
    if a.track_renames:
        print(f"[history] rename tracking: {stats['track_renames']['links']} link(s) "
              f"{stats['track_renames']['by_kind']}", flush=True)

    queries = build_queries(rows)
    qstats = query_stats(queries)
    print(f"[history] generated {qstats['n']} version-targeted queries {qstats['by_kind']} "
          f"(labels: {qstats['by_label_source']})", flush=True)

    enc, meta = make_doc_encoder(a.preset, a.device, mock=a.mock_encoder)
    t0 = time.time()
    emb, embed_stats = embeddings_for(rows, enc.encode_docs, cache={}, batch_size=a.batch_size)
    print(f"[history] embedded {embed_stats['recomputed']} distinct texts for {embed_stats['rows']} "
          f"rows ({embed_stats['reuse_pct']:.1f}% reused) in {time.time() - t0:.1f}s", flush=True)

    out_dir = resolve_index_dir(a.out)
    meta.update({"task": "code-history", "split": "history", "repo": str(repo), "subdir": subdir,
                 "commits": stats["commits"], "ingest_stats": stats, "embed_stats": embed_stats,
                 "encode_device": a.device, "source": "src/build_history_index.py",
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    manifest = build_versioned_index(out_dir, rows, emb, meta)
    write_json_atomic(a.queries_out, {"repo": str(repo), "subdir": subdir, "stats": qstats,
                                      "queries": queries})
    write_json_atomic(a.stats_out, {"stats": stats, "per_commit": per_commit,
                                    "embed_stats": embed_stats, "query_stats": qstats})
    print(f"[history] wrote {out_dir}: {manifest['n_docs']} rows x dim {manifest['dim']}")
    print(f"[history] queries -> {a.queries_out}")
    print(f"[history] ingest stats -> {a.stats_out}")
    print(f"\n  try:  python src/cli.py \"{queries[0]['text'][:60]}\" --index {a.out}"
          if queries else "\n  (no version-targeted queries were generated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
