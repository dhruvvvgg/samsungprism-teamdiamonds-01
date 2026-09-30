"""P1 + Bonus measured on a real repository's history.

    python src/bench_real_history.py --mock-encoder --max-commits 12     # structure only, seconds
    python src/bench_real_history.py --device cuda --max-commits 40      # the real numbers

Three things are measured, on real edits rather than invented ones:

P1, incremental rebuild
    Per commit: full rebuild (fresh embedding store, every live function re-embedded) against
    incremental (content-hash store carried forward). Reports seconds, embeddings recomputed, reused
    and % reused. The ingester independently counts how many lineages were added/modified/unchanged at
    each commit, so the reuse figure has a ground truth to be checked against rather than admired --
    a mismatch is reported as a warning.

P1, version targeting
    For each auto-generated version-targeted query, search ONLY the target version and check whether the
    correct lineage comes back at rank 1.

Bonus, retrieval across all versions
    Search every version at once, and report top-10 near-duplicate rate (slots taken by repeat versions
    of a lineage already in the list) and correct-lineage recall, with and without collapsing.

Experiment (off; --delta-vectors)
    Version-delta vectors -- one base vector per lineage plus a small int8 residual per distinct version
    (src/versioning/delta_index.py) -- against the current per-row vectors: index size and top-1 version
    accuracy on the same queries.

With --mock-encoder the counts, timings ratios and duplicate rates are real; the retrieval quality
numbers are not, and the script says so rather than letting them be quoted.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def snapshot_at(rows, version):
    """The live state at a version: one row per lineage present at that commit."""
    return [r for r in rows if r["version"] == version]


def rebuild_benchmark(rows, per_commit, encode_fn, batch_size, warn_threshold=0):
    """Full vs incremental rebuild, commit by commit."""
    from src.versioning.version_index import embeddings_for
    inc_cache, out, warnings = {}, [], []
    for rec in per_commit:
        v = rec["version"]
        snap = snapshot_at(rows, v)
        if not snap:
            continue
        t0 = time.time()
        _, full = embeddings_for(snap, encode_fn, cache={}, batch_size=batch_size)
        full_s = time.time() - t0
        t0 = time.time()
        _, inc = embeddings_for(snap, encode_fn, cache=inc_cache, batch_size=batch_size)
        inc_s = time.time() - t0
        expected_reuse = rec["unchanged"] if v > 1 else 0
        row = {"version": v, "commit": rec["short"], "subject": rec["subject"][:60],
               "lineages": len(snap),
               "added": rec["added"], "modified": rec["modified"], "unchanged": rec["unchanged"],
               "full": {"recomputed": full["recomputed"], "seconds": round(full_s, 3)},
               "incremental": {"recomputed": inc["recomputed"], "reused": inc["reused"],
                               "reuse_pct": round(inc["reuse_pct"], 1), "seconds": round(inc_s, 3)},
               "expected_unchanged": expected_reuse,
               "speedup_x": round(full_s / inc_s, 2) if inc_s > 0 else None}
        if v > 1 and inc["reused"] < expected_reuse - warn_threshold:
            warnings.append(f"v{v} ({rec['short']}): reused {inc['reused']} but {expected_reuse} "
                            f"lineages were unchanged -- the content hash is missing identical text")
        out.append(row)
        print(f"[p1] v{v:3d} {rec['short']} lineages={len(snap):5d} | full {full['recomputed']:5d} "
              f"({full_s:6.2f}s) | incr {inc['recomputed']:5d} reuse {inc['reuse_pct']:5.1f}% "
              f"({inc_s:6.2f}s) | unchanged={rec['unchanged']}", flush=True)
    return out, warnings


def evaluate_retrieval(svc, queries, k, index):
    """Version-targeted (P1) and across-all-versions (Bonus) retrieval."""
    from src.versioning.version_index import duplication_at_k
    targeted_hits = exact_version = token_ok = 0
    dup_all, dup_collapsed, recall_all, recall_collapsed = [], [], 0, 0
    lat = []
    for i, q in enumerate(queries):
        # --- P1: restrict to the target version --------------------------------------------------
        try:
            t0 = time.time()
            tgt = svc.search(q["text"], k=k, version=q["target_version"])
            lat.append(1000 * (time.time() - t0))
            if tgt["hits"] and tgt["hits"][0].get("snippet_id") == q["lineage_id"]:
                targeted_hits += 1
        except ValueError:
            pass                                    # no rows at that version; counted as a miss
        # --- Bonus: every version competes ---------------------------------------------------------
        raw = svc.search(q["text"], k=k, all_versions=True)
        col = svc.search(q["text"], k=k)
        rows_raw = [(h["doc_id"], h["score"], index.row_of(h["doc_id"])) for h in raw["hits"]]
        rows_col = [(h["doc_id"], h["score"], index.row_of(h["doc_id"])) for h in col["hits"]]
        dup_all.append(duplication_at_k(rows_raw, index, k=k)["duplicate_pct"])
        dup_collapsed.append(duplication_at_k(rows_col, index, k=k)["duplicate_pct"])
        recall_all += int(any(h.get("snippet_id") == q["lineage_id"] for h in raw["hits"]))
        recall_collapsed += int(any(h.get("snippet_id") == q["lineage_id"] for h in col["hits"]))
        top = raw["hits"][0] if raw["hits"] else None
        if top and top.get("snippet_id") == q["lineage_id"]:
            exact_version += int(top.get("version") == q["target_version"])
            token_ok += int(top.get("version") in q.get("acceptable_versions",
                                                        [q["target_version"]]))
        if (i + 1) % 25 == 0:
            print(f"[eval]   {i + 1}/{len(queries)} queries", flush=True)
    n = max(len(queries), 1)

    def mean(xs):
        return round(statistics.fmean(xs), 2) if xs else None

    return {
        "n_queries": len(queries),
        "p1_version_targeted": {
            "top1_correct_lineage": round(targeted_hits / n, 4),
            "median_latency_ms": round(statistics.median(lat), 1) if lat else None,
        },
        "bonus_all_versions": {
            f"duplicate_pct_at_{k}_all_versions": mean(dup_all),
            f"duplicate_pct_at_{k}_collapsed": mean(dup_collapsed),
            f"lineage_recall_at_{k}_all_versions": round(recall_all / n, 4),
            f"lineage_recall_at_{k}_collapsed": round(recall_collapsed / n, 4),
            "top1_exact_version": round(exact_version / n, 4),
            "top1_token_consistent_version": round(token_ok / n, 4),
        },
    }


def evaluate_delta(index, encode_query, queries, lineage_k=None):
    """Version-delta vectors (experiment) against the current approach, on the same queries.

    Both rank ALL versions of ALL lineages and are scored on the top-1 result: the right lineage, the
    exact target version, and a version consistent with the query's token. The current approach is a
    full scan of the stored per-row vectors; the delta approach ranks lineages by base vector first and
    then versions by reconstructed full vector. Also reports both index sizes and how often the two
    agree on the top-1 row. Off unless --delta-vectors."""
    from src.versioning.delta_index import DeltaVectors
    dv = DeltaVectors.build(index.E, [v["snippet_id"] for v in index.versions],
                            [v["content_hash"] for v in index.versions])
    acc = {"current": {"lineage": 0, "exact": 0, "consistent": 0},
           "delta": {"lineage": 0, "exact": 0, "consistent": 0}}
    agree = 0
    for q in queries:
        qvec = encode_query(q["text"])
        cur = index.search(qvec, k=1)
        dlt = dv.search(qvec, k=1, lineage_k=lineage_k)
        rows = {"current": cur[0][2] if cur else None, "delta": dlt[0][0] if dlt else None}
        agree += int(rows["current"] is not None and rows["current"] == rows["delta"])
        for name, row in rows.items():
            if row is None:
                continue
            v = index.versions[row]
            if v["snippet_id"] == q["lineage_id"]:
                acc[name]["lineage"] += 1
                acc[name]["exact"] += int(v["version"] == q["target_version"])
                acc[name]["consistent"] += int(v["version"] in q.get("acceptable_versions",
                                                                    [q["target_version"]]))
    n = max(len(queries), 1)
    return {"n_queries": len(queries), "lineage_k": lineage_k or "max(3k, 20)",
            "rows": dv.n_rows, "lineages": len(dv.lineages), "distinct_entries": dv.n_entries,
            "index_bytes_current": dv.baseline_bytes(), "index_bytes_delta": dv.size_bytes(),
            "size_ratio_delta_over_current": round(dv.size_bytes() / max(dv.baseline_bytes(), 1), 4),
            "top1_lineage": {k: round(v["lineage"] / n, 4) for k, v in acc.items()},
            "top1_exact_version": {k: round(v["exact"] / n, 4) for k, v in acc.items()},
            "top1_token_consistent_version": {k: round(v["consistent"] / n, 4) for k, v in acc.items()},
            "top1_agreement_delta_vs_current": round(agree / n, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="local checkout; default clones the demo repo")
    ap.add_argument("--repo-url", default=None)
    ap.add_argument("--subdir", default=None)
    ap.add_argument("--max-commits", type=int, default=40)
    ap.add_argument("--index", dest="index_dir", default="history",
                    help="an already-built history index to evaluate retrieval on (skips rebuilding)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--preset", default="f2llm-v2-0.6b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mock-encoder", action="store_true")
    ap.add_argument("--limit-queries", type=int, default=100)
    ap.add_argument("--delta-vectors", action="store_true",
                    help="EXPERIMENT (off by default): also compare version-delta vectors (one base "
                         "vector per lineage + int8 residual per distinct version) with the current "
                         "per-row vectors -- index size and top-1 version accuracy")
    ap.add_argument("--delta-lineage-k", type=int, default=None,
                    help="lineages kept by the first stage of the delta search (default max(3k, 20))")
    ap.add_argument("--skip-retrieval", action="store_true",
                    help="only the rebuild benchmark (no index needed)")
    ap.add_argument("--clone-dir", default=str(ROOT / "data" / "repos"))
    ap.add_argument("--out", default=str(ROOT / "results" / "real_history_benchmark.json"))
    a = ap.parse_args()

    from src.runtime_index import make_doc_encoder
    from src.utils_io import write_json_atomic
    from src.versioning.git_history import DEFAULT_REPO, DEFAULT_SUBDIR, ensure_repo, ingest
    from src.versioning.history_queries import build_queries, query_stats

    if a.repo:
        repo, subdir = Path(a.repo), a.subdir
    else:
        url = a.repo_url or DEFAULT_REPO
        name = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        repo = ensure_repo(url, Path(a.clone_dir) / name)
        subdir = a.subdir if a.subdir is not None else (DEFAULT_SUBDIR if url == DEFAULT_REPO else None)
    print(f"[hist] repo {repo}" + (f" (subdir {subdir})" if subdir else ""), flush=True)

    rows, per_commit, stats = ingest(repo, max_commits=a.max_commits or None, subdir=subdir)
    print(f"[hist] {stats['commits']} commits, {stats['rows']} rows, {stats['lineages']} lineages, "
          f"{stats['distinct_hashes']} distinct contents", flush=True)
    enc, meta = make_doc_encoder(a.preset, a.device, mock=a.mock_encoder)

    print("\n[p1] full vs incremental rebuild, per commit"
          + ("   (MOCK: counts real, seconds are not)" if a.mock_encoder else ""), flush=True)
    per_version, warnings = rebuild_benchmark(rows, per_commit, enc.encode_docs, a.batch_size)
    tot_full = sum(r["full"]["recomputed"] for r in per_version)
    tot_inc = sum(r["incremental"]["recomputed"] for r in per_version)
    tot_full_s = sum(r["full"]["seconds"] for r in per_version)
    tot_inc_s = sum(r["incremental"]["seconds"] for r in per_version)

    queries = build_queries(rows)[:a.limit_queries or None]
    qstats = query_stats(queries)
    retrieval, delta = None, None
    if not a.skip_retrieval and queries:
        from src.search_service import SearchService
        try:
            svc = SearchService(a.index_dir, device="cpu" if a.mock_encoder else a.device,
                                mock=a.mock_encoder)
        except FileNotFoundError as exc:
            print(f"\n[eval] SKIPPED: {exc}".splitlines()[0], flush=True)
            print("[eval] build it first: python src/build_history_index.py", flush=True)
        else:
            if not svc.versioned:
                raise SystemExit(f"{a.index_dir} is not a versioned index")
            print(f"\n[eval] {len(queries)} version-targeted queries against {a.index_dir} "
                  f"({svc.index.manifest['n_docs']} rows)", flush=True)
            retrieval = evaluate_retrieval(svc, queries, a.k, svc.index)
            if a.delta_vectors:
                delta = evaluate_delta(svc.index, svc.encoder.encode_query, queries, a.delta_lineage_k)

    out = {"repo": str(repo), "subdir": subdir, "ingest": stats, "query_stats": qstats,
           "mock_encoder": bool(a.mock_encoder), "encoder": meta["model"], "k": a.k,
           "per_version": per_version,
           "totals": {"embeddings_full": tot_full, "embeddings_incremental": tot_inc,
                      "embeddings_saved": tot_full - tot_inc,
                      "saved_pct": round(100.0 * (tot_full - tot_inc) / max(tot_full, 1), 1),
                      "seconds_full": round(tot_full_s, 2),
                      "seconds_incremental": round(tot_inc_s, 2),
                      "speedup_x": round(tot_full_s / tot_inc_s, 2) if tot_inc_s > 0 else None},
           "retrieval": retrieval, "warnings": warnings,
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}

    if delta is not None:
        out["delta_vectors"] = delta
    elif a.delta_vectors:
        print("[delta] SKIPPED: --delta-vectors needs a built history index and at least one query")

    print("\n" + "=" * 78)
    print(f"[hist] REAL HISTORY: {Path(repo).name}, {stats['commits']} commits")
    print("=" * 78)
    print(f"  P1 rebuild : full {tot_full} embeddings ({tot_full_s:.1f}s)  ->  incremental {tot_inc} "
          f"({tot_inc_s:.1f}s)")
    print(f"               saved {tot_full - tot_inc} ({out['totals']['saved_pct']}%), "
          f"x{out['totals']['speedup_x']}")
    if retrieval:
        p1, bonus = retrieval["p1_version_targeted"], retrieval["bonus_all_versions"]
        print(f"  P1 targeting: correct lineage at rank 1 when targeting a version: "
              f"{p1['top1_correct_lineage']}  ({p1['median_latency_ms']} ms median)")
        print(f"  Bonus       : top-{a.k} duplicate slots "
              f"{bonus[f'duplicate_pct_at_{a.k}_all_versions']}% (all versions) -> "
              f"{bonus[f'duplicate_pct_at_{a.k}_collapsed']}% (collapsed)")
        print(f"                lineage recall@{a.k} "
              f"{bonus[f'lineage_recall_at_{a.k}_all_versions']} -> "
              f"{bonus[f'lineage_recall_at_{a.k}_collapsed']} collapsed")
        print(f"                top-1 exact version {bonus['top1_exact_version']}, "
              f"token-consistent {bonus['top1_token_consistent_version']}")
    if delta is not None:
        print(f"  Delta vectors (experiment): index {delta['index_bytes_delta']:,} B vs "
              f"{delta['index_bytes_current']:,} B current (x{delta['size_ratio_delta_over_current']}); "
              f"top-1 exact version {delta['top1_exact_version']['delta']} vs "
              f"{delta['top1_exact_version']['current']} current; "
              f"agree on {delta['top1_agreement_delta_vs_current']} of top-1s")
    if a.mock_encoder:
        print("  NOTE: mock encoder -- rebuild counts and duplicate rates are real, retrieval "
              "quality is not.")
    for w in warnings:
        print(f"  WARNING: {w}")
    write_json_atomic(a.out, out, indent=2)
    print(f"\n  written to {a.out}")
    print(json.dumps(out["totals"], indent=2))
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
