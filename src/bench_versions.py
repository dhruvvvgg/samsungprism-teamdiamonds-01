"""P1 benchmark: full vs incremental rebuild, version by version.

    python src/bench_versions.py --mock-encoder          # structure only, seconds, no model
    python src/bench_versions.py --device cuda           # real embedder, real timings

The scenario is the ordinary one for a code index: a new version of the corpus lands, and the index has
to cover it. At version v the index holds every snippet at version v (one row per snippet).

    full         a fresh embedding store every time -- every snippet is re-embedded
    incremental  the store persists; a snippet whose content hash is unchanged is reused

Reported per version: wall-clock seconds, embeddings recomputed, embeddings reused, % reused. The
fixture knows how many versions are byte-identical repeats, so the script also prints that expected
figure next to the measured one -- a reuse rate that does not line up with it means the content hash is
either missing real changes or failing to recognise identical text, and the run says so.

With --mock-encoder the *counts* are exact and meaningful while the *seconds* are not (hashing is far
faster than a transformer); only the real-model run gives a usable time saving.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def snapshot_rows(fixture, version):
    """One row per snippet at `version` (its latest version if it has fewer)."""
    from src.versioning.content_hash import content_hash
    from src.versioning.version_index import doc_id_for
    rows = []
    for s in fixture["snippets"]:
        v = max((x for x in s["versions"] if x["version"] <= version),
                key=lambda x: x["version"], default=None)
        if v is None:
            continue
        rows.append({"doc_id": doc_id_for(s["snippet_id"], v["version"]),
                     "snippet_id": s["snippet_id"], "version": v["version"],
                     "content_hash": v.get("content_hash") or content_hash(v["text"]),
                     "mutation": v["mutation"], "text": v["text"]})
    return rows


def expected_unchanged(fixture, version):
    """How many snippets are byte-identical to their previous version at `version` (the fixture's own
    count, independent of anything the index does)."""
    n = 0
    for s in fixture["snippets"]:
        v = next((x for x in s["versions"] if x["version"] == version), None)
        if v is not None and v["mutation"] == "unchanged":
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-snippets", type=int, default=500)
    ap.add_argument("--n-versions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unchanged-rate", type=float, default=0.25)
    ap.add_argument("--preset", default="f2llm-v2-1.7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mock-encoder", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "version_rebuild_benchmark.json"))
    a = ap.parse_args()

    from src.runtime_index import make_doc_encoder
    from src.utils_io import write_json_atomic
    from src.versioning.fixture import build_fixture, fixture_stats
    from src.versioning.version_index import embeddings_for

    fx = build_fixture(a.n_snippets, a.n_versions, seed=a.seed, unchanged_rate=a.unchanged_rate)
    st = fixture_stats(fx)
    print(f"[p1] fixture: {len(fx['snippets'])} snippets x {a.n_versions} versions; mutations "
          f"{st['per_mutation']}")
    enc, meta = make_doc_encoder(a.preset, a.device, mock=a.mock_encoder)
    print(f"[p1] encoder: {meta['model']}"
          + ("   (MOCK: counts are real, seconds are not)" if a.mock_encoder else ""), flush=True)

    inc_cache, rows_per_version, per_version, warnings = {}, {}, [], []
    for v in range(1, a.n_versions + 1):
        rows = snapshot_rows(fx, v)
        rows_per_version[v] = len(rows)
        t0 = time.time()
        _, full = embeddings_for(rows, enc.encode_docs, cache={}, batch_size=a.batch_size)
        full_s = time.time() - t0
        t0 = time.time()
        _, inc = embeddings_for(rows, enc.encode_docs, cache=inc_cache, batch_size=a.batch_size)
        inc_s = time.time() - t0
        exp = expected_unchanged(fx, v) if v > 1 else 0
        rec = {"version": v, "rows": len(rows),
               "full": {"recomputed": full["recomputed"], "reused": full["reused"],
                        "seconds": round(full_s, 3)},
               "incremental": {"recomputed": inc["recomputed"], "reused": inc["reused"],
                               "reuse_pct": round(inc["reuse_pct"], 1), "seconds": round(inc_s, 3)},
               "expected_unchanged": exp,
               "speedup_x": round(full_s / inc_s, 2) if inc_s > 0 else None}
        if v > 1 and inc["reused"] < exp:
            warnings.append(f"version {v}: only {inc['reused']} rows reused but {exp} are "
                            f"byte-identical repeats -- the content hash is missing identical text")
        per_version.append(rec)
        print(f"[p1] v{v}: rows {len(rows):5d} | full recompute {full['recomputed']:5d} "
              f"({full_s:7.2f}s) | incremental recompute {inc['recomputed']:5d} reuse "
              f"{inc['reuse_pct']:5.1f}% ({inc_s:7.2f}s) | expected unchanged {exp}", flush=True)

    tot_full = sum(r["full"]["recomputed"] for r in per_version)
    tot_inc = sum(r["incremental"]["recomputed"] for r in per_version)
    tot_full_s = sum(r["full"]["seconds"] for r in per_version)
    tot_inc_s = sum(r["incremental"]["seconds"] for r in per_version)
    out = {"fixture": fx["meta"], "fixture_stats": st, "encoder": meta["model"],
           "mock_encoder": bool(a.mock_encoder), "device": a.device,
           "per_version": per_version,
           "totals": {"embeddings_full": tot_full, "embeddings_incremental": tot_inc,
                      "embeddings_saved": tot_full - tot_inc,
                      "saved_pct": round(100.0 * (tot_full - tot_inc) / max(tot_full, 1), 1),
                      "seconds_full": round(tot_full_s, 2), "seconds_incremental": round(tot_inc_s, 2),
                      "speedup_x": round(tot_full_s / tot_inc_s, 2) if tot_inc_s > 0 else None},
           "warnings": warnings, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("\n" + "=" * 70)
    print("[p1] TOTALS across all versions")
    print("=" * 70)
    print(f"  embeddings computed : full {tot_full}  ->  incremental {tot_inc}  "
          f"(saved {tot_full - tot_inc}, {out['totals']['saved_pct']}%)")
    print(f"  wall clock          : full {tot_full_s:.1f}s  ->  incremental {tot_inc_s:.1f}s  "
          f"(x{out['totals']['speedup_x']})")
    if a.mock_encoder:
        print("  NOTE: mock encoder -- the counts above are real, the seconds are not.")
    for w in warnings:
        print(f"  WARNING: {w}")
    write_json_atomic(a.out, out, indent=2)
    print(f"\n  written to {a.out}")
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
