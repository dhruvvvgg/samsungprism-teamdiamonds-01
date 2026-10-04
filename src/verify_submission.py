"""Independently re-check a finished official run: recompute NDCG@10 / MRR@10 from the exported
rankings and the official qrels, and re-hash every artefact against the recorded sha256.

    python src/verify_submission.py --confirm-test

Why it exists: the headline number in `appsretrieval_results.json` is computed by MTEB inside the run.
This script recomputes it from the *stored ranking* with its own metric implementation and its own read
of the qrels, so a disagreement points at exactly one of three things -- a wrong ranking export, a
metric misunderstanding, or a file that changed after the run. It loads no model and encodes nothing.

The metric code below is deliberately written out rather than imported from src/eval/dev_metrics.py: a
verifier that reuses the implementation it is verifying only proves the two agree with each other.

Exit code 0 = everything checked out, 1 = at least one check failed.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dcg(gains):
    """Discounted cumulative gain, rank i (1-based) discounted by log2(i + 1)."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids, rels, k=10):
    """NDCG@k for one query. rels: {doc_id: gain}. Ideal gains = the best k gains available."""
    gains = [rels.get(d, 0) for d in ranked_ids[:k]]
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0


def mrr_at_k(ranked_ids, rels, k=10):
    """Reciprocal rank of the first relevant document within the top k, else 0."""
    for i, d in enumerate(ranked_ids[:k], start=1):
        if rels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def score_rankings(rankings, qrels, k=10):
    """Mean NDCG@k / MRR@k over the queries in `qrels` (a query with no ranking scores 0, which is the
    honest reading: the system returned nothing for it)."""
    ndcgs, mrrs, missing = [], [], []
    for qid, rels in qrels.items():
        ranked = rankings.get(qid)
        if ranked is None:
            missing.append(qid)
            ranked = []
        ndcgs.append(ndcg_at_k(ranked, rels, k))
        mrrs.append(mrr_at_k(ranked, rels, k))
    n = max(len(ndcgs), 1)
    return {f"ndcg_at_{k}": sum(ndcgs) / n, f"mrr_at_{k}": sum(mrrs) / n,
            "n_queries_scored": len(ndcgs), "missing_from_rankings": missing}


def load_qrels(args):
    """{qid: {doc_id: gain}} from --qrels, else the official test qrels (needs --confirm-test).

    Reading the test *qrels* is what a verifier must do; it evaluates the ranking the official run
    already produced and starts no new retrieval, so the one-official-run rule is untouched."""
    if args.qrels:
        raw = json.loads(Path(args.qrels).read_text(encoding="utf-8"))
        return {str(q): {str(d): int(s) for d, s in rels.items()} for q, rels in raw.items()}
    if not args.confirm_test:
        raise SystemExit("Refusing to read the official TEST qrels without --confirm-test. (Pass "
                         "--qrels FILE to verify against a local qrels file instead.)")
    from src.eval.load_data import load_apps
    _, _, qrels = load_apps(allow_test=True)
    return qrels


def recorded_path(value, base_dir=None):
    path = Path(value)
    parts = path.parts
    for anchor in ("outputs", "runtime_index"):
        if anchor in parts:
            path = Path(*parts[parts.index(anchor):])
            break
    else:
        if path.is_absolute() and base_dir is None:
            return path
        path = Path(str(value).lstrip("/"))
    base = Path.cwd() if base_dir is None else Path(base_dir)
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"Checksum path escapes --base-dir: {value}")
    return resolved


def check_hashes(checksums, problems, notes, base_dir=None, results=None, rankings=None):
    from src.runtime_index import sha256_file
    for key in ("results", "rankings"):
        rec = checksums.get(key)
        if not rec:
            notes.append(f"checksums file has no {key!r} entry")
            continue
        override = results if key == "results" else rankings
        path = Path(override) if override is not None else recorded_path(rec["path"], base_dir)
        if not path.exists():
            problems.append(f"{key}: file recorded at {path} is missing")
            continue
        got = sha256_file(path)
        if got == rec["sha256"]:
            notes.append(f"{key}: sha256 OK ({got[:16]}...) {path}")
        else:
            problems.append(f"{key}: sha256 MISMATCH -- file on disk {got[:16]}... != recorded "
                            f"{rec['sha256'][:16]}... ({path}). The file changed after the run.")
    idx = checksums.get("runtime_index")
    if idx:
        d = recorded_path(idx["dir"], base_dir)
        for name, rec in idx["files"].items():
            path = d / name
            if not path.exists():
                problems.append(f"runtime_index/{name}: missing")
            elif sha256_file(path) == rec["sha256"]:
                notes.append(f"runtime_index/{name}: sha256 OK")
            else:
                problems.append(f"runtime_index/{name}: sha256 MISMATCH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=Path.cwd())
    ap.add_argument("--rankings", default="outputs/appsretrieval_rankings.json")
    ap.add_argument("--results", default="outputs/appsretrieval_results.json")
    ap.add_argument("--checksums", default="outputs/submission_checksums.json")
    ap.add_argument("--qrels", default=None,
                    help="qrels JSON {qid: {doc_id: gain}}; default is the official test qrels")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tolerance", type=float, default=1e-4,
                    help="allowed |recomputed - reported| before the check fails")
    ap.add_argument("--confirm-test", action="store_true",
                    help="acknowledge reading the official test qrels (no retrieval is run)")
    a = ap.parse_args()
    for field in ("rankings", "results", "checksums", "qrels"):
        value = getattr(a, field)
        if value and not Path(value).is_absolute():
            setattr(a, field, str(a.base_dir / value))
    problems, notes = [], []

    rpath = Path(a.rankings)
    if not rpath.exists():
        raise SystemExit(f"No rankings file at {rpath}. It is written by src/eval/run_official.py.")
    payload = json.loads(rpath.read_text(encoding="utf-8"))
    rankings = payload["rankings"] if isinstance(payload, dict) and "rankings" in payload else payload
    depth = payload.get("depth") if isinstance(payload, dict) else None
    print(f"[verify] rankings : {len(rankings)} queries, depth {depth}, from {rpath}")

    qrels = load_qrels(a)
    print(f"[verify] qrels    : {len(qrels)} queries"
          f"{'' if a.qrels else ' (official test split)'}")
    got = score_rankings(rankings, qrels, a.k)
    print("\n[verify] RECOMPUTED FROM THE RANKING FILE (independent implementation):")
    print(f"           ndcg_at_{a.k} = {got[f'ndcg_at_{a.k}']:.5f}")
    print(f"           mrr_at_{a.k}  = {got[f'mrr_at_{a.k}']:.5f}")
    print(f"           queries scored = {got['n_queries_scored']}")
    if got["missing_from_rankings"]:
        problems.append(f"{len(got['missing_from_rankings'])} judged queries have no ranking at all "
                        f"(e.g. {got['missing_from_rankings'][:3]}); they were scored 0")

    reported = None
    results_path = Path(a.results)
    if results_path.exists():
        res = json.loads(results_path.read_text(encoding="utf-8"))
        scores = (res.get("scores") or {}).get("test") or [{}]
        reported = scores[0]
        print(f"\n[verify] REPORTED BY MTEB in {results_path.name}:")
        for m in (f"ndcg_at_{a.k}", f"mrr_at_{a.k}"):
            rv, gv = reported.get(m), got[m]
            if isinstance(rv, (int, float)):
                diff = gv - rv
                ok = abs(diff) <= a.tolerance
                print(f"           {m} = {rv:.5f}   recomputed {gv:.5f}   diff {diff:+.6f}   "
                      f"{'OK' if ok else 'MISMATCH'}")
                if not ok:
                    problems.append(f"{m}: recomputed {gv:.5f} vs reported {rv:.5f} "
                                    f"(diff {diff:+.6f} > tolerance {a.tolerance})")
            else:
                notes.append(f"results file has no numeric {m}")
    else:
        notes.append(f"no results file at {results_path}; skipped the reported-vs-recomputed comparison")

    cpath = Path(a.checksums)
    if cpath.exists():
        print(f"\n[verify] checksums from {cpath.name}:")
        check_hashes(json.loads(cpath.read_text(encoding="utf-8")), problems, notes,
                     a.base_dir, a.results, a.rankings)
    else:
        notes.append(f"no checksums file at {cpath}; skipped hash verification")

    for n in notes:
        print(f"[verify]   note: {n}")
    print("\n" + "=" * 70)
    if problems:
        print(f"[verify] FAILED -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        print("=" * 70)
        return 1
    print("[verify] PASSED: recomputed metrics match the reported ones and every hash checks out.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
