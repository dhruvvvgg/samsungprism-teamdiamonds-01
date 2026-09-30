"""CPU serving benchmark: what one query actually costs without a GPU.

    python src/bench_cpu.py --n-queries 5                  # quick check
    python src/bench_cpu.py                                # the real measurement (50 queries)
    python src/bench_cpu.py --thread-sweep                 # latency at 1, 2 and all physical cores
    python src/bench_cpu.py --index lite                   # the 0.6B tradeoff index

Reports model load time, index load time, per-query latency (median and p95, split into query-encode and
search), per-query token count, peak RSS and the thread count -- the numbers the README's resource table
needs.

Latency is measured per query, one at a time, which is the interactive case. The first query is timed
but excluded from the statistics: it pays for lazily-initialised kernels and allocator warm-up, so
including it would quietly inflate the median on short runs.

The token count is reported alongside latency because on this workload they are the same story: APPS
queries are long problem statements, attention cost grows faster than linearly in sequence length, and
the measured split is ~9 s of query encoding against ~5 ms of vector search. Anything that reduces
latency meaningfully has to reduce tokens, cores-per-token, or model size.

Queries come from the APPS *train* partition by default. Latency depends on query length, not on which
split a query belongs to, and the train queries are the same real APPS problem statements -- so this
gives an honest number without the test split being touched outside the official run. `--queries-source
apps-test` is available with --confirm-test if you specifically want test-split texts.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime_index import DEFAULT_SERVING_QUERY_TOKENS  # noqa: E402


def load_queries(source, n, confirm_test=False, queries_file=None):
    """n query strings from the chosen source, plus a label describing where they came from."""
    if queries_file:
        raw = json.loads(Path(queries_file).read_text(encoding="utf-8"))
        texts = list(raw.values()) if isinstance(raw, dict) else list(raw)
        return texts[:n], f"file {queries_file}"
    if source == "mock":
        from src.versioning.fixture import build_fixture
        fx = build_fixture(n_snippets=max(n, 1), n_versions=1, seed=0)
        return [q["text"] for q in fx["queries"]][:n], "synthetic (mock)"
    if source == "apps-test":
        if not confirm_test:
            raise SystemExit("--queries-source apps-test reads the test split's query texts; pass "
                             "--confirm-test, or use the default apps-train.")
        from src.eval.load_data import load_apps
        _, queries, _ = load_apps(allow_test=True)
        return list(queries.values())[:n], "APPS test-split query texts"
    from src.eval.dev_data import load_dev
    d = load_dev()
    return d["query_texts"][:n], "APPS train-partition query texts"


def pct(values, p):
    """p-th percentile (nearest-rank), robust for the small n this script runs."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


def measure(svc, queries, k, announce=True):
    """Time each query once. Returns (total_ms, encode_ms, search_ms) with the warm-up still at [0]."""
    total, enc, srch = [], [], []
    for i, q in enumerate(queries):
        t0 = time.time()
        res = svc.search(q, k=k)
        total.append(1000 * (time.time() - t0))
        enc.append(res["timings_ms"]["encode_query_ms"])
        srch.append(res["timings_ms"]["search_ms"])
        if announce and i == 0:
            print(f"[bench] first query (warm-up, excluded): {total[0]:.0f} ms", flush=True)
        elif announce and (i + 1) % 10 == 0:
            print(f"[bench]   {i + 1}/{len(queries)} queries, running median "
                  f"{statistics.median(total[1:]):.0f} ms", flush=True)
    return total, enc, srch


def summarise(total, enc, srch):
    timed = total[1:] or total            # drop the warm-up unless it is all we have
    e, s = (enc[1:] or enc), (srch[1:] or srch)
    return {"n_measured": len(timed),
            "latency_ms": {"p50": round(statistics.median(timed), 1), "p95": round(pct(timed, 95), 1),
                           "mean": round(statistics.fmean(timed), 1), "min": round(min(timed), 1),
                           "max": round(max(timed), 1)},
            "query_encode_ms": {"p50": round(statistics.median(e), 1), "p95": round(pct(e, 95), 1)},
            "search_ms": {"p50": round(statistics.median(s), 2), "p95": round(pct(s, 95), 2)}}


def thread_levels(physical):
    """The thread counts to sweep: 1, 2 and all physical cores, deduplicated and ordered.

    Never above the physical core count -- a level that oversubscribes the machine measures scheduler
    contention rather than thread scaling, which would make the README's scaling claim wrong."""
    physical = max(1, int(physical))
    return sorted({n for n in (1, 2, physical) if n <= physical})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", dest="index_dir", default="full",
                    help="index name (full / lite / versions) or a path")
    ap.add_argument("--index-dir", dest="index_dir", default=argparse.SUPPRESS,
                    help=argparse.SUPPRESS)
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None, help="default: physical cores")
    ap.add_argument("--thread-sweep", action="store_true",
                    help="also measure at 1, 2 and all physical cores (one model load, reused)")
    ap.add_argument("--cpu-dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--int8", action="store_true", help="run src/check_cpu_precision.py first")
    ap.add_argument("--max-query-tokens", type=int, default=None,
                    help=f"truncate the query at this many tokens "
                         f"(default: {DEFAULT_SERVING_QUERY_TOKENS}, the serving cap measured on dev; "
                         f"0 = uncapped). Documents are never truncated by this")
    ap.add_argument("--queries-source", default="apps-train",
                    choices=["apps-train", "apps-test", "mock"])
    ap.add_argument("--queries-file", default=None, help="JSON list or {id: text} of queries")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--mock-encoder", action="store_true", help="hashing encoder; CI smoke only")
    ap.add_argument("--out", default=str(ROOT / "results" / "cpu_benchmark.json"))
    a = ap.parse_args()

    from src.runtime_index import peak_rss_mb, physical_cores, resolve_query_cap, set_cpu_threads
    from src.search_service import SearchService
    from src.utils_io import write_json_atomic

    t0 = time.time()
    svc = SearchService(a.index_dir, device="cpu", mock=a.mock_encoder, threads=a.threads,
                        cpu_dtype=a.cpu_dtype, int8=a.int8, max_query_tokens=resolve_query_cap(a.max_query_tokens))
    setup_s = time.time() - t0
    info = svc.describe()
    queries, label = load_queries(a.queries_source, a.n_queries, a.confirm_test, a.queries_file)
    if not queries:
        raise SystemExit("no queries loaded")
    n_phys, phys_source = physical_cores()
    print(f"[bench] index   : {info['index_dir']} ({info['n_docs']} docs, dim {info['dim']})")
    print(f"[bench] model   : {info['model']} on cpu, {info['threads']} threads "
          f"({svc.thread_info.get('source')}, {svc.thread_info.get('logical_cpus')} logical / "
          f"{n_phys} physical cores)")
    cap_note = "" if info["max_query_tokens"] is None else f", query cap {info['max_query_tokens']}"
    print(f"[bench] precision: {info['cpu_dtype']}{' +int8' if info['int8'] else ''}{cap_note}")
    print(f"[bench] queries : {len(queries)} from {label}")
    print(f"[bench] setup   : model load {info['model_load_seconds']:.2f}s + index load "
          f"{info['load_index_seconds']:.2f}s = {setup_s:.2f}s total", flush=True)

    tokens = [svc.token_count(q) for q in queries]
    tokens = [t for t in tokens if t is not None]
    tok_stats = ({"p50": pct(tokens, 50), "p95": pct(tokens, 95), "mean": round(statistics.fmean(tokens), 1),
                  "max": max(tokens), "min": min(tokens)} if tokens else None)
    if tok_stats:
        print(f"[bench] tokens  : per query p50 {tok_stats['p50']}, p95 {tok_stats['p95']}, "
              f"max {tok_stats['max']}"
              + ("" if info["max_query_tokens"] is None
                 else f"  (capped at {info['max_query_tokens']})"), flush=True)

    total, enc, srch = measure(svc, queries, a.top_k)
    base = summarise(total, enc, srch)

    sweep = []
    if a.thread_sweep:
        print("\n[bench] thread sweep (same loaded model, threads changed between passes)", flush=True)
        for n in thread_levels(n_phys):
            set_cpu_threads(n)
            t, e, s = measure(svc, queries, a.top_k, announce=False)
            row = dict(summarise(t, e, s), threads=n)
            sweep.append(row)
            print(f"[bench]   {n} thread(s): p50 {row['latency_ms']['p50']:.0f} ms, "
                  f"p95 {row['latency_ms']['p95']:.0f} ms", flush=True)
        set_cpu_threads(a.threads if a.threads else n_phys)     # leave it where the run started

    out = {
        "index_dir": info["index_dir"], "model": info["model"], "revision": info["revision"],
        "preset": info["preset"], "n_docs": info["n_docs"], "dim": info["dim"], "device": "cpu",
        "threads": info["threads"], "thread_source": svc.thread_info.get("source"),
        "logical_cpus": svc.thread_info.get("logical_cpus"), "physical_cores": n_phys,
        "physical_cores_source": phys_source,
        "cpu_dtype": info["cpu_dtype"], "int8": info["int8"],
        "max_query_tokens": info["max_query_tokens"],
        "model_load_seconds": round(info["model_load_seconds"], 3),
        "index_load_seconds": round(info["load_index_seconds"], 3),
        "setup_seconds": round(setup_s, 3),
        "queries": {"n_measured": base["n_measured"], "n_run": len(total), "source": label,
                    "warmup_excluded_ms": round(total[0], 1) if len(total) > 1 else None},
        "query_tokens": tok_stats,
        "latency_ms": base["latency_ms"], "query_encode_ms": base["query_encode_ms"],
        "search_ms": base["search_ms"],
        "thread_sweep": sweep,
        "peak_rss_mb": round(peak_rss_mb(), 1) if peak_rss_mb() else None,
        "mock_encoder": bool(a.mock_encoder), "top_k": a.top_k,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("\n" + "=" * 70)
    print("[bench] CPU RESULTS" + ("   (MOCK ENCODER -- timings are not the real model)"
                                   if a.mock_encoder else ""))
    print("=" * 70)
    print(f"  model load        : {out['model_load_seconds']:.2f} s")
    print(f"  index load        : {out['index_load_seconds']:.2f} s")
    print(f"  latency  p50/p95  : {out['latency_ms']['p50']:.0f} / {out['latency_ms']['p95']:.0f} ms"
          f"   (over {out['queries']['n_measured']} queries)")
    print(f"    query encode    : {out['query_encode_ms']['p50']:.0f} / "
          f"{out['query_encode_ms']['p95']:.0f} ms")
    print(f"    vector search   : {out['search_ms']['p50']:.2f} / {out['search_ms']['p95']:.2f} ms")
    if tok_stats:
        print(f"  query tokens      : p50 {tok_stats['p50']}, p95 {tok_stats['p95']}")
    print(f"  peak RSS          : {out['peak_rss_mb']} MB")
    print(f"  threads           : {out['threads']} ({out['thread_source']}; "
          f"{out['logical_cpus']} logical / {n_phys} physical)")
    if sweep:
        print("\n  thread scaling:")
        first = sweep[0]["latency_ms"]["p50"]
        for row in sweep:
            p50 = row["latency_ms"]["p50"]
            print(f"    {row['threads']:>2} thread(s): p50 {p50:8.0f} ms  p95 "
                  f"{row['latency_ms']['p95']:8.0f} ms   ({first / max(p50, 1e-9):.2f}x vs 1 thread)")
    write_json_atomic(a.out, out, indent=2)
    print(f"\n  written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
