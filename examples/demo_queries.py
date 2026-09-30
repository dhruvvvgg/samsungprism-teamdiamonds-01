"""Demo: index this repo's example package and ask it a few questions in plain English.

    # no model, no download -- proves the wiring end to end in seconds
    python examples/demo_queries.py --mock-encoder

    # the real thing (CPU; the model load dominates, the queries are fast afterwards)
    python examples/demo_queries.py

It builds a code index over `examples/textkit/` (function- and class-level chunks with file and line
numbers), then runs a handful of natural-language queries and prints where each answer lives.

With --mock-encoder the ranking is meaningless by construction -- the hashing encoder has no semantics.
It is there to prove the pipeline runs, and the script says so rather than letting a reader mistake the
output for a quality result.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO_QUERIES = [
    "split a paragraph into sentences",
    "remove accents from text",
    "count how often each word appears",
    "compute cosine similarity between two vectors",
    "rank documents against a query",
    "cut a string down to a maximum number of words",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(ROOT / "examples" / "textkit"))
    ap.add_argument("--index", dest="index_dir", default=str(ROOT / "examples" / "_demo_index"))
    ap.add_argument("--preset", default="f2llm-v2-0.6b", help="the lite model by default: this is a demo")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--mock-encoder", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the demo index even if it exists")
    a = ap.parse_args()

    index_dir = Path(a.index_dir)
    if a.rebuild or not (index_dir / "manifest.json").exists():
        cmd = [sys.executable, str(ROOT / "src" / "build_index.py"), "--source", a.folder,
               "--out", str(index_dir), "--preset", a.preset, "--device", a.device]
        if a.mock_encoder:
            cmd.append("--mock-encoder")
        print(f"[demo] building the index: {' '.join(cmd[1:])}\n", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            return proc.returncode
    else:
        print(f"[demo] reusing the index at {index_dir} (--rebuild to force)\n")

    from src.search_service import SearchService
    t0 = time.time()
    svc = SearchService(str(index_dir), device=a.device, mock=a.mock_encoder)
    info = svc.describe()
    print(f"[demo] index {index_dir.name}: {info['n_docs']} chunks from {a.folder}")
    print(f"[demo] model {info['model']} loaded in {time.time() - t0:.1f}s "
          f"(once -- every query below reuses it)")
    if a.mock_encoder:
        print("[demo] MOCK ENCODER: the ranking below is plumbing, not meaning.")
    print()

    for q in DEMO_QUERIES:
        res = svc.search(q, k=a.top_k, preview_chars=110)
        ms = res["timings_ms"]["encode_query_ms"] + res["timings_ms"]["search_ms"]
        print(f"  ? {q}   ({ms:.0f} ms)")
        for h in res["hits"]:
            where = h.get("location", h["doc_id"])
            what = h.get("qualname") or ""
            print(f"      {h['score']:.3f}  {where}  {what}")
        print()
    print(f"[demo] {len(DEMO_QUERIES)} queries against {info['n_docs']} chunks, "
          f"model loaded once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
