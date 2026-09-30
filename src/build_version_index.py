"""Build the versioned index (P1/Bonus): every version of every fixture snippet in one served index.

    python src/build_version_index.py --device cuda                       # real embedder
    python src/build_version_index.py --mock-encoder --n-snippets 20      # no model, no download

Writes `version_index/` (a runtime index plus versions.json) and `results/version_fixture.json` so the
benchmarks and the CLI work on exactly the same fixture. Embedding is content-hash deduplicated, so a
version that is byte-identical to its predecessor costs nothing.

Then:
    python src/cli.py "sum the scores" --index-dir version_index --version 2
    python src/cli.py --history snip0007 --index-dir version_index
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "version_index"))
    ap.add_argument("--fixture-out", default=str(ROOT / "results" / "version_fixture.json"))
    ap.add_argument("--n-snippets", type=int, default=500)
    ap.add_argument("--n-versions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unchanged-rate", type=float, default=0.25)
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "corpus"],
                    help="corpus: mutate real APPS corpus texts instead of synthetic snippets")
    ap.add_argument("--preset", default="f2llm-v2-1.7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mock-encoder", action="store_true")
    a = ap.parse_args()

    from src.runtime_index import make_doc_encoder
    from src.utils_io import write_json_atomic
    from src.versioning.fixture import build_fixture, fixture_stats
    from src.versioning.version_index import build_versioned_index, embeddings_for, rows_for

    source_texts = None
    if a.source == "corpus":
        from src.eval.dev_data import load_dev
        source_texts = load_dev()["doc_texts"][:a.n_snippets]
    fx = build_fixture(a.n_snippets, a.n_versions, seed=a.seed, source_texts=source_texts,
                       unchanged_rate=a.unchanged_rate)
    st = fixture_stats(fx)
    print(f"[versions] fixture: {fx['meta']['n_snippets']} snippets x {a.n_versions} versions "
          f"({st['later_versions']} later versions, {st['unchanged']} of them unchanged = "
          f"{st['unchanged_share'] * 100:.1f}%)")
    print(f"[versions] mutations: {st['per_mutation']}")
    print(f"[versions] queries : {fx['meta']['n_queries']} "
          f"({fx['meta']['n_version_specific']} version-specific)", flush=True)

    enc, meta = make_doc_encoder(a.preset, a.device, mock=a.mock_encoder)
    rows = rows_for(fx)
    t0 = time.time()
    emb, stats = embeddings_for(rows, enc.encode_docs, cache={}, batch_size=a.batch_size)
    meta.update({"task": "version-fixture", "split": "fixture", "fixture_meta": fx["meta"],
                 "fixture_stats": st, "encode_seconds": round(time.time() - t0, 2),
                 "encode_device": a.device, "embed_stats": stats,
                 "source": "src/build_version_index.py", "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    manifest = build_versioned_index(a.out, rows, emb, meta)
    write_json_atomic(a.fixture_out, fx)
    print(f"[versions] embedded {stats['recomputed']} distinct texts for {stats['rows']} rows "
          f"({stats['reused']} reused within the build, {stats['reuse_pct']:.1f}%) in "
          f"{stats['seconds']:.1f}s")
    print(f"[versions] wrote {a.out}: {manifest['n_docs']} rows x dim {manifest['dim']}")
    print(f"[versions] fixture saved to {a.fixture_out}")
    print(f"\n  try:  python src/cli.py \"{fx['queries'][0]['text'][:60]}\" "
          f"--index-dir {Path(a.out).name}"
          + (" --mock-encoder" if a.mock_encoder else " --device cpu"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
