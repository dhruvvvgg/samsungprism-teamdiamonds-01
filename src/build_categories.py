"""Tag an existing index with offline categories, in place. No model, no re-encoding.

    python src/build_categories.py --index full
    python src/build_categories.py --index examples/_demo_index --clusters 8

Reads the index's stored texts and embeddings, computes the AST family per row and a k-means cluster
with a readable label, and writes `categories.json` next to the embeddings. The manifest's file hashes
are refreshed so `--verify-index` still passes.

Tagging is display and filtering only: it does not change any ranking. Using categories to reorder
results is a retrieval change and goes through the adoption rule -- see src/eval/dev_categories.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", dest="index_dir", default="full",
                    help="index name (full / lite / versions / history) or a path")
    ap.add_argument("--clusters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-clusters", action="store_true",
                    help="AST families only (skip the embedding clustering)")
    a = ap.parse_args()

    from src.indexing.categories import family_counts, tag_rows
    from src.runtime_index import CATEGORIES, RuntimeIndex, resolve_index_dir, sha256_file
    from src.utils_io import write_json_atomic

    d = resolve_index_dir(a.index_dir)
    index = RuntimeIndex.load(d)
    print(f"[cat] {d}: {len(index.doc_ids)} rows", flush=True)
    t0 = time.time()
    tags = tag_rows(index.doc_texts, None if a.no_clusters else index.E,
                    n_clusters=a.clusters, seed=a.seed)
    write_json_atomic(d / CATEGORIES, tags)

    manifest = dict(index.manifest)
    files = dict(manifest.get("files") or {})
    files[CATEGORIES] = {"sha256": sha256_file(d / CATEGORIES),
                         "bytes": (d / CATEGORIES).stat().st_size}
    manifest["files"] = files
    manifest["categories"] = {"clusters": 0 if a.no_clusters else a.clusters, "seed": a.seed}
    write_json_atomic(d / "manifest.json", manifest, indent=2, default=str)

    counts = family_counts(tags)
    print(f"[cat] tagged in {time.time() - t0:.1f}s")
    print("[cat] AST families: " + json.dumps(counts))
    if not a.no_clusters:
        labels = sorted({t["cluster_label"] for t in tags if t.get("cluster_label")})
        print(f"[cat] {len(labels)} cluster labels:")
        for lab in labels:
            print(f"        {lab}")
    problems = RuntimeIndex.load(d).verify_files()
    print(f"[cat] index still verifies: {'yes' if not problems else problems}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
