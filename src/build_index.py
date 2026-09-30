"""Build `runtime_index/` from scratch: encode the corpus once, write the served index.

    # real index on a GPU session (~8 min for 8,765 docs on a T4 with the 1.7B)
    python src/build_index.py --preset f2llm-v2-1.7b --device cuda

    # tiny fake index, no model and no dataset download -- for smoke tests and CI
    python src/build_index.py --source mock --mock-encoder --limit 20 --out /tmp/tiny_index

An official run already exports this index for free, so this script is for the cases where that is not
what you want: rebuilding after deleting the index, building one without spending a test-split
evaluation, or building with a different model.

The corpus comes from the dev loader, which reads the corpus rows and the TRAIN qrels only. The corpus
is one shared 8,765-document pool in CoIR-APPS -- the same documents the test split ranks against -- so
the index this produces is the right one to serve without the test split being touched at all.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mock_corpus(n):
    """n deterministic synthetic snippets (the version fixture's bases at a single version)."""
    from src.versioning.fixture import build_fixture
    fx = build_fixture(n_snippets=n, n_versions=1, seed=0)
    return ([s["snippet_id"] for s in fx["snippets"]],
            [s["versions"][0]["text"] for s in fx["snippets"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="f2llm-v2-1.7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="full",
                    help="output index: a name (full / lite) or a path. 'lite' pairs with "
                         "--preset f2llm-v2-0.6b, the speed/quality tradeoff index")
    ap.add_argument("--source", default="apps",
                    help="apps: the real 8,765-document corpus (dev loader, train qrels only); "
                         "mock: synthetic snippets, no download; "
                         "or a PATH to a folder of Python files, chunked at function/class level")
    ap.add_argument("--chunk-methods", action="store_true",
                    help="folder source: also index each method separately (duplicates the text "
                         "inside its class chunk, but helps when classes are large)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents (0 = all)")
    ap.add_argument("--mock-encoder", action="store_true",
                    help="hashing encoder instead of a model: no download, no GPU, meaningless scores")
    ap.add_argument("--variant", default="registry+full",
                    help="query variant whose instruction prefix is stored in the manifest")
    a = ap.parse_args()

    from src.runtime_index import make_doc_encoder, resolve_index_dir, write_index

    out_dir = resolve_index_dir(a.out)
    chunks = None
    if a.source == "mock":
        doc_ids, doc_texts = mock_corpus(a.limit or 20)
    elif a.source == "apps":
        from src.eval.dev_data import load_dev
        d = load_dev()
        doc_ids, doc_texts = d["doc_ids"], d["doc_texts"]
        if a.limit:
            doc_ids, doc_texts = doc_ids[:a.limit], doc_texts[:a.limit]
    else:
        from src.indexing.code_chunker import chunk_folder
        try:
            chunks, stats = chunk_folder(a.source, chunk_methods=a.chunk_methods)
        except NotADirectoryError as exc:
            raise SystemExit(f"--source {a.source!r} is neither 'apps', 'mock', nor a directory: {exc}")
        if not chunks:
            raise SystemExit(f"No Python chunks found under {a.source!r}. Nothing to index.")
        if a.limit:
            chunks = chunks[:a.limit]
        print(f"[build] scanned {stats['files_scanned']} Python files under {a.source}: "
              f"{len(chunks)} chunks {stats['by_kind']}", flush=True)
        if stats["files_failed"]:
            # a file that does not parse is silently missing from every search result, so say so
            print(f"[build] WARNING: {stats['files_failed']} file(s) could not be parsed and are NOT "
                  f"indexed:", flush=True)
            for f in stats["failures"]:
                print(f"[build]   {f['file']}: {f['error']}", flush=True)
        if stats["skipped_oversize"]:
            print(f"[build] {stats['skipped_oversize']} oversize chunk(s) skipped", flush=True)
        doc_ids = [c["chunk_id"] for c in chunks]
        doc_texts = [c["text"] for c in chunks]
    print(f"[build] corpus: {len(doc_ids)} documents from {a.source}", flush=True)

    try:
        enc, meta = make_doc_encoder(a.preset, a.device, mock=a.mock_encoder, variant=a.variant)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if a.mock_encoder:
        print("[build] MOCK encoder: no model is loaded and the scores mean nothing", flush=True)

    t0 = time.time()
    emb = enc.encode_docs(doc_texts, batch_size=a.batch_size)
    encode_s = time.time() - t0
    if chunks is not None:
        meta["source_root"] = str(Path(a.source).resolve())
        meta["chunk_methods"] = bool(a.chunk_methods)
    meta.update({"task": "AppsRetrieval" if a.source == "apps" else "code-folder",
                 "split": "corpus", "corpus_source": a.source,
                 "encode_seconds": round(encode_s, 2), "encode_device": a.device,
                 "source": f"src/build_index.py --source {a.source}",
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    manifest = write_index(out_dir, emb, doc_ids, doc_texts, meta, chunks=chunks)
    print(f"[build] encoded {len(doc_texts)} docs in {encode_s:.1f}s "
          f"({len(doc_texts) / max(encode_s, 1e-9):.1f} docs/s)")
    print(f"[build] wrote {out_dir}: {manifest['n_docs']} x {manifest['dim']} fp16")
    for name, rec in manifest["files"].items():
        print(f"[build]   {name:20s} {rec['bytes'] / 1e6:8.2f} MB  sha256 {rec['sha256'][:16]}...")
    print("[build] manifest:\n" + json.dumps({k: v for k, v in manifest.items() if k != "files"},
                                             indent=2, default=str))


if __name__ == "__main__":
    main()
