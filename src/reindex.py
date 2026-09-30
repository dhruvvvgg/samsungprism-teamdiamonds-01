"""Incremental update of a custom-folder or git-history index (P1).

    python src/reindex.py --index my_code_index                 # re-scan the folder it was built from
    python src/reindex.py --index my_code_index --dry-run       # report what would change, write nothing
    python src/reindex.py --index history_index --mock-encoder

The index remembers where it came from (`source_root` for a folder, `repo` + `subdir` for a history), so
the operation takes no path at all: it re-scans that source, hashes every chunk, compares with the
chunks already in the index, and encodes ONLY the chunks whose text it has never embedded. Everything
else keeps its stored vector.

What is deliberately NOT possible here:

  * The official APPS indexes (`full`, `lite`, anything whose manifest task is AppsRetrieval) are never
    re-scanned or rewritten. They have no source folder to scan, and their embeddings are the
    submitted result.
  * The source location is read from the index's own manifest, never from a request. An HTTP caller can
    name an index (from an allowlist) but cannot point the scan at a path.

A folder chunk's identity is (file, kind, qualname, occurrence); its content is the SHA-256 of the exact
text. A chunk that only moved (lines shifted by an edit above it) is `unchanged` -- it keeps its vector
and only its file:start-end is refreshed. An exact-text hash is used rather than the comment-stripping
hash of the version index because a chunk's text is what the encoder sees, so a comment edit is a real
change here.

A history index is compared on distinct (lineage, content hash) pairs, and re-ingests the same window
size (`commits` in the manifest) from the repository's current HEAD.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OFFICIAL_TASKS = {"AppsRetrieval"}
REINDEXABLE_TASKS = {"code-folder": "folder", "code-history": "history"}
DERIVED_MANIFEST_KEYS = ("files", "kind", "n_docs", "dim", "stored_dtype", "index_version")


class ReindexError(Exception):
    """A refusal or a failure the caller should see verbatim. `status` is the HTTP code the API uses."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def reindex_kind(manifest):
    """'folder' | 'history' for an index that can be re-scanned, else None."""
    task = manifest.get("task")
    if task in OFFICIAL_TASKS:
        return None
    kind = REINDEXABLE_TASKS.get(task)
    if kind == "folder" and manifest.get("source_root"):
        return kind
    if kind == "history" and manifest.get("repo"):
        return kind
    return None


def reindexable_info(index_dir):
    """{reindexable, kind, reason} from the manifest alone -- no embeddings are loaded."""
    from src.runtime_index import INDEX_NAMES, MANIFEST
    path = Path(index_dir)
    if path.resolve() in {INDEX_NAMES[n].resolve() for n in ("full", "lite")}:
        return {"reindexable": False, "kind": None,
                "reason": "the official APPS indexes are never re-scanned"}
    try:
        manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"reindexable": False, "kind": None, "missing": True, "reason": "no readable manifest"}
    kind = reindex_kind(manifest)
    if kind is None:
        return {"reindexable": False, "kind": None,
                "reason": "not a custom-folder or history index (no source to re-scan)"}
    return {"reindexable": True, "kind": kind, "reason": ""}


def allowed_index_names(default_index, environ):
    """Names a request may ask to reindex: the named indexes, the server's own default and whatever the
    operator listed in ALLOWED_INDEXES. A path that is none of these is refused, so a request cannot
    aim the operation at an arbitrary directory."""
    from src.runtime_index import INDEX_NAMES
    names = set(INDEX_NAMES) | {default_index}
    names |= {x.strip() for x in environ.get("ALLOWED_INDEXES", "").split(",") if x.strip()}
    return names


def check_index_name(name, default_index, environ):
    """Raise ReindexError unless `name` may be reindexed by an HTTP caller."""
    if not isinstance(name, str) or not name or "\x00" in name or len(name) > 512:
        raise ReindexError("invalid index name")
    if name not in allowed_index_names(default_index, environ):
        raise ReindexError(f"index {name!r} is not on the reindex allowlist")
    return name


def check_reindexable(name):
    """After the allowlist: the index directory for `name`, or ReindexError if its manifest says it must
    not (or cannot) be re-scanned. Reads the manifest only, so a refusal never loads a model."""
    from src.runtime_index import resolve_index_dir
    path = resolve_index_dir(name)
    info = reindexable_info(path)
    if not info["reindexable"]:
        raise ReindexError(f"index {name!r} cannot be reindexed: {info['reason']}",
                           status=404 if info.get("missing") else 400)
    return path


def _scan_folder(manifest):
    from src.indexing.code_chunker import chunk_folder
    root = Path(manifest["source_root"])
    if not root.is_dir():
        raise ReindexError(f"the source folder {str(root)!r} recorded in the index no longer exists",
                           status=404)
    chunks, stats = chunk_folder(root, chunk_methods=bool(manifest.get("chunk_methods")))
    if not chunks:
        raise ReindexError(f"no Python chunks found under {str(root)!r}; refusing to empty the index")
    return chunks, stats


def _folder_keys(chunks):
    """Unique identity per chunk: (file, kind, qualname, n-th occurrence of that triple)."""
    seen, keys = {}, []
    for c in chunks:
        base = (c["file"], c["kind"], c["qualname"])
        n = seen.get(base, 0)
        seen[base] = n + 1
        keys.append(base + (n,))
    return keys


def _classify(old, new):
    """Counts from two {key: hash} dicts."""
    added = sum(1 for k in new if k not in old)
    modified = sum(1 for k, h in new.items() if k in old and old[k] != h)
    unchanged = sum(1 for k, h in new.items() if k in old and old[k] == h)
    removed = sum(1 for k in old if k not in new)
    return {"added": added, "modified": modified, "unchanged": unchanged, "removed": removed}


def _plan_folder(index, manifest):
    """(new rows, chunks, classification, scan stats) for a folder index."""
    from src.runtime_index import sha256_text
    chunks, stats = _scan_folder(manifest)
    old_chunks = index.chunks or []
    old = {k: sha256_text(index.doc_texts[i])
           for i, k in enumerate(_folder_keys(old_chunks))} if old_chunks else {}
    new_keys = _folder_keys(chunks)
    new = {k: sha256_text(c["text"]) for k, c in zip(new_keys, chunks)}
    rows = [{"doc_id": c["chunk_id"], "text": c["text"], "content_hash": new[k]}
            for k, c in zip(new_keys, chunks)]
    same = (index.doc_ids == [r["doc_id"] for r in rows]
            and index.doc_texts == [r["text"] for r in rows] and old_chunks == chunks)
    return rows, chunks, _classify(old, new), stats, same


def _plan_history(index, manifest):
    """(rows, classification, ingest stats, identical?) for a git-history index."""
    from src.versioning.git_history import ingest
    repo = Path(manifest["repo"])
    if not (repo / ".git").exists():
        raise ReindexError(f"the repository {str(repo)!r} recorded in the index is no longer there",
                           status=404)
    rows, _, stats = ingest(repo, max_commits=manifest.get("commits") or None,
                            subdir=manifest.get("subdir"), progress=False)
    old_versions = index.versions or []
    old_pairs = {(v["snippet_id"], v["content_hash"]) for v in old_versions}
    old_lineages = {p[0] for p in old_pairs}
    new_pairs = {(r["lineage_id"], r["content_hash"]) for r in rows}
    counts = {"added": sum(1 for p in new_pairs if p not in old_pairs and p[0] not in old_lineages),
              "modified": sum(1 for p in new_pairs if p not in old_pairs and p[0] in old_lineages),
              "unchanged": len(new_pairs & old_pairs),
              "removed": len(old_pairs - new_pairs)}
    same = (index.doc_ids == [r["doc_id"] for r in rows]
            and index.doc_texts == [r["text"] for r in rows]
            and [v.get("commit") for v in old_versions] == [r["commit"] for r in rows])
    return rows, counts, stats, same


def reindex_index(index_dir, encode_docs, dry_run=False, batch_size=16):
    """Re-scan the source of `index_dir` and bring the index up to date. Returns a report dict.

    encode_docs(texts, batch_size=...) -> (n, dim) array; it is only ever called with texts that have no
    stored vector, and not at all when nothing changed."""
    import numpy as np

    from src.runtime_index import RuntimeIndex, sha256_text, write_index
    from src.versioning.version_index import build_versioned_index, embeddings_for
    t_start = time.time()
    index_dir = Path(index_dir)
    info = reindexable_info(index_dir)          # manifest only: refuses before any embedding is loaded
    if not info["reindexable"]:
        raise ReindexError(f"{index_dir} cannot be reindexed: {info['reason']}")
    try:
        index = RuntimeIndex.load(index_dir)
    except FileNotFoundError as exc:
        raise ReindexError(str(exc), status=404) from exc
    manifest, kind = index.manifest, info["kind"]

    t0 = time.time()
    if kind == "folder":
        rows, chunks, counts, scan_stats, same = _plan_folder(index, manifest)
        old_hash = {}
        for i, text in enumerate(index.doc_texts):
            old_hash.setdefault(sha256_text(text), index.E[i])
    else:
        rows, counts, scan_stats, same = _plan_history(index, manifest)
        chunks = None
        old_hash = {}
        for i, v in enumerate(index.versions or []):
            old_hash.setdefault(v["content_hash"], index.E[i])
    scan_seconds = time.time() - t0

    cache = dict(old_hash)
    would_encode = len({r["content_hash"] for r in rows} - set(cache))
    report = {"index": str(index_dir), "kind": kind, "source": manifest.get("source_root")
              or manifest.get("repo"), "dry_run": bool(dry_run), **counts,
              "n_docs_before": len(index.doc_ids), "n_docs_after": len(rows),
              "embeddings_recomputed": would_encode, "embeddings_reused": len(rows) - would_encode,
              "scan_seconds": round(scan_seconds, 3), "encode_seconds": 0.0, "written": False,
              "categories_dropped": False}
    if dry_run:
        report["elapsed_seconds"] = round(time.time() - t_start, 3)
        return report

    t0 = time.time()
    if would_encode:
        emb, stats = embeddings_for(rows, encode_docs, cache=cache, batch_size=batch_size)
        report["embeddings_recomputed"], report["embeddings_reused"] = stats["recomputed"], stats["reused"]
    else:
        emb = (np.stack([cache[r["content_hash"]] for r in rows]) if rows
               else np.zeros((0, index.E.shape[1]), dtype=np.float32))
    report["encode_seconds"] = round(time.time() - t0, 3)

    if not same:
        meta = {k: v for k, v in manifest.items() if k not in DERIVED_MANIFEST_KEYS}
        meta["last_reindex"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "added": counts["added"],
                                "modified": counts["modified"], "removed": counts["removed"],
                                "embeddings_recomputed": report["embeddings_recomputed"]}
        if kind == "folder":
            write_index(index_dir, emb, [r["doc_id"] for r in rows], [r["text"] for r in rows], meta,
                        chunks=chunks)
        else:
            meta["ingest_stats"] = scan_stats
            meta["commits"] = scan_stats["commits"]
            build_versioned_index(index_dir, rows, emb, meta)
        # tags are per row; after the rows changed they would point at the wrong chunks
        stale = index_dir / "categories.json"
        if stale.exists():
            stale.unlink()
            report["categories_dropped"] = True
        report["written"] = True
    report["elapsed_seconds"] = round(time.time() - t_start, 3)
    return report


def format_report(rep):
    head = "DRY RUN -- nothing written" if rep["dry_run"] else (
        "index updated" if rep["written"] else "index already up to date")
    lines = [f"[reindex] {rep['index']} ({rep['kind']}, from {rep['source']}): {head}",
             f"[reindex] chunks     : +{rep['added']} added  ~{rep['modified']} modified  "
             f"={rep['unchanged']} unchanged  -{rep['removed']} removed  "
             f"({rep['n_docs_before']} -> {rep['n_docs_after']})",
             f"[reindex] embeddings : {rep['embeddings_reused']} reused, "
             f"{rep['embeddings_recomputed']} {'to recompute' if rep['dry_run'] else 'recomputed'}",
             f"[reindex] time       : {rep['elapsed_seconds']:.2f}s "
             f"(scan {rep['scan_seconds']:.2f}s, encode {rep['encode_seconds']:.2f}s)"]
    if rep["categories_dropped"]:
        lines.append("[reindex] categories.json was dropped (per-row tags are stale); rebuild it with "
                     "src/build_categories.py")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", required=True,
                    help="a custom-folder or history index (name or path). The official indexes are "
                         "refused")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mock-encoder", action="store_true",
                    help="hashing encoder: no model, for smoke tests only")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)

    from src.runtime_index import MANIFEST, load_query_encoder, resolve_index_dir
    index_dir = resolve_index_dir(a.index)
    holder = {}

    def encode_docs(texts, batch_size=16):
        # loaded on first use: an index that is already up to date never loads a model
        if "enc" not in holder:
            manifest = json.loads((index_dir / MANIFEST).read_text(encoding="utf-8"))
            holder["enc"] = load_query_encoder(manifest, device=a.device, mock=a.mock_encoder,
                                               max_query_tokens=None)
        return holder["enc"].encode_docs(texts, batch_size=batch_size)

    try:
        report = reindex_index(index_dir, encode_docs, dry_run=a.dry_run, batch_size=a.batch_size)
    except ReindexError as exc:
        print(f"[reindex] refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2) if a.json else format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
