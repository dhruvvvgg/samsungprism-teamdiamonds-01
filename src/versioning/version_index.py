"""Content-hash index over a version history: re-embed only what changed, retrieve a chosen version,
and collapse a lineage to its best version (P1 + Bonus).

The index is a normal `runtime_index/` (src/runtime_index.py) with one extra file, `versions.json`,
carrying {snippet_id, version, content_hash} per row. Document ids are `"{snippet_id}@v{version}"`, so a
served index holds every version at once and the choice of which versions to search is made per query:

    default           collapse each lineage to its best-scoring version   (near-duplicates stop flooding)
    --all-versions    every version competes on its own
    --version N       only version N
    --history ID      that snippet's versions, oldest to newest (no ranking involved)

The incremental rebuild is the point of the content hash: `embeddings_for` takes a {hash: vector} cache,
encodes only the hashes it has never seen, and reuses the rest. Two versions that hash the same are the
same text after normalisation, so reusing the embedding is exact, not an approximation.
"""
import time

import numpy as np

from src.versioning.content_hash import content_hash

DOC_ID_SEP = "@v"


def doc_id_for(snippet_id, version):
    return f"{snippet_id}{DOC_ID_SEP}{version}"


def parse_doc_id(doc_id):
    """('snip0001', 3) from 'snip0001@v3'. Raises ValueError on anything else."""
    sid, _, ver = str(doc_id).rpartition(DOC_ID_SEP)
    if not sid or not ver.isdigit():
        raise ValueError(f"not a versioned document id: {doc_id!r}")
    return sid, int(ver)


def rows_for(fixture, versions=None):
    """Flatten the fixture into index rows, oldest version first.

    `versions`: iterable of version numbers to include (default: all). Row order is stable -- snippet
    order, then version order -- so an index rebuilt from the same fixture is byte-comparable."""
    keep = None if versions is None else set(int(v) for v in versions)
    rows = []
    for s in fixture["snippets"]:
        for v in s["versions"]:
            if keep is not None and v["version"] not in keep:
                continue
            rows.append({"doc_id": doc_id_for(s["snippet_id"], v["version"]),
                         "snippet_id": s["snippet_id"], "version": v["version"],
                         "content_hash": v.get("content_hash") or content_hash(v["text"]),
                         "mutation": v.get("mutation", "unknown"), "text": v["text"]})
    return rows


def embeddings_for(rows, encode_fn, cache=None, batch_size=32):
    """(embeddings (N, dim), stats) for `rows`, embedding each distinct content hash at most once.

    cache: {content_hash: np.ndarray} reused across calls -- pass the same dict to make a rebuild
    incremental, or a fresh one to force a full rebuild. Mutated in place.

    stats: recomputed / reused / rows / distinct_hashes / seconds / reuse_pct. `reused` counts rows
    served from the cache, which includes duplicates *within* this call (the same text at two versions)
    as well as hits from earlier calls -- both are work genuinely avoided."""
    cache = {} if cache is None else cache
    t0 = time.time()
    wanted = []
    for r in rows:
        if r["content_hash"] not in cache and r["content_hash"] not in wanted:
            wanted.append(r["content_hash"])
    by_hash = {}
    for r in rows:
        by_hash.setdefault(r["content_hash"], r["text"])
    if wanted:
        new = np.asarray(encode_fn([by_hash[h] for h in wanted], batch_size=batch_size),
                         dtype=np.float32)
        if new.shape[0] != len(wanted):
            raise ValueError(f"encoder returned {new.shape[0]} vectors for {len(wanted)} texts")
        for h, vec in zip(wanted, new):
            cache[h] = vec
    emb = (np.stack([cache[r["content_hash"]] for r in rows]) if rows
           else np.zeros((0, 0), dtype=np.float32))
    return emb, {"rows": len(rows), "distinct_hashes": len({r["content_hash"] for r in rows}),
                 "recomputed": len(wanted), "reused": len(rows) - len(wanted),
                 "reuse_pct": (100.0 * (len(rows) - len(wanted)) / len(rows)) if rows else 0.0,
                 "seconds": time.time() - t0}


VERSION_EXTRAS = ("commit", "commit_short", "commit_date", "commit_subject", "change", "lineage_id",
                  "link_kind", "link_similarity", "link_scope", "link_from")
CHUNK_FIELDS = ("file", "start_line", "end_line", "kind", "name", "qualname")


def build_versioned_index(out_dir, rows, embeddings, meta):
    """Write rows + embeddings as a versioned runtime index.

    Rows from real git history also carry a source location and the commit they came from. Those ride
    along in chunks.json and in the version records, so a history hit can say both where the code lives
    and which commit it belongs to -- a version number alone is not something anyone can act on."""
    from src.runtime_index import write_index
    versions = []
    for r in rows:
        rec = {"snippet_id": r["snippet_id"], "version": r["version"],
               "content_hash": r["content_hash"], "mutation": r["mutation"]}
        rec.update({k: r[k] for k in VERSION_EXTRAS if k in r})
        versions.append(rec)
    chunks = None
    if rows and all(k in rows[0] for k in ("file", "start_line", "end_line")):
        chunks = [{k: r.get(k) for k in CHUNK_FIELDS} for r in rows]
    return write_index(out_dir, embeddings, [r["doc_id"] for r in rows], [r["text"] for r in rows],
                       meta, versions=versions, chunks=chunks)


def version_rows(index, version):
    """Row indices of an index whose version == `version` (for version-targeted retrieval)."""
    if index.versions is None:
        raise ValueError(f"{index.dir} is not a versioned index (no versions.json); "
                         "--version / --history need one built by src/build_version_index.py")
    return np.array([i for i, v in enumerate(index.versions) if int(v["version"]) == int(version)],
                    dtype=np.int64)


def collapse_lineages(hits, index, k=None):
    """Keep only the best-scoring version per lineage, preserving score order.

    `hits`: [(doc_id, score, row), ...] best first, as RuntimeIndex.search returns. Near-identical
    versions of one snippet otherwise occupy several of the top-10 slots between them and push different
    snippets out; collapsing is what makes the default result list one entry per thing that exists."""
    seen, out = set(), []
    for doc_id, score, row in hits:
        lineage = index.versions[row]["snippet_id"] if index.versions else doc_id
        if lineage in seen:
            continue
        seen.add(lineage)
        out.append((doc_id, score, row))
        if k is not None and len(out) >= k:
            break
    return out


def history(index, snippet_id):
    """That snippet's versions oldest to newest: [{version, doc_id, content_hash, mutation, row}]."""
    if index.versions is None:
        raise ValueError(f"{index.dir} is not a versioned index (no versions.json)")
    rows = [dict(v, row=i, doc_id=index.doc_ids[i]) for i, v in enumerate(index.versions)
            if v["snippet_id"] == snippet_id]
    if not rows:
        raise KeyError(f"no snippet {snippet_id!r} in {index.dir}")
    return sorted(rows, key=lambda r: int(r["version"]))


def duplication_at_k(hits, index, k=10):
    """How much of a top-k list is taken up by repeat versions of a lineage already in it.

    Returns {slots, distinct_lineages, duplicate_slots, duplicate_pct}. duplicate_slots is the number of
    results beyond the first for each lineage -- exactly the slots collapsing would free."""
    top = hits[:k]
    lineages = [index.versions[row]["snippet_id"] if index.versions else doc for doc, _, row in top]
    distinct = len(set(lineages))
    return {"slots": len(top), "distinct_lineages": distinct,
            "duplicate_slots": len(top) - distinct,
            "duplicate_pct": (100.0 * (len(top) - distinct) / len(top)) if top else 0.0}
