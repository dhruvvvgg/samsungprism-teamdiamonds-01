"""One query against a built `runtime_index/`: load, encode the query, one mat-vec, format the hits.

The CLI (src/cli.py) and the API (src/api.py) are both thin shells over this class, so what you get from
a curl and from a terminal is the same code path -- including the lineage handling for a versioned index.

The corpus is NEVER encoded here. Only the query is.
"""
import threading
import time

from src.runtime_index import (DEFAULT_SERVING_QUERY_TOKENS, RuntimeIndex, load_query_encoder,
                               peak_rss_mb, resolve_index_dir, set_cpu_threads)


class SearchService:
    SERVING_DEFAULT = object()          # sentinel: "use the serving default", distinct from None

    def __init__(self, index_dir="full", device="cpu", mock=False, threads=None,
                 verify=False, cpu_dtype="fp32", int8=False, max_query_tokens=SERVING_DEFAULT,
                 query_cache_size=0, allow_lossy_int8=False):
        """device defaults to cpu: this is the serving path, and the whole point of the exported index
        is that answering a query needs no GPU.

        The model is loaded ONCE here, not per query. A process that keeps a SearchService alive (the
        API, or the CLI's --interactive mode) pays the model load once and then only the per-query
        forward pass, which is the difference between a usable demo and an unusable one.

        index_dir accepts a name ('full', 'lite', 'versions') or a path.
        cpu_dtype / int8 / max_query_tokens are serving-time knobs; none of them alter the stored
        document embeddings, so they never make the index and the query disagree about the model.

        max_query_tokens defaults to DEFAULT_SERVING_QUERY_TOKENS (1024), measured on dev as the
        largest cap that is not significantly worse than uncapped. Pass None (or 0 on the CLI) for no
        cap. The official run does not go through this class at all, so it stays uncapped.

        query_cache_size: entries in the exact-query embedding cache (src/query_cache.py). 0 -- the
        default -- means no cache, which is what every benchmark gets; only the serving entry points
        (the API, the CLI's interactive mode) pass a size."""
        if max_query_tokens is SearchService.SERVING_DEFAULT:
            max_query_tokens = DEFAULT_SERVING_QUERY_TOKENS
        self.index_dir = resolve_index_dir(index_dir)
        self.thread_info = set_cpu_threads(threads) if str(device).startswith("cpu") else {}
        t0 = time.time()
        self.index = RuntimeIndex.load(self.index_dir)
        self.load_index_seconds = time.time() - t0
        self.verify_problems = self.index.verify_files() if verify else None
        t0 = time.time()
        self.encoder = load_query_encoder(self.index.manifest, device=device, mock=mock,
                                          cpu_dtype=cpu_dtype, int8=int8,
                                          max_query_tokens=max_query_tokens,
                                          allow_lossy_int8=allow_lossy_int8)
        self.model_load_seconds = time.time() - t0
        self.device = device
        from src.query_cache import QueryEmbeddingCache
        self.query_cache = QueryEmbeddingCache(query_cache_size)
        self._index_lock = threading.RLock()        # guards self.index against a reload mid-search
        self._reindex_lock = threading.Lock()
        self._sync_index_flags()

    def _sync_index_flags(self):
        self.versioned = self.index.versions is not None
        self.version_numbers = (sorted({int(v["version"]) for v in self.index.versions})
                                if self.versioned else [])
        self.chunked = self.index.chunks is not None
        self.categorized = self.index.categories is not None
        self.index_bytes = sum(int(f.get("bytes", 0))
                               for f in (self.index.manifest.get("files") or {}).values())

    def reload_index(self):
        """Re-read the index from disk (after a reindex) without reloading the model."""
        fresh = RuntimeIndex.load(self.index_dir)
        with self._index_lock:
            self.index = fresh
            self._sync_index_flags()

    def reindex(self, dry_run=False):
        """Incremental update of a custom-folder / history index; the next search sees the change.
        Raises ReindexError for an index that must not be re-scanned (the official APPS indexes)."""
        from src.reindex import ReindexError, reindex_index
        if not self._reindex_lock.acquire(blocking=False):
            raise ReindexError("a reindex of this index is already running", status=409)
        try:
            report = reindex_index(self.index_dir, self.encoder.encode_docs, dry_run=dry_run)
            if report["written"]:
                self.reload_index()
            return report
        finally:
            self._reindex_lock.release()

    def describe(self):
        m = self.index.manifest
        return {"index_dir": str(self.index.dir), "kind": m.get("kind", "flat"),
                "n_docs": m.get("n_docs"), "dim": m.get("dim"), "model": m.get("model"),
                "revision": m.get("revision"), "preset": m.get("preset"),
                "query_prefix": m.get("query_prefix", ""), "versioned": self.versioned,
                "chunked": self.chunked, "categorized": self.categorized,
                "versions": self.version_numbers, "source_root": m.get("source_root"),
                "device": self.device, "threads": self.thread_info.get("threads"),
                "cpu_dtype": getattr(self.encoder, "cpu_dtype", None),
                "int8": getattr(self.encoder, "int8", False),
                "max_query_tokens": getattr(self.encoder, "max_query_tokens", None),
                "query_cache": self.query_cache.stats(),
                "load_index_seconds": round(self.load_index_seconds, 3),
                "model_load_seconds": round(self.model_load_seconds, 3),
                "peak_rss_mb": peak_rss_mb(), "verify_problems": self.verify_problems}

    def performance_info(self, query):
        """What the query cost and what it ran on, for the UI's performance panel. `truncated` is true
        when the query is longer than the serving cap, i.e. the encoder did not see all of it."""
        cap = getattr(self.encoder, "max_query_tokens", None)
        tokens = self.token_count(query)
        return {"query_tokens": tokens, "query_cap": cap,
                "truncated": bool(cap and tokens is not None and tokens > cap),
                "device": self.device, "threads": self.thread_info.get("threads"),
                "model": self.index.manifest.get("model"), "cpu_dtype": getattr(self.encoder, "cpu_dtype", None),
                "index_docs": self.index.manifest.get("n_docs"), "index_dim": self.index.manifest.get("dim"),
                "index_bytes": self.index_bytes, "index_size_mb": round(self.index_bytes / 1e6, 2)}

    def _cache_key(self, query):
        from src.query_cache import cache_key
        enc, m = self.encoder, self.index.manifest
        return cache_key(query, m.get("model"), m.get("revision"), getattr(enc, "query_prefix", ""),
                         getattr(enc, "max_query_tokens", None), getattr(enc, "cpu_dtype", None),
                         getattr(enc, "int8", False))

    def _encode_query(self, query):
        """(vector, encode_ms, cached). Served from the cache only for an exact repeat of the same
        text under the same model / revision / prompt / token cap."""
        t0 = time.time()
        key = self._cache_key(query) if self.query_cache.enabled else None
        if key is not None:
            vec = self.query_cache.get(key)
            if vec is not None:
                return vec, 1000 * (time.time() - t0), True
        vec = self.encoder.encode_query(query)
        if key is not None:
            self.query_cache.put(key, vec)
        return vec, 1000 * (time.time() - t0), False

    def _rows_for_category(self, category, rows=None):
        """Row indices whose tag matches `category` (an ast_family or a cluster label substring)."""
        import numpy as np
        if not self.categorized:
            raise ValueError(f"{self.index.dir} has no category tags; build it with "
                             f"src/build_categories.py")
        want = str(category).strip().lower()
        keep = [i for i, t in enumerate(self.index.categories)
                if t.get("ast_family", "").lower() == want
                or want in str(t.get("cluster_label", "")).lower()]
        if rows is not None:
            allowed = set(int(r) for r in rows)
            keep = [i for i in keep if i in allowed]
        return np.array(keep, dtype=np.int64)

    def categories(self):
        """Available tags and how many rows carry each, for a UI filter."""
        if not self.categorized:
            return {"families": {}, "clusters": {}}
        from collections import Counter
        fams = Counter(t.get("ast_family") for t in self.index.categories)
        clusters = Counter(t.get("cluster_label") for t in self.index.categories
                           if t.get("cluster_label"))
        return {"families": dict(fams.most_common()), "clusters": dict(clusters.most_common())}

    def route(self, query, allow_llm=False):
        """How this query would be routed, without running a search."""
        from src.retrieval.query_router import classify
        return classify(query, allow_llm=allow_llm)

    def token_count(self, text):
        """Tokens the query would use before any cap (None if the encoder cannot say)."""
        fn = getattr(self.encoder, "token_count", None)
        return fn(text) if fn else None

    # --- metadata providers -----------------------------------------------------------------------
    #
    # Each provider is `fn(service, row) -> dict`, and the dicts are merged in order into the hit.
    # This is an extension point on purpose: several features want to attach their own fields to a
    # result (source location, lineage, category, ...), and without a list they would all be editing
    # the same few lines of one function and colliding with each other. A provider that has nothing to
    # say returns {}.
    META_PROVIDERS = []

    @classmethod
    def register_meta_provider(cls, fn):
        """Add a provider. Returns fn, so it can be used as a decorator. Registering the same function
        twice is a no-op, which keeps a re-imported module from duplicating fields."""
        if fn not in cls.META_PROVIDERS:
            cls.META_PROVIDERS.append(fn)
        return fn

    def _meta_for(self, row):
        """Per-row extras a result should carry. A hit is only actionable if you can find the code
        again, so every provider that can locate or classify this row contributes."""
        meta = {}
        for provider in self.META_PROVIDERS:
            extra = provider(self, row)
            if extra:
                meta.update(extra)
        return meta

    def search(self, query, k=10, version=None, all_versions=False, preview_chars=240,
               candidate_factor=5, category=None):
        """Top-k hits for `query`.

        On a versioned index the default collapses each lineage to its best-scoring version; pass
        all_versions=True to let every version compete, or version=N to search only that version.
        A collapsing search asks the index for more candidates than k, because collapsing removes rows
        -- without that, a top-10 of one lineage's versions would collapse to a single result."""
        qvec, encode_ms, cached = self._encode_query(query)
        with self._index_lock:
            return self._rank(query, qvec, encode_ms, k, version, all_versions, preview_chars,
                              candidate_factor, category, cached)

    def _rank(self, query, qvec, encode_ms, k, version, all_versions, preview_chars,
              candidate_factor, category, cached=False):
        """Everything after the query is encoded: it reads self.index, so it runs under the lock."""
        from src.versioning.version_index import collapse_lineages, version_rows
        t = {"encode_query_ms": encode_ms}
        rows = None
        if version is not None:
            rows = version_rows(self.index, version)
            if len(rows) == 0:
                raise ValueError(f"no documents at version {version} in {self.index.dir}")
        collapse = self.versioned and not all_versions and version is None
        if category:
            # filtering narrows the row space BEFORE ranking, so k still returns k results
            rows = self._rows_for_category(category, rows)
            if len(rows) == 0:
                raise ValueError(f"no documents tagged {category!r} in {self.index.dir}")
        t0 = time.time()
        depth = max(k * candidate_factor, k + 20) if (collapse or category) else k
        hits = self.index.search(qvec, k=depth, rows=rows)
        if collapse:
            hits = collapse_lineages(hits, self.index, k=k)
        hits = hits[:k]
        t["search_ms"] = 1000 * (time.time() - t0)
        out = []
        for rank, (doc_id, score, row) in enumerate(hits, start=1):
            text = self.index.doc_texts[row]
            out.append(dict({"rank": rank, "doc_id": doc_id, "score": round(float(score), 6),
                             "preview": text[:preview_chars],
                             "truncated": len(text) > preview_chars}, **self._meta_for(row)))
        return {"query": query, "k": k, "collapsed_lineages": collapse,
                "version_filter": version, "category_filter": category, "hits": out,
                "timings_ms": dict({kk: round(vv, 2) for kk, vv in t.items()}, cached=bool(cached))}

    def explain_hits(self, query, hits, route=None, groups=None):
        """Attach a `why` block to each hit: route, score gap to the next result, and matched terms.
        Lexical and arithmetic only -- no model call. When results are grouped by lineage the gap of a
        group's head is measured to the NEXT GROUP's head, which is the result the reader sees below it."""
        from src.retrieval.explain import explain_hit
        with self._index_lock:
            for i, h in enumerate(hits):
                row = self.index.row_of(h["doc_id"])
                text = self.index.doc_texts[row] if row is not None else h.get("preview", "")
                nxt = hits[i + 1]["score"] if i + 1 < len(hits) else None
                h["why"] = explain_hit(query, text, h["score"], nxt, route)
        for i, g in enumerate(groups or []):
            nxt = groups[i + 1]["best"]["score"] if i + 1 < len(groups) else None
            g["best"]["why"]["score_gap_to_next"] = (None if nxt is None else
                                                      round(g["best"]["score"] - nxt, 6))
            g["best"]["why"]["gap_basis"] = "next lineage"
        return hits

    def get_doc(self, doc_id):
        """One document in full, with the same location / lineage metadata a search hit carries.

        `has_source_file` says whether `file` / `location` are real: only a code-folder or history
        index knows which file a document came from. An APPS document is identified by its id alone,
        and the UI must not invent a path for it."""
        with self._index_lock:
            row = self.index.row_of(doc_id)
            if row is None:
                raise KeyError(f"no document {doc_id!r} in {self.index.dir}")
            text = self.index.doc_texts[row]
            meta = self._meta_for(row)
            return dict({"doc_id": doc_id, "text": text, "n_lines": max(1, len(text.splitlines())),
                         "first_line": meta.get("start_line") or 1,
                         "has_source_file": bool(self.chunked)}, **meta)

    def compare(self, query, version_a, version_b, k=10, preview_chars=240):
        """The same query against two versions, marking what appeared, disappeared or moved rank.
        The query is encoded once; only the ranking is repeated."""
        from src.versioning.compare import compare_hits
        if not self.versioned:
            raise ValueError(f"{self.index.dir} is not a versioned index; compare needs one")
        qvec, encode_ms, cached = self._encode_query(query)
        with self._index_lock:
            ra = self._rank(query, qvec, encode_ms, k, version_a, False, preview_chars, 5, None, cached)
            rb = self._rank(query, qvec, encode_ms, k, version_b, False, preview_chars, 5, None, cached)
        hits_a, hits_b, summary = compare_hits(ra["hits"], rb["hits"])
        return {"query": query, "k": k, "version_a": version_a, "version_b": version_b,
                "a": hits_a, "b": hits_b, "summary": summary,
                "timings_ms": {"encode_query_ms": round(encode_ms, 2), "cached": cached,
                               "search_ms": round(ra["timings_ms"]["search_ms"]
                                                  + rb["timings_ms"]["search_ms"], 2)}}

    def diff(self, snippet_id, version_a, version_b, context=3):
        """Unified diff of one lineage between two of its versions (no model involved)."""
        from src.versioning.compare import unified_diff
        from src.versioning.version_index import history
        with self._index_lock:
            by_version = {int(r["version"]): r for r in history(self.index, snippet_id)}
            for v in (version_a, version_b):
                if int(v) not in by_version:
                    raise KeyError(f"{snippet_id!r} has no version {v} "
                                   f"(it has {sorted(by_version)})")
            ra, rb = by_version[int(version_a)], by_version[int(version_b)]
            out = unified_diff(self.index.doc_texts[ra["row"]], self.index.doc_texts[rb["row"]],
                               f"{snippet_id} v{version_a}", f"{snippet_id} v{version_b}", context)
        out.update({"snippet_id": snippet_id, "version_a": int(version_a), "version_b": int(version_b),
                    "commit_a": ra.get("commit_short"), "commit_b": rb.get("commit_short"),
                    "same_content_hash": ra["content_hash"] == rb["content_hash"]})
        return out

    def history(self, snippet_id, preview_chars=240):
        """Every version of one snippet, oldest to newest (no ranking involved)."""
        from src.versioning.version_index import history
        with self._index_lock:
            return self._history(history(self.index, snippet_id), snippet_id, preview_chars)

    def _history(self, rows, snippet_id, preview_chars):
        def one(r):
            rec = {"version": r["version"], "doc_id": r["doc_id"],
                   "content_hash": r["content_hash"][:12], "mutation": r.get("mutation"),
                   "preview": self.index.doc_texts[r["row"]][:preview_chars]}
            # a git-history index also knows which commit each version came from
            rec.update({k: r[k] for k in ("commit_short", "commit_date", "commit_subject", "change")
                        if r.get(k) is not None})
            return rec
        return {"snippet_id": snippet_id, "n_versions": len(rows), "versions": [one(r) for r in rows]}


def _location_meta(service, row):
    """Where the code lives (code-folder and history indexes)."""
    if not service.chunked:
        return {}
    c = service.index.chunks[row]
    return {"file": c["file"], "start_line": c["start_line"], "end_line": c["end_line"],
            "location": f"{c['file']}:{c['start_line']}-{c['end_line']}",
            "kind": c.get("kind"), "name": c.get("name"), "qualname": c.get("qualname")}


def _category_meta(service, row):
    """Offline tags, when the index carries them."""
    if not service.categorized:
        return {}
    t = service.index.categories[row]
    out = {"ast_family": t.get("ast_family"), "ast_reason": t.get("ast_reason")}
    if t.get("cluster_label"):
        out["cluster_label"] = t["cluster_label"]
    return out


def _lineage_meta(service, row):
    """Where the row sits in a version history. A history index is BOTH located and versioned, which
    is why these are separate providers merged together rather than one shadowing the other."""
    if not service.versioned:
        return {}
    v = service.index.versions[row]
    meta = {"snippet_id": v["snippet_id"], "version": v["version"],
            "content_hash": v.get("content_hash", "")[:12], "mutation": v.get("mutation")}
    for optional in ("commit_short", "commit_date", "commit_subject", "change"):
        if v.get(optional) is not None:
            meta[optional] = v[optional]
    return meta


SearchService.register_meta_provider(_location_meta)
SearchService.register_meta_provider(_lineage_meta)
SearchService.register_meta_provider(_category_meta)
