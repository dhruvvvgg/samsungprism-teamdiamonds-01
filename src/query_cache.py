"""Exact-query embedding cache for the SERVING path (CLI interactive mode, API, UI).

A demo asks the same question repeatedly -- an example query, a page refresh, a version comparison --
and encoding a query is nearly all of the latency (retrieval itself is 4-6 ms). Caching the query vector
turns the second identical request from seconds into milliseconds without changing any result: the vector
returned is the one the encoder produced for exactly that text under exactly that configuration.

Where it is NOT used, on purpose:

  * `run_official` and every benchmark script. `SearchService` builds its cache with size 0 unless the
    caller asks for one, and only the serving entry points (src/api.py, src/cli.py --interactive) ask.
    A benchmark that read a cached vector would be timing a dictionary lookup, and the official run
    encodes every query fresh. tests/test_query_cache.py pins both.
  * Anything that is not an exact repeat. There is no fuzzy or semantic matching: a query differing by
    one character is a miss.

The key is (query text, model, revision, query prompt, query token cap, dtype, int8). Change any of them
-- a different cap, prompt or model revision -- and the old vector is no longer served, because it was
computed under a different configuration.

Turn it off with QUERY_CACHE=0 (or QUERY_CACHE_SIZE=0); size it with QUERY_CACHE_SIZE (default 128).
"""
import os
import threading
from collections import OrderedDict

import numpy as np

DEFAULT_CACHE_SIZE = 128
OFF_VALUES = {"0", "false", "no", "off"}


def resolve_cache_size(environ=None, default=DEFAULT_CACHE_SIZE):
    """Cache size for a SERVING entry point, from the environment. 0 means disabled."""
    env = os.environ if environ is None else environ
    if str(env.get("QUERY_CACHE", "")).strip().lower() in OFF_VALUES:
        return 0
    raw = env.get("QUERY_CACHE_SIZE")
    if raw not in (None, ""):
        try:
            return max(0, int(raw))
        except ValueError:
            return default
    return default


def cache_key(query, model, revision, prefix, cap, dtype=None, int8=False):
    return (str(query), model, revision, prefix or "", cap, dtype, bool(int8))


class QueryEmbeddingCache:
    """A small thread-safe LRU of query vectors. maxsize 0 disables it (get always misses, put is a no-op)."""

    def __init__(self, maxsize=0):
        self.maxsize = max(0, int(maxsize))
        self._data = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self):
        return self.maxsize > 0

    def get(self, key):
        if not self.enabled:
            return None
        with self._lock:
            vec = self._data.get(key)
            if vec is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return vec

    def put(self, key, vec):
        if not self.enabled:
            return
        stored = np.array(vec, copy=True)
        stored.setflags(write=False)
        with self._lock:
            self._data[key] = stored
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def stats(self):
        with self._lock:
            return {"enabled": self.enabled, "maxsize": self.maxsize, "size": len(self._data),
                    "hits": self.hits, "misses": self.misses}
