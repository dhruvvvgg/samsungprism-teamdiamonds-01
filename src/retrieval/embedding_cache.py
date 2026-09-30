"""fp16 .npy embedding cache with a metadata sidecar. Check-if-exists-else-rebuild: the cache key covers
model / revision / max_seq_length / dtype and a hash of the exact input strings, so a changed prompt,
model or text list can never be served stale."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.utils_io import save_npy_atomic, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "embeddings"


def texts_hash(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def cache_paths(name, texts, model_key, cache_dir=None):
    """The (.npy, .json) pair this (name, texts, model_key) hashes to. Single source of truth for the
    cache key, so the build path and the cache-only path can never drift apart."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    key = hashlib.sha256(f"{model_key}|{name}|{texts_hash(texts)}".encode()).hexdigest()[:20]
    return cache_dir / f"{name}__{key}.npy", cache_dir / f"{name}__{key}.json"


def load_cached(name, texts, model_key, cache_dir=None):
    """Return (embeddings float32, meta) if already cached, else None. Never encodes -- use this when a
    miss should be reported rather than silently costing a GPU model load."""
    npy, meta_path = cache_paths(name, texts, model_key, cache_dir)
    if not (npy.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    meta["cache_hit"] = True
    return np.load(npy).astype(np.float32), meta


def get_or_build(name, texts, encode_fn, model_key, cache_dir=None):
    """Return (embeddings float32 (N, D), meta). `encode_fn(texts) -> np.ndarray` runs only on a miss.
    meta has encode_seconds (of the build that produced the cache), n, model_key."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    npy, meta_path = cache_paths(name, texts, model_key, cache_dir)
    hit = load_cached(name, texts, model_key, cache_dir)
    if hit is not None:
        return hit
    t0 = time.time()
    emb = np.asarray(encode_fn(texts), dtype=np.float32)
    meta = {"name": name, "n": len(texts), "model_key": model_key,
            "encode_seconds": round(time.time() - t0, 3), "cache_hit": False}
    # .npy first, then the meta sidecar: load_cached() requires BOTH, so a kill between them leaves
    # the entry looking absent (re-encoded next time) rather than looking present but truncated.
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_npy_atomic(npy, emb.astype(np.float16))
    write_json_atomic(meta_path, meta)
    return emb.astype(np.float16).astype(np.float32), meta   # return exactly what a later hit returns
