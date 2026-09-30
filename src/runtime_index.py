"""The served index: a directory of files that answers queries WITHOUT re-encoding the corpus.

    runtime_index/
      manifest.json      what produced it (model, revision, prompts, dim, dtype) + sha256 of every file
      embeddings.npy     fp16 (N, dim), L2-normalised, row i belongs to doc_ids[i]
      doc_ids.json       [doc_id, ...]
      corpus_texts.json  [text, ...]           aligned with doc_ids
      versions.json      [{snippet_id, version, content_hash}, ...]   versioned indexes only (P1/Bonus)

The official run exports this from the corpus embeddings it already computed (no second encode), and
`src/build_index.py` builds it from scratch. Everything downstream -- `src/cli.py`, `src/api.py`,
`src/bench_cpu.py` -- only ever loads it, encodes ONE query, and does a single mat-vec.

The manifest is the contract: the query prompt is stored in it, so a served query is prefixed exactly
the way the corpus was encoded. Serving an index with a different model or prompt than it was built
with is the quiet failure this file exists to prevent.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.utils_io import save_npy_atomic, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
# Named indexes, so --index full / --index lite works without anyone memorising directory names.
# "full" is the submitted system: F2LLM-v2-1.7B, official test NDCG@10 0.9376 / MRR@10 0.9238.
# "lite" is the F2LLM-v2-0.6B speed/quality tradeoff (official test 0.90443 / 0.88397) and is NOT the
# submitted system -- anything reported from it has to say so.
INDEX_NAMES = {"full": ROOT / "runtime_index", "lite": ROOT / "runtime_index_lite",
               "versions": ROOT / "version_index", "history": ROOT / "history_index"}

# Default query-length cap for SERVING (CLI, API, frontend, bench_cpu). Not used by the official run,
# by build_index.py, or by any dev script -- those stay uncapped, so the submitted result is unaffected.
#
# 1024 was chosen from the dev sweep (src/eval/dev_query_cap.py, 1,000-query slice, 1.7B):
#   uncapped 0.9299 | 1024 -> 0.9289 (-0.0010, CI [-0.004, 0.000], 2.6% of queries truncated)
#                   |  512 -> -0.0077 (significant loss) | 256 -> -0.0784
# 1024 is the largest cap whose CI still touches zero, i.e. the only one not measurably worse, while
# bounding the tail: query lengths run p50 325, p99 1,230, max 13,601 tokens, and attention cost grows
# faster than linearly, so the cap is what stops one pathological query dominating a demo.
DEFAULT_SERVING_QUERY_TOKENS = 1024

MANIFEST = "manifest.json"
EMBEDDINGS = "embeddings.npy"
DOC_IDS = "doc_ids.json"
CORPUS_TEXTS = "corpus_texts.json"
VERSIONS = "versions.json"
CHUNKS = "chunks.json"        # code-folder indexes: file + line span per row
CATEGORIES = "categories.json"  # optional: offline tags per row (ast family, cluster)
INDEX_VERSION = 1


def resolve_query_cap(value):
    """CLI/API query cap: None -> the serving default, 0 (or negative) -> uncapped, else the value.

    0 has to mean "uncapped" explicitly, because None already means "unspecified, use the default" --
    without a separate spelling there would be no way to ask a serving entry point for no cap at all."""
    if value is None:
        return DEFAULT_SERVING_QUERY_TOKENS
    value = int(value)
    return None if value <= 0 else value


def resolve_index_dir(name_or_path):
    """'full' / 'lite' / 'versions' -> the matching directory; anything else is used as a path."""
    if name_or_path is None:
        return INDEX_NAMES["full"]
    key = str(name_or_path).strip().lower()
    return INDEX_NAMES[key] if key in INDEX_NAMES else Path(name_or_path)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def physical_cores():
    """Physical core count, or the best available estimate. Oversubscribing torch with hyperthreads
    makes single-query latency worse, not better, so the CPU path pins threads to physical cores."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n), "psutil(logical=False)"
    except Exception:  # noqa: BLE001  psutil is optional
        pass
    logical = os.cpu_count() or 1
    return max(1, logical // 2), f"os.cpu_count()={logical} // 2 (psutil unavailable)"


def set_cpu_threads(n=None):
    """Pin torch (and BLAS) to `n` threads, defaulting to physical cores. Returns an info dict.

    Must run before the first heavy torch op. The env vars matter for the BLAS behind numpy's mat-vec,
    which is the search itself."""
    source = "explicit --threads"
    if n is None:
        n, source = physical_cores()
    n = max(1, int(n))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
    info = {"threads": n, "source": source, "logical_cpus": os.cpu_count()}
    try:
        import torch
        torch.set_num_threads(n)
        info["torch_num_threads"] = torch.get_num_threads()
    except ImportError:
        info["torch_num_threads"] = None
    return info


def write_index(out_dir, embeddings, doc_ids, doc_texts, meta, versions=None, chunks=None,
                categories=None):
    """Write a runtime index atomically and return its manifest (which includes each file's sha256).

    embeddings: (N, dim) array; stored as fp16 (half the bytes, and what the embedding cache already
    holds -- the ranking difference against fp32 is far below the metric's resolution)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = np.asarray(embeddings)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {emb.shape}")
    if len(doc_ids) != emb.shape[0] or len(doc_texts) != emb.shape[0]:
        raise ValueError(f"doc_ids ({len(doc_ids)}) / texts ({len(doc_texts)}) do not match "
                         f"embeddings rows ({emb.shape[0]})")
    save_npy_atomic(out_dir / EMBEDDINGS, emb.astype(np.float16))
    write_json_atomic(out_dir / DOC_IDS, [str(d) for d in doc_ids])
    write_json_atomic(out_dir / CORPUS_TEXTS, list(doc_texts))
    files = [EMBEDDINGS, DOC_IDS, CORPUS_TEXTS]
    if versions is not None:
        if len(versions) != emb.shape[0]:
            raise ValueError("versions must have one entry per row")
        write_json_atomic(out_dir / VERSIONS, list(versions))
        files.append(VERSIONS)
    if chunks is not None:
        if len(chunks) != emb.shape[0]:
            raise ValueError("chunks must have one entry per row")
        write_json_atomic(out_dir / CHUNKS, list(chunks))
        files.append(CHUNKS)
    if categories is not None:
        if len(categories) != emb.shape[0]:
            raise ValueError("categories must have one entry per row")
        write_json_atomic(out_dir / CATEGORIES, list(categories))
        files.append(CATEGORIES)
    manifest = dict(meta)
    manifest.update({
        "index_version": INDEX_VERSION,
        "kind": ("versioned" if versions is not None
                 else "code" if chunks is not None else "flat"),
        "n_docs": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "stored_dtype": "float16",
        "files": {f: {"sha256": sha256_file(out_dir / f), "bytes": (out_dir / f).stat().st_size}
                  for f in files},
    })
    write_json_atomic(out_dir / MANIFEST, manifest, indent=2, default=str)
    return manifest


class RuntimeIndex:
    """A loaded index. Holds embeddings as float32 for a fast mat-vec (8,765 x 2,048 is ~72 MB)."""

    def __init__(self, directory, manifest, embeddings, doc_ids, doc_texts, versions=None,
                 chunks=None, categories=None):
        self.dir = Path(directory)
        self.manifest, self.E = manifest, embeddings
        self.doc_ids, self.doc_texts, self.versions = doc_ids, doc_texts, versions
        self.chunks = chunks
        self.categories = categories
        self._by_id = {d: i for i, d in enumerate(doc_ids)}

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        mpath = d / MANIFEST
        if not mpath.exists():
            raise FileNotFoundError(
                f"No runtime index at {d} (missing {MANIFEST}). Build one with:\n"
                f"  python src/build_index.py --preset f2llm-v2-1.7b --device cuda --out {d}\n"
                f"or export it from an official run (src/eval/run_official.py writes it automatically).")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        emb = np.load(d / EMBEDDINGS).astype(np.float32)
        doc_ids = json.loads((d / DOC_IDS).read_text(encoding="utf-8"))
        doc_texts = json.loads((d / CORPUS_TEXTS).read_text(encoding="utf-8"))
        versions = None
        if (d / VERSIONS).exists():
            versions = json.loads((d / VERSIONS).read_text(encoding="utf-8"))
        chunks = None
        if (d / CHUNKS).exists():
            chunks = json.loads((d / CHUNKS).read_text(encoding="utf-8"))
        categories = None
        if (d / CATEGORIES).exists():
            categories = json.loads((d / CATEGORIES).read_text(encoding="utf-8"))
        if emb.shape[0] != len(doc_ids):
            raise ValueError(f"{d}: {emb.shape[0]} embedding rows but {len(doc_ids)} doc ids")
        return cls(d, manifest, emb, doc_ids, doc_texts, versions, chunks, categories)

    def verify_files(self):
        """Re-hash every file and compare with the manifest. Returns a list of problem strings (empty
        means the index on disk is byte-identical to the one the manifest describes)."""
        problems = []
        for name, rec in (self.manifest.get("files") or {}).items():
            path = self.dir / name
            if not path.exists():
                problems.append(f"{name}: missing")
                continue
            got = sha256_file(path)
            if got != rec.get("sha256"):
                problems.append(f"{name}: sha256 {got[:12]}... != manifest "
                                f"{str(rec.get('sha256'))[:12]}...")
        return problems

    def row_of(self, doc_id):
        return self._by_id.get(doc_id)

    def search(self, qvec, k=10, rows=None):
        """Top-k by cosine (rows are L2-normalised). `rows`: optional array of row indices to restrict
        to (used by version-targeted retrieval). Returns [(doc_id, score, row), ...] best first."""
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        if rows is None:
            scores, idx_space = self.E @ q, None
        else:
            rows = np.asarray(rows, dtype=np.int64)
            scores, idx_space = self.E[rows] @ q, rows
        k = int(min(k, len(scores)))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        out = []
        for t in top:
            row = int(t) if idx_space is None else int(idx_space[t])
            out.append((self.doc_ids[row], float(scores[t]), row))
        return out


class HashingQueryEncoder:
    """Deterministic keyed-hash encoder: NO model, NO download. For smoke tests and CI only.

    It produces a stable unit vector per input string (token hashes summed into buckets), so a tiny
    index built with it and a query encoded with it retrieve consistently -- enough to exercise the
    plumbing end to end. Its rankings carry no semantic meaning; never report a metric from it."""

    def __init__(self, dim, max_query_tokens=None):
        self.dim = int(dim)
        self.model_name = "mock/hashing-encoder"
        self.load_seconds = 0.0
        self.cpu_dtype, self.int8 = "fp32", False
        # the mock honours the cap too (by word count), so a capped service reports and behaves
        # consistently whichever encoder is behind it -- a mock that ignored it would make the
        # smoke tests pass while the real serving path was misconfigured
        self.max_query_tokens = max_query_tokens

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        tokens = [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t]
        if self.max_query_tokens:
            tokens = tokens[:self.max_query_tokens]
        for tok in tokens or ["__empty__"]:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % self.dim
            v[bucket] += 1.0 if h[4] % 2 else -1.0
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def encode_query(self, text):
        return self._vec(text)

    def encode_queries(self, texts, batch_size=8):
        return self.encode_docs(texts, batch_size=batch_size)

    def encode_docs(self, texts, batch_size=32):
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def token_count(self, text):
        """Whitespace-ish token count -- there is no tokenizer here, and the benchmarks that report
        token counts must still produce a number under --mock-encoder."""
        return len(text.split())


class PresetQueryEncoder:
    """The real encoder, configured from the manifest so the query prompt matches the corpus encoding.

    Serving-time options, none of which touch the stored document embeddings:
      cpu_dtype         'fp32' (default) or 'bf16' -- weights only, CPU
      int8              dynamic int8 quantisation of Linear layers (measure before trusting it)
      max_query_tokens  truncate the QUERY at this many tokens. Documents are never truncated here:
                        they were encoded once, at the preset's full length, when the index was built.
    """

    def __init__(self, manifest, device="cpu", max_seq_length=None, cpu_dtype="fp32", int8=False,
                 max_query_tokens=None):
        import time

        from src.retrieval.dense_encoder import DenseEncoder
        if cpu_dtype not in ("fp32", "bf16"):
            raise ValueError(f"cpu_dtype must be fp32 or bf16, got {cpu_dtype!r}")
        self.query_prefix = manifest.get("query_prefix", "")
        self.doc_prefix = manifest.get("doc_prefix", "")
        self.model_name = manifest["model"]
        on_cpu = str(device).startswith("cpu")
        # fp16 is a CUDA-only win; on CPU it is emulated and slower, so CPU gets fp32 or explicit bf16
        dtype = cpu_dtype if on_cpu else manifest.get("dtype")
        t0 = time.time()
        self.dense = DenseEncoder(self.model_name, device,
                                  max_seq_length or manifest.get("max_seq_length", 8192),
                                  trust_remote_code=manifest.get("trust_remote_code", False),
                                  dtype=dtype, revision=manifest.get("revision"),
                                  expect_eos=manifest.get("expect_eos", False),
                                  allow_cpu_half=(cpu_dtype == "bf16"))
        if int8:
            self.dense.quantize_dynamic_int8()
        self.max_query_tokens = max_query_tokens
        self.doc_max_seq_length = max_seq_length or manifest.get("max_seq_length", 8192)
        if max_query_tokens:
            self.dense.set_max_seq_length(max_query_tokens)
        self.cpu_dtype, self.int8 = cpu_dtype, bool(self.dense.quantized_int8)
        self.load_seconds = time.time() - t0

    def encode_query(self, text):
        return self.dense.embed([self.query_prefix + text], batch_size=1)[0]

    def encode_queries(self, texts, batch_size=8):
        return self.dense.embed([self.query_prefix + t for t in texts], batch_size=batch_size)

    def encode_docs(self, texts, batch_size=32):
        """Documents are encoded at the preset's full length, never at the query cap: this is what an
        incremental reindex uses, and its vectors must agree with the ones the index was built with."""
        texts = [self.doc_prefix + t for t in texts]
        if not self.max_query_tokens:
            return self.dense.embed(texts, batch_size=batch_size)
        self.dense.set_max_seq_length(self.doc_max_seq_length)
        try:
            return self.dense.embed(texts, batch_size=batch_size)
        finally:
            self.dense.set_max_seq_length(self.max_query_tokens)

    def token_count(self, text):
        """Tokens in the full query prompt before any cap, so a cap's bite is measurable."""
        return self.dense.token_count(self.query_prefix + text)


def make_doc_encoder(preset_name="f2llm-v2-1.7b", device="cuda", mock=False, variant="registry+full",
                     mock_dim=64):
    """(encoder, manifest_meta) for building an index. The encoder exposes .encode_docs(texts, batch_size).

    One place decides how documents are encoded and what the manifest then claims about them, so every
    builder (corpus index, versioned index, benchmarks) writes a manifest that matches what it did."""
    from src.retrieval.query_variants import format_query, parse_variant
    _, transform = parse_variant(variant)
    if transform != "full":
        raise ValueError(f"variant {variant!r} rewrites the query text; a runtime index stores only a "
                         f"prefix, so use a '+full' variant")
    if mock:
        enc = HashingQueryEncoder(dim=mock_dim)
        return enc, {"preset": "mock", "model": enc.model_name, "revision": "n/a", "query_prefix": "",
                     "doc_prefix": "", "dense_variant": variant, "max_seq_length": 0, "dtype": "fp32",
                     "trust_remote_code": False, "expect_eos": False,
                     "warning": "built with the hashing mock encoder: plumbing only, scores are "
                                "meaningless -- never report a metric from this index"}
    from src.retrieval.dense_encoder import DenseEncoder
    from src.retrieval.model_presets import PRESETS
    if preset_name not in PRESETS:
        raise ValueError(f"unknown preset {preset_name!r}; available: {sorted(PRESETS)}")
    p = PRESETS[preset_name]
    if str(device).startswith("cpu"):
        set_cpu_threads()
    dense = DenseEncoder(p["model"], device, p["max_seq_length"],
                         trust_remote_code=p["trust_remote_code"],
                         dtype=("fp32" if str(device).startswith("cpu") else p["dtype"]),
                         revision=p["revision"], expect_eos=p["expect_eos"])
    doc_prefix = p.get("doc_prefix", "")

    class _DocEncoder:
        model_name = p["model"]
        dense_model = dense

        def encode_docs(self, texts, batch_size=32):
            return dense.embed([doc_prefix + t for t in texts], batch_size=batch_size)

    return _DocEncoder(), {"preset": preset_name, "model": p["model"], "revision": p["revision"],
                           "query_prefix": format_query("", variant), "doc_prefix": doc_prefix,
                           "dense_variant": variant, "max_seq_length": p["max_seq_length"],
                           "dtype": p["dtype"], "trust_remote_code": p["trust_remote_code"],
                           "expect_eos": p.get("expect_eos", False)}


def load_query_encoder(manifest, device="cpu", mock=False, cpu_dtype="fp32", int8=False,
                       max_query_tokens=None):
    """Mock encoder if asked (or if the index was built by one), else the manifest's real model."""
    if mock or manifest.get("model") == "mock/hashing-encoder":
        return HashingQueryEncoder(manifest["dim"], max_query_tokens=max_query_tokens)
    return PresetQueryEncoder(manifest, device=device, cpu_dtype=cpu_dtype, int8=int8,
                              max_query_tokens=max_query_tokens)


def peak_rss_mb():
    """Peak resident memory of this process in MB, or None if it cannot be determined."""
    try:
        import resource                                   # POSIX (Linux/Kaggle); ru_maxrss is kilobytes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        try:
            import psutil
            info = psutil.Process().memory_info()
            return getattr(info, "peak_wset", info.rss) / 1e6
        except Exception:  # noqa: BLE001
            return None
