"""Dense retriever: sentence-transformers encoder, cosine ranking, MTEB AbsEncoder adapter."""
import os

import numpy as np

DEFAULT_MODEL = os.environ.get("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _texts(batch):
    """Extract list[str] from an MTEB batch (dict with 'text', or plain list)."""
    if isinstance(batch, dict):
        return list(batch["text"])
    return list(batch)


def cuda_peak_reset():
    """Zero CUDA's peak-memory counters (no-op off CUDA). Call before the thing you want to measure --
    model load included, so the reported peak covers weights + activations together, not just one."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_report(label=""):
    """Print (and return) the peak CUDA memory seen since the last cuda_peak_reset(). None off CUDA."""
    import torch
    if not torch.cuda.is_available():
        print(f"[dense] peak GPU mem {label}: n/a (no CUDA device)", flush=True)
        return None
    stats = {"max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
             "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
    print(f"[dense] PEAK GPU mem {label}: max_allocated={stats['max_allocated_gb']:.2f} GB "
          f"max_reserved={stats['max_reserved_gb']:.2f} GB "
          f"(T4 total = 14.56 GB)", flush=True)
    return stats


def cpu_has_native_bf16():
    """True if this CPU can do bf16 natively (AVX512-BF16 or AMX). Best effort: on anything else,
    bf16 is emulated through fp32 and is slower than fp32, so a False here is a real warning."""
    try:
        import torch
        if hasattr(torch.backends, "cpu") and hasattr(torch.backends.cpu, "_is_amx_tile_supported"):
            if torch.backends.cpu._is_amx_tile_supported():
                return True
    except Exception:  # noqa: BLE001  detection must never break a load
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            flags = f.read()
        return ("avx512_bf16" in flags) or ("amx_bf16" in flags)
    except OSError:
        return False          # no /proc (Windows/macOS): assume not, and warn


def linear_census(module):
    """(fp32 Linear layer names, dynamically-quantised Linear layer names) in a live module tree.

    Walks the modules that would actually run a forward pass, so it reports what encodes rather than
    what was intended. torch's dynamic quantised Linear is a different class, not a subclass of
    nn.Linear, so the two are cleanly separable by isinstance."""
    import torch
    fp32, quantized = [], []
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            fp32.append(name)
        elif m.__class__.__name__ in ("Linear", "LinearPackedParams") and \
                "quantized" in type(m).__module__:
            quantized.append(name)
    return fp32, quantized


class ModelLoadError(RuntimeError):
    """Model failed to load; message names the likely cause and a suggested fix."""


def _explain_load_failure(model_name, exc, trust_remote_code):
    import importlib.metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "not installed"

    env = (f"transformers={ver('transformers')}, sentence-transformers="
           f"{ver('sentence-transformers')}, torch={ver('torch')}")
    msg = str(exc)
    if isinstance(exc, ModuleNotFoundError):
        pkg = (exc.name or "").split(".")[0] or "the missing package"
        cause = f"missing dependency '{pkg}' needed by the model's code"
        fix = f"pip install {pkg}   (and add it to requirements.txt)"
    elif isinstance(exc, ImportError) or (isinstance(exc, AttributeError) and trust_remote_code):
        cause = ("version mismatch between the model's bundled remote code "
                 "(trust_remote_code) and the installed transformers: the code imports "
                 "a name that this transformers version removed or moved")
        fix = ("pin an older transformers, e.g.  pip install 'transformers<4.50' "
               "(restart the runtime afterwards), or use a model whose code works with "
               "your version (nomic-ai/CodeRankEmbed loads on transformers 5.12), or a "
               "model that needs no remote code")
    elif isinstance(exc, (OSError, ValueError)) and not trust_remote_code:
        cause = "model could not be loaded; if its repo ships custom code it needs trust_remote_code"
        fix = "pass --trust-remote-code (only for repos you trust); also check the model name / network"
    else:
        cause = "unexpected error while loading the model"
        fix = "see the original error below (chained)"
    return ModelLoadError(
        f"Failed to load embedding model {model_name!r}.\n"
        f"  likely cause: {cause}\n"
        f"  suggested fix: {fix}\n"
        f"  environment: {env}\n"
        f"  original error: {type(exc).__name__}: {msg}")


class DenseEncoder:
    """Plain encoder + cosine ranking. Model name is a parameter (or DENSE_MODEL env var)."""

    def __init__(self, model_name=None, device=None, max_seq_length=256,
                 query_prefix="", doc_prefix="", trust_remote_code=False, fp16=False,
                 revision=None, dtype=None, expect_eos=False, tokenizer_kwargs=None,
                 allow_cpu_half=False):
        """dtype: None (leave as loaded) | 'fp32' | 'fp16' | 'bf16'; fp16=True is shorthand for
        dtype='fp16'. Half precisions apply only on CUDA unless allow_cpu_half=True, which permits
        bf16 on CPU (fp16 stays CUDA-only: most CPU kernels have no fp16 path and fall back to slow
        emulation). expect_eos=True asserts the tokenizer appends exactly one EOS token (needed by
        last-token-pooling models such as F2LLM)."""
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name or DEFAULT_MODEL
        extra = {}
        if revision:
            extra["revision"] = revision
        if tokenizer_kwargs:
            extra["tokenizer_kwargs"] = tokenizer_kwargs
        try:
            self.model = SentenceTransformer(self.model_name, device=device or "cpu",
                                             trust_remote_code=trust_remote_code, **extra)
        except Exception as exc:  # surface a diagnosis instead of a raw traceback
            raise _explain_load_failure(self.model_name, exc, trust_remote_code) from exc
        self.fell_back_to_fp32 = False
        self.allow_cpu_half = allow_cpu_half
        self.quantized_int8 = False
        self._apply_dtype(dtype or ("fp16" if fp16 else None))
        self.model.max_seq_length = max_seq_length
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.describe_device(requested=device)
        if expect_eos:
            self.check_eos()

    def _apply_dtype(self, dtype):
        import torch
        if dtype is None:
            return
        if dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"dtype must be fp32, fp16 or bf16, got {dtype!r}")
        on_cuda = str(self.model.device).startswith("cuda")
        if dtype != "fp32" and not on_cuda:
            if dtype == "bf16" and self.allow_cpu_half:
                # Deliberate CPU bf16: halves the weight memory, and is fast ONLY on hardware with
                # native bf16 (AVX512-BF16 or AMX). Without it, every op converts to fp32 and back,
                # which measured 4x SLOWER than fp32 on the Kaggle CPU (30.8 s vs 7.6 s per query).
                # So this warns loudly rather than quietly handing someone a 4x regression.
                native = cpu_has_native_bf16()
                if native:
                    print("[dense] CPU bf16: this CPU reports native bf16 support (AVX512-BF16/AMX)",
                          flush=True)
                else:
                    print("[dense] " + "!" * 70, flush=True)
                    print("[dense] WARNING: CPU bf16 requested, but this CPU has NO native bf16 "
                          "support.", flush=True)
                    print("[dense] Every operation will be emulated via fp32. Measured on a CPU like "
                          "this one: 30,774 ms/query for bf16 vs 7,554 ms for fp32 -- 4x SLOWER, for "
                          "no quality gain.", flush=True)
                    print("[dense] Use --cpu-dtype fp32 unless you have measured otherwise on THIS "
                          "machine.", flush=True)
                    print("[dense] " + "!" * 70, flush=True)
            else:
                print(f"[dense] WARNING: {dtype} ignored because the model is not on a CUDA device; "
                      "using fp32", flush=True)
                dtype = "fp32"
        self.model.to({"fp32": torch.float32, "fp16": torch.float16,
                       "bf16": torch.bfloat16}[dtype])

    def _is_half(self):
        import torch
        return next(self.model.parameters()).dtype in (torch.float16, torch.bfloat16)

    def check_eos(self):
        """Fail fast unless the tokenizer appends exactly one EOS token to a single input."""
        tok = self.model.tokenizer
        ids = tok("def f(x): return x")["input_ids"]
        eos = tok.eos_token_id
        if ids[-1] != eos or (len(ids) > 1 and ids[-2] == eos):
            raise ValueError(
                f"Tokenizer does not append exactly one EOS token (ids tail={ids[-3:]}, "
                f"eos_token_id={eos}). Last-token pooling would read the wrong position. "
                "Pass tokenizer_kwargs={'add_eos_token': True} (--add-eos-token) if EOS is missing.")
        print(f"[dense] EOS check ok: inputs end with eos_token_id={eos}", flush=True)

    def describe_device(self, requested=None):
        """Print where the model really lives (catches 'asked for cuda, running on cpu')."""
        import torch
        cuda = torch.cuda.is_available()
        dev = str(self.model.device)
        dtype = str(next(self.model.parameters()).dtype)
        gpu = torch.cuda.get_device_name(0) if cuda else "-"
        print(f"[dense] torch.cuda.is_available()={cuda} requested_device={requested or 'cpu'} "
              f"model_device={dev} dtype={dtype} gpu={gpu} max_seq_length={self.model.max_seq_length}",
              flush=True)
        if requested and str(requested).startswith("cuda") and not dev.startswith("cuda"):
            print("[dense] WARNING: cuda requested but model is NOT on cuda", flush=True)

    def _encode(self, texts, batch_size, show_progress_bar):
        return self.model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=show_progress_bar).astype(np.float32)

    def embed(self, texts, batch_size=64, show_progress_bar=False):
        self._check_alive()
        out = self._encode(texts, batch_size, show_progress_bar)
        if not np.isfinite(out).all():
            if not self._is_half():
                raise FloatingPointError("Model produced non-finite (NaN/inf) embeddings in fp32.")
            print("[dense] WARNING: non-finite embeddings in half precision (fp16 overflow); "
                  "switching model to fp32 and re-encoding this batch", flush=True)
            self.model.float()
            self.fell_back_to_fp32 = True
            out = self._encode(texts, batch_size, show_progress_bar)
            if not np.isfinite(out).all():
                raise FloatingPointError("Model produced non-finite embeddings even in fp32.")
        return out

    def quantize_dynamic_int8(self, strict=True):
        """Dynamically quantise Linear layers to int8, in place (CPU only). Returns True if applied.

        Dynamic quantisation keeps activations in fp32 and quantises weights per tensor, so it needs no
        calibration data. It roughly quarters the weight memory of the Linear layers, but for an
        *embedding* model the risk is not speed, it is fidelity: the output vector is compared by cosine
        similarity, and per-tensor int8 weights can move it enough to reorder the top-10. Another team
        reported exactly that for Qwen-based embedders (F2LLM-v2 is Qwen3), so this is off by default
        and src/check_cpu_precision.py exists to measure it rather than trust it.

        Two things here are the fix for a measured bug, not defensive decoration. A first attempt
        quantised with the default inplace=False and assigned the *copy* back; the result was int8
        output identical to fp32 to six decimal places, unchanged latency and +5.4 GB of RAM -- i.e. a
        second model had been built while the original went on doing the encoding. So:

          * inplace=True, so there is one model rather than two, and the freed fp32 weights are the
            point of the exercise rather than an extra copy of them;
          * a census afterwards that RAISES if any nn.Linear survived (strict=True). A quantisation
            that silently does nothing is worse than one that fails: it produces numbers labelled
            "int8" that were really fp32, which is exactly what happened.
        """
        self._check_alive()
        import gc

        import torch
        if str(getattr(self.model, "device", "cpu")).startswith("cuda"):
            print("[dense] int8 dynamic quantisation is a CPU path; skipping on CUDA", flush=True)
            return False
        first = next(self.model.parameters(), None)          # None: a model with no parameters at all
        if first is not None and first.dtype != torch.float32:
            print(f"[dense] int8 dynamic quantisation needs fp32 weights (the model is {first.dtype}); "
                  "skipping", flush=True)
            return False
        before_fp, before_q = linear_census(self.model)
        if not before_fp:
            print("[dense] no fp32 Linear layers found to quantise; continuing unquantised", flush=True)
            return False
        try:
            torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear},
                                                   dtype=torch.qint8, inplace=True)
        except Exception as exc:  # noqa: BLE001  never let an optimisation break the query path
            print(f"[dense] int8 dynamic quantisation failed ({type(exc).__name__}: {exc}); "
                  "continuing unquantised", flush=True)
            return False
        gc.collect()
        after_fp, after_q = linear_census(self.model)
        print(f"[dense] int8 census: Linear layers fp32 {len(before_fp)} -> {len(after_fp)}, "
              f"quantised {len(before_q)} -> {len(after_q)}", flush=True)
        if after_fp:
            msg = (f"int8 dynamic quantisation did not take: {len(after_fp)} of {len(before_fp)} "
                   f"Linear layers are still fp32 (e.g. {after_fp[:3]}). The model that would encode "
                   f"is therefore NOT the quantised one, and any 'int8' measurement from it would be "
                   f"fp32 wearing an int8 label.")
            if strict:
                raise RuntimeError(msg)
            print(f"[dense] WARNING: {msg}", flush=True)
            return False
        self.quantized_int8 = True
        print(f"[dense] int8 dynamic quantisation applied in place to {len(after_q)} Linear layers "
              "(the fp32 weights are gone, not shadowed)", flush=True)
        return True

    def linear_dtype_report(self):
        """(fp32 Linear names, quantised Linear names) for the live model -- what actually encodes."""
        self._check_alive()
        return linear_census(self.model)

    def set_max_seq_length(self, n):
        """Change the truncation length. Used to cap QUERY length at serving time; documents are
        encoded once, at the preset's full length, and are never truncated by this."""
        self._check_alive()
        self.model.max_seq_length = int(n)
        return self.model.max_seq_length

    def token_count(self, text):
        """Tokens this model would use for `text` BEFORE truncation (so a cap's effect is visible)."""
        self._check_alive()
        return len(self.model.tokenizer(text)["input_ids"])

    def gpu_mem_snapshot(self, label=""):
        """Print allocated/reserved CUDA memory (no-op off CUDA); cheap visibility for OOM debugging."""
        import torch
        if not torch.cuda.is_available():
            return
        a, r = torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9
        print(f"[dense] GPU mem {label}: allocated={a:.2f} GB reserved={r:.2f} GB", flush=True)

    def release(self):
        """Free the model's GPU memory once its embeddings are computed and cached. Safe to call more
        than once. After this, `embed()`/`rank()` raise clearly instead of silently reloading a fresh
        (differently-configured) model -- construct a new DenseEncoder if you need it again."""
        if self.model is None:
            return
        self.gpu_mem_snapshot("before release")
        del self.model
        self.model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self.gpu_mem_snapshot("after release")

    def _check_alive(self):
        if self.model is None:
            raise RuntimeError(
                f"DenseEncoder for {self.model_name!r} was release()d; construct a new DenseEncoder "
                "to embed more text.")

    def rank(self, query, doc_embeddings, k=10):
        """Return (indices, scores) of top-k docs by cosine similarity (embeddings are normalized)."""
        q = self.embed([self.query_prefix + query])[0]
        scores = doc_embeddings @ q
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]


def encode_length_sorted(texts, embed_fn, batch_size, on_batch=None):
    """Embed `texts` in length-descending batches (minimal padding), returned in ORIGINAL order.

    embed_fn(list[str]) -> np.ndarray (len, dim). Longest batch runs first, so OOM shows up
    immediately. Ties keep input order (stable sort)."""
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    order = sorted(range(n), key=lambda i: -len(texts[i]))
    out = None
    for bi, start in enumerate(range(0, n, batch_size)):
        idx = order[start:start + batch_size]
        emb = np.asarray(embed_fn([texts[i] for i in idx]))
        if out is None:
            out = np.empty((n, emb.shape[1]), dtype=emb.dtype)
        out[idx] = emb
        if on_batch:
            on_batch(bi, len(idx), start + len(idx))
    return out


def make_mteb_encoder(model_name=None, device=None, max_seq_length=256, sort_batches=True,
                      batch_size=32, **enc_kwargs):
    """Build the MTEB-compatible encoder (import deferred so the module loads without mteb).

    sort_batches=True (default): gather every text, encode in length-sorted batches, restore order.
    sort_batches=False: legacy behaviour, encode MTEB's dataloader batches in dataset order."""
    from mteb.models.abs_encoder import AbsEncoder
    from mteb.models.model_meta import ModelMeta

    class DenseMTEBEncoder(AbsEncoder):
        def __init__(self):
            self.dense = DenseEncoder(model_name, device, max_seq_length, **enc_kwargs)
            self.mteb_model_meta = ModelMeta.create_empty(
                {"name": f"local/dense-{self.dense.model_name.split('/')[-1]}",
                 "revision": "local"})

        def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
                   prompt_type=None, **kwargs):
            is_query = getattr(prompt_type, "value", prompt_type) == "query"
            pre = self.dense.query_prefix if is_query else self.dense.doc_prefix
            import time
            import torch
            kind = "query" if is_query else "document"
            if sort_batches:
                bs = kwargs.get("batch_size") or getattr(inputs, "batch_size", None) or batch_size
                texts = [pre + t for b in inputs for t in _texts(b)]
                t0 = time.time()

                def log(bi, n, done):
                    if bi % 25 == 0:
                        print(f"[encode:{kind}] sorted batch {bi}: {n} texts | {done}/{len(texts)} "
                              f"done, {done / max(time.time() - t0, 1e-9):.1f} texts/s", flush=True)

                emb = encode_length_sorted(texts, lambda t: self.dense.embed(t, batch_size=bs),
                                           bs, on_batch=log)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"[encode:{kind}] DONE {len(texts)} texts in {time.time() - t0:.1f}s "
                      f"({len(texts) / max(time.time() - t0, 1e-9):.1f} texts/s, length-sorted, "
                      f"batch_size={bs})", flush=True)
                return emb
            out, n_done, t0 = [], 0, time.time()
            for i, b in enumerate(inputs):
                texts = [pre + t for t in _texts(b)]
                tb = time.time()
                out.append(self.dense.embed(texts, batch_size=len(texts)))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                n_done += len(texts)
                if i % 25 == 0:
                    dt = time.time() - tb
                    print(f"[encode:{kind}] batch {i}: {len(texts)} texts in {dt:.2f}s "
                          f"({len(texts) / max(dt, 1e-9):.1f} texts/s) | cumulative "
                          f"{n_done / max(time.time() - t0, 1e-9):.1f} texts/s", flush=True)
            print(f"[encode:{kind}] DONE {n_done} texts in {time.time() - t0:.1f}s "
                  f"({n_done / max(time.time() - t0, 1e-9):.1f} texts/s)", flush=True)
            return np.concatenate(out, axis=0)

    return DenseMTEBEncoder()
