"""Mocked tests (no model is loaded or run): length-sorted batching, F2LLM prompt formatting,
preset/CLI handling, dtype handling, EOS check, fp16->fp32 fallback."""
import random

import numpy as np
import pytest
import torch

import sentence_transformers
from src.eval import run_baseline
from src.retrieval.dense_encoder import DenseEncoder, encode_length_sorted, make_mteb_encoder
from src.retrieval.model_presets import F2LLM_INSTRUCTION, PRESETS

EOS = 151645
F2LLM_QUERY_PREFIX = "Instruct: Retrieve the most relevant code snippet for the given query.\nQuery: "


def vec(text):
    """Deterministic per-text 'embedding' so any reordering bug changes the output."""
    return np.array([len(text), sum(map(ord, text)) % 997, text.count("a"), 1.0], dtype=np.float32)


class FakeTok:
    def __init__(self, mode="eos"):
        self.mode, self.eos_token_id = mode, EOS

    def __call__(self, text):
        ids = [1, 2, 3]
        if self.mode == "eos":
            ids.append(EOS)
        elif self.mode == "double":
            ids += [EOS, EOS]
        return {"input_ids": ids}


class FakeST:
    """Stands in for SentenceTransformer: records everything, embeds via vec()."""
    instances = []
    tok_mode = "eos"
    nan_when_half = False

    def __init__(self, name, device="cpu", trust_remote_code=False, **kwargs):
        self.name, self.kwargs = name, kwargs
        self.device = torch.device(device)
        self._dtype = torch.float32
        self.max_seq_length = None
        self.tokenizer = FakeTok(FakeST.tok_mode)
        self.calls = []
        FakeST.instances.append(self)

    def parameters(self):
        yield torch.zeros(1, dtype=self._dtype)

    def to(self, dtype):
        self._dtype = dtype
        return self

    def float(self):
        return self.to(torch.float32)

    def encode(self, texts, batch_size=32, **kwargs):
        self.calls.append(list(texts))
        if FakeST.nan_when_half and self._dtype != torch.float32:
            return np.full((len(texts), 4), np.nan, dtype=np.float32)
        return np.stack([vec(t) for t in texts])


@pytest.fixture(autouse=True)
def fake_st(monkeypatch):
    FakeST.instances, FakeST.tok_mode, FakeST.nan_when_half = [], "eos", False
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)


def texts_of_varied_length(n, seed=0):
    rnd = random.Random(seed)
    base = ["".join(rnd.choice("abcxyz ") for _ in range(rnd.randint(0, 200))) for _ in range(n)]
    return base + base[:5]  # duplicates: equal-length ties must not swap rows


# ---------- step 1: length-sorted batching preserves output order ----------
@pytest.mark.parametrize("n,bs", [(0, 4), (1, 4), (7, 3), (64, 16), (101, 32), (10, 100)])
def test_length_sorted_preserves_order(n, bs):
    texts = texts_of_varied_length(n) if n else []
    batches = []

    def embed_fn(chunk):
        batches.append(list(chunk))
        return np.stack([vec(t) for t in chunk])

    out = encode_length_sorted(texts, embed_fn, bs)
    if n == 0:
        assert out.shape[0] == 0
        return
    assert np.array_equal(out, np.stack([vec(t) for t in texts]))   # original order restored
    assert all(len(b) <= bs for b in batches)
    assert sum(len(b) for b in batches) == len(texts)                # nothing dropped / duplicated
    flat = [len(t) for b in batches for t in b]
    assert flat == sorted(flat, reverse=True)                         # batches really are length-sorted


def test_length_sorted_ties_keep_input_order():
    texts = ["aa", "bb", "cc", "dd"]  # identical length
    seen = []
    encode_length_sorted(texts, lambda c: (seen.append(list(c)), np.stack([vec(t) for t in c]))[1], 2)
    assert [t for b in seen for t in b] == texts


@pytest.mark.parametrize("sort", [True, False])
def test_adapter_output_order_matches_input_order(sort):
    from mteb.types import PromptType
    texts = texts_of_varied_length(50, seed=3)
    batches = [{"text": texts[i:i + 8]} for i in range(0, len(texts), 8)]  # MTEB-style dataloader batches
    m = make_mteb_encoder("fake/model", sort_batches=sort, batch_size=8)
    out = m.encode(batches, prompt_type=PromptType.document, batch_size=8)
    assert np.array_equal(out, np.stack([vec(t) for t in texts]))
    if sort:
        sent = [len(t) for chunk in FakeST.instances[0].calls for t in chunk]
        assert sent == sorted(sent, reverse=True)
        assert all(len(c) <= 8 for c in FakeST.instances[0].calls)


# ---------- step 3: F2LLM prompt formatting ----------
def test_preset_query_prefix_is_exact():
    assert PRESETS["f2llm-v2-0.6b"]["query_prefix"] == F2LLM_QUERY_PREFIX
    assert F2LLM_INSTRUCTION == "Retrieve the most relevant code snippet for the given query."
    assert PRESETS["f2llm-v2-0.6b"]["doc_prefix"] == ""


def test_f2llm_prefix_on_queries_only_and_config_passed_through():
    from mteb.types import PromptType
    a = run_baseline.parse_args(["--preset", "f2llm-v2-0.6b", "--device", "cuda"])
    m = make_mteb_encoder(a.model, a.device, a.max_seq_length, batch_size=a.batch_size,
                          query_prefix=a.query_prefix, doc_prefix=a.doc_prefix,
                          trust_remote_code=a.trust_remote_code, dtype=a.dtype,
                          revision=a.revision, expect_eos=a.expect_eos)
    fake = FakeST.instances[0]
    assert fake.name == "codefuse-ai/F2LLM-v2-0.6B"
    assert fake.kwargs == {"revision": "54b4e2dc74e01be7126d4cf5f016af6b21edc563"}
    assert fake.max_seq_length == 8192
    assert fake._dtype == torch.float16          # fp16 on a (fake) cuda device, not bf16
    docs, queries = ["def f(): pass", "x = 1"], ["reverse a string", "sum two numbers"]
    m.encode([{"text": queries}], prompt_type=PromptType.query)
    m.encode([{"text": docs}], prompt_type=PromptType.document)
    sent_q, sent_d = fake.calls
    assert sorted(sent_q) == sorted(F2LLM_QUERY_PREFIX + q for q in queries)
    assert all(t.startswith("Instruct: Retrieve the most relevant code snippet for the given query.\nQuery: ")
               for t in sent_q)
    assert sorted(sent_d) == sorted(docs)         # documents get no prompt


def test_preset_cli_precedence_and_escapes():
    a = run_baseline.parse_args(["--preset", "f2llm-v2-0.6b", "--max-seq-length", "2048", "--dtype", "fp32"])
    assert (a.max_seq_length, a.dtype) == (2048, "fp32")          # explicit flags beat the preset
    assert a.revision.startswith("54b4e2dc") and a.expect_eos is True
    b = run_baseline.parse_args(["--query-prefix", "A\\nB: "])
    assert b.query_prefix == "A\nB: "                             # literal \n from a notebook cell
    b = run_baseline.parse_args(["--model", "x"])
    assert b.preset is None and b.dtype is None and b.max_seq_length == 256   # other models unchanged
    with pytest.raises(SystemExit):
        run_baseline.parse_args(["--preset", "nope"])


# ---------- dtype handling ----------
def test_half_precision_ignored_on_cpu():
    DenseEncoder("fake/m", device="cpu", dtype="fp16")
    assert FakeST.instances[0]._dtype == torch.float32


def test_dtype_default_leaves_model_untouched_and_fp16_alias_works():
    DenseEncoder("fake/m", device="cuda")
    assert FakeST.instances[0]._dtype == torch.float32
    DenseEncoder("fake/m", device="cuda", fp16=True)
    assert FakeST.instances[1]._dtype == torch.float16
    with pytest.raises(ValueError):
        DenseEncoder("fake/m", device="cuda", dtype="fp8")


def test_fp16_overflow_falls_back_to_fp32():
    FakeST.nan_when_half = True
    enc = DenseEncoder("fake/m", device="cuda", dtype="fp16")
    out = enc.embed(["abc", "hello"])
    assert np.isfinite(out).all() and enc.fell_back_to_fp32
    assert enc.model._dtype == torch.float32


def test_nan_in_fp32_raises():
    enc = DenseEncoder("fake/m", device="cpu")
    enc.model.encode = lambda t, **k: np.full((len(t), 4), np.nan, dtype=np.float32)
    with pytest.raises(FloatingPointError):
        enc.embed(["x"])


# ---------- EOS check ----------
def test_eos_check():
    DenseEncoder("fake/m", expect_eos=True)                       # exactly one EOS: ok
    for mode in ("none", "double"):
        FakeST.tok_mode = mode
        with pytest.raises(ValueError, match="EOS"):
            DenseEncoder("fake/m", expect_eos=True)


def test_unset_preset_paths_still_load_old_style():
    """Existing call style (no new kwargs) keeps working and passes no extra ST kwargs."""
    DenseEncoder("sentence-transformers/all-MiniLM-L6-v2", device=None, max_seq_length=256,
                 query_prefix="", doc_prefix="", trust_remote_code=False, fp16=False)
    assert FakeST.instances[0].kwargs == {}


# ---------- release(): free GPU memory before a second (reranker) model loads ----------
def test_release_frees_model_and_blocks_further_use():
    enc = DenseEncoder("fake/m")
    enc.embed(["ok before release"])                       # sanity: works before release
    enc.release()
    assert enc.model is None
    with pytest.raises(RuntimeError, match="release"):
        enc.embed(["should fail"])
    enc.release()                                           # idempotent: second call is a no-op, no error


def test_release_calls_cuda_cleanup_when_cuda_available(monkeypatch):
    enc = DenseEncoder("fake/m")
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    enc.release()
    assert calls == ["sync", "empty_cache"]


def test_release_skips_cuda_calls_when_no_cuda(monkeypatch):
    enc = DenseEncoder("fake/m")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    called = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.append(1))
    enc.release()                                           # must not raise / must not touch cuda.empty_cache
    assert called == []


def test_dev_context_release_encoder_delegates_and_resets(monkeypatch):
    import types

    from src.eval import dev_lib

    args = types.SimpleNamespace(preset="f2llm-v2-0.6b", device="cpu", batch_size=4, n_queries=0,
                                 use_holdout=False)
    ctx = object.__new__(dev_lib.DevContext)
    ctx._enc = None
    ctx.release_encoder()                                   # never-built encoder: no-op, no crash

    built = DenseEncoder("fake/m")
    released = []
    monkeypatch.setattr(built, "release", lambda: released.append(1))
    ctx._enc = built
    ctx.release_encoder()
    assert released == [1] and ctx._enc is None

    # after release, .encoder lazily rebuilds a fresh model rather than staying dead
    ctx.args, ctx.preset = args, {"model": "fake/m2", "revision": None, "max_seq_length": 128,
                                  "trust_remote_code": False, "dtype": None, "expect_eos": False}
    fresh = ctx.encoder
    assert fresh is not built and fresh.model_name == "fake/m2"


# ---------- f2llm-v2-1.7b preset: same lineage as 0.6b (parametrized against both) ----------
F2LLM_PRESET_EXPECT = {
    "f2llm-v2-0.6b": {"model": "codefuse-ai/F2LLM-v2-0.6B",
                      "revision": "54b4e2dc74e01be7126d4cf5f016af6b21edc563"},
    "f2llm-v2-1.7b": {"model": "codefuse-ai/F2LLM-v2-1.7B",
                      "revision": "3766d46e7a68545ed6190c15330983f9b39ab718"},
    "f2llm-v2-4b": {"model": "codefuse-ai/F2LLM-v2-4B",
                    "revision": "e04d1a04f4a0e154bf43969d3dcc596bbd92b1f8"},
}


@pytest.mark.parametrize("preset", sorted(F2LLM_PRESET_EXPECT))
def test_f2llm_presets_share_prompt_pooling_dtype_and_eos_wiring(preset):
    from mteb.types import PromptType
    exp = F2LLM_PRESET_EXPECT[preset]
    assert PRESETS[preset]["query_prefix"] == F2LLM_QUERY_PREFIX      # same instruction text, both sizes
    assert PRESETS[preset]["doc_prefix"] == ""
    assert PRESETS[preset]["max_seq_length"] == 8192
    assert PRESETS[preset]["dtype"] == "fp16"                          # T4 has no native bf16
    assert PRESETS[preset]["trust_remote_code"] is False
    assert PRESETS[preset]["expect_eos"] is True
    assert PRESETS[preset]["revision"] == exp["revision"]

    a = run_baseline.parse_args(["--preset", preset, "--device", "cuda"])
    m = make_mteb_encoder(a.model, a.device, a.max_seq_length, batch_size=a.batch_size,
                          query_prefix=a.query_prefix, doc_prefix=a.doc_prefix,
                          trust_remote_code=a.trust_remote_code, dtype=a.dtype,
                          revision=a.revision, expect_eos=a.expect_eos)
    fake = FakeST.instances[0]
    assert fake.name == exp["model"] and fake.kwargs == {"revision": exp["revision"]}
    assert fake.max_seq_length == 8192
    assert fake._dtype == torch.float16                                # fp16 on a (fake) cuda device
    docs, queries = ["def f(): pass"], ["reverse a string"]
    m.encode([{"text": queries}], prompt_type=PromptType.query)
    m.encode([{"text": docs}], prompt_type=PromptType.document)
    sent_q, sent_d = fake.calls
    assert sent_q == [F2LLM_QUERY_PREFIX + queries[0]] and sent_d == docs   # prefix on queries only


# ---------- peak-GPU-memory reporting (real number for dev_split.py, not an estimate) ----------
def test_cuda_peak_reset_and_report(monkeypatch):
    from src.retrieval.dense_encoder import cuda_peak_report, cuda_peak_reset

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cuda_peak_reset()                             # must not raise off CUDA
    assert cuda_peak_report("label") is None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: calls.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 3.4e9)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 4.1e9)
    cuda_peak_reset()
    assert calls == ["reset"]
    stats = cuda_peak_report("label")
    assert stats == {"max_allocated_gb": pytest.approx(3.4), "max_reserved_gb": pytest.approx(4.1)}


# ---------- preset memory budget warning (warn, never fail) ----------
def test_memory_warning_fires_only_over_threshold():
    from src.retrieval.model_presets import PRESET_MEMORY, memory_warning

    assert memory_warning("f2llm-v2-0.6b") is None                  # 2.3 GB, far under
    assert memory_warning("f2llm-v2-1.7b") is None                  # 6.6 GB, under the 13 GB default
    w = memory_warning("f2llm-v2-4b")
    assert w is not None and "EXCEEDS" in w                         # 15.3 GB > a T4's 14.56 GB
    assert "15344" in w and "--batch-size" in w
    assert "does NOT shrink with --batch-size" in w                 # weights vs activations, stated plainly

    assert memory_warning("f2llm-v2-1.7b", warn_threshold_mb=6000) is not None   # threshold configurable
    assert memory_warning("f2llm-v2-4b", warn_threshold_mb=99000) is None
    assert memory_warning("some-unknown-preset") is None            # unknown preset: silent, not a crash
    assert PRESET_MEMORY["f2llm-v2-4b"]["registry_memory_mb"] == 15344


def test_memory_warning_distinguishes_over_vs_near_budget():
    from src.retrieval.model_presets import memory_warning

    near = memory_warning("f2llm-v2-1.7b", warn_threshold_mb=6000, gpu_total_mb=40000)
    assert "is within but close to" in near and "EXCEEDS" not in near
