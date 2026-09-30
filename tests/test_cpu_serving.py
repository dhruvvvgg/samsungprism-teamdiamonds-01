"""CPU serving options: index naming, the query cap, precision flags, interactive mode, thread sweep.

Mocked -- these cover the plumbing and the guard rails. What a cap or a precision mode actually costs in
quality or latency is measured on Kaggle by src/eval/dev_query_cap.py, src/check_cpu_precision.py and
src/bench_cpu.py; nothing here pretends to answer that.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.runtime_index import INDEX_NAMES, HashingQueryEncoder, resolve_index_dir, write_index

ROOT = Path(__file__).resolve().parents[1]

META = {"preset": "mock", "model": "mock/hashing-encoder", "revision": "n/a", "query_prefix": "",
        "doc_prefix": "", "max_seq_length": 0, "dtype": "fp32", "trust_remote_code": False,
        "expect_eos": False}


@pytest.fixture(scope="module")
def mock_index(tmp_path_factory):
    d = tmp_path_factory.mktemp("serve") / "idx"
    enc = HashingQueryEncoder(dim=32)
    texts = [f"def f{i}(xs):\n    return sum(x for x in xs if x > {i})\n" for i in range(12)]
    write_index(d, enc.encode_docs(texts), [f"d{i}" for i in range(12)], texts, dict(META))
    return d


# --- named indexes -------------------------------------------------------------------------------

def test_index_names_resolve_to_their_directories():
    assert resolve_index_dir("full") == INDEX_NAMES["full"]
    assert resolve_index_dir("lite") == INDEX_NAMES["lite"]
    assert resolve_index_dir("versions") == INDEX_NAMES["versions"]
    assert resolve_index_dir("LITE") == INDEX_NAMES["lite"]          # case-insensitive
    assert resolve_index_dir(None) == INDEX_NAMES["full"]


def test_full_and_lite_are_different_directories():
    """The lite index must never overwrite the submitted one."""
    assert INDEX_NAMES["full"] != INDEX_NAMES["lite"]
    assert INDEX_NAMES["full"].name == "runtime_index"
    assert INDEX_NAMES["lite"].name == "runtime_index_lite"


def test_a_path_is_passed_through_unchanged(tmp_path):
    assert resolve_index_dir(str(tmp_path / "somewhere")) == tmp_path / "somewhere"


def test_service_accepts_a_name_or_a_path(mock_index):
    from src.search_service import SearchService
    svc = SearchService(str(mock_index), mock=True)
    assert svc.index_dir == mock_index
    assert svc.describe()["index_dir"] == str(mock_index)


# --- serving knobs -------------------------------------------------------------------------------

def test_describe_reports_the_serving_knobs(mock_index):
    from src.search_service import SearchService
    info = SearchService(str(mock_index), mock=True).describe()
    for key in ("cpu_dtype", "int8", "max_query_tokens", "threads", "model_load_seconds"):
        assert key in info, key


def test_cpu_dtype_is_validated():
    from src.runtime_index import PresetQueryEncoder
    with pytest.raises(ValueError, match="cpu_dtype must be fp32 or bf16"):
        PresetQueryEncoder({"model": "x", "dim": 4}, cpu_dtype="int4")


def test_query_cap_reaches_the_encoder(monkeypatch):
    """--max-query-tokens must set the tokenizer's truncation length, and must not be confused with
    the document length: documents are encoded once at index build time and never truncated here."""
    calls = {}

    class FakeDense:
        quantized_int8 = False

        def __init__(self, *a, **kw):
            calls["init"] = kw

        def set_max_seq_length(self, n):
            calls["cap"] = n
            return n

        def quantize_dynamic_int8(self):
            calls["int8"] = True
            return True

    import src.runtime_index as ri
    monkeypatch.setattr("src.retrieval.dense_encoder.DenseEncoder", FakeDense)
    enc = ri.PresetQueryEncoder({"model": "m", "revision": "r", "query_prefix": "Q: ",
                                 "max_seq_length": 8192}, device="cpu", max_query_tokens=512)
    assert calls["cap"] == 512
    assert enc.max_query_tokens == 512
    assert "int8" not in calls                      # not requested, not applied


def test_int8_is_opt_in(monkeypatch):
    calls = {}

    class FakeDense:
        quantized_int8 = False

        def __init__(self, *a, **kw):
            pass

        def set_max_seq_length(self, n):
            return n

        def quantize_dynamic_int8(self, allow_lossy_int8=False):
            if not allow_lossy_int8:
                raise RuntimeError("int8 dynamic quantisation is rejected on CPU")
            calls["int8"] = True
            calls["allow_lossy_int8"] = allow_lossy_int8
            return True

    import src.runtime_index as ri
    monkeypatch.setattr("src.retrieval.dense_encoder.DenseEncoder", FakeDense)
    ri.PresetQueryEncoder({"model": "m", "revision": "r", "max_seq_length": 8192}, device="cpu")
    assert "int8" not in calls, "int8 must never be applied unless asked for"

    with pytest.raises(RuntimeError, match="int8 dynamic quantisation is rejected on CPU"):
        ri.PresetQueryEncoder({"model": "m", "revision": "r", "max_seq_length": 8192}, device="cpu",
                              int8=True)

    ri.PresetQueryEncoder({"model": "m", "revision": "r", "max_seq_length": 8192}, device="cpu",
                          int8=True, allow_lossy_int8=True)
    assert calls.get("int8") is True
    assert calls.get("allow_lossy_int8") is True


def test_dense_encoder_quantize_fails_closed_without_flag():
    import torch
    from src.retrieval.dense_encoder import DenseEncoder
    enc = DenseEncoder.__new__(DenseEncoder)
    enc.model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    enc.quantized_int8 = False
    enc._check_alive = lambda: None
    with pytest.raises(RuntimeError, match="int8 dynamic quantisation is rejected"):
        enc.quantize_dynamic_int8(allow_lossy_int8=False)


def test_bf16_only_reaches_the_encoder_when_asked(monkeypatch):
    seen = {}

    class FakeDense:
        quantized_int8 = False

        def __init__(self, *a, **kw):
            seen.update(kw)

        def set_max_seq_length(self, n):
            return n

    import src.runtime_index as ri
    monkeypatch.setattr("src.retrieval.dense_encoder.DenseEncoder", FakeDense)
    ri.PresetQueryEncoder({"model": "m", "revision": "r", "max_seq_length": 8192}, device="cpu")
    assert seen["dtype"] == "fp32" and seen["allow_cpu_half"] is False
    ri.PresetQueryEncoder({"model": "m", "revision": "r", "max_seq_length": 8192}, device="cpu",
                          cpu_dtype="bf16")
    assert seen["dtype"] == "bf16" and seen["allow_cpu_half"] is True


def test_cuda_keeps_the_manifest_dtype(monkeypatch):
    """The CPU precision knob must not silently change what a GPU run does."""
    seen = {}

    class FakeDense:
        quantized_int8 = False

        def __init__(self, *a, **kw):
            seen.update(kw)

        def set_max_seq_length(self, n):
            return n

    import src.runtime_index as ri
    monkeypatch.setattr("src.retrieval.dense_encoder.DenseEncoder", FakeDense)
    ri.PresetQueryEncoder({"model": "m", "revision": "r", "max_seq_length": 8192, "dtype": "fp16"},
                          device="cuda")
    assert seen["dtype"] == "fp16"


def test_mock_encoder_exposes_the_same_surface():
    enc = HashingQueryEncoder(dim=16)
    assert enc.cpu_dtype == "fp32" and enc.int8 is False and enc.max_query_tokens is None
    assert enc.encode_queries(["a", "b"]).shape == (2, 16)
    assert enc.token_count("one two three") == 3


# --- thread sweep --------------------------------------------------------------------------------

def test_thread_levels_are_deduplicated_and_ordered():
    from src.bench_cpu import thread_levels
    assert thread_levels(8) == [1, 2, 8]
    assert thread_levels(2) == [1, 2]
    assert thread_levels(1) == [1]                 # no duplicate "1" on a single-core box


def test_percentile_helper_matches_hand_computed_values():
    from src.bench_cpu import pct
    assert pct([10, 20, 30, 40], 50) == 20
    assert pct([10, 20, 30, 40], 95) == 40
    assert pct([5], 95) == 5
    assert pct([], 50) is None


def test_summarise_excludes_the_warmup():
    from src.bench_cpu import summarise
    got = summarise([1000.0, 10.0, 20.0, 30.0], [900.0, 9.0, 18.0, 27.0], [1.0, 0.5, 0.5, 0.5])
    assert got["n_measured"] == 3
    assert got["latency_ms"]["max"] == 30.0        # the 1000 ms warm-up is gone
    assert got["latency_ms"]["p50"] == 20.0


def test_summarise_keeps_a_single_measurement():
    from src.bench_cpu import summarise
    got = summarise([42.0], [40.0], [2.0])
    assert got["n_measured"] == 1 and got["latency_ms"]["p50"] == 42.0


# --- precision check math ------------------------------------------------------------------------

def test_compare_to_reference_detects_reordering():
    from src.check_cpu_precision import compare_to_reference
    ref_e = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    same = compare_to_reference(ref_e, [["a", "b"], ["c", "d"]], ref_e, [["a", "b"], ["c", "d"]], 2)
    assert same["cosine_to_fp32_mean"] == pytest.approx(1.0)
    assert same["top2_overlap_mean"] == 1.0 and same["rank1_changed"] == 0
    assert same["top2_identical_and_in_order"] == 2

    # same documents, different order: overlap stays 1.0 but rank 1 moved -- which is the point
    reordered = compare_to_reference(ref_e, [["a", "b"]], ref_e, [["b", "a"]], 2)
    assert reordered["top2_overlap_mean"] == 1.0
    assert reordered["rank1_changed"] == 1
    assert reordered["top2_identical_and_in_order"] == 0


def test_compare_to_reference_reports_a_moved_vector():
    from src.check_cpu_precision import compare_to_reference
    ref_e = np.array([[1.0, 0.0]], dtype=np.float32)
    moved = np.array([[0.8, 0.6]], dtype=np.float32)
    got = compare_to_reference(ref_e, [["a"]], moved, [["z"]], 1)
    assert got["cosine_to_fp32_mean"] == pytest.approx(0.8, abs=1e-6)
    assert got["top1_overlap_mean"] == 0.0


# --- end to end: interactive mode ----------------------------------------------------------------

def run_cli(stdin_text, *args):
    proc = subprocess.run([sys.executable, "src/cli.py", *[str(a) for a in args]], cwd=ROOT,
                          input=stdin_text, capture_output=True, text=True)
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_interactive_mode_loads_once_and_answers_several_queries(mock_index):
    out = run_cli("sum the scores\n:k 2\nbinary search\n:quit\n",
                  "--interactive", "--index", mock_index, "--mock-encoder")
    assert "the model stays warm" in out
    assert out.count("timing  :") == 2                     # both queries answered
    assert "k = 2" in out


def test_interactive_mode_handles_eof_and_unknown_commands(mock_index):
    out = run_cli("sum the scores\n:nonsense\n", "--interactive", "--index", mock_index,
                  "--mock-encoder")
    assert "unknown command" in out
    assert out.count("timing  :") == 1


def test_interactive_info_command_prints_json(mock_index):
    out = run_cli(":info\n:quit\n", "--interactive", "--index", mock_index, "--mock-encoder")
    start = out.index("{", out.index("commands:"))
    assert json.loads(out[start:out.rindex("}") + 1])["n_docs"] == 12


def test_cli_accepts_the_old_index_dir_spelling(mock_index):
    """--index-dir still works; it must not be silently overridden by --index's default."""
    out = run_cli("", "sum the scores", "-k", 2, "--index-dir", mock_index, "--mock-encoder", "--json")
    assert json.loads(out)["index"]["index_dir"] == str(mock_index)


# --- serving query cap (default 1024, measured on dev) --------------------------------------------

def test_resolve_query_cap_semantics():
    from src.runtime_index import DEFAULT_SERVING_QUERY_TOKENS, resolve_query_cap
    assert resolve_query_cap(None) == DEFAULT_SERVING_QUERY_TOKENS   # unspecified -> the serving cap
    assert resolve_query_cap(0) is None                              # 0 -> explicitly uncapped
    assert resolve_query_cap(-1) is None
    assert resolve_query_cap(512) == 512
    assert DEFAULT_SERVING_QUERY_TOKENS == 1024


def test_serving_default_is_the_cap_dev_measured_as_not_significantly_worse():
    """1024 is not arbitrary: on the 1,000-query dev slice it cost -0.0010 NDCG@10 with a CI touching
    zero, while 512 cost -0.0077 (significant) and 256 cost -0.0784."""
    from src.runtime_index import DEFAULT_SERVING_QUERY_TOKENS
    assert DEFAULT_SERVING_QUERY_TOKENS == 1024


def test_search_service_applies_the_cap_by_default(mock_index, monkeypatch):
    seen = {}

    class FakeDense:
        quantized_int8 = False

        def __init__(self, *a, **kw):
            pass

        def set_max_seq_length(self, n):
            seen["cap"] = n
            return n

    import src.runtime_index as ri
    monkeypatch.setattr("src.retrieval.dense_encoder.DenseEncoder", FakeDense)
    ri.PresetQueryEncoder({"model": "m", "revision": "r", "max_seq_length": 8192}, device="cpu",
                          max_query_tokens=ri.DEFAULT_SERVING_QUERY_TOKENS)
    assert seen["cap"] == 1024


def test_search_service_can_be_asked_for_no_cap(mock_index):
    from src.search_service import SearchService
    capped = SearchService(str(mock_index), mock=True)
    assert capped.describe()["max_query_tokens"] == 1024
    uncapped = SearchService(str(mock_index), mock=True, max_query_tokens=None)
    assert uncapped.describe()["max_query_tokens"] is None


def test_cli_reports_the_cap_and_honours_zero(mock_index):
    default = json.loads(run_cli("", "sum the scores", "-k", 2, "--index", mock_index,
                                 "--mock-encoder", "--json"))
    assert default["index"]["max_query_tokens"] == 1024
    uncapped = json.loads(run_cli("", "sum the scores", "-k", 2, "--index", mock_index,
                                  "--mock-encoder", "--max-query-tokens", "0", "--json"))
    assert uncapped["index"]["max_query_tokens"] is None
    custom = json.loads(run_cli("", "sum the scores", "-k", 2, "--index", mock_index,
                                "--mock-encoder", "--max-query-tokens", "256", "--json"))
    assert custom["index"]["max_query_tokens"] == 256


def test_api_applies_the_serving_cap(mock_index, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import src.api as api
    monkeypatch.setenv("INDEX_DIR", str(mock_index))
    monkeypatch.setenv("MOCK_ENCODER", "1")
    monkeypatch.delenv("MAX_QUERY_TOKENS", raising=False)
    monkeypatch.setattr(api, "_service", None)
    monkeypatch.setattr(api, "_error", None)
    with TestClient(api.app) as client:                     # with-block runs the startup lifespan
        health = client.get("/health").json()
        assert health["loaded"] is True, "the API must load the model at startup, not on first request"
        assert health["index"]["max_query_tokens"] == 1024


# --- int8: the quantised model must be the one that encodes --------------------------------------

def test_linear_census_separates_fp32_from_quantised():
    import torch

    from src.retrieval.dense_encoder import linear_census
    net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    fp32, quant = linear_census(net)
    assert len(fp32) == 2 and quant == []
    torch.ao.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    fp32_after, quant_after = linear_census(net)
    assert fp32_after == [], f"fp32 Linear layers survived quantisation: {fp32_after}"
    assert len(quant_after) >= 2


def test_quantization_is_in_place_so_there_is_only_one_model():
    """The bug this guards: quantize_dynamic(inplace=False) built a second model while the fp32
    original went on encoding -- identical output, unchanged speed, +5.4 GB of RAM."""
    import torch

    from src.retrieval.dense_encoder import linear_census
    net = torch.nn.Sequential(torch.nn.Linear(8, 8))
    same = torch.ao.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8,
                                                  inplace=True)
    assert same is net, "inplace=True must mutate the model we keep a reference to"
    assert linear_census(net)[0] == []


def test_quantize_raises_when_it_did_not_take(monkeypatch):
    """A quantisation that silently does nothing produces fp32 numbers labelled int8. It must fail."""
    import torch

    from src.retrieval.dense_encoder import DenseEncoder
    enc = DenseEncoder.__new__(DenseEncoder)          # no model load
    enc.model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    enc.quantized_int8 = False
    enc._check_alive = lambda: None
    monkeypatch.setattr(torch.ao.quantization, "quantize_dynamic",
                        lambda *a, **kw: None)         # pretend it silently no-ops
    with pytest.raises(RuntimeError, match="did not take"):
        enc.quantize_dynamic_int8(strict=True, allow_lossy_int8=True)
    assert enc.quantized_int8 is False
    assert enc.quantize_dynamic_int8(strict=False, allow_lossy_int8=True) is False   # non-strict reports instead of raising


def test_quantize_succeeds_and_flags_the_encoder():
    import torch

    from src.retrieval.dense_encoder import DenseEncoder
    enc = DenseEncoder.__new__(DenseEncoder)
    enc.model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    enc.quantized_int8 = False
    enc._check_alive = lambda: None
    assert enc.quantize_dynamic_int8(allow_lossy_int8=True) is True
    assert enc.quantized_int8 is True
    assert enc.linear_dtype_report()[0] == []          # nothing fp32 left to encode with


def test_quantize_skips_a_model_with_no_linear_layers():
    import torch

    from src.retrieval.dense_encoder import DenseEncoder
    enc = DenseEncoder.__new__(DenseEncoder)
    enc.model = torch.nn.Sequential(torch.nn.ReLU())
    enc.quantized_int8 = False
    enc._check_alive = lambda: None
    assert enc.quantize_dynamic_int8(allow_lossy_int8=True) is False
    assert enc.quantized_int8 is False


def test_bf16_native_detection_never_raises():
    from src.retrieval.dense_encoder import cpu_has_native_bf16
    assert isinstance(cpu_has_native_bf16(), bool)
