"""Exact-query embedding cache: hits, misses, key invalidation, and the guarantee that only the serving
entry points ever have one. Mocked throughout -- the hashing encoder stands in for the model."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.query_cache import QueryEmbeddingCache, cache_key, resolve_cache_size
from src.runtime_index import HashingQueryEncoder, write_index
from src.search_service import SearchService

ROOT = Path(__file__).resolve().parents[1]
META = {"model": "mock/hashing-encoder", "revision": "rev-1"}


@pytest.fixture()
def index_dir(tmp_path):
    d = tmp_path / "idx"
    enc = HashingQueryEncoder(64)
    texts = [f"def f{i}(xs):\n    return sum(x for x in xs if x > {i})\n" for i in range(10)]
    write_index(d, enc.encode_docs(texts), [f"d{i}" for i in range(10)], texts, dict(META))
    return d


def counting(svc):
    calls = []
    real = svc.encoder.encode_query
    svc.encoder.encode_query = lambda text: calls.append(text) or real(text)
    return calls


# --- the flag -----------------------------------------------------------------------------------------

def test_cache_size_defaults_on_and_the_env_flag_turns_it_off():
    assert resolve_cache_size({}) == 128
    for off in ("0", "false", "no", "off", "OFF"):
        assert resolve_cache_size({"QUERY_CACHE": off}) == 0
    assert resolve_cache_size({"QUERY_CACHE_SIZE": "7"}) == 7
    assert resolve_cache_size({"QUERY_CACHE_SIZE": "0"}) == 0
    assert resolve_cache_size({"QUERY_CACHE_SIZE": "lots"}) == 128
    assert resolve_cache_size({"QUERY_CACHE": "0", "QUERY_CACHE_SIZE": "50"}) == 0     # the switch wins


# --- the LRU itself -------------------------------------------------------------------------------------

def test_hits_misses_and_lru_eviction():
    c = QueryEmbeddingCache(2)
    assert c.get("a") is None
    c.put("a", np.ones(3))
    c.put("b", np.zeros(3))
    assert c.get("a") is not None            # touches a: b is now the oldest
    c.put("c", np.ones(3))                   # evicts b
    assert c.get("b") is None and c.get("a") is not None and c.get("c") is not None
    s = c.stats()
    assert (s["size"], s["hits"], s["misses"]) == (2, 3, 2)


def test_a_disabled_cache_stores_and_serves_nothing():
    c = QueryEmbeddingCache(0)
    c.put("a", np.ones(3))
    assert c.get("a") is None and c.stats()["size"] == 0 and c.enabled is False


def test_stored_vectors_are_copies_and_read_only():
    c = QueryEmbeddingCache(4)
    v = np.ones(3, dtype=np.float32)
    c.put("k", v)
    v[:] = 9
    got = c.get("k")
    assert got.tolist() == [1, 1, 1]
    with pytest.raises(ValueError):
        got[0] = 5


def test_the_key_changes_with_every_part_of_the_configuration():
    base = cache_key("q", "m", "r1", "P: ", 1024, "fp32", False)
    assert base == cache_key("q", "m", "r1", "P: ", 1024, "fp32", False)
    for changed in (cache_key("q2", "m", "r1", "P: ", 1024, "fp32", False),
                    cache_key("q", "m2", "r1", "P: ", 1024, "fp32", False),
                    cache_key("q", "m", "r2", "P: ", 1024, "fp32", False),
                    cache_key("q", "m", "r1", "Q: ", 1024, "fp32", False),
                    cache_key("q", "m", "r1", "P: ", 512, "fp32", False),
                    cache_key("q", "m", "r1", "P: ", None, "fp32", False),
                    cache_key("q", "m", "r1", "P: ", 1024, "bf16", False),
                    cache_key("q", "m", "r1", "P: ", 1024, "fp32", True)):
        assert changed != base


# --- through the service ---------------------------------------------------------------------------------

def test_a_repeated_query_is_encoded_once_and_reports_cached(index_dir):
    svc = SearchService(str(index_dir), mock=True, query_cache_size=8)
    calls = counting(svc)
    first = svc.search("sum values above a threshold", k=3)
    second = svc.search("sum values above a threshold", k=3)
    assert calls == ["sum values above a threshold"]
    assert first["timings_ms"]["cached"] is False and second["timings_ms"]["cached"] is True
    assert [h["doc_id"] for h in first["hits"]] == [h["doc_id"] for h in second["hits"]]
    assert [h["score"] for h in first["hits"]] == [h["score"] for h in second["hits"]]
    assert svc.describe()["query_cache"]["hits"] == 1


def test_a_different_query_is_a_miss_and_near_duplicates_do_not_match(index_dir):
    svc = SearchService(str(index_dir), mock=True, query_cache_size=8)
    calls = counting(svc)
    svc.search("sum values", k=2)
    svc.search("sum values ", k=2)            # one trailing space: a different exact string
    svc.search("Sum values", k=2)             # different case
    assert len(calls) == 3


def test_changing_the_token_cap_prompt_revision_or_dtype_invalidates(index_dir):
    svc = SearchService(str(index_dir), mock=True, query_cache_size=8)
    calls = counting(svc)
    svc.search("sum values", k=2)
    svc.search("sum values", k=2)
    assert len(calls) == 1
    svc.encoder.max_query_tokens = 5                          # a different cap
    svc.search("sum values", k=2)
    assert len(calls) == 2
    svc.encoder.query_prefix = "Instruct: find code\nQuery: "  # a different prompt
    svc.search("sum values", k=2)
    assert len(calls) == 3
    svc.index.manifest["revision"] = "rev-2"                   # a different model revision
    svc.search("sum values", k=2)
    assert len(calls) == 4
    svc.encoder.cpu_dtype = "bf16"
    svc.search("sum values", k=2)
    assert len(calls) == 5
    svc.search("sum values", k=2)                              # ...and the new configuration caches
    assert len(calls) == 5


def test_the_least_recently_used_query_is_evicted(index_dir):
    svc = SearchService(str(index_dir), mock=True, query_cache_size=2)
    calls = counting(svc)
    for q in ("one", "two", "one", "three"):                   # "two" is now the oldest
        svc.search(q, k=1)
    svc.search("one", k=1)
    assert calls == ["one", "two", "three"]
    svc.search("two", k=1)
    assert calls[-1] == "two"


def test_the_default_service_has_no_cache_at_all(index_dir):
    svc = SearchService(str(index_dir), mock=True)
    calls = counting(svc)
    a = svc.search("sum values", k=2)
    b = svc.search("sum values", k=2)
    assert len(calls) == 2
    assert a["timings_ms"]["cached"] is False and b["timings_ms"]["cached"] is False
    assert svc.describe()["query_cache"]["enabled"] is False


def test_compare_shares_the_cache(index_dir):
    from src.versioning.fixture import build_fixture
    from src.versioning.version_index import build_versioned_index, embeddings_for, rows_for
    rows = rows_for(build_fixture(4, 2, seed=1))
    enc = HashingQueryEncoder(64)
    emb, _ = embeddings_for(rows, enc.encode_docs, cache={})
    build_versioned_index(index_dir.parent / "v", rows, emb, dict(META))
    svc = SearchService(str(index_dir.parent / "v"), mock=True, query_cache_size=4)
    calls = counting(svc)
    svc.search("sum the scores", k=2)
    res = svc.compare("sum the scores", 1, 2, k=2)
    assert len(calls) == 1 and res["timings_ms"]["cached"] is True


# --- only the serving entry points have one -----------------------------------------------------------------

def test_benchmarks_and_the_official_run_never_ask_for_a_cache():
    """SearchService defaults to no cache, so a benchmark that just constructs it has none. This pins
    that no benchmark, dev script or the official run opts in -- only the API and the CLI do."""
    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if ("query_cache_size" in text or "resolve_cache_size" in text) and path.name not in (
                "api.py", "cli.py", "search_service.py", "query_cache.py"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
    for name in ("bench_cpu.py", "bench_agent.py", "bench_evolution.py", "bench_real_history.py",
                 "bench_versions.py", "check_cpu_precision.py", "eval/run_official.py"):
        text = (ROOT / "src" / name).read_text(encoding="utf-8")
        assert "query_cache" not in text and "QUERY_CACHE" not in text, name


def test_the_official_run_does_not_import_the_cache():
    text = (ROOT / "src" / "eval" / "run_official.py").read_text(encoding="utf-8")
    assert "query_cache" not in text and "SearchService" not in text


def client_for(index_dir, monkeypatch, **env):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import src.api as api
    monkeypatch.setenv("INDEX_DIR", str(index_dir))
    monkeypatch.setenv("MOCK_ENCODER", "1")
    monkeypatch.delenv("ALLOWED_INDEXES", raising=False)
    monkeypatch.delenv("QUERY_CACHE", raising=False)
    monkeypatch.delenv("QUERY_CACHE_SIZE", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(api, "_service", None)
    monkeypatch.setattr(api, "_error", None)
    monkeypatch.setattr(api, "_services", {})
    return TestClient(api.app), api


def test_the_api_caches_by_default_and_reports_it(index_dir, monkeypatch):
    client, api = client_for(index_dir, monkeypatch)
    first = client.post("/search", json={"query": "sum values", "k": 2}).json()
    second = client.post("/search", json={"query": "sum values", "k": 2}).json()
    assert first["timings_ms"]["cached"] is False and second["timings_ms"]["cached"] is True
    assert client.get("/health").json()["index"]["query_cache"]["hits"] == 1


def test_the_api_env_flag_turns_the_cache_off(index_dir, monkeypatch):
    client, api = client_for(index_dir, monkeypatch, QUERY_CACHE="0")
    client.post("/search", json={"query": "sum values", "k": 2})
    again = client.post("/search", json={"query": "sum values", "k": 2}).json()
    assert again["timings_ms"]["cached"] is False
    assert api.get_service().query_cache.enabled is False


def run_cli(*args, stdin=""):
    proc = subprocess.run([sys.executable, "src/cli.py", *map(str, args)], cwd=ROOT, input=stdin,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_cli_interactive_mode_caches_and_prints_it(index_dir):
    out = run_cli("--interactive", "--index", index_dir, "--mock-encoder",
                  stdin="sum values\nsum values\n:quit\n")
    timing = [line for line in out.split("\n") if line.startswith("timing")]
    assert "cached: false" in timing[0] and "cached: true" in timing[1]


def test_cli_interactive_cache_can_be_switched_off(index_dir):
    out = run_cli("--interactive", "--no-query-cache", "--index", index_dir, "--mock-encoder",
                  stdin="sum values\nsum values\n:quit\n")
    assert out.count("cached: false") == 2 and "cached: true" not in out


def test_a_one_shot_cli_run_never_caches(index_dir):
    out = run_cli("sum values", "--index", index_dir, "--mock-encoder")
    assert "cached: false" in out
