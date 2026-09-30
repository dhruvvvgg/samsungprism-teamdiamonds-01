"""LLM re-judge (mock/injected LLM), pipeline stages, the MTEB SearchProtocol model, official-run guards.
Everything uses fakes: no model is loaded and no network / API key is needed."""
import json
import sys
import zlib

import numpy as np
import pytest

from src.agent import llm_client as L
from src.agent import llm_rejudge as J
from src.retrieval import pipeline as P
from src.retrieval.search_model import HybridSearchModel


# ---------- LLM wrapper mock mode ----------
@pytest.fixture
def mock_llm(monkeypatch, tmp_path):
    for v in ("LLM_MODEL", *L.KEY_ENV.values()):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setattr(L, "CACHE_DIR", tmp_path / "cache")
    calls = []
    real = L._call_mock
    monkeypatch.setattr(L, "_call_mock", lambda *a: (calls.append(1), real(*a))[1])
    return calls


def test_mock_provider_needs_no_key_is_deterministic_and_cached(mock_llm):
    prompt = J.build_prompt("count the number of primes below n", "def primes(n): count primes below n")
    a = L.call_llm(prompt, system=J.JUDGE_SYSTEM)
    b = L.call_llm(prompt, system=J.JUDGE_SYSTEM)
    assert a == b and 0.0 <= J.parse_score(a) <= 1.0
    assert len(mock_llm) == 1                                        # second call served from disk cache
    assert L.call_llm("plain text") == "mock response"


def test_mock_judge_prefers_overlapping_code(mock_llm):
    good = J.judge_score("sum of two integers a and b", "def sum_two(a, b): return integers a b sum")
    bad = J.judge_score("sum of two integers a and b", "import os; print(os.listdir())")
    assert good > bad


def test_real_provider_without_key_is_loud(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(L, "CACHE_DIR", tmp_path / "c")
    with pytest.raises(L.LLMConfigError, match="OPENAI_API_KEY"):
        J.judge_score("p", "c")                                      # must not be swallowed into "no opinion"


# ---------- rejudge logic ----------
@pytest.mark.parametrize("text,expect", [
    ('{"score": 0.7, "reason": "ok"}', 0.7), ("Score: 0.3 because...", 0.3),
    ('```json\n{"score": 1.5}\n```', 1.0), ('{"score": -2}', 0.0), ("no idea", None), ("", None),
    ('{"score": "high"}', None), ("I'd say 1", 1.0)])
def test_parse_score(text, expect):
    assert J.parse_score(text) == expect


def test_margin_z():
    assert J.margin_z([1.0]) == 0.0 and J.margin_z([1.0, 1.0, 1.0]) == 0.0
    s = [3.0, 2.0, 1.0, 0.0]
    assert J.margin_z(s) == pytest.approx(1.0 / np.std(s))


def fake_call(prompt, system=None):
    return json.dumps({"score": 0.95 if "GOOD" in prompt else 0.05})


DOCS = ["bad code", "GOOD code", "other", "more", "x"]


def test_rejudge_reorders_when_uncertain_and_leaves_tail():
    idx, sc = [0, 1, 2, 3, 4], np.array([0.50, 0.49, 0.48, 0.10, 0.05])
    new, new_sc, info = J.rejudge("prob", idx, sc, DOCS, judge_top=3, margin_z_threshold=5.0, beta=1.0,
                                  call=fake_call)
    assert info["triggered"] and info["n_calls"] == 3 and new[0] == 1 and new[3:] == [3, 4]
    assert sorted(new) == idx and np.all(np.diff(new_sc) <= 0)         # scores stay monotone
    same, _, info0 = J.rejudge("prob", idx, sc, DOCS, judge_top=3, margin_z_threshold=5.0, beta=0.0,
                               call=fake_call)
    assert same == idx                                                  # beta=0 -> retrieval order kept


def test_rejudge_not_triggered_when_confident():
    idx, sc = [0, 1, 2], np.array([0.9, 0.2, 0.1])
    calls = []
    new, _, info = J.rejudge("p", idx, sc, DOCS, margin_z_threshold=0.5, beta=1.0,
                             call=lambda *a, **k: calls.append(1) or "0.9")
    assert new == idx and not info["triggered"] and calls == []         # no LLM cost when confident


def test_rejudge_degrades_on_llm_failure_and_garbage():
    idx, sc = [0, 1, 2], np.array([0.5, 0.49, 0.48])

    def boom(*a, **k):
        raise TimeoutError("network")
    new, _, info = J.rejudge("p", idx, sc, DOCS, margin_z_threshold=9.0, call=boom)
    assert new == idx and info.get("fallback")
    new, _, info = J.rejudge("p", idx, sc, DOCS, margin_z_threshold=9.0, call=lambda *a, **k: "garbled")
    assert new == idx and info.get("fallback")
    partial = iter(['{"score": 0.9}', "garbled", "garbled"])
    new, _, _ = J.rejudge("p", idx, sc, DOCS, margin_z_threshold=9.0, beta=1.0, call=lambda *a, **k: next(partial))
    assert sorted(new) == idx                                            # partial answers still valid


# ---------- pipeline ----------
def make_scores(n_q=4, n_d=30, seed=0):
    return np.random.RandomState(seed).rand(n_q, n_d).astype(np.float32)


def test_merge_config_rejects_typos():
    assert P.merge_config({"rrf": {"k": 10}})["rrf"]["w_dense"] == 1.0
    for bad in ({"rff": {}}, {"rrf": {"kk": 1}}):
        with pytest.raises(KeyError):
            P.merge_config(bad)


def test_first_stage_dense_and_hybrid():
    dense = np.array([0.9, 0.8, 0.1, 0.05])
    bm = np.array([0.0, 0.1, 9.0, 8.0])
    cfg = P.merge_config({})
    idx, sc = P.first_stage(dense, bm, cfg)
    assert idx[:2] == [0, 1] and np.all(np.diff(sc) <= 0)               # BM25 ignored when disabled
    cfg = P.merge_config({"bm25": {"enabled": True}, "rrf": {"depth": 2, "w_bm25": 3.0}})
    idx, sc = P.first_stage(dense, bm, cfg)
    assert idx[0] == 2 and set(idx) == {0, 1, 2, 3} and np.all(np.diff(sc) <= 1e-12)


class GoodScorer:
    """Reranker stub: pairs whose doc text contains 'GOOD' score high."""
    def score_pairs(self, pairs, batch_size=16):
        return np.array([1.0 if "GOOD" in d else 0.0 for _, d in pairs], dtype=np.float32)


def test_run_pipeline_rerank_moves_good_doc_to_top_and_pads():
    S = make_scores(3, 30)
    texts = [f"doc {i}" for i in range(30)]
    target = [int(np.argsort(-S[q])[4]) for q in range(3)]               # 5th-ranked by dense
    for t in target:
        texts[t] = "GOOD solution"
    cfg = {"rerank": {"enabled": True, "k": 10, "alpha": 0.0}, "final_depth": 25}
    lists, timings = P.run_pipeline(S, None, ["q"] * 3, texts, cfg, GoodScorer())
    assert [l[0] for l in lists] == target and "rerank" in timings
    for l in lists:
        assert len(l) == 25 and len(set(l)) == 25                        # padded, no duplicates
    keep, _ = P.run_pipeline(S, None, ["q"] * 3, texts, {**cfg, "rerank": {"enabled": True, "k": 10, "alpha": 1.0}},
                             GoodScorer())
    assert [l[0] for l in keep] == [int(np.argmax(S[q])) for q in range(3)]   # alpha=1: first stage order


def test_pipeline_survives_reranker_failure():
    class Broken:
        def score_pairs(self, *a, **k):
            raise RuntimeError("CUDA OOM")
    S = make_scores(2, 30)
    base, _ = P.run_pipeline(S, None, ["q"] * 2, ["d"] * 30, {"final_depth": 20})
    out, _ = P.run_pipeline(S, None, ["q"] * 2, ["d"] * 30,
                            {"rerank": {"enabled": True, "k": 10}, "final_depth": 20}, Broken())
    assert out == base                                                    # fell back, no crash, not empty


def test_pipeline_rejudge_stage_uses_injected_llm():
    S = np.tile(np.array([0.50, 0.49, 0.48, 0.1, 0.05] + [0.0] * 25, dtype=np.float32), (2, 1))
    texts = ["bad"] * 30
    texts[1] = "GOOD"
    cfg = {"rejudge": {"enabled": True, "judge_top": 3, "margin_z": 9.0, "beta": 1.0}, "final_depth": 10}
    lists, t = P.run_pipeline(S, None, ["q", "q"], texts, cfg, judge_call=fake_call)
    assert [l[0] for l in lists] == [1, 1] and t["rejudge_triggered"] == 2 and t["rejudge_calls"] == 6


def test_run_pipeline_empty_input_is_safe():
    lists, _ = P.run_pipeline(np.zeros((0, 5), np.float32), None, [], ["a"] * 5, {})
    assert lists == []


# ---------- MTEB SearchProtocol model ----------
def _vec(text, dim=1024):
    v = np.zeros(dim, dtype=np.float32)
    for w in text.lower().split():
        v[zlib.crc32(w.encode()) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeDense:
    def __init__(self):
        self.calls = []
        self.released = False

    def embed(self, texts, batch_size=16, **kw):
        self.calls.append(list(texts))
        return np.stack([_vec(t) for t in texts])

    def release(self):
        self.released = True


def toy_task(n=20):
    rnd = np.random.RandomState(1)
    words = [f"w{i}" for i in range(400)]
    docs = [" ".join(rnd.choice(words, 12)) for _ in range(n)]
    queries = [" ".join(d.split()[:8]) + " statement" for d in docs]
    corpus = {"id": [f"d{100 + i}" for i in range(n)], "text": docs, "title": [""] * n}
    qs = {"id": [f"q{i}" for i in range(n)], "text": queries}
    return corpus, qs


def test_search_model_is_a_search_protocol_and_returns_mteb_format():
    from mteb.models.models_protocols import SearchProtocol
    corpus, qs = toy_task()
    m = HybridSearchModel({}, FakeDense())
    assert isinstance(m, SearchProtocol)
    m.index(corpus, task_metadata=None, hf_split="test", hf_subset="default", encode_kwargs={"batch_size": 4})
    out = m.search(qs, task_metadata=None, hf_split="test", hf_subset="default", top_k=15,
                   encode_kwargs={"batch_size": 4})
    assert set(out) == set(qs["id"])
    for i, qid in enumerate(qs["id"]):
        res = out[qid]
        assert len(res) == 15 and set(res) <= set(corpus["id"])
        ordered = sorted(res, key=res.get, reverse=True)
        assert ordered[0] == f"d{100 + i}"                                # relevant doc first
        assert len(set(res.values())) == 15                                # strictly decreasing scores


def test_search_model_full_stack_with_bm25_variants_rerank_and_lazy_scorer():
    corpus, qs = toy_task()
    events = []

    def factory():
        events.append("load_reranker")
        return GoodScorer()

    cfg = {"dense_variants": ["registry+full", "solution+no_examples"], "bm25": {"enabled": True},
           "rerank": {"enabled": True, "k": 5, "alpha": 0.5}, "rejudge": {"enabled": False}}
    dense = FakeDense()
    dense.release = lambda: events.append("release_dense")
    m = HybridSearchModel(cfg, dense, factory)
    m.index(corpus, encode_kwargs={"batch_size": 8})
    assert events == []                                                   # reranker not loaded at index time
    out = m.search(qs, top_k=10, encode_kwargs={"batch_size": 8})
    assert events == ["release_dense", "load_reranker"]                   # dense freed BEFORE the reranker loads
    assert all(len(v) == 10 for v in out.values())
    sent = [t for call in dense.calls for t in call]
    assert sum(t.startswith("Instruct: Retrieve the most relevant") for t in sent) == len(qs["id"])
    assert sum(t.startswith("Instruct: Given a competitive programming") for t in sent) == len(qs["id"])
    assert not any(t.startswith("Instruct:") for t in corpus["text"])
    assert "index_bm25_s" in m.timings and "bm25_scoring_s" in m.timings


def test_search_model_keeps_dense_resident_when_rerank_disabled():
    """No second model loads when rerank is off, so the dense encoder must NOT be released -- it may
    still be needed (e.g. by the caller, or a later search() call)."""
    corpus, qs = toy_task()
    dense = FakeDense()
    m = HybridSearchModel({}, dense)
    m.index(corpus, encode_kwargs={"batch_size": 8})
    m.search(qs, top_k=5, encode_kwargs={"batch_size": 8})
    assert not dense.released and m.dense is dense


def test_search_before_index_is_an_error():
    with pytest.raises(ValueError):
        HybridSearchModel({}, FakeDense()).search({"id": ["q"], "text": ["t"]})


def test_all_stages_off_equals_plain_dense_ranking():
    corpus, qs = toy_task()
    m = HybridSearchModel({}, FakeDense())
    m.index(corpus, encode_kwargs={"batch_size": 8})
    out = m.search(qs, top_k=20, encode_kwargs={"batch_size": 8})
    D = np.stack([_vec(t) for t in corpus["text"]])
    from src.retrieval.query_variants import format_query
    for i, qid in enumerate(qs["id"]):
        scores = D @ _vec(format_query(qs["text"][i], "registry+full"))
        expect = [corpus["id"][j] for j in np.argsort(-scores, kind="stable")[:20]]
        assert sorted(out[qid], key=out[qid].get, reverse=True) == expect


# ---------- official-run guards ----------
def test_official_run_refuses_without_confirmation_and_on_rerun(monkeypatch, tmp_path):
    import json as _json

    from src.eval import run_official

    cfg = tmp_path / "c.json"
    cfg.write_text(_json.dumps({"preset": "f2llm-v2-1.7b", "rerank": {"enabled": True}}))

    # --config is required, so a bare invocation now fails on that first (argparse exit code 2)
    monkeypatch.setattr(sys, "argv", ["run_official.py"])
    with pytest.raises(SystemExit):
        run_official.main()

    # with a valid config but no --confirm-test, the test-split guard is what stops it
    monkeypatch.setattr(run_official, "LOCK", tmp_path / "nolock.json")
    monkeypatch.setattr(sys, "argv", ["run_official.py", "--config", str(cfg)])
    with pytest.raises(SystemExit, match="TEST split"):
        run_official.main()

    lock = tmp_path / "done.json"
    lock.write_text("{}")
    monkeypatch.setattr(run_official, "LOCK", lock)
    monkeypatch.setattr(sys, "argv", ["run_official.py", "--config", str(cfg), "--confirm-test"])
    with pytest.raises(SystemExit, match="already happened"):
        run_official.main()


# ---------- dev context arithmetic (no model) ----------
def test_query_scores_average_and_added_latency():
    from src.eval.dev_lib import BASELINE_VARIANT, DevContext
    ctx = object.__new__(DevContext)
    ctx.query_texts = ["a"] * 4
    ctx.idx = np.array([0, 2])
    D = np.eye(3, dtype=np.float32)
    embs = {"registry+full": (np.array([[1, 0, 0]] * 4, dtype=np.float32), {"encode_seconds": 2.0}),
            "contest+full": (np.array([[0, 1, 0]] * 4, dtype=np.float32), {"encode_seconds": 4.0})}
    ctx.doc_emb = lambda: (D, {})
    ctx.query_emb = lambda v: embs[v]
    S, ms = ctx.query_scores([BASELINE_VARIANT])
    assert S.shape == (2, 3) and ms == 0.0
    S2, ms2 = ctx.query_scores(["registry+full", "contest+full"])
    assert np.allclose(S2[0], [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-6)
    assert ms2 == pytest.approx(1000 * (2.0 + 4.0 - 2.0) / 4)             # extra encode seconds / n queries


# ---------- official run: config/preset safety rails for a one-shot test-split evaluation ----------
def test_official_config_file_matches_the_chosen_configuration():
    """main's DEFAULT official config must be the cheap, confirmed no-reranker path. The expensive
    reranked config lives on the qwen3-reranker-official branch, deliberately not here, so main cannot
    default anyone into a multi-hour run."""
    import json as _json
    import pathlib

    from src.eval.run_official import NON_PIPELINE_KEYS
    from src.retrieval.pipeline import DEFAULT_CONFIG, merge_config

    configs = pathlib.Path(__file__).resolve().parents[1] / "configs"
    raw = _json.loads((configs / "official_f2llm17b_noreranker.json").read_text(encoding="utf-8"))
    assert raw["preset"] == "f2llm-v2-1.7b"                       # F2LLM-v2-1.7B first stage
    assert raw["reference"]["ndcg_at_10"] == 0.93692              # its own expected score, not a global
    # every key is either a pipeline setting or one run_official strips; a new key in neither would
    # abort the run at merge_config, so it is caught here instead of on the one test-split run
    unknown = set(raw) - set(DEFAULT_CONFIG) - set(NON_PIPELINE_KEYS)
    assert not unknown, f"config keys run_official would choke on: {sorted(unknown)}"
    for key in NON_PIPELINE_KEYS:
        raw.pop(key, None)
    cfg = merge_config(raw)                                       # must satisfy the pipeline schema
    assert cfg["dense_variants"] == ["registry+full"]
    assert cfg["bm25"]["enabled"] is False and cfg["rejudge"]["enabled"] is False
    assert cfg["rerank"]["enabled"] is False, "main's default must not enable the reranker"

    # the expensive reranked config must NOT be on main
    assert not (configs / "official_f2llm17b_qwen3_k30_a05.json").exists(), (
        "the reranked config is supposed to live only on the qwen3-reranker-official branch")


def test_official_run_refuses_missing_config_instead_of_defaulting(monkeypatch, tmp_path):
    """A typo'd --config must abort: merge_config({}) is an UNRERANKED dense run, and silently
    evaluating that would waste the one-shot test-split evaluation on the wrong system."""
    from src.eval import run_official

    monkeypatch.setattr(run_official, "LOCK", tmp_path / "nolock.json")
    monkeypatch.setattr(sys, "argv", ["run_official.py", "--config", str(tmp_path / "nope.json"),
                                      "--confirm-test"])
    with pytest.raises(SystemExit, match="Refusing to fall back to defaults"):
        run_official.main()


def test_official_run_rejects_preset_contradicting_the_config(monkeypatch, tmp_path):
    import json as _json

    from src.eval import run_official

    cfg = tmp_path / "c.json"
    cfg.write_text(_json.dumps({"preset": "f2llm-v2-1.7b", "rerank": {"enabled": True}}))
    monkeypatch.setattr(run_official, "LOCK", tmp_path / "nolock.json")
    monkeypatch.setattr(sys, "argv", ["run_official.py", "--config", str(cfg), "--confirm-test",
                                      "--preset", "f2llm-v2-0.6b"])
    with pytest.raises(SystemExit, match="contradicts the 'preset' pinned"):
        run_official.main()


def test_official_run_requires_a_preset_from_somewhere(monkeypatch, tmp_path):
    import json as _json

    from src.eval import run_official

    cfg = tmp_path / "c.json"
    cfg.write_text(_json.dumps({"rerank": {"enabled": True}}))     # no preset pinned, none on the CLI
    monkeypatch.setattr(run_official, "LOCK", tmp_path / "nolock.json")
    monkeypatch.setattr(sys, "argv", ["run_official.py", "--config", str(cfg), "--confirm-test"])
    with pytest.raises(SystemExit, match="No first-stage preset"):
        run_official.main()


# ---------- crash-safety: a killed run must not leave misleading partial state ----------
def test_write_json_atomic_leaves_no_partial_file_on_failure(tmp_path):
    import json as _json

    from src.utils_io import write_json_atomic

    target = tmp_path / "appsretrieval_results.json"
    write_json_atomic(target, {"scores": {"test": [{"ndcg_at_10": 0.9}]}}, indent=2)
    assert _json.loads(target.read_text())["scores"]["test"][0]["ndcg_at_10"] == 0.9

    # a write that blows up mid-serialisation must leave the PREVIOUS good file intact and no .partial
    class Unserialisable:
        pass
    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": Unserialisable()})
    assert _json.loads(target.read_text())["scores"]["test"][0]["ndcg_at_10"] == 0.9
    assert not list(tmp_path.glob("*.partial")), "a failed write left a misleading .partial behind"


def test_save_npy_atomic_roundtrip_and_no_partial(tmp_path):
    from src.utils_io import save_npy_atomic

    arr = np.arange(12, dtype=np.float16).reshape(3, 4)
    p = save_npy_atomic(tmp_path / "docs.npy", arr)
    assert np.array_equal(np.load(p), arr)
    assert not list(tmp_path.glob("*.partial*"))


def test_embedding_cache_entry_is_absent_unless_both_files_landed(tmp_path):
    """The kill-safety invariant the cache relies on: load_cached() needs the .npy AND the meta, so an
    interrupted build reads as a miss (re-encode) rather than as a present-but-truncated entry."""
    from src.retrieval.embedding_cache import cache_paths, get_or_build, load_cached

    texts, key = ["a", "bb"], "modelX"
    get_or_build("docs", texts, lambda t: np.ones((len(t), 3)), key, tmp_path)
    npy, meta = cache_paths("docs", texts, key, tmp_path)
    assert load_cached("docs", texts, key, tmp_path) is not None

    meta.unlink()                                    # simulate a kill after the .npy, before the meta
    assert load_cached("docs", texts, key, tmp_path) is None
    calls = []
    get_or_build("docs", texts, lambda t: (calls.append(1), np.ones((len(t), 3)))[1], key, tmp_path)
    assert calls == [1]                              # re-encoded, not served from the orphaned .npy


def test_official_lock_records_completion_and_is_written_after_the_result(monkeypatch, tmp_path):
    """The lock must mean 'a run finished', so a killed run cannot wrongly block a retry."""
    import inspect

    from src.eval import run_official

    src = inspect.getsource(run_official.main)
    out_pos = src.index("write_json_atomic(a.out")
    lock_pos = src.index("write_json_atomic(LOCK")
    assert out_pos < lock_pos, "the lock must be written after the results file, never before"
    # and both are after mteb.evaluate, i.e. nothing is written before the work completes
    assert src.index("mteb.evaluate") < out_pos
    assert '"completed": True' in src


def test_reranker_logs_progress_on_a_long_pass_and_stays_quiet_on_a_short_one(capsys):
    """Silence during a ~113k-pair pass is what made a killed run indistinguishable from a working one."""
    from src.retrieval import reranker as rk

    class TinyScorer(rk._TorchScorer):
        def __init__(self):
            self.name, self.fell_back_to_fp32 = "tiny", False

        def _score_batch(self, batch):
            return np.zeros(len(batch), dtype=np.float32)

    TinyScorer().score_pairs([("q", "d")] * 50, batch_size=10, progress_pairs=2000)
    assert "[rerank]" not in capsys.readouterr().out          # short pass: no noise

    TinyScorer().score_pairs([("q", "d" * 3)] * 500, batch_size=10, progress_pairs=100)
    out = capsys.readouterr().out
    assert "scoring 500 pairs" in out and "eta" in out and "pairs/s" in out


def test_search_model_logs_encoding_progress(capsys):
    corpus, qs = toy_task()
    m = HybridSearchModel({}, FakeDense())
    m.index(corpus, encode_kwargs={"batch_size": 4})
    out = capsys.readouterr().out
    assert "encoding 20 documents" in out and "eta" in out
    m.search(qs, top_k=5, encode_kwargs={"batch_size": 4})
    assert "queries[registry+full]" in capsys.readouterr().out
