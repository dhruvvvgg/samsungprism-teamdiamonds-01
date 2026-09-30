"""The official P0 path must behave identically on every branch.

`main` holds a completed test-split run: F2LLM-v2-1.7B alone, no reranker, no fusion, uncapped queries,
NDCG@10 0.9376 / MRR@10 0.9238. That run cost the one test-split evaluation the project is allowed, so a
feature branch that quietly changes what `configs/official_f2llm17b_noreranker.json` resolves to would
invalidate the submitted number without anything failing.

This file is deliberately duplicated on every feature branch, unchanged. It drives the real
`run_official.main()` with the real config and fakes only the model and `mteb.evaluate`, so it asserts
what the official run would actually construct -- not what a config file says it should.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CONFIG = ROOT / "configs" / "official_f2llm17b_noreranker.json"

EXPECTED_MODEL = "codefuse-ai/F2LLM-v2-1.7B"
EXPECTED_REVISION = "3766d46e7a68545ed6190c15330983f9b39ab718"
EXPECTED_MAX_SEQ = 8192          # the preset's full length: the official path is NEVER query-capped


class FakeDense:
    """Records how it was constructed, then behaves like a tiny encoder."""
    last_kwargs = None
    last_args = None

    def __init__(self, *args, **kwargs):
        FakeDense.last_args, FakeDense.last_kwargs = args, kwargs
        self.fell_back_to_fp32 = False
        self.max_seq_length_calls = []

    def embed(self, texts, batch_size=8, show_progress_bar=False):
        out = []
        for t in texts:
            h = abs(hash(t.strip()[-40:])) % 997
            v = np.array([1.0, h % 7, h % 11, h % 13], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    def set_max_seq_length(self, n):
        self.max_seq_length_calls.append(n)
        return n

    def release(self):
        pass


def run_official_with_fakes(monkeypatch, tmp_path, extra_argv=()):
    """Run the real main() against the real config, with the model and mteb.evaluate faked.

    Only `evaluate` and `get_task` are replaced, on the real mteb module: the SearchProtocol model
    imports mteb.models.model_meta at construction time, so a stand-in module object would break the
    very code path this is meant to exercise."""
    import mteb

    from src.eval import run_official

    captured = {}

    class FakeTaskResult:
        def to_dict(self):
            return {"scores": {"test": [{"ndcg_at_10": 0.9376, "mrr_at_10": 0.9238}]}}

    class FakeResult:
        task_results = [FakeTaskResult()]

    def fake_evaluate(model, tasks, **kwargs):
        captured["model"] = model
        # exercise the same index()/search() calls MTEB would make, so the pipeline really runs
        model.index({"id": ["d0", "d1", "d2"], "text": ["def a(): pass", "def b(): pass", "def c(): x"]})
        model.search({"id": ["q0"], "text": ["find b"]}, top_k=3)
        return FakeResult()

    monkeypatch.setattr(mteb, "evaluate", fake_evaluate)
    monkeypatch.setattr(mteb, "get_task", lambda name: name)
    monkeypatch.setattr("src.retrieval.dense_encoder.DenseEncoder", FakeDense)
    monkeypatch.setattr(run_official, "LOCK", tmp_path / "lock.json")
    argv = ["run_official.py", "--config", str(OFFICIAL_CONFIG), "--confirm-test",
            "--out", str(tmp_path / "results.json"),
            "--rankings-out", str(tmp_path / "rankings.json"),
            "--checksums-out", str(tmp_path / "checksums.json"),
            "--index-dir", str(tmp_path / "index"), *extra_argv]
    monkeypatch.setattr(sys, "argv", argv)
    run_official.main()
    return captured


def test_official_config_resolves_to_f2llm_17b_with_no_second_stage(monkeypatch, tmp_path):
    captured = run_official_with_fakes(monkeypatch, tmp_path)

    # the model the official run would load
    assert FakeDense.last_args[0] == EXPECTED_MODEL
    assert FakeDense.last_kwargs["revision"] == EXPECTED_REVISION
    assert FakeDense.last_kwargs["dtype"] == "fp16"
    assert FakeDense.last_kwargs["expect_eos"] is True

    # and the pipeline it would run: no reranker, no BM25, no LLM re-judge, one dense variant
    cfg = captured["model"].cfg
    assert cfg["rerank"]["enabled"] is False, "the official path must not enable the reranker"
    assert cfg["bm25"]["enabled"] is False, "the official path must not enable BM25 hybrid"
    assert cfg["rejudge"]["enabled"] is False, "the official path must not enable the LLM re-judge"
    assert cfg["dense_variants"] == ["registry+full"], "no variant averaging / dense-dense fusion"


def test_official_path_is_never_query_capped(monkeypatch, tmp_path):
    """The serving query cap (1024) is a CLI/API convenience measured on dev. If it ever leaked into
    the official run it would silently truncate 2.6% of test queries and move the submitted number."""
    run_official_with_fakes(monkeypatch, tmp_path)
    assert FakeDense.last_args[2] == EXPECTED_MAX_SEQ, (
        "the official run must encode at the preset's full length, not a serving cap")
    assert "max_query_tokens" not in FakeDense.last_kwargs
    from src.runtime_index import DEFAULT_SERVING_QUERY_TOKENS
    assert FakeDense.last_args[2] != DEFAULT_SERVING_QUERY_TOKENS


def test_official_run_does_not_go_through_the_serving_stack(monkeypatch, tmp_path):
    """SearchService applies the serving defaults (query cap, CPU precision). The official run must not
    use it, or those defaults would apply to the submitted result."""
    import src.search_service as ss

    def explode(*a, **kw):
        raise AssertionError("run_official must not construct a SearchService")

    monkeypatch.setattr(ss, "SearchService", explode)
    run_official_with_fakes(monkeypatch, tmp_path)      # must complete without touching SearchService


def test_official_config_file_is_unchanged_in_substance():
    cfg = json.loads(OFFICIAL_CONFIG.read_text(encoding="utf-8"))
    assert cfg["preset"] == "f2llm-v2-1.7b"
    assert cfg["rerank"] == {"enabled": False}
    assert cfg["bm25"] == {"enabled": False}
    assert cfg["rejudge"] == {"enabled": False}
    assert cfg["dense_variants"] == ["registry+full"]
    assert cfg["final_depth"] == 1000
    assert cfg["reference"]["ndcg_at_10"] == 0.93692
    assert "max_query_tokens" not in cfg and "query_cap" not in cfg


def test_the_preset_itself_still_points_at_the_evaluated_checkpoint():
    from src.retrieval.model_presets import PRESETS
    p = PRESETS["f2llm-v2-1.7b"]
    assert p["model"] == EXPECTED_MODEL
    assert p["revision"] == EXPECTED_REVISION, "a different revision is a different model"
    assert p["max_seq_length"] == EXPECTED_MAX_SEQ
    assert p["expect_eos"] is True


def test_official_run_still_refuses_without_confirm_test(monkeypatch, tmp_path):
    from src.eval import run_official
    monkeypatch.setattr(run_official, "LOCK", tmp_path / "lock.json")
    monkeypatch.setattr(sys, "argv", ["run_official.py", "--config", str(OFFICIAL_CONFIG)])
    with pytest.raises(SystemExit, match="Refusing to run"):
        run_official.main()
