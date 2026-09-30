"""Dev-protocol plumbing: metrics, adoption rule, holdout split, selection, table rows. No models."""
import json
import types

import numpy as np
import pytest

from src.eval import dev_lib, dev_metrics as M, dev_select
from src.eval.dev_data import make_split


# ---------- metrics ----------
def test_ranks_from_scores_and_summary():
    S = np.array([[0.9, 0.1, 0.5],      # target col 0 -> rank 1
                  [0.9, 0.1, 0.5],      # target col 2 -> rank 2
                  [0.9, 0.1, 0.5]])     # target col 1 -> rank 3
    r = M.ranks_from_scores(S, np.array([0, 2, 1]))
    assert r.tolist() == [1, 2, 3]
    s = M.summarize(r)
    assert s["recall@1"] == pytest.approx(1 / 3) and s["recall@10"] == 1.0 and s["n"] == 3
    assert s["mrr@10"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert s["ndcg@10"] == pytest.approx((1 + 1 / np.log2(3) + 1 / 2) / 3)


def test_rank_cap_and_lists():
    assert M.rank_from_list([5, 6, 7], 6) == 2
    assert M.rank_from_list([5, 6, 7], 99) == M.RANK_CAP
    assert M.summarize([M.RANK_CAP])["recall@100"] == 0.0


def test_compare_counts_and_ci():
    base = np.array([3, 3, 3, 1, 50] * 200)
    better = np.array([1, 1, 3, 1, 50] * 200)           # 400 improved, 0 worsened
    c = M.compare(better, base)
    assert (c["improved"], c["worsened"]) == (400, 0) and c["ci_low"] > 0 and c["delta_ndcg@10"] > 0
    worse = np.array([5, 5, 3, 1, 50] * 200)
    c2 = M.compare(worse, base)
    assert (c2["improved"], c2["worsened"]) == (0, 400) and c2["ci_high"] < 0
    assert M.compare(base, base)["same"] == len(base)
    with pytest.raises(AssertionError):
        M.compare(better[:10], base)


def test_adopt_rule():
    ok = {"improved": 100, "worsened": 30, "delta_ndcg@10": 0.02, "ci_low": 0.005, "ci_high": 0.03}
    assert M.adopt(ok)
    assert not M.adopt({**ok, "delta_ndcg@10": 0.002})                # gain too small
    assert not M.adopt({**ok, "ci_low": -0.001})                      # CI includes zero
    assert not M.adopt({**ok, "worsened": 80})                        # lopsided worsening
    assert not M.adopt({**ok, "improved": 0, "worsened": 0})


# ---------- holdout split ----------
def test_split_is_deterministic_disjoint_and_order_independent():
    ids = [f"q{i}" for i in range(5000)]
    tune, hold = make_split(ids)
    assert len(hold) == 1000 and len(tune) == 4000 and not set(tune) & set(hold)
    assert (tune, hold) == make_split(list(reversed(ids)))            # independent of input order
    assert (tune, hold) == make_split(ids)                            # stable across calls
    assert make_split(ids, seed=14)[1] != hold                        # seed matters


def test_dev_data_never_reads_test_qrels():
    import inspect
    from src.eval import dev_data
    src = inspect.getsource(dev_data.load_dev)
    assert 'split="train"' in src and 'split="test"' not in src


# ---------- selection ----------
def _row(exp, ndcg, adopt, notes="", ms=1.0, patch=None):
    return {"exp": exp, "group": "hybrid", "split": "tune", "ndcg@10": ndcg, "adopt": adopt, "notes": notes,
            "added_ms_per_query": ms, "config_patch": patch or {}, "delta_ndcg@10": 0.01, "ci_low": 0.001,
            "ci_high": 0.02, "improved": 10, "worsened": 1}


def test_choose_picks_best_adopted_and_ignores_reference_rows():
    rows = [_row("a", 0.90, True), _row("b", 0.92, True), _row("c", 0.99, False),
            _row("bm25-only", 0.95, True, notes="reference row, not a candidate")]
    assert dev_select.choose(rows)["exp"] == "b"
    assert dev_select.choose([_row("c", 0.99, False)]) is None
    tie = [_row("x", 0.9, True, ms=5.0), _row("y", 0.9, True, ms=1.0)]
    assert dev_select.choose(tie)["exp"] == "y"                       # cheaper wins ties


def test_apply_patch_merges_and_validates():
    cfg = dev_select.apply_patch({}, {"bm25": {"enabled": True}, "rrf": {"k": 30}})
    assert cfg["bm25"]["enabled"] and cfg["rrf"]["k"] == 30 and cfg["rrf"]["w_dense"] == 1.0
    with pytest.raises(KeyError):
        dev_select.apply_patch({}, {"rrf": {"kk": 1}})
    with pytest.raises(KeyError):
        dev_select.apply_patch({}, {"nonsense": {}})


# ---------- table rows ----------
def test_record_writes_row_with_adopt_flag_and_table(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_lib, "RESULTS_JSONL", tmp_path / "r.jsonl")
    monkeypatch.setattr(dev_lib, "TABLE_MD", tmp_path / "t.md")

    class Ctx:
        split_name = "tune"
    base = np.array([3, 3, 3, 1, 50] * 200)
    better = np.array([1, 1, 3, 1, 50] * 200)
    row = dev_lib.record(Ctx, "exp1", "variants", better, base, {"dense_variants": ["x+y"]}, latency_ms=2.5,
                         ref_ranks=base)
    assert row["adopt"] is True and row["improved"] == 400 and row["vs_f2llm_improved"] == 400
    dev_lib.record(Ctx, "baseline", "baseline", base, None, {})
    saved = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert len(saved) == 2 and "adopt" not in saved[1]
    dev_lib.render_table()
    md = (tmp_path / "t.md").read_text()
    assert "exp1" in md and "ADOPT" in md and "vs F2LLM" in md


def test_test_split_loader_refuses_by_default():
    from src.eval.load_data import load_apps
    with pytest.raises(PermissionError, match="TEST split"):
        load_apps()


# ---------- multi-preset support (0.6B vs 1.7B rows must never collide/mix) ----------
def test_row_preset_fallback_for_untagged_legacy_rows():
    assert dev_lib.row_preset({"exp": "old row, written before multi-preset support"}) == "f2llm-v2-0.6b"
    assert dev_lib.row_preset({"preset": "f2llm-v2-1.7b"}) == "f2llm-v2-1.7b"

    class CtxNoArgs:
        pass
    assert dev_lib.row_preset(CtxNoArgs()) == "f2llm-v2-0.6b"

    class Ctx06:
        args = types.SimpleNamespace(preset="f2llm-v2-0.6b")
    class Ctx17:
        args = types.SimpleNamespace(preset="f2llm-v2-1.7b")
    assert dev_lib.row_preset(Ctx06()) == "f2llm-v2-0.6b"
    assert dev_lib.row_preset(Ctx17()) == "f2llm-v2-1.7b"


def test_record_tags_preset_and_render_table_never_collides_across_presets(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_lib, "RESULTS_JSONL", tmp_path / "r.jsonl")
    monkeypatch.setattr(dev_lib, "TABLE_MD", tmp_path / "t.md")

    class Ctx17:
        split_name = "tune"
        args = types.SimpleNamespace(preset="f2llm-v2-1.7b")
    base = np.array([3, 3, 3, 1, 50] * 200)
    better = np.array([1, 1, 3, 1, 50] * 200)

    row17 = dev_lib.record(Ctx17, "qwen3[card] k=20 alpha=0.25", "rerank", better, base, {}, ref_ranks=base)
    assert row17["preset"] == "f2llm-v2-1.7b"

    # an old, untagged 0.6B row with the SAME exp/group/split/n must survive alongside it, not be
    # overwritten -- this is the exact scenario (identical exp string, different preset) the fix targets
    legacy_row = {**row17, "preset": None, "ndcg@10": 0.8959 + 0.0096}
    del legacy_row["preset"]
    with open(dev_lib.RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(legacy_row) + "\n")

    dev_lib.render_table()
    md = (dev_lib.TABLE_MD).read_text()
    assert md.count("qwen3[card] k=20 alpha=0.25") == 2      # both rows present, not collapsed to one
    assert "f2llm-v2-1.7b" in md and "f2llm-v2-0.6b" in md    # legacy row's fallback preset shows up too


def _fake_row(exp, preset, ndcg, extra=None):
    row = {"exp": exp, "group": "rerank", "split": "tune", "n": 1000, "adopt": True, "ndcg@10": ndcg,
           "added_ms_per_query": 1.0, "config_patch": {}, "delta_ndcg@10": 0.01, "ci_low": 0.001,
           "ci_high": 0.02, "improved": 10, "worsened": 1}
    if preset is not None:
        row["preset"] = preset
    row.update(extra or {})
    return row


def test_dev_select_load_rows_and_choose_are_preset_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_select, "RESULTS_JSONL", tmp_path / "r.jsonl")
    rows = [_fake_row("a", "f2llm-v2-0.6b", 0.90),
            _fake_row("b", None, 0.99),                     # legacy, no 'preset' field at all
            _fake_row("c", "f2llm-v2-1.7b", 0.95)]
    with open(dev_select.RESULTS_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    six_b = dev_select.load_rows("rerank", preset="f2llm-v2-0.6b")
    assert {r["exp"] for r in six_b} == {"a", "b"}                        # legacy row counts as 0.6B
    assert dev_select.choose(six_b)["exp"] == "b"                         # best WITHIN 0.6B only

    one_7b = dev_select.load_rows("rerank", preset="f2llm-v2-1.7b")
    assert {r["exp"] for r in one_7b} == {"c"}
    assert dev_select.choose(one_7b)["exp"] == "c"                        # never sees the 0.6B rows


def test_dev_select_writes_separate_chosen_file_per_non_default_preset(tmp_path, monkeypatch):
    import sys

    from src.eval import dev_select as S
    monkeypatch.setattr(S, "RESULTS_JSONL", tmp_path / "r.jsonl")
    chosen_06 = tmp_path / "chosen.json"
    monkeypatch.setattr(S, "CHOSEN", chosen_06)
    row = _fake_row("x", "f2llm-v2-1.7b", 0.9, {"config_patch": {"rerank": {"enabled": True, "k": 20}}})
    (tmp_path / "r.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["dev_select.py", "--group", "rerank", "--preset", "f2llm-v2-1.7b"])
    S.main()

    assert not chosen_06.exists()                                        # 0.6B's chosen.json untouched
    chosen_17 = tmp_path / "chosen_f2llm-v2-1.7b.json"
    assert chosen_17.exists() and json.loads(chosen_17.read_text())["rerank"]["k"] == 20


def test_dev_split_reference_gap_is_looked_up_per_preset():
    assert dev_lib.TEST_NDCG_REFERENCE["f2llm-v2-1.7b"] == pytest.approx(0.93692)
    assert dev_lib.TEST_NDCG_REFERENCE["f2llm-v2-0.6b"] == pytest.approx(0.90446)
    assert dev_lib.TEST_NDCG_REFERENCE.get("some-unknown-preset") is None   # dev_split.py must skip, not crash


# ---------- dense+dense fusion: cache-only loading, never a silent GPU encode ----------
def test_load_cached_returns_none_on_miss_and_shares_key_with_get_or_build(tmp_path):
    from src.retrieval.embedding_cache import cache_paths, get_or_build, load_cached

    texts, key = ["a", "bb"], "modelA"
    assert load_cached("docs", texts, key, tmp_path) is None          # nothing cached yet
    built, _ = get_or_build("docs", texts, lambda t: np.ones((len(t), 3)), key, tmp_path)
    hit = load_cached("docs", texts, key, tmp_path)
    assert hit is not None and np.array_equal(hit[0], built) and hit[1]["cache_hit"] is True
    npy, meta = cache_paths("docs", texts, key, tmp_path)
    assert npy.exists() and meta.exists()                              # same key both paths agree on
    assert load_cached("docs", ["a", "different"], key, tmp_path) is None
    assert load_cached("docs", texts, "otherModel", tmp_path) is None


def test_cached_embeddings_raises_naming_the_build_command(monkeypatch, tmp_path):
    from src.retrieval import embedding_cache

    monkeypatch.setattr(embedding_cache, "CACHE_DIR", tmp_path)
    with pytest.raises(dev_lib.CacheMissError) as ei:
        dev_lib.cached_embeddings("f2llm-v2-1.7b", ["doc"], ["query"])
    msg = str(ei.value)
    assert "f2llm-v2-1.7b" in msg and "dev_split.py --preset f2llm-v2-1.7b" in msg
    assert "docs" in msg and "gitignored" in msg                       # says why a fresh session is empty


def test_cached_embeddings_returns_arrays_when_both_halves_present(monkeypatch, tmp_path):
    from src.retrieval import embedding_cache
    from src.retrieval.query_variants import format_query

    monkeypatch.setattr(embedding_cache, "CACHE_DIR", tmp_path)
    docs, queries = ["doc one", "doc two"], ["q one"]
    key = dev_lib.model_key_for("f2llm-v2-0.6b")
    embedding_cache.get_or_build("docs", docs, lambda t: np.ones((len(t), 4)), key)
    embedding_cache.get_or_build("q_registry+full", [format_query(q, "registry+full") for q in queries],
                                 lambda t: np.full((len(t), 4), 2.0), key)
    d_emb, q_emb = dev_lib.cached_embeddings("f2llm-v2-0.6b", docs, queries)
    assert d_emb.shape == (2, 4) and q_emb.shape == (1, 4)
    assert np.allclose(d_emb, 1.0) and np.allclose(q_emb, 2.0)


def test_model_key_for_distinguishes_presets():
    keys = {p: dev_lib.model_key_for(p) for p in ("f2llm-v2-0.6b", "f2llm-v2-1.7b", "f2llm-v2-4b")}
    assert len(set(keys.values())) == 3                                # no two presets share a cache
    assert "F2LLM-v2-1.7B" in keys["f2llm-v2-1.7b"]


def test_dev_fuse_exits_cleanly_on_cache_miss_without_encoding(monkeypatch, tmp_path):
    """A missing cache must abort with the build command -- never silently start a GPU encode."""
    import sys

    from src.eval import dev_fuse as F

    loaded = []

    class FakeCtx:
        def __init__(self, args):
            self.args = args
            self.doc_texts, self.query_texts = ["d"], ["q"]
            self.idx = np.array([0])
            self.split_name = "tune"

        def query_scores(self, variants):
            loaded.append("ENCODED")                       # must never happen
            return np.zeros((1, 1)), 0.0

    monkeypatch.setattr(F, "DevContext", FakeCtx)
    monkeypatch.setattr(F, "cached_embeddings",
                        lambda *a, **k: (_ for _ in ()).throw(dev_lib.CacheMissError("no cache for X")))
    monkeypatch.setattr(sys, "argv", ["dev_fuse.py", "--preset", "f2llm-v2-1.7b",
                                      "--preset-b", "f2llm-v2-0.6b", "--n-queries", "1"])
    with pytest.raises(SystemExit, match="CACHE MISS"):
        F.main()
    assert loaded == []                                    # nothing encoded, no GPU time spent


def test_dev_fuse_end_to_end_on_cached_embeddings(monkeypatch, tmp_path):
    """Full mocked sweep: both caches present, CPU only, correct rows written and no model loaded."""
    import sys

    from src.eval import dev_fuse as F

    monkeypatch.setattr(dev_lib, "RESULTS_JSONL", tmp_path / "r.jsonl")
    monkeypatch.setattr(dev_lib, "TABLE_MD", tmp_path / "t.md")

    n_docs = 40
    rng = np.random.RandomState(0)
    emb = {"f2llm-v2-1.7b": (rng.rand(n_docs, 8), rng.rand(3, 8)),
           "f2llm-v2-0.6b": (rng.rand(n_docs, 8), rng.rand(3, 8))}

    class FakeCtx:
        def __init__(self, args):
            self.args = args
            self.doc_texts = ["d"] * n_docs
            self.query_texts = ["q0", "q1", "q2"]
            self.idx = np.array([0, 1, 2])
            self.split_name = "tune"
            self._target = np.array([0, 1, 2])

        def rel_of(self):
            return self._target

        def ranks_from_lists(self, lists):
            return np.array([M.rank_from_list(l, t) for l, t in zip(lists, self._target)])

    monkeypatch.setattr(F, "DevContext", FakeCtx)
    monkeypatch.setattr(F, "cached_embeddings", lambda preset, docs, queries, variant=None: emb[preset])
    monkeypatch.setattr(sys, "argv", ["dev_fuse.py", "--preset", "f2llm-v2-1.7b",
                                      "--preset-b", "f2llm-v2-0.6b", "--n-queries", "3"])
    F.main()

    rows = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text().splitlines()]
    groups = {r["group"] for r in rows}
    assert groups == {"fuse"}
    assert all(r["preset"] == "f2llm-v2-1.7b" for r in rows)          # tagged with model A
    refs = [r for r in rows if r["notes"].startswith("reference row")]
    assert len(refs) == 2                                             # both models standalone
    rrf = [r for r in rows if r["exp"].startswith("RRF dense+dense")]
    avg = [r for r in rows if r["exp"].startswith("score-avg")]
    assert len(rrf) == len(F.K_GRID) * len(F.W_GRID)
    assert len(avg) == len(F.AVG_W_GRID)
    assert all(r["config_patch"] == {} for r in rrf + avg)            # not adoptable into chosen.json
    assert {r["fusion"] for r in rrf} == {"rrf"} and {r["fusion"] for r in avg} == {"score_average"}
    assert {r["model_b"] for r in rrf + avg} == {"f2llm-v2-0.6b"}     # both fusion inputs recorded
    assert (tmp_path / "t.md").exists()


def test_dev_fuse_refuses_to_fuse_a_model_with_itself(monkeypatch):
    import sys

    from src.eval import dev_fuse as F

    class FakeCtx:
        def __init__(self, args):
            self.args = args
            self.doc_texts, self.query_texts, self.idx = ["d"], ["q"], np.array([0])
            self.split_name = "tune"

    monkeypatch.setattr(F, "DevContext", FakeCtx)
    monkeypatch.setattr(sys, "argv", ["dev_fuse.py", "--preset", "f2llm-v2-1.7b",
                                      "--preset-b", "f2llm-v2-1.7b"])
    with pytest.raises(SystemExit, match="no-op"):
        F.main()


# ---------- stacking fusion under reranking (fused first stage + reranker) ----------
def _stacking_ctx(n_docs=40):
    """A FakeCtx + fake cached 0.6B embeddings for dev_rerank's fusion path."""
    rng = np.random.RandomState(3)

    class FakeCtx:
        def __init__(self, args):
            self.args = args
            self.doc_texts = ["d"] * n_docs
            self.query_texts = ["q0", "q1", "q2"]
            self.idx = np.array([0, 1, 2])
            self.split_name = "tune"
            self.data = {"query_ids": ["q0", "q1", "q2"]}
            self._target = np.array([0, 1, 2])
            self.released = 0

        def rel_of(self):
            return self._target

        def query_scores(self, variants):
            return rng.rand(3, n_docs).astype(np.float32), 0.0

        def ranks_from_lists(self, lists):
            return np.array([M.rank_from_list(l, t) for l, t in zip(lists, self._target)])

        def release_encoder(self):
            self.released += 1

    return FakeCtx, (rng.rand(n_docs, 8), rng.rand(3, 8))


def test_dev_rerank_fused_first_stage_and_reranked_adoption_base(monkeypatch, tmp_path):
    import sys

    from src.eval import dev_rerank as R
    from src.retrieval.pipeline import merge_config

    monkeypatch.setattr(dev_lib, "RESULTS_JSONL", tmp_path / "r.jsonl")
    monkeypatch.setattr(dev_lib, "TABLE_MD", tmp_path / "t.md")
    FakeCtx, fake_emb = _stacking_ctx()
    monkeypatch.setattr(R, "DevContext", FakeCtx)
    monkeypatch.setattr(R, "load_base_config", lambda p: merge_config({}))
    monkeypatch.setattr(R, "cached_embeddings", lambda *a, **k: fake_emb)

    seen = []

    def fake_cached_rerank(name, instruction, max_length, dtype, args, ctx, cand_lists, kmax):
        seen.append([tuple(l[:3]) for l in cand_lists])       # record WHICH candidates were scored
        return np.random.RandomState(9).rand(len(cand_lists), kmax).astype(np.float32), 0.02

    monkeypatch.setattr(R, "cached_rerank", fake_cached_rerank)
    monkeypatch.setattr(sys, "argv", [
        "dev_rerank.py", "--preset", "f2llm-v2-1.7b", "--n-queries", "3",
        "--rerankers", "qwen3-reranker-0.6b", "--instructions", "apps",
        "--fuse-with", "f2llm-v2-0.6b", "--fuse-w-b", "0.25", "--fuse-norm", "zscore",
        "--compare-to-rerank", "--base-k", "30", "--base-alpha", "0.5",
        "--alphas", "0.0", "0.5", "1.0"])
    R.main()

    # the base rerank pass ran on the UNFUSED candidates, the sweep on the FUSED ones -- different lists
    assert len(seen) == 2 and seen[0] != seen[1]

    rows = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text().splitlines()]
    base_rows = [r for r in rows if r["exp"].startswith("BASE ")]
    sweep = [r for r in rows if r["exp"].startswith("fused(")]
    assert len(base_rows) == 1 and "standing best" in base_rows[0]["exp"]
    assert base_rows[0]["notes"].startswith("reference row")          # base itself is not a candidate
    assert len(sweep) == 3 * len(R.K_LIST)                            # 3 alphas x k grid
    assert all(r["first_stage"] == "fused" for r in sweep)
    assert all(r["fuse_w_b"] == 0.25 and r["fuse_with"] == "f2llm-v2-0.6b" for r in sweep)
    assert all("standing best" in r["adoption_base"] for r in sweep)  # judged against the reranked base
    assert all(r["config_patch"] == {} for r in sweep)                # fused rows not adoptable
    assert {a for r in sweep for a in [float(r["exp"].split("alpha=")[1])]} == {0.0, 0.5, 1.0}


def test_dev_rerank_unfused_path_is_unchanged(monkeypatch, tmp_path):
    """Without the new flags, behaviour matches the original: base is the raw first stage, rows keep
    their real config_patch, and nothing is tagged as fused."""
    import sys

    from src.eval import dev_rerank as R
    from src.retrieval.pipeline import merge_config

    monkeypatch.setattr(dev_lib, "RESULTS_JSONL", tmp_path / "r.jsonl")
    monkeypatch.setattr(dev_lib, "TABLE_MD", tmp_path / "t.md")
    FakeCtx, _ = _stacking_ctx()
    monkeypatch.setattr(R, "DevContext", FakeCtx)
    monkeypatch.setattr(R, "load_base_config", lambda p: merge_config({}))
    monkeypatch.setattr(R, "cached_embeddings",
                        lambda *a, **k: pytest.fail("must not touch fusion caches when --fuse-with is off"))
    monkeypatch.setattr(R, "cached_rerank",
                        lambda *a, **k: (np.random.RandomState(0).rand(3, a[7]).astype(np.float32), 0.01))
    monkeypatch.setattr(sys, "argv", ["dev_rerank.py", "--preset", "f2llm-v2-1.7b", "--n-queries", "3",
                                      "--rerankers", "qwen3-reranker-0.6b", "--instructions", "apps"])
    R.main()

    rows = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert rows and not any(r["exp"].startswith(("BASE ", "fused(")) for r in rows)
    assert all(r["first_stage"] == "dense" for r in rows)
    assert all(r["config_patch"]["rerank"]["enabled"] for r in rows)   # still adoptable


# ---------- --n-queries 0 must mean "whole tune set", and --ks must narrow the grid ----------
def test_n_queries_zero_means_all_and_default_stays_small(monkeypatch):
    """add_common_args documents '0 = all'. Each script defaults to a small slice, but an explicit 0
    must reach DevContext as 0 -- silently rewriting it to the default would answer a different
    question than the one asked (and look identical to the smaller run)."""
    import sys

    seen = {}

    def fake_ctx_factory(name):
        class FakeCtx:
            def __init__(self, args):
                seen[name] = args.n_queries
                raise SystemExit("stop after arg parsing")
        return FakeCtx

    for mod_name, tag, default in (("dev_rerank", "rerank", 1000), ("dev_fuse", "fuse", 1000),
                                   ("dev_rejudge", "rejudge", 300)):
        mod = __import__(f"src.eval.{mod_name}", fromlist=["main"])
        monkeypatch.setattr(mod, "DevContext", fake_ctx_factory(mod_name), raising=False)
        if mod_name == "dev_rejudge":     # imports DevContext inside main()
            monkeypatch.setattr(dev_lib, "DevContext", fake_ctx_factory(mod_name))

        monkeypatch.setattr(sys, "argv", [f"{mod_name}.py", "--preset", "f2llm-v2-1.7b"])
        with pytest.raises(SystemExit):
            mod.main()
        assert seen[mod_name] == default, f"{mod_name} default should be {default}"

        monkeypatch.setattr(sys, "argv", [f"{mod_name}.py", "--preset", "f2llm-v2-1.7b", "--n-queries", "0"])
        with pytest.raises(SystemExit):
            mod.main()
        assert seen[mod_name] == 0, f"{mod_name} silently overrode an explicit --n-queries 0"


def test_dev_rerank_ks_narrows_grid_and_rejects_k_above_kmax(monkeypatch, tmp_path):
    import sys

    from src.eval import dev_rerank as R
    from src.retrieval.pipeline import merge_config

    monkeypatch.setattr(dev_lib, "RESULTS_JSONL", tmp_path / "r.jsonl")
    monkeypatch.setattr(dev_lib, "TABLE_MD", tmp_path / "t.md")
    FakeCtx, _ = _stacking_ctx()
    monkeypatch.setattr(R, "DevContext", FakeCtx)
    monkeypatch.setattr(R, "load_base_config", lambda p: merge_config({}))
    monkeypatch.setattr(R, "cached_rerank",
                        lambda *a, **k: (np.random.RandomState(0).rand(3, a[7]).astype(np.float32), 0.01))

    monkeypatch.setattr(sys, "argv", ["dev_rerank.py", "--preset", "f2llm-v2-1.7b", "--n-queries", "3",
                                      "--rerankers", "qwen3-reranker-0.6b", "--instructions", "apps",
                                      "--ks", "20", "30", "--alphas", "0.2", "0.3", "0.4", "0.5"])
    R.main()
    rows = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 4                                    # exactly the narrowed grid, no k=10
    ks = {int(r["exp"].split("k=")[1].split(" ")[0]) for r in rows}
    alphas = {float(r["exp"].split("alpha=")[1]) for r in rows}
    assert ks == {20, 30} and alphas == {0.2, 0.3, 0.4, 0.5}

    # a k above --kmax can never be scored, so it must fail loudly rather than be silently dropped
    monkeypatch.setattr(sys, "argv", ["dev_rerank.py", "--preset", "f2llm-v2-1.7b", "--n-queries", "3",
                                      "--ks", "50", "--kmax", "30"])
    with pytest.raises(SystemExit, match="exceed --kmax"):
        R.main()


# ---------- the document pool must ALWAYS be the full corpus; only queries are ever subsampled ----------
def test_query_subsampling_never_shrinks_the_document_pool():
    """The inflation failure mode we are ruling out: scoring against a reduced corpus makes NDCG@10 look
    better for free. Asserts the score matrix keeps one column per corpus document no matter how few
    queries are sampled, so ranks are always computed against the whole pool."""
    from src.eval.dev_lib import DevContext

    n_docs, n_train = 8765, 5000
    doc_emb = np.random.RandomState(0).rand(n_docs, 6).astype(np.float32)
    query_emb = np.random.RandomState(1).rand(n_train, 6).astype(np.float32)

    for n_queries in (0, 1, 50, 1000, 4000):
        ctx = object.__new__(DevContext)
        ctx.args = types.SimpleNamespace(n_queries=n_queries, batch_size=4, use_holdout=False)
        ctx.query_texts = ["q"] * n_train
        pool = list(range(4000))                                  # the tune half
        if n_queries:
            pool = sorted(np.random.RandomState(7).choice(pool, min(n_queries, len(pool)), replace=False))
        ctx.idx = np.array(pool)
        ctx.doc_emb = lambda: (doc_emb, {})
        ctx.query_emb = lambda v: (query_emb, {"encode_seconds": 0.0})

        S, _ = ctx.query_scores(["registry+full"])
        assert S.shape[1] == n_docs, f"corpus was subsampled to {S.shape[1]} docs at n_queries={n_queries}"
        assert S.shape[0] == len(ctx.idx)                         # only the query axis shrinks


def test_load_dev_keeps_every_corpus_row():
    """load_dev must not filter the corpus -- doc_ids/doc_texts come straight off the HF corpus split."""
    import inspect

    from src.eval import dev_data
    src = inspect.getsource(dev_data.load_dev)
    assert 'doc_ids = [str(r["_id"]) for r in corpus]' in src
    assert 'doc_texts = [r.get("text") or "" for r in corpus]' in src
    # qrels filtering may drop QUERIES, but nothing may rebuild/filter the doc lists
    assert "doc_texts = [" not in src.split('doc_texts = [r.get("text") or "" for r in corpus]')[1]


def test_ranks_from_scores_ranks_against_every_column():
    """The metric counts all documents that outscore the target, i.e. the full pool, not a top-k window."""
    S = np.zeros((1, 8765), dtype=np.float32)
    S[0, 500] = 1.0                       # target
    S[0, :300] = 2.0                      # 300 documents beat it
    assert M.ranks_from_scores(S, np.array([500]))[0] == 301


def test_only_recall_diagnostic_subsamples_docs_and_it_is_not_a_scoring_script():
    """recall_diagnostic.py subsamples the corpus BY DESIGN (--n-docs). Confirm no dev-table scoring
    script does, so no headline number could have come from a reduced pool."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("dev_split.py", "dev_rerank.py", "dev_fuse.py", "dev_hybrid.py", "dev_variants.py",
                 "dev_rejudge.py"):
        src = (root / "src" / "eval" / name).read_text(encoding="utf-8")
        # strip comments/docstring prose: only executable references count
        code = "\n".join(l.split("#")[0] for l in src.splitlines())
        assert "--n-docs" not in code, f"{name} exposes a doc-count flag"
        assert "n_docs=" not in code, f"{name} passes a doc-count argument"
        for sampler in ("random.sample(", ".choice("):
            for hit in code.split(sampler)[1:]:
                assert "doc" not in hit[:60].lower(), f"{name} appears to sample documents: {hit[:60]!r}"
    assert "--n-docs" in (root / "src" / "eval" / "recall_diagnostic.py").read_text(encoding="utf-8")
