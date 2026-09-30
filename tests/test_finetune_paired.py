"""Experiment B, paired evaluation: base 0.6B and the tuned model scored on the same holdout queries in
one run, with per-query scores, a paired bootstrap and the adoption verdict. Fake encoders only."""
import json
import re

import numpy as np
import pytest

import src.train.finetune_lite as ft
from src.eval import dev_metrics as M


def ranks_with(n, improved_to=None, improved=0, worsened=0, start=1):
    """A ranks array: all `start`, with the first `improved` moved to `improved_to`, then `worsened` worse."""
    r = np.full(n, start)
    if improved_to is not None:
        r[:improved] = improved_to
    return r


# --- the statistics -------------------------------------------------------------------------------------

def test_a_clear_gain_is_adopted_with_a_ci_above_zero():
    base = np.full(1000, 1)
    base[:150] = 3                                   # 15% of queries miss the top 1
    tuned = np.ones(1000, dtype=int)                 # the tuned model gets all of them right
    c = ft.paired_comparison(tuned, base)
    assert (c["improved"], c["worsened"], c["same"]) == (150, 0, 850)
    assert c["delta_ndcg@10"] == pytest.approx(150 * (1 - 0.5) / 1000)
    assert c["ci_low"] > 0 and c["adopt"] is True
    assert c["delta_mrr@10"] > 0 and c["mrr_ci_low"] < c["mrr_ci_high"]


def test_no_change_is_not_adopted_and_its_ci_straddles_zero():
    r = np.random.RandomState(0).randint(1, 4, size=500)
    c = ft.paired_comparison(r, r.copy())
    assert (c["improved"], c["worsened"]) == (0, 0) and c["adopt"] is False
    assert c["delta_ndcg@10"] == 0 and c["ci_low"] <= 0 <= c["ci_high"]


def test_a_gain_under_the_threshold_is_not_adopted_even_if_significant():
    base = np.full(1000, 2)
    tuned = base.copy()
    tuned[:8] = 1                                    # 8 queries improve: dNDCG@10 = 8 * 0.369 / 1000
    c = ft.paired_comparison(tuned, base)
    assert 0 < c["delta_ndcg@10"] < 0.005 and c["adopt"] is False


def test_a_gain_that_worsens_too_many_queries_is_not_adopted():
    base = np.full(400, 2)
    tuned = base.copy()
    tuned[:100] = 1                                  # 100 improve
    tuned[100:170] = 5                               # 70 worsen: more than half of 100
    c = ft.paired_comparison(tuned, base)
    assert c["improved"] == 100 and c["worsened"] == 70
    assert c["adopt"] is False
    tuned[100:140] = 2                               # only 30 worsen now
    tuned[140:170] = 2
    c2 = ft.paired_comparison(tuned, base)
    assert c2["worsened"] == 0 and c2["delta_ndcg@10"] >= 0.005 and c2["adopt"] is True


def test_the_bootstrap_is_reproducible():
    a, b = np.array([1, 2, 3, 1, 5] * 40), np.array([2, 2, 4, 3, 1] * 40)
    assert ft.paired_comparison(a, b) == ft.paired_comparison(a, b)


def test_per_query_scores_average_to_the_summary():
    r = np.array([1, 2, 5, 11, 1001, 1, 3])
    pq = ft.per_query_scores(r)
    s = M.summarize(r)
    assert len(pq["ndcg@10"]) == len(pq["mrr@10"]) == len(pq["rank"]) == 7
    assert np.mean(pq["ndcg@10"]) == pytest.approx(s["ndcg@10"])
    assert np.mean(pq["mrr@10"]) == pytest.approx(s["mrr@10"])
    assert pq["mrr@10"][3] == 0 and pq["ndcg@10"][4] == 0          # beyond the top 10 scores nothing


def test_the_report_pairs_against_the_reference_only_when_it_has_it():
    tuned, base = np.full(100, 1), np.full(100, 2)
    without = ft.paired_report(tuned, base, None, n_boot=200)
    assert "vs_1.7b" not in without and "vs_base_0.6b" in without
    with_ref = ft.paired_report(tuned, base, np.full(100, 1), n_boot=200)
    assert with_ref["vs_1.7b"]["improved"] == 0 and with_ref["vs_1.7b"]["adopt"] is False
    assert set(with_ref["per_query"]) == {"tuned", "base_0.6b", "reference_1.7b"}


def test_loading_reference_ranks_requires_the_same_holdout(tmp_path):
    p = tmp_path / "ref.json"
    assert ft.load_reference_ranks(p, [1, 2, 3]) is None
    p.write_text(json.dumps({"holdout_idx": [1, 2, 3], "ranks": [1, 1, 2]}))
    assert ft.load_reference_ranks(p, [1, 2, 3]).tolist() == [1, 1, 2]
    with pytest.raises(SystemExit, match="different holdout"):
        ft.load_reference_ranks(p, [1, 2, 4])


# --- the stage, end to end with fake encoders -------------------------------------------------------------

N = 40


class FakeEncoder:
    """One-hot documents; queries land on their document, except for the base model, which misses the
    first `misses` queries (it puts them on a neighbouring document instead)."""

    def __init__(self, tuned, misses=12):
        self.tuned, self.misses = tuned, misses
        self.released = False

    def embed(self, texts, batch_size=8):
        out = np.zeros((len(texts), N), dtype=np.float32)
        for row, t in enumerate(texts):
            m = re.search(r"query (\d+)$", t)
            if m:
                q = int(m.group(1))
                target = q if (self.tuned or q >= self.misses) else (q + 1) % N
            else:
                target = int(re.search(r"doc (\d+)$", t).group(1))
            out[row, target] = 1.0
        return out

    def release(self):
        self.released = True


@pytest.fixture()
def stage(monkeypatch, tmp_path):
    data = {"doc_texts": [f"doc {i}" for i in range(N)], "query_texts": [f"query {i}" for i in range(N)],
            "rel_idx": list(range(N)), "tune_idx": [], "holdout_idx": list(range(N))}
    made = []

    def encoder_for(preset, device, adapter=None):
        enc = FakeEncoder(tuned=adapter is not None)
        made.append((preset, adapter, enc))
        return enc, {}

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    monkeypatch.setattr(ft, "load_split", lambda preset: (data, [], data["holdout_idx"]))
    monkeypatch.setattr(ft, "encoder_for", encoder_for)
    monkeypatch.setattr(ft, "ADAPTER", adapter)
    monkeypatch.setattr(ft, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(ft, "PER_QUERY", tmp_path / "out" / "holdout_per_query.json")
    monkeypatch.setattr(ft, "RESULTS_JSON", tmp_path / "results.json")
    args = type("A", (), {"preset": "f2llm-v2-0.6b", "device": "cpu", "batch_size": 4, "adapter": None,
                          "n_boot": 300, "ref_ranks": str(tmp_path / "out" / "none.json")})()
    return args, made, tmp_path


def test_eval_scores_base_and_tuned_on_the_same_queries_and_writes_everything(stage, capsys):
    args, made, tmp = stage
    assert ft.stage_eval(args) == 0
    assert [(p, a is not None) for p, a, _ in made] == [("f2llm-v2-0.6b", False), ("f2llm-v2-0.6b", True)]
    assert all(e.released for _, _, e in made), "each model is released before the next is loaded"
    res = json.loads((tmp / "results.json").read_text())
    assert res["n_holdout"] == N and res["beats_base_0.6b"] is True and res["beats_base_1.7b"] is False
    p = res["paired"]["vs_base_0.6b"]
    assert p["improved"] == 12 and p["worsened"] == 0 and p["adopt"] is True and p["ci_low"] > 0
    assert res["vs_1.7b_paired"] is False and "vs_1.7b" not in res["paired"]
    pq = json.loads((tmp / "out" / "holdout_per_query.json").read_text())
    assert pq["holdout_idx"] == list(range(N))
    assert len(pq["tuned"]["ndcg@10"]) == len(pq["base_0.6b"]["mrr@10"]) == N
    assert np.mean(pq["tuned"]["ndcg@10"]) == pytest.approx(res["finetuned"]["ndcg@10"])
    assert np.mean(pq["base_0.6b"]["mrr@10"]) == pytest.approx(res["base_0.6b"]["mrr@10"])
    out = capsys.readouterr().out
    assert "PAIRED HOLDOUT EVALUATION" in out and "ADOPT" in out and "not available" in out


def test_eval_pairs_against_the_1_7b_when_its_ranks_exist(stage, capsys):
    args, made, tmp = stage
    (tmp / "out").mkdir(exist_ok=True)
    ref = tmp / "out" / "holdout_ranks_f2llm-v2-1.7b.json"
    ref.write_text(json.dumps({"holdout_idx": list(range(N)), "ranks": [1] * N}))
    args.ref_ranks = str(ref)
    ft.stage_eval(args)
    res = json.loads((tmp / "results.json").read_text())
    assert res["vs_1.7b_paired"] is True
    v = res["paired"]["vs_1.7b"]
    assert v["improved"] == 0 and v["worsened"] == 0 and v["adopt"] is False      # tuned == 1.7B here
    assert res["beats_base_1.7b"] is False
    assert "tuned vs 1.7B" in capsys.readouterr().out


def test_base_only_saves_per_query_ranks_and_loads_no_adapter(stage):
    args, made, tmp = stage
    args.base_only = True
    args.preset = "f2llm-v2-1.7b"
    assert ft.stage_eval(args) == 0
    assert [(p, a) for p, a, _ in made] == [("f2llm-v2-1.7b", None)]
    saved = json.loads((tmp / "out" / "holdout_ranks_f2llm-v2-1.7b.json").read_text())
    assert saved["holdout_idx"] == list(range(N)) and len(saved["ranks"]) == N
    assert saved["ranks"][:12] == [2] * 12 and saved["ranks"][12:] == [1] * (N - 12)
    assert not (tmp / "results.json").exists()


def test_eval_still_refuses_without_an_adapter_before_encoding_anything(stage):
    args, made, tmp = stage
    args.adapter = str(tmp / "missing")
    with pytest.raises(SystemExit, match="No adapter"):
        ft.stage_eval(args)
    assert made == []


def test_only_the_eval_stage_reads_the_holdout():
    import inspect
    for stage_fn in (ft.stage_mine, ft.stage_train):
        src = inspect.getsource(stage_fn)
        assert "holdout_ranks" not in src and "paired_report" not in src
    assert "holdout" in inspect.getsource(ft.stage_eval)
