"""Experiments A / F / B, the metrics report and the failure analysis.

Mocked throughout: no model, no GPU, no network. What is under test is the gating maths, the
description cache's resume behaviour, the fine-tuning split discipline, and the two reports' tolerance
of missing or partial input.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.build_descriptions import clean, load_done, mock_description, snippet_key
from src.eval.dev_gated_rerank import confidence_gap

ROOT = Path(__file__).resolve().parents[1]


# --- A: the confidence gate ------------------------------------------------------------------------

def test_a_confident_query_has_a_large_gap():
    clear = np.array([0.9, 0.2, 0.19, 0.18, 0.1])
    tied = np.array([0.9, 0.89, 0.2, 0.19, 0.1])
    assert confidence_gap(clear, depth=5) > confidence_gap(tied, depth=5)


def test_the_gap_is_scale_invariant():
    """z-scoring is the point: two queries with the same shape must get the same gap even when one
    scores higher overall, or a single threshold could not mean the same thing for both."""
    row = np.array([0.9, 0.5, 0.4, 0.3, 0.2])
    assert confidence_gap(row, 5) == pytest.approx(confidence_gap(row * 10, 5))
    assert confidence_gap(row, 5) == pytest.approx(confidence_gap(row + 5, 5))


def test_a_flat_candidate_set_is_maximally_uncertain():
    assert confidence_gap(np.array([0.5, 0.5, 0.5, 0.5]), 4) == 0.0


def test_a_single_candidate_is_never_gated_in():
    assert confidence_gap(np.array([0.5]), 4) == float("inf")


def test_depth_limits_what_the_z_score_sees():
    row = np.array([0.9, 0.8] + [0.0] * 50)
    narrow, wide = confidence_gap(row, 2), confidence_gap(row, 52)
    assert narrow != wide


def test_gating_reproduces_the_two_extremes():
    """A threshold of 0 must rerank nothing and a huge threshold everything -- the sweep's endpoints
    have to coincide with never-rerank and always-rerank or the comparison is meaningless."""
    gaps = np.array([0.1, 1.0, 5.0])
    assert (gaps < 0.0).sum() == 0
    assert (gaps < 99.0).sum() == 3


# --- F: descriptions -----------------------------------------------------------------------------

def test_snippet_key_is_content_addressed():
    assert snippet_key("def f(): pass") == snippet_key("def f(): pass")
    assert snippet_key("def f(): pass") != snippet_key("def g(): pass")


def test_clean_strips_boilerplate_and_markdown():
    assert clean("This function sorts a list.") == "sorts a list."
    assert clean("`counts the primes`") == "counts the primes"
    assert clean("first line\nsecond line") == "first line"
    assert clean(None) == ""


def test_mock_description_is_deterministic():
    code = "def add(a, b):\n    return a + b\n"
    assert mock_description(code) == mock_description(code)
    assert "add" in mock_description(code)


def test_load_done_survives_a_truncated_final_line(tmp_path):
    """A killed run leaves a half-written line; it must cost that one item, not the whole file."""
    path = tmp_path / "d.jsonl"
    path.write_text('{"key": "a", "description": "one"}\n'
                    '{"key": "b", "description": "two"}\n'
                    '{"key": "c", "descrip', encoding="utf-8")
    done = load_done(path)
    assert done == {"a": "one", "b": "two"}


def test_load_done_on_a_missing_file_is_empty(tmp_path):
    assert load_done(tmp_path / "nope.jsonl") == {}


def test_description_build_is_resumable(tmp_path):
    out = tmp_path / "desc.jsonl"
    first = subprocess.run([sys.executable, "src/build_descriptions.py", "--source", "mock",
                            "--mock", "--limit", "6", "--out", str(out)], cwd=ROOT,
                           capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    assert len(load_done(out)) == 6

    # a rerun must do nothing at all
    second = subprocess.run([sys.executable, "src/build_descriptions.py", "--source", "mock",
                             "--mock", "--limit", "6", "--out", str(out)], cwd=ROOT,
                            capture_output=True, text=True)
    assert second.returncode == 0
    assert "nothing to do" in second.stdout
    assert len(load_done(out)) == 6

    # and a partial file only generates the remainder
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    out.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    third = subprocess.run([sys.executable, "src/build_descriptions.py", "--source", "mock",
                            "--mock", "--limit", "6", "--out", str(out)], cwd=ROOT,
                           capture_output=True, text=True)
    assert third.returncode == 0
    assert "2 already described, 4 to do" in third.stdout
    assert len(load_done(out)) == 6


def test_description_fusion_refuses_to_run_without_descriptions(tmp_path):
    proc = subprocess.run([sys.executable, "src/eval/dev_descriptions.py",
                           "--descriptions", str(tmp_path / "missing.jsonl")],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "build_descriptions.py" in (proc.stdout + proc.stderr)


# --- B: fine-tuning split discipline ----------------------------------------------------------------

def test_tune_and_holdout_never_overlap():
    """The guarantee the whole experiment rests on."""
    from src.eval.dev_data import make_split
    ids = [f"q{i}" for i in range(5000)]
    tune, holdout = make_split(ids)
    assert len(holdout) == 1000 and len(tune) == 4000
    assert not (set(tune) & set(holdout))
    again_tune, again_holdout = make_split(ids)
    assert again_holdout == holdout, "the split must be reproducible from its seed"


def test_holdout_references_are_recorded_for_both_bases():
    from src.train.finetune_lite import HOLDOUT_REFERENCE
    assert set(HOLDOUT_REFERENCE) == {"f2llm-v2-0.6b", "f2llm-v2-1.7b"}
    assert HOLDOUT_REFERENCE["f2llm-v2-0.6b"]["ndcg@10"] == 0.9054
    assert HOLDOUT_REFERENCE["f2llm-v2-1.7b"]["ndcg@10"] == 0.9306
    # a P0 candidate has to clear the 1.7B, which is the higher bar
    assert (HOLDOUT_REFERENCE["f2llm-v2-1.7b"]["ndcg@10"]
            > HOLDOUT_REFERENCE["f2llm-v2-0.6b"]["ndcg@10"])


def test_mining_uses_tune_queries_only(monkeypatch, tmp_path):
    """Hard negatives mined over holdout queries would leak the evaluation set into training."""
    import src.train.finetune_lite as ft

    data = {"doc_texts": [f"doc {i}" for i in range(6)],
            "query_texts": [f"query {i}" for i in range(5)],
            "rel_idx": [0, 1, 2, 3, 4],
            "tune_idx": [0, 1, 2], "holdout_idx": [3, 4]}
    seen = {}

    class FakeEnc:
        def embed(self, texts, batch_size=8):
            seen.setdefault("texts", []).extend(texts)
            return np.eye(len(texts), 6, dtype=np.float32)

        def release(self):
            pass

    monkeypatch.setattr(ft, "load_split", lambda preset: (data, data["tune_idx"], data["holdout_idx"]))
    monkeypatch.setattr(ft, "encoder_for", lambda *a, **kw: (FakeEnc(), {}))
    monkeypatch.setattr(ft, "NEGATIVES", tmp_path / "neg.json")
    args = type("A", (), {"preset": "f2llm-v2-0.6b", "device": "cpu", "batch_size": 2,
                          "n_negatives": 2, "force": True})()
    assert ft.stage_mine(args) == 0
    written = json.loads((tmp_path / "neg.json").read_text(encoding="utf-8"))
    assert written["split"] == "tune"
    assert set(written["negatives"]) == {"0", "1", "2"}, "holdout queries must not be mined"
    assert not any("query 3" in t or "query 4" in t for t in seen["texts"])


def test_mined_negatives_exclude_the_positive(monkeypatch, tmp_path):
    import src.train.finetune_lite as ft

    data = {"doc_texts": [f"doc {i}" for i in range(4)], "query_texts": ["q0"],
            "rel_idx": [2], "tune_idx": [0], "holdout_idx": []}

    class FakeEnc:
        def embed(self, texts, batch_size=8):
            # make the positive (doc 2) the nearest neighbour, so excluding it is actually tested
            out = []
            for t in texts:
                out.append([1.0, 0.0] if t.startswith("doc 2") or t.startswith("q") else [0.9, 0.1])
            return np.array(out, dtype=np.float32)

        def release(self):
            pass

    monkeypatch.setattr(ft, "load_split", lambda preset: (data, [0], []))
    monkeypatch.setattr(ft, "encoder_for", lambda *a, **kw: (FakeEnc(), {}))
    monkeypatch.setattr(ft, "NEGATIVES", tmp_path / "neg.json")
    args = type("A", (), {"preset": "f2llm-v2-0.6b", "device": "cpu", "batch_size": 2,
                          "n_negatives": 2, "force": True})()
    ft.stage_mine(args)
    negs = json.loads((tmp_path / "neg.json").read_text(encoding="utf-8"))["negatives"]["0"]
    assert 2 not in negs, "the positive document must never be mined as its own negative"


def test_eval_stage_refuses_without_an_adapter(monkeypatch, tmp_path):
    import src.train.finetune_lite as ft
    monkeypatch.setattr(ft, "ADAPTER", tmp_path / "missing")
    args = type("A", (), {"preset": "f2llm-v2-0.6b", "device": "cpu", "batch_size": 2,
                          "adapter": None})()
    with pytest.raises(SystemExit, match="No adapter"):
        ft.stage_eval(args)


def test_train_stage_requires_mined_negatives(monkeypatch, tmp_path):
    import src.train.finetune_lite as ft
    monkeypatch.setattr(ft, "NEGATIVES", tmp_path / "missing.json")
    args = type("A", (), {"preset": "f2llm-v2-0.6b", "device": "cpu"})()
    with pytest.raises(SystemExit, match="Mine hard negatives first"):
        ft.stage_train(args)


# --- the metrics report -----------------------------------------------------------------------------

def test_metrics_report_tolerates_every_input_being_missing(tmp_path, monkeypatch):
    import src.build_metrics_report as R
    monkeypatch.setattr(R, "SOURCES", {k: (tmp_path / f"{k}.json", cmd)
                                       for k, (_, cmd) in R.SOURCES.items()})
    out = tmp_path / "metrics.md"
    monkeypatch.setattr(sys, "argv", ["build_metrics_report.py", "--out", str(out)])
    assert R.main() == 0
    text = out.read_text(encoding="utf-8")
    assert "not yet measured" in text
    assert "run_official.py" in text, "a missing row must name the command that fills it"
    for heading in ("Official test-split result", "Latency", "Indexing cost", "Agent vs plain dense"):
        assert heading in text


def test_metrics_report_uses_the_files_that_exist(tmp_path, monkeypatch):
    import src.build_metrics_report as R
    (tmp_path / "official.json").write_text(json.dumps({
        "scores": {"test": [{"ndcg_at_10": 0.9376, "mrr_at_10": 0.9238, "recall_at_10": 0.97,
                             "precision_at_10": 0.097, "map_at_10": 0.91}]},
        "_run_info": {"model": "codefuse-ai/F2LLM-v2-1.7B", "revision": "3766d46e7a68",
                      "total_eval_seconds": 996.0, "device": "cuda",
                      "config_path": "configs/official_f2llm17b_noreranker.json"}}), encoding="utf-8")
    (tmp_path / "agent.json").write_text(json.dumps({
        "label_summary": {"n": 30, "by_kind": {"structural": 14}, "by_label_method": {"grep": 18}},
        "overall": {"n": 30, "dense": {"precision_at_k": 0.2, "recall": 0.3, "median_ms": 40.0},
                    "agent": {"precision_at_k": 0.6, "recall": 0.8, "median_ms": 120.0}}}),
        encoding="utf-8")
    monkeypatch.setattr(R, "SOURCES", {k: (tmp_path / f"{k}.json", cmd)
                                       for k, (_, cmd) in R.SOURCES.items()})
    out = tmp_path / "metrics.md"
    monkeypatch.setattr(sys, "argv", ["build_metrics_report.py", "--out", str(out)])
    assert R.main() == 0
    text = out.read_text(encoding="utf-8")
    assert "0.9376" in text and "0.9238" in text
    assert "F2LLM-v2-1.7B" in text
    assert "measured: 2/" in text
    assert "0.600" in text                       # the agent's precision made it into the table


def test_metrics_report_survives_a_corrupt_file(tmp_path, monkeypatch):
    import src.build_metrics_report as R
    (tmp_path / "official.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(R, "SOURCES", {k: (tmp_path / f"{k}.json", cmd)
                                       for k, (_, cmd) in R.SOURCES.items()})
    out = tmp_path / "metrics.md"
    monkeypatch.setattr(sys, "argv", ["build_metrics_report.py", "--out", str(out)])
    assert R.main() == 0
    assert "not yet measured" in out.read_text(encoding="utf-8")


# --- failure analysis ---------------------------------------------------------------------------------

DISSIMILAR = "def alpha(rows):\n    return sorted(rows, key=len)\n"
EXPECTED = "def solve(values):\n    total = sum(values)\n    return total\n"


def test_failure_grouping_recognises_each_cause():
    from src.analyze_failures import classify
    group, why = classify("x " * 2000, EXPECTED, [DISSIMILAR], median_chars=100)
    assert group == "very_long_query" and "cap" in why

    group, why = classify("a query with plenty of distinct descriptive terms here " * 3,
                          EXPECTED, [EXPECTED], median_chars=100)
    assert group == "near_duplicate_corpus" and "%" in why

    group, why = classify("sort it", EXPECTED, [DISSIMILAR], median_chars=100)
    assert group == "generic_wording"


def test_near_duplicate_is_diagnosed_before_generic_wording():
    """Both can be true of one query; the more specific and more actionable diagnosis wins, and this
    pins that ordering so it cannot drift silently."""
    from src.analyze_failures import classify
    group, _ = classify("sort it", EXPECTED, [EXPECTED], median_chars=100)
    assert group == "near_duplicate_corpus"


def test_other_group_is_for_genuinely_unexplained_cases():
    from src.analyze_failures import classify
    wordy = " ".join(f"distinctterm{i}" for i in range(40))
    group, _ = classify(wordy, "def alpha(): return 1", ["def zulu(): return 99"], median_chars=1000)
    assert group == "other"


def test_jaccard_is_symmetric_and_bounded():
    from src.analyze_failures import jaccard, tokens
    a, b = tokens("alpha beta gamma"), tokens("beta gamma delta")
    assert jaccard(a, b) == jaccard(b, a)
    assert 0.0 < jaccard(a, b) < 1.0
    assert jaccard(a, a) == 1.0
    assert jaccard(a, set()) == 0.0


def test_failure_analysis_runs_from_the_rankings_file_alone(tmp_path):
    """It must work on CPU with no model and no corpus download."""
    rankings = {"q1": ["d9", "d8", "d7", "d6", "d5", "d4", "d3", "d2", "d1", "d0", "gold1"],
                "q2": ["gold2", "d1", "d2"],
                "q3": ["d5", "d6", "d7"]}
    rpath = tmp_path / "rankings.json"
    rpath.write_text(json.dumps({"depth": 11, "rankings": rankings}), encoding="utf-8")
    qrels = tmp_path / "qrels.json"
    qrels.write_text(json.dumps({"q1": {"gold1": 1}, "q2": {"gold2": 1}, "q3": {"gold3": 1}}),
                     encoding="utf-8")
    out, md = tmp_path / "fa.json", tmp_path / "fa.md"
    proc = subprocess.run([sys.executable, "src/analyze_failures.py", "--rankings", str(rpath),
                           "--qrels", str(qrels), "--out", str(out), "--markdown-out", str(md)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_queries"] == 3
    assert data["n_in_top_k"] == 1                        # only q2 found its answer in the top 10
    assert data["n_failures"] == 2
    ids = {f["query_id"] for f in data["failures"]}
    assert ids == {"q1", "q3"}
    by_id = {f["query_id"]: f for f in data["failures"]}
    assert by_id["q1"]["rank_of_relevant"] == 11          # found, but outside the top 10
    assert by_id["q3"]["rank_of_relevant"] is None        # never retrieved
    assert md.exists() and "Failure analysis" in md.read_text(encoding="utf-8")


def test_failure_analysis_refuses_the_test_qrels_without_confirmation(tmp_path):
    rpath = tmp_path / "rankings.json"
    rpath.write_text(json.dumps({"rankings": {"q1": ["d1"]}}), encoding="utf-8")
    proc = subprocess.run([sys.executable, "src/analyze_failures.py", "--rankings", str(rpath)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "--confirm-test" in (proc.stdout + proc.stderr)


def test_failure_analysis_reports_a_missing_rankings_file(tmp_path):
    proc = subprocess.run([sys.executable, "src/analyze_failures.py", "--rankings",
                           str(tmp_path / "nope.json"), "--qrels", str(tmp_path / "q.json")],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "run_official.py" in (proc.stdout + proc.stderr)


def test_metrics_report_script_runs_end_to_end(tmp_path):
    out = tmp_path / "metrics.md"
    proc = subprocess.run([sys.executable, "src/build_metrics_report.py", "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.exists() and "# Metrics report" in out.read_text(encoding="utf-8")
