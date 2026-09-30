"""The metrics report and the failure analysis.

Mocked throughout: no model, no GPU, no network. What is under test is the
two reports' tolerance of missing or partial input.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


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
