"""One local end-to-end pass with the mock encoder: build an index, query it through the CLI, verify a
rankings file. No model, no dataset, no network.

These run the scripts as subprocesses, the way a person or CI does, so an argparse mistake or a broken
import is caught here rather than on Kaggle -- the failure mode that has cost this project the most time.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args, expect=0):
    proc = subprocess.run([sys.executable, *[str(a) for a in args]], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == expect, (
        f"exit {proc.returncode} for {args}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc.stdout


@pytest.fixture(scope="module")
def mock_index(tmp_path_factory):
    out = tmp_path_factory.mktemp("idx") / "runtime_index"
    run("src/build_index.py", "--source", "mock", "--mock-encoder", "--limit", 12, "--out", out)
    return out


def test_build_index_writes_a_complete_index(mock_index):
    manifest = json.loads((mock_index / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_docs"] == 12 and manifest["kind"] == "flat"
    assert "meaningless" in manifest["warning"]                 # a mock index says so about itself
    for name in ("embeddings.npy", "doc_ids.json", "corpus_texts.json"):
        assert (mock_index / name).exists()
    from src.runtime_index import RuntimeIndex
    assert RuntimeIndex.load(mock_index).verify_files() == []


def test_cli_returns_ranked_hits(mock_index):
    out = run("src/cli.py", "sum the scores whose value is < 100", "-k", 3,
              "--index", mock_index, "--mock-encoder", "--verify-index", "--json")
    payload = json.loads(out)
    hits = payload["result"]["hits"]
    assert len(hits) == 3
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert [h["score"] for h in hits] == sorted([h["score"] for h in hits], reverse=True)
    assert payload["index"]["verify_problems"] == []
    assert payload["index"]["threads"] >= 1
    assert payload["result"]["timings_ms"]["search_ms"] >= 0


def test_cli_human_output_reports_each_stage_timing(mock_index):
    out = run("src/cli.py", "binary search", "-k", 2, "--index", mock_index, "--mock-encoder")
    timing = next(line for line in out.split("\n") if line.startswith("timing"))
    assert "query encode" in timing and "search" in timing
    load = next(line for line in out.split("\n") if line.startswith("load"))
    assert "model" in load and "index" in load
    assert "#1" in out and "#2" in out


def test_cli_needs_a_query_or_a_history_id(mock_index):
    proc = subprocess.run([sys.executable, "src/cli.py", "--index", str(mock_index)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "--history" in proc.stderr


def test_cli_on_a_missing_index_explains_how_to_build_one(tmp_path):
    proc = subprocess.run([sys.executable, "src/cli.py", "q", "--index", str(tmp_path / "nope")],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "build_index.py" in (proc.stdout + proc.stderr)


@pytest.fixture(scope="module")
def version_index(tmp_path_factory):
    d = tmp_path_factory.mktemp("ver")
    out, fixture = d / "version_index", d / "fixture.json"
    run("src/build_version_index.py", "--mock-encoder", "--n-snippets", 12, "--n-versions", 3,
        "--out", out, "--fixture-out", fixture)
    return out, fixture


def test_versioned_cli_collapses_by_default_and_can_show_everything(version_index):
    index, _ = version_index
    collapsed = json.loads(run("src/cli.py", "sum the scores", "-k", 5, "--index", index,
                               "--mock-encoder", "--json"))["result"]
    everything = json.loads(run("src/cli.py", "sum the scores", "-k", 5, "--index", index,
                                "--mock-encoder", "--all-versions", "--json"))["result"]
    assert collapsed["collapsed_lineages"] is True
    assert everything["collapsed_lineages"] is False
    lineages = [h["snippet_id"] for h in collapsed["hits"]]
    assert len(lineages) == len(set(lineages)) == 5             # one slot per snippet
    # with collapsing off, the same lineage may take several slots; it must never take fewer distinct
    assert len(set(h["snippet_id"] for h in everything["hits"])) <= len(set(lineages))


def test_version_filter_returns_only_that_version(version_index):
    index, _ = version_index
    res = json.loads(run("src/cli.py", "sum the scores", "-k", 4, "--index", index,
                         "--mock-encoder", "--version", 2, "--json"))["result"]
    assert res["version_filter"] == 2
    assert {h["version"] for h in res["hits"]} == {2}
    assert all(h["doc_id"].endswith("@v2") for h in res["hits"])


def test_history_lists_versions_oldest_first(version_index):
    index, _ = version_index
    out = json.loads(run("src/cli.py", "--history", "snip0003", "--index", index,
                         "--mock-encoder", "--json"))["history"]
    assert out["n_versions"] == 3
    assert [v["version"] for v in out["versions"]] == [1, 2, 3]
    assert all(v["content_hash"] for v in out["versions"])


def test_p1_and_bonus_benchmarks_run_and_write_their_results(version_index, tmp_path):
    index, fixture = version_index
    p1_out, bonus_out = tmp_path / "p1.json", tmp_path / "bonus.json"
    run("src/bench_versions.py", "--mock-encoder", "--n-snippets", 20, "--n-versions", 3,
        "--out", p1_out)
    p1 = json.loads(p1_out.read_text(encoding="utf-8"))
    assert p1["warnings"] == []                                # reuse matched the fixture's own count
    assert p1["totals"]["embeddings_incremental"] < p1["totals"]["embeddings_full"]
    assert p1["totals"]["saved_pct"] > 0
    assert [r["version"] for r in p1["per_version"]] == [1, 2, 3]
    assert p1["per_version"][0]["incremental"]["recomputed"] == 20   # nothing to reuse at v1

    run("src/bench_evolution.py", "--mock-encoder", "--index", index, "--fixture", fixture,
        "--limit-queries", 10, "--out", bonus_out)
    bonus = json.loads(bonus_out.read_text(encoding="utf-8"))
    d = bonus["duplication_at_k"]
    assert d["collapsed_duplicate_pct_mean"] == 0.0             # collapsing leaves no duplicates
    assert d["all_versions_duplicate_pct_mean"] >= d["collapsed_duplicate_pct_mean"]
    assert bonus["n_queries"] == 10


def test_cpu_benchmark_quick_mode(mock_index, tmp_path):
    out = tmp_path / "bench.json"
    stdout = run("src/bench_cpu.py", "--index", mock_index, "--mock-encoder",
                 "--queries-source", "mock", "--n-queries", 5, "--out", out)
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["queries"]["n_run"] == 5
    assert rec["queries"]["n_measured"] == 4                   # the warm-up query is excluded
    assert rec["latency_ms"]["p50"] >= 0 and rec["latency_ms"]["p95"] >= rec["latency_ms"]["p50"]
    assert rec["threads"] >= 1 and rec["mock_encoder"] is True
    assert "MOCK ENCODER" in stdout


def test_verifier_recomputes_metrics_and_checks_hashes(tmp_path):
    """A tiny rankings file whose NDCG@10 and MRR@10 can be computed by hand:
    q1's relevant doc is at rank 2 -> 1/log2(3); q2's is at rank 1 -> 1.0."""
    import math

    from src.runtime_index import sha256_file
    from src.utils_io import write_json_atomic
    ndcg = (1 / math.log2(3) + 1.0) / 2
    mrr = (0.5 + 1.0) / 2
    qrels = write_json_atomic(tmp_path / "qrels.json", {"q1": {"d1": 1}, "q2": {"d2": 1}})
    rankings = write_json_atomic(tmp_path / "rankings.json",
                                 {"depth": 3, "rankings": {"q1": ["d5", "d1", "d3"], "q2": ["d2", "d9"]}})
    results = write_json_atomic(tmp_path / "results.json",
                                {"scores": {"test": [{"ndcg_at_10": ndcg, "mrr_at_10": mrr}]}})
    checksums = write_json_atomic(tmp_path / "checksums.json", {
        "results": {"path": str(results), "sha256": sha256_file(results)},
        "rankings": {"path": str(rankings), "sha256": sha256_file(rankings)}})

    out = run("src/verify_submission.py", "--rankings", rankings, "--results", results,
              "--checksums", checksums, "--qrels", qrels)
    assert "PASSED" in out
    assert f"{ndcg:.5f}" in out

    # a results file claiming a better score than the ranking supports must fail
    write_json_atomic(results, {"scores": {"test": [{"ndcg_at_10": 0.99, "mrr_at_10": 0.99}]}})
    proc = subprocess.run([sys.executable, "src/verify_submission.py", "--rankings", str(rankings),
                           "--results", str(results), "--checksums", str(checksums),
                           "--qrels", str(qrels)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "MISMATCH" in proc.stdout and "FAILED" in proc.stdout


def test_verifier_refuses_the_test_qrels_without_confirmation(tmp_path):
    from src.utils_io import write_json_atomic
    rankings = write_json_atomic(tmp_path / "rankings.json", {"rankings": {"q1": ["d1"]}})
    proc = subprocess.run([sys.executable, "src/verify_submission.py", "--rankings", str(rankings)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "--confirm-test" in (proc.stdout + proc.stderr)


def test_api_health_and_search_share_the_cli_logic(mock_index, monkeypatch):
    """The FastAPI app answers from the same SearchService the CLI uses."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import src.api as api
    monkeypatch.setenv("INDEX_DIR", str(mock_index))
    monkeypatch.setenv("MOCK_ENCODER", "1")
    monkeypatch.setattr(api, "_service", None)
    monkeypatch.setattr(api, "_error", None)
    client = TestClient(api.app)

    health = client.get("/health").json()
    assert health["status"] == "ok" and health["loaded"] is False     # lazy load

    res = client.post("/search", json={"query": "sum the scores", "k": 3}).json()
    assert len(res["hits"]) == 3 and res["total_ms"] >= 0
    assert [h["rank"] for h in res["hits"]] == [1, 2, 3]

    health = client.get("/health").json()
    assert health["loaded"] is True and health["index"]["n_docs"] == 12

    assert client.post("/search", json={"query": "", "k": 3}).status_code == 422
    assert client.get("/history/snip0000").status_code == 404        # flat index has no history
