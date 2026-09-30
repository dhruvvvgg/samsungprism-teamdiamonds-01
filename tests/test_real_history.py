"""Ingesting a real git history: lineage tracking, incremental reuse, and version-targeted queries.

Every test builds a small git repository in a temp directory, so the suite needs no network and no
demo repo checkout. The encoder is the hashing mock throughout -- what is under test here is the
history walk and the bookkeeping, not retrieval quality.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.versioning.git_history import (changed_files, files_at, ingest, lineage_id_for, list_commits,
                                        read_file_at, snapshot_chunks)
from src.versioning.history_queries import (build_queries, first_docstring_line, new_identifiers,
                                            query_stats, readable_phrase)

ROOT = Path(__file__).resolve().parents[1]

V1 = '''"""Tools."""


def add(a, b):
    """Add two numbers together."""
    return a + b


def scale(values, factor):
    """Multiply every value by a factor."""
    return [v * factor for v in values]
'''

# v2: scale() gains a named local (a real change); add() untouched
V2 = V1.replace("    return [v * factor for v in values]",
                "    multiplier = factor\n    return [v * multiplier for v in values]")
# v3: add() gains a named local; scale() untouched
V3 = V2.replace("    return a + b", "    total = a + b\n    return total")
# v4: a brand new function; nothing else changes
V4 = V3 + '''

def clamp(value, lowest, highest):
    """Constrain a value to a range."""
    return max(lowest, min(highest, value))
'''
# v5: comments and blank lines only -- must NOT count as a change
V5 = V4.replace('    """Constrain a value to a range."""',
                '    """Constrain a value to a range."""\n    # keep it inside the bounds\n')


def git(repo, *args):
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"
    return proc.stdout


@pytest.fixture(scope="module")
def fixture_repo(tmp_path_factory):
    """A five-commit Python repository with known, deliberate edits."""
    repo = tmp_path_factory.mktemp("hist") / "r"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "Test")
    pkg = repo / "pkg"
    pkg.mkdir()
    (repo / "README.md").write_text("not python\n", encoding="utf-8")
    for body in (V1, V2, V3, V4, V5):
        (pkg / "tools.py").write_text(body, encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", f"commit {len(body)}")
    return repo


# --- walking the history -------------------------------------------------------------------------

def test_commits_are_returned_oldest_first(fixture_repo):
    commits = list_commits(fixture_repo)
    assert len(commits) == 5
    shas = [c["sha"] for c in commits]
    assert len(set(shas)) == 5
    # oldest first: the first commit has no parent
    first = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=str(fixture_repo),
                           capture_output=True, text=True).stdout.strip()
    assert shas[0] == first


def test_max_commits_keeps_the_most_recent_still_oldest_first(fixture_repo):
    everything = list_commits(fixture_repo)
    last_two = list_commits(fixture_repo, max_commits=2)
    assert [c["sha"] for c in last_two] == [c["sha"] for c in everything[-2:]]


def test_files_at_returns_only_python(fixture_repo):
    sha = list_commits(fixture_repo)[-1]["sha"]
    files = files_at(fixture_repo, sha)
    assert files == ["pkg/tools.py"]              # README.md excluded


def test_changed_files_reports_the_diff(fixture_repo):
    commits = list_commits(fixture_repo)
    changed = changed_files(fixture_repo, commits[1]["sha"], commits[0]["sha"])
    assert changed == {"pkg/tools.py"}
    assert changed_files(fixture_repo, commits[0]["sha"], None) is None


def test_read_file_at_returns_the_version_at_that_commit(fixture_repo):
    commits = list_commits(fixture_repo)
    v1 = read_file_at(fixture_repo, commits[0]["sha"], "pkg/tools.py")
    v2 = read_file_at(fixture_repo, commits[1]["sha"], "pkg/tools.py")
    assert "multiplier" not in v1 and "multiplier" in v2
    assert read_file_at(fixture_repo, commits[0]["sha"], "nope.py") is None


def test_snapshot_keys_are_file_plus_qualname(fixture_repo):
    sha = list_commits(fixture_repo)[0]["sha"]
    snap, failures = snapshot_chunks(fixture_repo, sha)
    assert failures == []
    assert set(snap) == {lineage_id_for("pkg/tools.py", "add"),
                         lineage_id_for("pkg/tools.py", "scale")}
    assert snap["pkg/tools.py::add"]["qualname"] == "add"


# --- change classification is the ground truth the reuse numbers are checked against ---------------

def test_ingest_classifies_every_change_correctly(fixture_repo):
    rows, per_commit, stats = ingest(fixture_repo, progress=False)
    assert stats["commits"] == 5
    assert stats["lineages"] == 3                    # add, scale, clamp
    by_version = {c["version"]: c for c in per_commit}
    assert (by_version[1]["added"], by_version[1]["modified"]) == (2, 0)
    # v2 changed scale only
    assert (by_version[2]["added"], by_version[2]["modified"], by_version[2]["unchanged"]) == (0, 1, 1)
    # v3 changed add only
    assert (by_version[3]["added"], by_version[3]["modified"], by_version[3]["unchanged"]) == (0, 1, 1)
    # v4 added clamp, nothing else moved
    assert (by_version[4]["added"], by_version[4]["modified"], by_version[4]["unchanged"]) == (1, 0, 2)


def test_a_comment_only_edit_is_not_a_change(fixture_repo):
    """The whole point of hashing normalised code: a comment must not cost an embedding."""
    _, per_commit, _ = ingest(fixture_repo, progress=False)
    v5 = [c for c in per_commit if c["version"] == 5][0]
    assert v5["modified"] == 0, "a comment-only edit was counted as a modification"
    assert v5["unchanged"] == 3


def test_rows_carry_location_and_commit(fixture_repo):
    rows, _, _ = ingest(fixture_repo, progress=False)
    r = rows[0]
    for key in ("doc_id", "snippet_id", "lineage_id", "version", "commit", "commit_short",
                "content_hash", "file", "start_line", "end_line", "qualname", "change"):
        assert key in r, key
    assert r["doc_id"].endswith("@v1")
    assert r["snippet_id"] == r["lineage_id"]        # the versioned index keys on snippet_id
    assert r["file"] == "pkg/tools.py"


def test_a_lineage_keeps_its_identity_as_line_numbers_move(fixture_repo):
    """clamp is appended at v4, which pushes nothing; but `add` gains a line at v3 and `scale` moves
    down. Identity is file::qualname, so the lineage survives."""
    rows, _, _ = ingest(fixture_repo, progress=False)
    scale_rows = sorted([r for r in rows if r["qualname"] == "scale"], key=lambda r: r["version"])
    assert [r["version"] for r in scale_rows] == [1, 2, 3, 4, 5]
    assert len({r["start_line"] for r in scale_rows}) > 1, "the fixture should move this function"
    assert len({r["lineage_id"] for r in scale_rows}) == 1


def test_unchanged_versions_share_a_content_hash(fixture_repo):
    rows, _, _ = ingest(fixture_repo, progress=False)
    add_rows = sorted([r for r in rows if r["qualname"] == "add"], key=lambda r: r["version"])
    assert add_rows[0]["content_hash"] == add_rows[1]["content_hash"]     # unchanged v1 -> v2
    assert add_rows[1]["content_hash"] != add_rows[2]["content_hash"]     # changed at v3


def test_max_commits_limits_the_walk(fixture_repo):
    rows, per_commit, stats = ingest(fixture_repo, max_commits=2, progress=False)
    assert stats["commits"] == 2
    assert {r["version"] for r in rows} == {1, 2}


# --- auto-generated version-targeted queries ------------------------------------------------------

def test_docstring_extraction_is_parsed_not_guessed():
    assert first_docstring_line('def f():\n    """Do the thing properly."""\n    return 1\n') \
        == "Do the thing properly."
    assert first_docstring_line('def f():\n    return "Not a docstring at all, just a value"\n') is None
    assert first_docstring_line('def f():\n    """TODO: write this later."""\n    return 1\n') is None
    assert first_docstring_line('def f():\n    """short"""\n    return 1\n') is None
    assert first_docstring_line("def f(:\n") is None                      # unparseable


def test_readable_phrase_splits_identifiers():
    row = {"qualname": "Command.invoke_it", "file": "src/click/core.py"}
    assert readable_phrase(row) == "command invoke it in core.py"


def test_new_identifiers_is_the_diff_of_names():
    assert new_identifiers("a = one", "a = one + two") == {"two"}
    assert new_identifiers("x = same", "x = same") == set()


def test_queries_come_from_docstrings_and_new_tokens(fixture_repo):
    rows, _, _ = ingest(fixture_repo, progress=False)
    queries = build_queries(rows)
    assert queries, "no version-targeted queries generated"
    stats = query_stats(queries)
    assert stats["by_label_source"] == {"docstring": len(queries)}
    for q in queries:
        assert q["kind"] == "version_specific"
        assert q["token"] and q["token"] in q["text"]
        assert q["target_version"] in q["acceptable_versions"]


def test_acceptable_versions_is_read_off_the_real_texts(fixture_repo):
    rows, _, _ = ingest(fixture_repo, progress=False)
    by_lineage = {}
    for r in rows:
        by_lineage.setdefault(r["lineage_id"], {})[r["version"]] = r["text"]
    for q in build_queries(rows):
        texts = by_lineage[q["lineage_id"]]
        expected = sorted(v for v, t in texts.items() if q["token"] in t)
        assert q["acceptable_versions"] == expected
        earlier = [v for v in texts if v < q["target_version"]]
        assert all(q["token"] not in texts[v] for v in earlier), "the token predates its target version"


def test_a_token_never_appears_in_the_version_before_its_target(fixture_repo):
    """This is what makes a generated query discriminating, and it holds by construction: the token
    comes from new_identifiers(), which excludes everything already in the previous version."""
    rows, _, _ = ingest(fixture_repo, progress=False)
    texts = {}
    for r in rows:
        texts.setdefault(r["lineage_id"], {})[r["version"]] = r["text"]
    for q in build_queries(rows):
        if q["kind"] != "version_specific":
            continue
        previous = q["target_version"] - 1
        assert previous not in q["acceptable_versions"]
        assert q["token"] not in texts[q["lineage_id"]].get(previous, "")


def test_a_change_with_no_new_identifier_is_labelled_lineage_only():
    """It still tests version targeting (the version is supplied explicitly), but it cannot
    discriminate on text, so it is tagged rather than silently counted as version-specific."""
    body_v1 = 'def g(x):\n    """Do a thing here now properly."""\n    return x + 1\n'
    body_v2 = 'def g(x):\n    """Do a thing here now properly."""\n    return x + 2\n'
    rows = [{"lineage_id": "f.py::g", "version": 1, "change": "added", "doc_id": "f.py::g@v1",
             "commit_short": "abc", "file": "f.py", "qualname": "g", "text": body_v1},
            {"lineage_id": "f.py::g", "version": 2, "change": "modified", "doc_id": "f.py::g@v2",
             "commit_short": "def", "file": "f.py", "qualname": "g", "text": body_v2}]
    queries = build_queries(rows)
    assert len(queries) == 1
    assert queries[0]["kind"] == "lineage_only"
    assert queries[0]["token"] is None
    assert queries[0]["acceptable_versions"] == [2]
    assert query_stats(queries)["by_kind"] == {"lineage_only": 1}


def test_queries_are_capped_per_lineage(fixture_repo):
    rows, _, _ = ingest(fixture_repo, progress=False)
    queries = build_queries(rows, max_per_lineage=1)
    seen = [q["lineage_id"] for q in queries]
    assert len(seen) == len(set(seen))


# --- the rebuild benchmark ------------------------------------------------------------------------

def mock_encode(dim=32):
    from src.runtime_index import HashingQueryEncoder
    enc = HashingQueryEncoder(dim=dim)
    calls = []

    def encode(texts, batch_size=32):
        calls.append(list(texts))
        return enc.encode_docs(texts, batch_size=batch_size)
    return encode, calls


def test_incremental_rebuild_reuses_at_least_the_unchanged_lineages(fixture_repo):
    from src.bench_real_history import rebuild_benchmark
    rows, per_commit, _ = ingest(fixture_repo, progress=False)
    encode, _ = mock_encode()
    per_version, warnings = rebuild_benchmark(rows, per_commit, encode, batch_size=8)
    assert warnings == [], warnings
    assert len(per_version) == 5
    assert per_version[0]["incremental"]["recomputed"] == 2          # nothing to reuse at v1
    for row in per_version[1:]:
        assert row["incremental"]["reused"] >= row["expected_unchanged"]
        assert row["incremental"]["recomputed"] <= row["full"]["recomputed"]
    total_full = sum(r["full"]["recomputed"] for r in per_version)
    total_inc = sum(r["incremental"]["recomputed"] for r in per_version)
    assert total_inc < total_full


def test_snapshot_at_returns_one_row_per_lineage(fixture_repo):
    from src.bench_real_history import snapshot_at
    rows, _, _ = ingest(fixture_repo, progress=False)
    snap = snapshot_at(rows, 4)
    assert len(snap) == 3 and {r["qualname"] for r in snap} == {"add", "scale", "clamp"}


# --- the built index, and the CLI / API on top of it ----------------------------------------------

@pytest.fixture(scope="module")
def history_index(fixture_repo, tmp_path_factory):
    out = tmp_path_factory.mktemp("hidx") / "idx"
    queries = tmp_path_factory.mktemp("hq") / "q.json"
    stats = tmp_path_factory.mktemp("hs") / "s.json"
    proc = subprocess.run([sys.executable, "src/build_history_index.py", "--repo", str(fixture_repo),
                           "--mock-encoder", "--out", str(out), "--queries-out", str(queries),
                           "--stats-out", str(stats)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out, queries


def test_history_index_is_versioned_and_located(history_index):
    from src.runtime_index import RuntimeIndex
    idx, _ = history_index
    ix = RuntimeIndex.load(idx)
    assert ix.manifest["kind"] == "versioned"
    assert ix.versions is not None and ix.chunks is not None
    assert len(ix.versions) == len(ix.chunks) == len(ix.doc_ids)
    assert ix.verify_files() == []
    assert ix.versions[0]["commit_short"]
    assert ix.chunks[0]["file"] == "pkg/tools.py"


def test_results_carry_both_location_and_lineage(history_index):
    from src.search_service import SearchService
    idx, _ = history_index
    svc = SearchService(str(idx), mock=True)
    assert svc.versioned and svc.chunked
    hit = svc.search("multiply every value by a factor", k=1)["hits"][0]
    for key in ("location", "file", "start_line", "end_line", "snippet_id", "version",
                "commit_short", "change"):
        assert key in hit, key
    assert hit["location"].startswith("pkg/tools.py:")


def test_version_targeting_and_history_work_on_real_history(history_index):
    from src.search_service import SearchService
    idx, _ = history_index
    svc = SearchService(str(idx), mock=True)
    res = svc.search("add two numbers", k=3, version=2)
    assert {h["version"] for h in res["hits"]} == {2}
    hist = svc.history("pkg/tools.py::add")
    assert [v["version"] for v in hist["versions"]] == [1, 2, 3, 4, 5]


def test_generated_queries_file_has_known_answers(history_index):
    _, queries_path = history_index
    payload = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    assert payload["stats"]["n"] >= 1
    for q in payload["queries"]:
        assert q["target_doc_id"].endswith(f"@v{q['target_version']}")
        assert q["label_source"] in ("docstring", "qualname")


def test_cli_shows_commit_and_location(history_index):
    idx, _ = history_index
    proc = subprocess.run([sys.executable, "src/cli.py", "multiply every value", "-k", "2",
                           "--index", str(idx), "--mock-encoder"], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pkg/tools.py:" in proc.stdout


def test_cli_history_lists_every_version(history_index):
    idx, _ = history_index
    proc = subprocess.run([sys.executable, "src/cli.py", "--history", "pkg/tools.py::add",
                           "--index", str(idx), "--mock-encoder"], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count("hash ") == 5


def test_api_serves_a_history_index(history_index, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import src.api as api
    idx, _ = history_index
    monkeypatch.setenv("INDEX_DIR", str(idx))
    monkeypatch.setenv("MOCK_ENCODER", "1")
    monkeypatch.setattr(api, "_service", None)
    monkeypatch.setattr(api, "_error", None)
    monkeypatch.setattr(api, "_services", {})
    with TestClient(api.app) as client:
        assert client.get("/health").json()["index"]["versioned"] is True
        data = client.post("/search", json={"query": "constrain a value", "k": 3}).json()
        assert data["hits"] and "commit_short" in data["hits"][0]
        hist = client.get("/history/pkg%2Ftools.py%3A%3Aadd").json()
        assert hist["n_versions"] == 5


def test_benchmark_runs_end_to_end(fixture_repo, history_index, tmp_path):
    idx, _ = history_index
    out = tmp_path / "bench.json"
    proc = subprocess.run([sys.executable, "src/bench_real_history.py", "--repo", str(fixture_repo),
                           "--mock-encoder", "--index", str(idx), "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["warnings"] == []
    assert data["totals"]["embeddings_incremental"] < data["totals"]["embeddings_full"]
    r = data["retrieval"]
    assert r["p1_version_targeted"]["top1_correct_lineage"] >= 0.0
    bonus = r["bonus_all_versions"]
    assert bonus["duplicate_pct_at_10_collapsed"] == 0.0
    assert bonus["duplicate_pct_at_10_all_versions"] >= bonus["duplicate_pct_at_10_collapsed"]


def test_the_synthetic_fixture_benchmark_still_exists_and_runs(tmp_path):
    """Real history replaces nothing: the synthetic benchmark stays, because its mutations are known
    by construction and so its reuse rate has an exact right answer."""
    out = tmp_path / "p1.json"
    proc = subprocess.run([sys.executable, "src/bench_versions.py", "--mock-encoder",
                           "--n-snippets", "15", "--n-versions", "3", "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["warnings"] == []
