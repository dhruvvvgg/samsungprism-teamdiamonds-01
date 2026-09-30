"""Incremental reindex (P1): only changed chunks are re-encoded, the official indexes are untouchable,
and the next search sees the edit.

Everything is mocked: a spy wraps the hashing encoder and records every text it is asked to encode, which
is how "unchanged chunks are not re-encoded" is proven rather than asserted.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.reindex import (ReindexError, check_index_name, check_reindexable, format_report,
                         reindex_index, reindexable_info)
from src.runtime_index import HashingQueryEncoder, RuntimeIndex, write_index

ROOT = Path(__file__).resolve().parents[1]

MOD_A = '''def alpha(x):
    """Add one."""
    return x + 1


def beta(y):
    """Double it."""
    return y * 2
'''

MOD_B = '''def gamma(items):
    """Total of the items."""
    return sum(items)
'''


class Spy:
    """Encodes with the hashing encoder and remembers exactly which texts it was asked for."""

    def __init__(self, dim=64):
        self.enc = HashingQueryEncoder(dim)
        self.texts = []
        self.calls = 0

    def __call__(self, texts, batch_size=16):
        self.calls += 1
        self.texts.extend(texts)
        return self.enc.encode_docs(texts)


@pytest.fixture()
def folder(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    (src / "a.py").write_text(MOD_A, encoding="utf-8")
    (src / "b.py").write_text(MOD_B, encoding="utf-8")
    return src


@pytest.fixture()
def folder_index(tmp_path, folder):
    out = tmp_path / "idx"
    proc = subprocess.run([sys.executable, "src/build_index.py", "--source", str(folder),
                           "--mock-encoder", "--out", str(out)], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


def rows_by_text(index):
    return {t: index.E[i].copy() for i, t in enumerate(index.doc_texts)}


# --- the core: unchanged chunks are never re-encoded ---------------------------------------------

def test_nothing_changed_means_nothing_encoded_and_nothing_written(folder_index):
    before = (folder_index / "embeddings.npy").read_bytes()
    spy = Spy()
    rep = reindex_index(folder_index, spy)
    assert spy.calls == 0 and spy.texts == []
    assert (rep["added"], rep["modified"], rep["removed"]) == (0, 0, 0)
    assert rep["unchanged"] == 3
    assert rep["embeddings_recomputed"] == 0 and rep["embeddings_reused"] == 3
    assert rep["written"] is False
    assert (folder_index / "embeddings.npy").read_bytes() == before


def test_a_modified_function_is_the_only_thing_encoded(folder, folder_index):
    old = rows_by_text(RuntimeIndex.load(folder_index))
    (folder / "a.py").write_text(MOD_A.replace("return y * 2", "return y * 3"), encoding="utf-8")
    spy = Spy()
    rep = reindex_index(folder_index, spy)
    assert (rep["added"], rep["modified"], rep["unchanged"], rep["removed"]) == (0, 1, 2, 0)
    assert len(spy.texts) == 1 and "return y * 3" in spy.texts[0]
    assert (rep["embeddings_recomputed"], rep["embeddings_reused"]) == (1, 2)
    assert rep["written"] is True
    new = RuntimeIndex.load(folder_index)
    assert new.verify_files() == []
    # the two untouched chunks kept their stored vectors bit for bit
    for text, vec in old.items():
        if "return y * 2" not in text:
            assert np.array_equal(rows_by_text(new)[text], vec)


def test_added_and_removed_chunks_are_counted(folder, folder_index):
    (folder / "b.py").write_text(MOD_B + '\n\ndef delta(z):\n    return z - 1\n', encoding="utf-8")
    (folder / "a.py").write_text(MOD_A.split("\n\ndef beta")[0] + "\n", encoding="utf-8")   # beta gone
    spy = Spy()
    rep = reindex_index(folder_index, spy)
    assert (rep["added"], rep["removed"]) == (1, 1)
    assert len(spy.texts) == 1 and "def delta" in spy.texts[0]
    assert rep["n_docs_before"] == 3 and rep["n_docs_after"] == 3


def test_a_chunk_that_only_moved_is_unchanged_but_its_lines_are_refreshed(folder, folder_index):
    (folder / "a.py").write_text("\n\n\n" + MOD_A, encoding="utf-8")        # everything shifts down
    spy = Spy()
    rep = reindex_index(folder_index, spy)
    assert spy.texts == []
    assert rep["unchanged"] == 3 and rep["modified"] == 0 and rep["added"] == 0
    assert rep["written"] is True                                        # new file:start-end recorded
    starts = {c["name"]: c["start_line"] for c in RuntimeIndex.load(folder_index).chunks}
    assert starts["alpha"] == 4


def test_a_comment_edit_counts_as_modified_because_the_encoder_sees_it(folder, folder_index):
    (folder / "b.py").write_text(MOD_B.replace("return sum(items)", "return sum(items)  # hot path"),
                                 encoding="utf-8")
    spy = Spy()
    rep = reindex_index(folder_index, spy)
    assert rep["modified"] == 1 and len(spy.texts) == 1


def test_a_copied_chunk_reuses_the_existing_vector(folder, folder_index):
    (folder / "c.py").write_text(MOD_B, encoding="utf-8")                  # same text as b.py
    spy = Spy()
    rep = reindex_index(folder_index, spy)
    assert rep["added"] == 1 and spy.texts == []
    assert rep["embeddings_recomputed"] == 0


def test_dry_run_reports_and_writes_nothing(folder, folder_index):
    (folder / "a.py").write_text(MOD_A.replace("x + 1", "x + 2"), encoding="utf-8")
    before = (folder_index / "manifest.json").read_bytes()
    spy = Spy()
    rep = reindex_index(folder_index, spy, dry_run=True)
    assert rep["dry_run"] is True and rep["written"] is False
    assert rep["modified"] == 1 and rep["embeddings_recomputed"] == 1
    assert spy.calls == 0
    assert (folder_index / "manifest.json").read_bytes() == before


def test_stale_category_tags_are_dropped(folder, folder_index):
    (folder_index / "categories.json").write_text(json.dumps([{"ast_family": "x"}] * 3))
    (folder / "b.py").write_text(MOD_B + "\n\ndef extra():\n    return 0\n", encoding="utf-8")
    rep = reindex_index(folder_index, Spy())
    assert rep["categories_dropped"] is True
    assert not (folder_index / "categories.json").exists()
    assert RuntimeIndex.load(folder_index).categories is None


def test_an_emptied_source_folder_is_refused_rather_than_emptying_the_index(folder, folder_index):
    for f in folder.glob("*.py"):
        f.unlink()
    with pytest.raises(ReindexError, match="no Python chunks"):
        reindex_index(folder_index, Spy())
    assert RuntimeIndex.load(folder_index).E.shape[0] == 3


def test_a_missing_source_folder_is_a_404(tmp_path, folder, folder_index):
    import shutil
    shutil.rmtree(folder)
    with pytest.raises(ReindexError) as err:
        reindex_index(folder_index, Spy())
    assert err.value.status == 404


def test_the_report_is_readable():
    text = format_report({"index": "i", "kind": "folder", "source": "s", "dry_run": False,
                          "written": True, "added": 1, "modified": 2, "unchanged": 3, "removed": 4,
                          "n_docs_before": 8, "n_docs_after": 5, "embeddings_reused": 3,
                          "embeddings_recomputed": 3, "scan_seconds": 0.1, "encode_seconds": 0.2,
                          "elapsed_seconds": 0.3, "categories_dropped": False})
    assert "+1 added" in text and "~2 modified" in text and "-4 removed" in text
    assert "3 reused" in text and "3 recomputed" in text


# --- what must never be reindexed ------------------------------------------------------------------

def test_the_official_apps_index_is_refused(tmp_path):
    d = tmp_path / "apps"
    enc = HashingQueryEncoder(16)
    write_index(d, enc.encode_docs(["def f(): pass"]), ["d0"], ["def f(): pass"],
                {"model": "mock/hashing-encoder", "task": "AppsRetrieval", "source_root": str(tmp_path)})
    assert reindexable_info(d)["reindexable"] is False
    spy = Spy(16)
    with pytest.raises(ReindexError, match="cannot be reindexed"):
        reindex_index(d, spy)
    assert spy.calls == 0


def test_an_index_without_a_source_is_refused(tmp_path):
    d = tmp_path / "flat"
    enc = HashingQueryEncoder(16)
    write_index(d, enc.encode_docs(["x"]), ["d0"], ["x"], {"model": "mock/hashing-encoder"})
    assert reindexable_info(d)["reindexable"] is False


def test_the_named_official_indexes_are_refused_before_anything_is_loaded():
    from src.runtime_index import INDEX_NAMES
    for name in ("full", "lite"):
        assert reindexable_info(INDEX_NAMES[name])["reindexable"] is False


def test_check_reindexable_refuses_official_and_missing_indexes(tmp_path):
    for name in ("full", "lite"):
        with pytest.raises(ReindexError, match="cannot be reindexed"):
            check_reindexable(name)
    with pytest.raises(ReindexError) as err:
        check_reindexable(str(tmp_path / "never_built"))
    assert err.value.status in (400, 404)


def test_request_names_are_checked_against_an_allowlist(tmp_path):
    env = {}
    assert check_index_name("history", "full", env) == "history"
    assert check_index_name("full", "full", env) == "full"      # the allowlist is only step one
    assert check_index_name(str(tmp_path / "mine"), str(tmp_path / "mine"), env)   # the server default
    for bad in ("../../etc", "/etc", "..", "", "idx\x00", "runtime_index/../secret"):
        with pytest.raises(ReindexError):
            check_index_name(bad, "full", env)
    env = {"ALLOWED_INDEXES": "team_a, team_b"}
    assert check_index_name("team_b", "full", env) == "team_b"


def test_the_source_path_comes_from_the_manifest_not_the_caller(folder, folder_index, tmp_path):
    """reindex_index takes an index and an encoder -- there is no argument through which a path to scan
    could be supplied. Pointing the manifest elsewhere is an operator action, not a request one."""
    import inspect
    params = set(inspect.signature(reindex_index).parameters)
    assert params == {"index_dir", "encode_docs", "dry_run", "batch_size"}


# --- search sees the edit ----------------------------------------------------------------------------

def test_the_next_search_reflects_the_edit(folder, folder_index):
    from src.search_service import SearchService
    svc = SearchService(str(folder_index), mock=True)
    assert not any("zebra_counter" in h["preview"] for h in svc.search("zebra_counter", k=5)["hits"])
    (folder / "b.py").write_text(MOD_B + '\n\ndef zebra_counter(stripes):\n    return len(stripes)\n',
                                 encoding="utf-8")
    rep = svc.reindex()
    assert rep["added"] == 1 and rep["written"] is True
    hits = svc.search("zebra_counter stripes", k=3)["hits"]
    assert "zebra_counter" in hits[0]["preview"]
    assert svc.describe()["n_docs"] == 4


def test_two_overlapping_reindexes_do_not_both_run(folder_index):
    from src.search_service import SearchService
    svc = SearchService(str(folder_index), mock=True)
    assert svc._reindex_lock.acquire(blocking=False)
    try:
        with pytest.raises(ReindexError) as err:
            svc.reindex()
        assert err.value.status == 409
    finally:
        svc._reindex_lock.release()


# --- the API -------------------------------------------------------------------------------------------

def client_for(index_dir, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import src.api as api
    monkeypatch.setenv("INDEX_DIR", str(index_dir))
    monkeypatch.setenv("MOCK_ENCODER", "1")
    monkeypatch.delenv("ALLOWED_INDEXES", raising=False)
    monkeypatch.setattr(api, "_service", None)
    monkeypatch.setattr(api, "_error", None)
    monkeypatch.setattr(api, "_services", {})
    return TestClient(api.app)


def test_post_reindex_updates_the_default_index_and_search_follows(folder, folder_index, monkeypatch):
    client = client_for(folder_index, monkeypatch)
    (folder / "a.py").write_text(MOD_A + '\n\ndef quokka_walk(n):\n    return n * 7\n', encoding="utf-8")
    r = client.post("/reindex", json={})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["added"] == 1 and rep["unchanged"] == 3 and rep["written"] is True
    assert rep["embeddings_recomputed"] == 1 and rep["elapsed_seconds"] >= 0
    # the request that names the index explicitly must hit the same, refreshed service
    s = client.post("/search", json={"query": "quokka_walk n", "k": 3, "index": str(folder_index)})
    assert "quokka_walk" in s.json()["hits"][0]["preview"]
    assert client.get("/health").json()["index"]["n_docs"] == 4


def test_post_reindex_dry_run(folder, folder_index, monkeypatch):
    client = client_for(folder_index, monkeypatch)
    (folder / "a.py").write_text(MOD_A.replace("x + 1", "x + 9"), encoding="utf-8")
    rep = client.post("/reindex", json={"dry_run": True}).json()
    assert rep["dry_run"] is True and rep["modified"] == 1 and rep["written"] is False


@pytest.mark.parametrize("bad", ["full", "lite", "../../etc/passwd", "/tmp", "..", "not_listed"])
def test_post_reindex_refuses_official_and_unlisted_indexes(bad, folder_index, monkeypatch):
    client = client_for(folder_index, monkeypatch)
    r = client.post("/reindex", json={"index": bad})
    assert r.status_code == 400, r.text
    assert "allowlist" in r.json()["detail"] or "cannot be reindexed" in r.json()["detail"]


def test_indexes_endpoint_marks_what_can_be_reindexed(folder_index, monkeypatch):
    client = client_for(folder_index, monkeypatch)
    entries = client.get("/indexes").json()["indexes"]
    mine = [e for e in entries if e["name"] == str(folder_index)][0]
    assert mine["reindexable"] is True and mine["kind"] == "folder"
    for e in entries:
        if e["name"] in ("full", "lite"):
            assert e["reindexable"] is False


def test_the_page_offers_the_update_button_without_external_urls():
    html = (ROOT / "src" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="reindexBtn"' in html and "Update index" in html and "/reindex" in html
    assert "http://" not in html and "https://" not in html


# --- a git-history index -------------------------------------------------------------------------------

HIST_V1 = '''def add(a, b):
    """Add two numbers together."""
    return a + b


def scale(values, factor):
    """Multiply every value by a factor."""
    return [v * factor for v in values]
'''


def _git(repo, *args):
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_history_reindex_encodes_only_the_new_version(tmp_path):
    from src.versioning.git_history import ingest
    from src.versioning.version_index import build_versioned_index, embeddings_for
    repo = tmp_path / "r"
    (repo / "pkg").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    bodies = [HIST_V1, HIST_V1.replace("return a + b", "total = a + b\n    return total")]
    for body in bodies:
        (repo / "pkg" / "tools.py").write_text(body, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "c")
    rows, _, stats = ingest(repo, progress=False)
    enc = HashingQueryEncoder(32)
    emb, _ = embeddings_for(rows, enc.encode_docs, cache={})
    out = tmp_path / "hist"
    build_versioned_index(out, rows, emb, {
        "model": "mock/hashing-encoder", "task": "code-history", "repo": str(repo), "subdir": None,
        "commits": stats["commits"]})
    assert reindexable_info(out)["kind"] == "history"

    spy = Spy(32)
    assert reindex_index(out, spy)["written"] is False and spy.texts == []      # nothing new yet

    (repo / "pkg" / "tools.py").write_text(bodies[1].replace("values, factor", "values, factor=2"),
                                           encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c3")
    rep = reindex_index(out, spy)
    assert rep["kind"] == "history" and rep["written"] is True
    assert rep["modified"] == 1 and rep["added"] == 0
    assert len(spy.texts) == 1 and "factor=2" in spy.texts[0]
    fresh = RuntimeIndex.load(out)
    assert fresh.verify_files() == []
    assert any("factor=2" in t for t in fresh.doc_texts)
