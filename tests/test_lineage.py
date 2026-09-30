"""Lineage grouping of search results, and the history payload the View history button renders."""
import pytest

from src.runtime_index import HashingQueryEncoder, write_index
from src.versioning.fixture import build_fixture
from src.versioning.lineage import group_by_lineage
from src.versioning.version_index import build_versioned_index, embeddings_for, rows_for


def hit(rank, sid, version, score=0.5):
    return {"rank": rank, "doc_id": f"{sid}@v{version}", "snippet_id": sid, "version": version,
            "score": score, "preview": "x"}


def test_one_group_per_lineage_headed_by_its_best_hit():
    hits = [hit(1, "a", 3), hit(2, "b", 1), hit(3, "a", 1), hit(4, "a", 2), hit(5, "c", 2)]
    groups = group_by_lineage(hits)
    assert [g["snippet_id"] for g in groups] == ["a", "b", "c"]
    assert [g["rank"] for g in groups] == [1, 2, 3]
    a = groups[0]
    assert a["best"]["version"] == 3                                   # the best-ranked one heads it
    assert [o["version"] for o in a["others"]] == [1, 2]               # the rest, oldest first
    assert a["n_in_results"] == 3 and a["versions_in_results"] == [1, 2, 3]
    assert groups[1]["others"] == [] and groups[1]["n_in_results"] == 1


def test_grouping_is_a_view_and_never_reorders_or_drops_hits():
    hits = [hit(1, "a", 2), hit(2, "b", 1), hit(3, "a", 1)]
    groups = group_by_lineage(hits)
    flat = [g["best"] for g in groups] + [o for g in groups for o in g["others"]]
    assert sorted(h["doc_id"] for h in flat) == sorted(h["doc_id"] for h in hits)
    assert hits[0]["rank"] == 1 and "others" not in hits[0]           # the input is untouched


def test_hits_without_a_lineage_are_their_own_groups():
    groups = group_by_lineage([{"rank": 1, "doc_id": "d1", "score": 1.0},
                               {"rank": 2, "doc_id": "d2", "score": 0.9}])
    assert [g["snippet_id"] for g in groups] == ["d1", "d2"]
    assert all(g["versions_in_results"] == [] for g in groups)


def test_no_hits_no_groups():
    assert group_by_lineage([]) == []


# --- through the API -------------------------------------------------------------------------------

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


@pytest.fixture()
def vindex(tmp_path):
    rows = rows_for(build_fixture(5, 4, seed=2))
    for r in rows:                                   # what a git-history ingest adds to each version
        r.update(commit_short=f"c{r['version']:02d}", commit_date="2024-01-01",
                 commit_subject=f"change {r['version']}", change="modified")
    enc = HashingQueryEncoder(64)
    emb, _ = embeddings_for(rows, enc.encode_docs, cache={})
    build_versioned_index(tmp_path / "v", rows, emb, {"model": "mock/hashing-encoder"})
    return tmp_path / "v"


def test_all_versions_search_returns_groups_that_partition_the_hits(vindex, monkeypatch):
    client = client_for(vindex, monkeypatch)
    body = client.post("/search", json={"query": "sum the scores", "k": 12, "all_versions": True}).json()
    assert len(body["hits"]) == 12
    groups = body["groups"]
    in_groups = [g["best"]["doc_id"] for g in groups] + [o["doc_id"] for g in groups for o in g["others"]]
    assert sorted(in_groups) == sorted(h["doc_id"] for h in body["hits"])
    assert len({g["snippet_id"] for g in groups}) == len(groups) <= 5
    assert any(g["others"] for g in groups)            # 12 hits over 5 lineages must repeat some


def test_the_default_collapsed_search_has_single_version_groups(vindex, monkeypatch):
    client = client_for(vindex, monkeypatch)
    groups = client.post("/search", json={"query": "sum the scores", "k": 4}).json()["groups"]
    assert len(groups) == 4 and all(g["others"] == [] for g in groups)


def test_a_flat_index_has_no_groups(tmp_path, monkeypatch):
    d = tmp_path / "flat"
    enc = HashingQueryEncoder(32)
    write_index(d, enc.encode_docs(["def f(): pass", "def g(): pass"]), ["d0", "d1"],
                ["def f(): pass", "def g(): pass"], {"model": "mock/hashing-encoder"})
    body = client_for(d, monkeypatch).post("/search", json={"query": "f", "k": 2}).json()
    assert "groups" not in body


def test_history_carries_the_commit_for_each_version(vindex, monkeypatch):
    client = client_for(vindex, monkeypatch)
    r = client.get("/history/snip0002")
    assert r.status_code == 200
    versions = r.json()["versions"]
    assert [v["version"] for v in versions] == [1, 2, 3, 4]
    assert versions[1]["commit_short"] == "c02" and versions[1]["commit_subject"] == "change 2"
    assert versions[0]["doc_id"] == "snip0002@v1"


def test_the_page_renders_groups_and_a_history_button():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "src" / "static" / "index.html").read_text(encoding="utf-8")
    assert "data.groups" in html and "View history" in html and 'data-history="' in html
    assert '"/history/"' in html
