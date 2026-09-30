"""Version comparison and lineage diff: pure logic first, then the service and the API on a small
versioned index built with the hashing encoder (no model, no network)."""
import difflib

import pytest

from src.runtime_index import HashingQueryEncoder, write_index
from src.versioning.compare import compare_hits, lineage_key, unified_diff
from src.versioning.fixture import build_fixture
from src.versioning.version_index import build_versioned_index, embeddings_for, history, rows_for


def hit(rank, sid, h="h"):
    return {"rank": rank, "doc_id": f"{sid}@v1", "snippet_id": sid, "content_hash": h, "preview": "x"}


# --- compare_hits -----------------------------------------------------------------------------------

def test_appeared_disappeared_moved_and_same_are_told_apart():
    a = [hit(1, "s1"), hit(2, "s2"), hit(3, "s3"), hit(4, "s4")]
    b = [hit(1, "s2"), hit(2, "s1"), hit(3, "s5"), hit(4, "s4")]
    out_a, out_b, summary = compare_hits(a, b)
    by_b = {lineage_key(h): h for h in out_b}
    assert by_b["s5"]["status"] == "appeared" and by_b["s5"]["rank_in_a"] is None
    assert by_b["s2"]["status"] == "moved" and by_b["s2"]["rank_change"] == 1        # 2 -> 1: up one
    assert by_b["s1"]["status"] == "moved" and by_b["s1"]["rank_change"] == -1       # 1 -> 2: down one
    assert by_b["s4"]["status"] == "same" and by_b["s4"]["rank_change"] == 0
    by_a = {lineage_key(h): h for h in out_a}
    assert by_a["s3"]["status"] == "disappeared" and by_a["s3"]["rank_in_b"] is None
    assert by_a["s1"]["status"] == "kept" and by_a["s1"]["rank_in_b"] == 2
    assert summary == {"appeared": 1, "disappeared": 1, "moved": 2, "same": 1, "content_changed": 0}


def test_a_lineage_whose_text_changed_is_flagged_even_if_its_rank_did_not():
    _, out_b, summary = compare_hits([hit(1, "s1", "old")], [hit(1, "s1", "new")])
    assert out_b[0]["status"] == "same" and out_b[0]["content_changed"] is True
    assert summary["content_changed"] == 1


def test_comparing_a_list_with_itself_changes_nothing():
    hits = [hit(i, f"s{i}") for i in range(1, 6)]
    _, out_b, summary = compare_hits(hits, hits)
    assert all(h["status"] == "same" and h["content_changed"] is False for h in out_b)
    assert summary["appeared"] == summary["disappeared"] == summary["moved"] == 0


def test_inputs_are_not_mutated():
    a, b = [hit(1, "s1")], [hit(1, "s2")]
    compare_hits(a, b)
    assert "status" not in a[0] and "status" not in b[0]


def test_hits_without_a_lineage_fall_back_to_the_doc_id():
    h = {"rank": 1, "doc_id": "d1"}
    assert lineage_key(h) == "d1"


# --- unified_diff ------------------------------------------------------------------------------------

def test_unified_diff_matches_difflib_and_counts_lines():
    a, b = "def f(x):\n    return x\n", "def f(x):\n    y = x\n    return y\n"
    d = unified_diff(a, b, "f v1", "f v2")
    expected = "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile="f v1",
                                              tofile="f v2", lineterm="", n=3))
    assert d["diff"] == expected
    assert d["identical"] is False
    assert (d["added_lines"], d["removed_lines"]) == (2, 1)
    assert d["diff"].startswith("--- f v1\n+++ f v2")


def test_identical_text_gives_an_empty_diff():
    d = unified_diff("a\nb\n", "a\nb\n", "x", "y")
    assert d["identical"] is True and d["diff"] == "" and d["added_lines"] == 0


# --- the service ---------------------------------------------------------------------------------------

def build_index(path, n=6, v=3, unchanged_rate=0.0, seed=1):
    fx = build_fixture(n, v, seed=seed, unchanged_rate=unchanged_rate)
    rows = rows_for(fx)
    enc = HashingQueryEncoder(64)
    emb, _ = embeddings_for(rows, enc.encode_docs, cache={})
    build_versioned_index(path, rows, emb, {"model": "mock/hashing-encoder", "task": "version-fixture"})
    return fx


@pytest.fixture()
def vindex(tmp_path):
    build_index(tmp_path / "vi")
    return tmp_path / "vi"


def test_service_compare_marks_every_result(vindex):
    from src.search_service import SearchService
    svc = SearchService(str(vindex), mock=True)
    res = svc.compare("sum the scores below a limit", 1, 3, k=4)
    assert (res["version_a"], res["version_b"]) == (1, 3)
    assert len(res["a"]) == len(res["b"]) == 4
    assert all(h["version"] == 1 for h in res["a"]) and all(h["version"] == 3 for h in res["b"])
    a_ranks = {h["snippet_id"]: h["rank"] for h in res["a"]}
    for h in res["b"]:
        assert h["status"] in ("appeared", "moved", "same")
        if h["snippet_id"] in a_ranks:
            assert h["rank_change"] == a_ranks[h["snippet_id"]] - h["rank"]
        else:
            assert h["status"] == "appeared"
    s = res["summary"]
    assert s["appeared"] + s["moved"] + s["same"] == 4
    assert s["disappeared"] == sum(1 for h in res["a"] if h["status"] == "disappeared")


def test_the_query_is_encoded_once_for_both_versions(vindex):
    from src.search_service import SearchService
    svc = SearchService(str(vindex), mock=True)
    calls = []
    real = svc.encoder.encode_query
    svc.encoder.encode_query = lambda text: calls.append(text) or real(text)
    svc.compare("sum the scores", 1, 2, k=3)
    assert calls == ["sum the scores"]


def test_comparing_a_version_with_itself_is_all_same(vindex):
    from src.search_service import SearchService
    res = SearchService(str(vindex), mock=True).compare("sum the scores", 2, 2, k=5)
    assert {h["status"] for h in res["b"]} == {"same"} and res["summary"]["moved"] == 0


def test_compare_on_a_missing_version_or_flat_index_is_a_value_error(vindex, tmp_path):
    from src.search_service import SearchService
    svc = SearchService(str(vindex), mock=True)
    with pytest.raises(ValueError, match="version 9"):
        svc.compare("q", 1, 9)
    flat = tmp_path / "flat"
    enc = HashingQueryEncoder(16)
    write_index(flat, enc.encode_docs(["def f(): pass"]), ["d0"], ["def f(): pass"],
                {"model": "mock/hashing-encoder"})
    with pytest.raises(ValueError, match="not a versioned index"):
        SearchService(str(flat), mock=True).compare("q", 1, 2)


def test_service_diff_shows_the_edit_between_versions(vindex):
    from src.search_service import SearchService
    svc = SearchService(str(vindex), mock=True)
    d = svc.diff("snip0000", 1, 2)
    assert d["identical"] is False and (d["added_lines"] or d["removed_lines"])
    assert d["diff"].startswith("--- snip0000 v1\n+++ snip0000 v2")
    assert d["same_content_hash"] is False
    # and the diff is against the real stored texts
    texts = {r["version"]: svc.index.doc_texts[r["row"]] for r in history(svc.index, "snip0000")}
    assert d["diff"] == unified_diff(texts[1], texts[2], "snip0000 v1", "snip0000 v2")["diff"]


def test_diff_of_an_unchanged_version_is_empty(tmp_path):
    from src.search_service import SearchService
    build_index(tmp_path / "same", unchanged_rate=1.0)
    d = SearchService(str(tmp_path / "same"), mock=True).diff("snip0001", 1, 3)
    assert d["identical"] is True and d["same_content_hash"] is True


def test_diff_of_an_unknown_lineage_or_version_is_a_key_error(vindex):
    from src.search_service import SearchService
    svc = SearchService(str(vindex), mock=True)
    with pytest.raises(KeyError):
        svc.diff("nope", 1, 2)
    with pytest.raises(KeyError, match="no version 7"):
        svc.diff("snip0000", 1, 7)


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


def test_compare_endpoint(vindex, monkeypatch):
    client = client_for(vindex, monkeypatch)
    r = client.get("/compare", params={"q": "sum the scores", "a": 1, "b": 3, "k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["a"]) == len(body["b"]) == 3 and set(body["summary"]) >= {"appeared", "moved"}
    assert client.get("/compare", params={"q": "x", "a": 1, "b": 9}).status_code == 400
    assert client.get("/compare", params={"q": "x", "a": 1}).status_code == 422       # b is required


def test_diff_endpoint(vindex, monkeypatch):
    client = client_for(vindex, monkeypatch)
    r = client.get("/diff", params={"snippet_id": "snip0000", "a": 1, "b": 2})
    assert r.status_code == 200 and r.json()["diff"].startswith("--- snip0000 v1")
    assert client.get("/diff", params={"snippet_id": "snip0000", "a": 1, "b": 9}).status_code == 404
    assert client.get("/diff", params={"snippet_id": "zzz", "a": 1, "b": 2}).status_code == 404


def test_diff_endpoint_accepts_slashes_in_a_lineage_id(tmp_path, monkeypatch):
    """History indexes use `file.py::qualname` ids; passing it as a query parameter must round-trip."""
    fx = build_fixture(2, 2, seed=1, unchanged_rate=0.0)
    for s in fx["snippets"]:
        s["snippet_id"] = f"pkg/mod{s['snippet_id']}.py::fn"
    rows = rows_for(fx)
    enc = HashingQueryEncoder(64)
    emb, _ = embeddings_for(rows, enc.encode_docs, cache={})
    build_versioned_index(tmp_path / "h", rows, emb, {"model": "mock/hashing-encoder"})
    client = client_for(tmp_path / "h", monkeypatch)
    r = client.get("/diff", params={"snippet_id": rows[0]["snippet_id"], "a": 1, "b": 2})
    assert r.status_code == 200 and "pkg/mod" in r.json()["diff"]


def test_versions_endpoint_and_index_facts(vindex, monkeypatch):
    client = client_for(vindex, monkeypatch)
    assert client.get("/versions").json() == {"versioned": True, "versions": [1, 2, 3]}
    entry = [e for e in client.get("/indexes").json()["indexes"] if e["name"] == str(vindex)][0]
    assert entry["versioned"] is True and entry["available"] is True


def test_the_page_has_the_compare_panel():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "src" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="cmpPanel"' in html and "Compare versions" in html
    assert "/compare?" in html and "/diff?" in html
    assert "http://" not in html and "https://" not in html
