"""Custom-folder indexing (AST chunks with file:line) and the static frontend.

Mocked throughout: the chunker needs no model, and the frontend is checked through FastAPI's test
client with the hashing encoder.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.indexing.code_chunker import chunk_folder, chunk_id_for, chunk_source, iter_python_files

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "textkit"

SAMPLE = '''"""Module docstring."""
import os

CONSTANT = 3


def alpha(x):
    """First."""
    return x + 1


@decorator
def beta(y):
    return y * 2


class Gamma:
    """A class."""

    def method(self):
        return 1


if __name__ == "__main__":
    print(alpha(1))
'''


# --- chunking ------------------------------------------------------------------------------------

def test_chunk_source_finds_functions_and_classes():
    chunks = chunk_source(SAMPLE, "sample.py")
    by_name = {c["name"]: c for c in chunks}
    assert {"alpha", "beta", "Gamma"} <= set(by_name)
    assert by_name["alpha"]["kind"] == "function"
    assert by_name["Gamma"]["kind"] == "class"


def test_line_spans_are_correct_and_include_decorators():
    chunks = {c["name"]: c for c in chunk_source(SAMPLE, "sample.py")}
    lines = SAMPLE.split("\n")
    beta = chunks["beta"]
    # the span starts at the decorator, not at `def`
    assert lines[beta["start_line"] - 1].strip() == "@decorator"
    assert "return y * 2" in lines[beta["end_line"] - 1]
    alpha = chunks["alpha"]
    assert lines[alpha["start_line"] - 1].startswith("def alpha")
    # and the recorded text is exactly those lines
    assert alpha["text"] == "\n".join(lines[alpha["start_line"] - 1:alpha["end_line"]])


def test_chunk_id_is_the_location():
    chunks = {c["name"]: c for c in chunk_source(SAMPLE, "pkg/sample.py")}
    a = chunks["alpha"]
    assert a["chunk_id"] == chunk_id_for("pkg/sample.py", a["start_line"], a["end_line"])
    assert a["chunk_id"].startswith("pkg/sample.py:")
    assert a["file"] == "pkg/sample.py"


def test_module_level_code_becomes_one_chunk():
    chunks = chunk_source(SAMPLE, "sample.py")
    module = [c for c in chunks if c["kind"] == "module"]
    assert len(module) == 1
    assert "import os" in module[0]["text"] and "CONSTANT = 3" in module[0]["text"]
    assert "def alpha" not in module[0]["text"], "function bodies must not be duplicated"


def test_methods_are_not_separate_chunks_by_default():
    assert [c for c in chunk_source(SAMPLE, "s.py") if c["kind"] == "method"] == []
    with_methods = chunk_source(SAMPLE, "s.py", chunk_methods=True)
    methods = [c for c in with_methods if c["kind"] == "method"]
    assert [m["qualname"] for m in methods] == ["Gamma.method"]


def test_chunks_are_ordered_by_position():
    chunks = chunk_source(SAMPLE, "s.py")
    starts = [c["start_line"] for c in chunks]
    assert starts == sorted(starts)


def test_a_file_with_only_a_docstring_yields_no_oversized_noise():
    chunks = chunk_source('"""Just a docstring that is long enough to be kept as a module chunk."""\n',
                          "d.py")
    assert all(c["kind"] == "module" for c in chunks)


def test_syntax_errors_are_reported_not_swallowed(tmp_path):
    (tmp_path / "ok.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def f(:\n", encoding="utf-8")
    chunks, stats = chunk_folder(tmp_path)
    assert stats["files_scanned"] == 2
    assert stats["files_failed"] == 1
    assert stats["failures"][0]["file"] == "broken.py"
    assert "SyntaxError" in stats["failures"][0]["error"]
    assert [c["name"] for c in chunks] == ["f"]        # the good file is still indexed


def test_noise_directories_are_skipped(tmp_path):
    (tmp_path / "keep.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    for noise in ("__pycache__", ".git", "node_modules", ".venv"):
        d = tmp_path / noise
        d.mkdir()
        (d / "junk.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    files = iter_python_files(tmp_path)
    assert [f.name for f in files] == ["keep.py"]


def test_oversize_chunks_are_dropped_and_counted(tmp_path):
    big = "def huge():\n" + "\n".join(f"    x{i} = {i}" for i in range(4000)) + "\n"
    (tmp_path / "big.py").write_text(big, encoding="utf-8")
    chunks, stats = chunk_folder(tmp_path, max_chunk_chars=500)
    assert stats["skipped_oversize"] == 1 and chunks == []


def test_chunk_folder_is_deterministic(tmp_path):
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    first, _ = chunk_folder(tmp_path)
    second, _ = chunk_folder(tmp_path)
    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]


def test_the_shipped_example_package_chunks_cleanly():
    chunks, stats = chunk_folder(EXAMPLES)
    assert stats["files_failed"] == 0, stats["failures"]
    names = {c["qualname"] for c in chunks}
    assert {"tokenize", "strip_accents", "cosine_similarity", "InvertedIndex"} <= names
    for c in chunks:
        assert c["start_line"] >= 1 and c["end_line"] >= c["start_line"]
        assert c["file"].endswith(".py")


# --- building and serving a code index -----------------------------------------------------------

@pytest.fixture(scope="module")
def code_index(tmp_path_factory):
    out = tmp_path_factory.mktemp("code") / "idx"
    proc = subprocess.run([sys.executable, "src/build_index.py", "--source", str(EXAMPLES),
                           "--mock-encoder", "--out", str(out)], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


def test_built_code_index_carries_chunk_metadata(code_index):
    from src.runtime_index import RuntimeIndex
    idx = RuntimeIndex.load(code_index)
    assert idx.manifest["kind"] == "code"
    assert idx.chunks is not None and len(idx.chunks) == len(idx.doc_ids)
    assert idx.verify_files() == []
    assert idx.manifest["source_root"].endswith("textkit")
    for c in idx.chunks:
        assert {"file", "start_line", "end_line"} <= set(c)


def test_search_results_expose_file_and_lines(code_index):
    from src.search_service import SearchService
    svc = SearchService(str(code_index), mock=True)
    assert svc.chunked is True
    res = svc.search("split a paragraph into sentences", k=3)
    assert len(res["hits"]) == 3
    for h in res["hits"]:
        assert h["location"] == f"{h['file']}:{h['start_line']}-{h['end_line']}"
        assert h["doc_id"] == h["location"]
        assert h["kind"] in ("function", "class", "module", "method")
        assert h["start_line"] <= h["end_line"]


def test_cli_prints_the_location(code_index):
    proc = subprocess.run([sys.executable, "src/cli.py", "count word frequencies", "-k", "2",
                           "--index", str(code_index), "--mock-encoder"], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".py:" in proc.stdout


def test_build_index_rejects_a_nonexistent_folder(tmp_path):
    proc = subprocess.run([sys.executable, "src/build_index.py", "--source",
                           str(tmp_path / "nope"), "--mock-encoder", "--out", str(tmp_path / "o")],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "neither" in (proc.stdout + proc.stderr)


# --- the frontend --------------------------------------------------------------------------------

def client_for(index_dir, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import src.api as api
    monkeypatch.setenv("INDEX_DIR", str(index_dir))
    monkeypatch.setenv("MOCK_ENCODER", "1")
    monkeypatch.setattr(api, "_service", None)
    monkeypatch.setattr(api, "_error", None)
    monkeypatch.setattr(api, "_services", {})
    return TestClient(api.app), api


def test_the_page_is_served_at_root(code_index, monkeypatch):
    client, _ = client_for(code_index, monkeypatch)
    with client:
        r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    for needed in ("<title>Code Retrieval</title>", 'id="q"', 'id="index"', 'id="k"',
                   'id="allv"', 'id="ver"', "highlightPython"):
        assert needed in body, needed


def test_the_page_has_no_external_dependencies(code_index, monkeypatch):
    """It must work offline: no CDN scripts, stylesheets or fonts."""
    client, _ = client_for(code_index, monkeypatch)
    with client:
        body = client.get("/").text
    for forbidden in ("http://", "https://", "cdn.", "<script src=", "<link rel=\"stylesheet\""):
        assert forbidden not in body, f"the page reaches out to the network: {forbidden}"


def test_search_endpoint_returns_every_field_the_page_uses(code_index, monkeypatch):
    client, _ = client_for(code_index, monkeypatch)
    with client:
        r = client.post("/search", json={"query": "cosine similarity", "k": 3, "preview_chars": 400})
    assert r.status_code == 200
    data = r.json()
    # fields the page reads at the top level
    for key in ("hits", "timings_ms", "total_ms", "collapsed_lineages", "version_filter"):
        assert key in data, key
    assert {"encode_query_ms", "search_ms"} <= set(data["timings_ms"])
    for h in data["hits"]:
        for key in ("rank", "score", "preview", "truncated", "doc_id", "location", "kind",
                    "qualname", "file", "start_line", "end_line"):
            assert key in h, key


def test_indexes_endpoint_lists_the_default(code_index, monkeypatch):
    client, _ = client_for(code_index, monkeypatch)
    with client:
        data = client.get("/indexes").json()
    assert data["default"] == str(code_index)
    assert any(ix["name"] == str(code_index) for ix in data["indexes"])


def test_health_reports_a_code_index(code_index, monkeypatch):
    client, _ = client_for(code_index, monkeypatch)
    with client:
        h = client.get("/health").json()
    assert h["loaded"] is True
    assert h["index"]["chunked"] is True
    assert h["index"]["source_root"].endswith("textkit")


def test_search_rejects_an_index_outside_the_allow_list(code_index, monkeypatch):
    client, _ = client_for(code_index, monkeypatch)
    monkeypatch.setenv("ALLOWED_INDEXES", "lite")
    with client:
        r = client.post("/search", json={"query": "x", "k": 2, "index": "full"})
    assert r.status_code == 400 and "ALLOWED_INDEXES" in r.json()["detail"]


def test_history_on_a_code_index_is_a_clean_404(code_index, monkeypatch):
    client, _ = client_for(code_index, monkeypatch)
    with client:
        r = client.get("/history/whatever")
    assert r.status_code == 404


# --- the demo script -----------------------------------------------------------------------------

def test_demo_script_runs_end_to_end(tmp_path):
    proc = subprocess.run([sys.executable, "examples/demo_queries.py", "--mock-encoder", "--rebuild",
                           "--index", str(tmp_path / "demo_idx")], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MOCK ENCODER" in proc.stdout
    assert proc.stdout.count(".py:") >= 6            # every query printed located results
    assert json.loads((tmp_path / "demo_idx" / "manifest.json").read_text())["kind"] == "code"
