"""The single-page UI and the API behaviour it depends on.

There is no JavaScript runtime in CI, so the page is checked structurally (the controls and hooks are
present, nothing points outside the file) and the behaviour that can be tested in Python -- what the
server does with the text the page sends -- is tested through FastAPI's test client with the hashing
encoder. Nothing here loads a model.
"""
import re
from pathlib import Path

import pytest

from src.runtime_index import HashingQueryEncoder, write_index

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "src" / "static" / "index.html").read_text(encoding="utf-8")


@pytest.fixture()
def flat_index(tmp_path):
    d = tmp_path / "flat"
    enc = HashingQueryEncoder(64)
    texts = [f"def f{i}(xs):\n    return sum(x for x in xs if x > {i})\n" for i in range(8)]
    write_index(d, enc.encode_docs(texts), [f"d{i}" for i in range(8)], texts,
                {"model": "mock/hashing-encoder", "task": "AppsRetrieval"})
    return d


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
    return TestClient(api.app), api


def test_the_page_never_reaches_outside_itself():
    assert "http://" not in HTML and "https://" not in HTML
    assert "<script src" not in HTML and "<link " not in HTML


# --- item 3: multiline query box -------------------------------------------------------------------

def test_the_query_box_is_a_textarea_and_ctrl_or_cmd_enter_submits():
    assert re.search(r'<textarea[^>]*id="q"', HTML)
    assert not re.search(r'<input[^>]*id="q"', HTML)
    assert "ev.ctrlKey || ev.metaKey" in HTML and 'ev.key === "Enter"' in HTML
    assert "requestSubmit" in HTML


def test_examples_are_four_problem_statements_and_one_code_query():
    labels = re.findall(r'label: "([^"]+)"', HTML)
    assert len([x for x in labels if x.startswith("Problem")]) == 4
    assert len([x for x in labels if x.startswith("Code")]) == 1
    assert 'id="examples"' in HTML


def test_the_example_statements_are_multiline():
    block = HTML.split("var EXAMPLES = [")[1].split("];")[0]
    assert block.count("\\n") > 30            # statements keep their line structure


def test_a_multiline_query_reaches_the_encoder_unchanged(flat_index, monkeypatch):
    client, api = client_for(flat_index, monkeypatch)
    svc = api.get_service()
    seen = []
    real = svc.encoder.encode_query
    svc.encoder.encode_query = lambda text: seen.append(text) or real(text)
    query = "Count the pairs.\n\nInput\n5 3\n1 2 3 4 5\n\nOutput\n4"
    r = client.post("/search", json={"query": query, "k": 3})
    assert r.status_code == 200, r.text
    assert seen == [query]
    assert r.json()["query"] == query


# --- item 4: full-code viewer and GET /doc ---------------------------------------------------------

def test_the_viewer_has_line_numbers_copy_and_collapse():
    for hook in ('id="viewer"', 'id="vwCopy"', 'id="vwToggle"', "lineNumbers(", "COLLAPSE_LINES",
                 '"/doc?id="', 'data-doc="'):
        assert hook in HTML, hook


def test_doc_endpoint_returns_the_full_text_and_only_real_locations(flat_index, monkeypatch):
    client, _ = client_for(flat_index, monkeypatch)
    r = client.get("/doc", params={"id": "d3"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["text"] == "def f3(xs):\n    return sum(x for x in xs if x > 3)\n"
    assert d["n_lines"] == 2 and d["first_line"] == 1
    assert d["has_source_file"] is False and "file" not in d and "location" not in d


def test_doc_endpoint_404s_for_an_unknown_id(flat_index, monkeypatch):
    client, _ = client_for(flat_index, monkeypatch)
    r = client.get("/doc", params={"id": "nope"})
    assert r.status_code == 404 and "nope" in r.json()["detail"]
    assert client.get("/doc").status_code == 422


def test_doc_endpoint_reports_the_file_and_real_line_numbers_for_a_code_index(tmp_path, monkeypatch):
    import subprocess
    import sys
    src = tmp_path / "proj"
    src.mkdir()
    (src / "m.py").write_text('X = 1\n\n\ndef first():\n    return 1\n\n\ndef second(a):\n'
                              '    b = a\n    return b\n', encoding="utf-8")
    out = tmp_path / "code"
    proc = subprocess.run([sys.executable, "src/build_index.py", "--source", str(src),
                           "--mock-encoder", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    client, _ = client_for(out, monkeypatch)
    d = client.get("/doc", params={"id": "m.py:8-10"}).json()
    assert d["has_source_file"] is True and d["file"] == "m.py"
    assert d["location"] == "m.py:8-10" and d["first_line"] == 8 and d["n_lines"] == 3
    assert d["text"].startswith("def second(a):")


def test_doc_endpoint_carries_lineage_for_a_versioned_index(tmp_path, monkeypatch):
    from src.versioning.fixture import build_fixture
    from src.versioning.version_index import build_versioned_index, embeddings_for, rows_for
    rows = rows_for(build_fixture(3, 2, seed=1))
    enc = HashingQueryEncoder(64)
    emb, _ = embeddings_for(rows, enc.encode_docs, cache={})
    build_versioned_index(tmp_path / "v", rows, emb, {"model": "mock/hashing-encoder"})
    client, _ = client_for(tmp_path / "v", monkeypatch)
    d = client.get("/doc", params={"id": "snip0001@v2"}).json()
    assert d["snippet_id"] == "snip0001" and d["version"] == 2 and d["has_source_file"] is False


# --- item 6: performance panel ---------------------------------------------------------------------

def test_search_response_carries_the_performance_block(flat_index, monkeypatch):
    client, _ = client_for(flat_index, monkeypatch)
    body = client.post("/search", json={"query": "sum the values above a threshold", "k": 3}).json()
    p = body["performance"]
    assert p["device"] == "cpu" and p["model"] == "mock/hashing-encoder"
    assert p["index_docs"] == 8 and p["index_dim"] == 64 and p["index_bytes"] > 0
    assert p["query_tokens"] == 6 and p["truncated"] is False and p["query_cap"] == 1024
    assert {"encode_query_ms", "search_ms"} <= set(body["timings_ms"]) and body["total_ms"] >= 0


def test_truncation_is_reported_when_the_query_exceeds_the_cap(flat_index):
    from src.search_service import SearchService
    svc = SearchService(str(flat_index), mock=True, max_query_tokens=3)
    assert svc.performance_info("one two three")["truncated"] is False
    long = svc.performance_info("one two three four five")
    assert long["truncated"] is True and long["query_cap"] == 3 and long["query_tokens"] == 5


def test_no_cap_means_never_truncated(flat_index):
    from src.search_service import SearchService
    svc = SearchService(str(flat_index), mock=True, max_query_tokens=None)
    info = svc.performance_info("word " * 5000)
    assert info["query_cap"] is None and info["truncated"] is False


def test_the_page_renders_the_performance_panel():
    for hook in ("function perfPanel", "data.performance", '"truncated"', '"query cap"', '"device"'):
        assert hook in HTML, hook


# --- item 7: the status follows the selected index; states; wording ---------------------------------

def make_second_index(tmp_path, name="other"):
    d = tmp_path / name
    enc = HashingQueryEncoder(32)
    write_index(d, enc.encode_docs(["def g(): pass"] * 3), ["a", "b", "c"], ["def g(): pass"] * 3,
                {"model": "mock/hashing-encoder"})
    return d


def test_health_without_an_index_describes_the_default(flat_index, monkeypatch):
    client, api = client_for(flat_index, monkeypatch)
    api.get_service()
    body = client.get("/health").json()
    assert body["loaded"] is True and body["index"]["n_docs"] == 8


def test_health_for_another_index_describes_that_one_and_does_not_load_it(flat_index, tmp_path, monkeypatch):
    client, api = client_for(flat_index, monkeypatch)
    api.get_service()
    other = make_second_index(tmp_path)
    monkeypatch.setenv("ALLOWED_INDEXES", f"{other}")
    body = client.get("/health", params={"index": str(other)}).json()
    assert body["loaded"] is False and body["available"] is True
    assert body["facts"]["n_docs"] == 3                          # from the manifest, not the default's 8
    assert str(other) not in api._services                       # a status probe never loads a model
    client.post("/search", json={"query": "g", "k": 1, "index": str(other)})
    loaded = client.get("/health", params={"index": str(other)}).json()
    assert loaded["loaded"] is True and loaded["index"]["n_docs"] == 3 and loaded["index"]["dim"] == 32


def test_health_for_an_unbuilt_index_says_so(flat_index, tmp_path, monkeypatch):
    client, _ = client_for(flat_index, monkeypatch)
    ghost = tmp_path / "never_built"
    monkeypatch.setenv("ALLOWED_INDEXES", str(ghost))
    body = client.get("/health", params={"index": str(ghost)}).json()
    assert body["loaded"] is False and body["available"] is False and "reason" in body


def test_health_refuses_to_probe_arbitrary_paths(flat_index, monkeypatch):
    client, _ = client_for(flat_index, monkeypatch)
    assert client.get("/health", params={"index": "/etc"}).status_code == 400
    assert client.get("/health", params={"index": "../../"}).status_code == 400


def test_searching_a_missing_index_is_a_clean_404(flat_index, tmp_path, monkeypatch):
    client, _ = client_for(flat_index, monkeypatch)
    r = client.post("/search", json={"query": "x", "index": str(tmp_path / "nope")})
    assert r.status_code == 404 and "No runtime index" in r.json()["detail"]


def test_indexes_flags_an_unbuilt_default_as_unavailable(tmp_path, monkeypatch):
    client, _ = client_for(tmp_path / "missing_default", monkeypatch)
    entry = [e for e in client.get("/indexes").json()["indexes"]
             if e["name"] == str(tmp_path / "missing_default")][0]
    assert entry["available"] is False


def test_the_page_follows_the_selected_index_and_has_every_state():
    assert '"/health?index="' in HTML
    for hook in ("function problemBox", "function emptyBox", "function loadingBox",
                 "Service unavailable", "is not available"):
        assert hook in HTML, hook
    assert 'textContent = "searching&hellip;"' not in HTML             # the literal-entity bug


def test_agent_mode_disables_the_controls_it_ignores():
    assert 'function syncAgentMode' in HTML
    assert '["allv", "ver", "cat", "sugg"]' in HTML and "Agent mode plans its own searches" in HTML


def test_scores_are_labelled_similarity_scores_never_percentages():
    assert HTML.count("score.toFixed(4)") == HTML.count('similarity score " + ')
    assert "not a percentage, a probability or a confidence" in HTML
    assert not re.search(r"score[^\n]{0,40}\*\s*100", HTML)         # no 0.93 -> "93%"


# --- the Prism layout: blue statement panel + search panel + ranked list ------------------------------

def test_prism_layout_and_palette():
    css = HTML.split("</style>")[0]
    for hook in ('class="hero"', 'class="ask"', 'class="results"', 'class="topbar"', 'class="footer"'):
        assert hook in HTML, hook
    assert "--blue: #1428a0" in css and "--lime: #d8ee74" in css and "--cyan: #b8ebff" in css
    assert "PRISM / 26" in HTML and "SRI-B" in HTML


def test_hero_shows_only_figures_that_were_measured():
    """The concept mock-up had invented numbers (repos indexed, intent match, median response). This page
    may show only the submitted, measured result and live values from the running index."""
    assert "0.9376" in HTML and "0.9238" in HTML and "official test split" in HTML
    for invented in ("4.8k", "92%", "0.41s", "indexed repos", "intent match"):
        assert invented not in HTML, invented


def test_examples_are_chips_that_fill_the_query():
    assert re.search(r'<div class="chips" id="examples">', HTML)
    assert "button[data-i]" in HTML and 'c.className = "chip"' in HTML


def test_each_result_has_a_copy_button_that_copies_the_whole_snippet():
    assert 'data-copy="' in HTML and "async function copyDoc" in HTML
    assert '"/doc?id="' in HTML                       # the full text comes from /doc, not the preview


def test_dark_theme_is_neutral_not_blue_tinted():
    dark = HTML.split("prefers-color-scheme: dark")[1].split("}")[0]
    paper = re.search(r"--paper: #([0-9a-f]{6})", dark).group(1)
    r, g, b = (int(paper[i:i + 2], 16) for i in (0, 2, 4))
    assert max(r, g, b) - min(r, g, b) <= 8


def test_the_highlighter_scans_numbers_and_names_in_one_pass():
    """The old chained replaces re-scanned already-written markup, so `class` inside <span class="num">
    was recoloured as a keyword and leaked into the code text."""
    assert "function plain(chunk)" in HTML and "KW_SET" in HTML and "KEYWORDS" not in HTML
    plain = HTML.split("function plain(chunk) {")[1].split("\n}\n")[0]
    assert "s.replace(" not in plain


def test_every_control_the_script_uses_still_exists():
    for el_id in ("q", "go", "examples", "index", "k", "allv", "ver", "cat", "agent", "sugg",
                  "reindexBtn", "modeNote", "agentNote", "cmpPanel", "viewer", "meta", "results", "status",
                  "qInfo", "resultCount"):
        assert f'id="{el_id}"' in HTML, el_id


def test_full_width_hero_then_a_search_bar_then_results():
    css = HTML.split("</style>")[0]
    assert "grid-template-columns: 3fr 1fr" not in css, "the hero is no longer beside the search panel"
    hero, ask, results = (HTML.index(m) for m in ('class="hero"', 'class="ask"', 'class="results"'))
    assert hero < ask < results
    inside_hero = HTML[hero:ask]
    assert 'id="hero-title"' in inside_hero and "<h1" in inside_hero, "the headline lives inside the blue card"
    assert 'id="q"' not in inside_hero and '<form' not in inside_hero, "the search bar sits below it"


def test_hero_has_four_fact_cards_and_no_logo_mark():
    assert HTML.count('class="fact"') == 4
    assert 'class="mark"' not in HTML and ".mark " not in HTML.split("</style>")[0]
    assert "PRISM / 26" in HTML


def test_the_search_heading_and_copy():
    assert "What do you want to find?" in HTML
    assert "What are you trying to build" not in HTML
    assert 'id="histBtn"' not in HTML and "async function showHistory" in HTML


def test_the_search_bar_grows_with_its_content():
    assert "function autoGrow" in HTML and "autoGrow();" in HTML


def test_below_the_hero_there_is_only_a_headline_and_a_search_bar():
    """Examples, index, size and the other controls are all inside one collapsed disclosure."""
    ask = HTML[HTML.index('class="ask"'):HTML.index('class="results"')]
    visible = ask.split('<details class="adv">')[0]
    assert "What do you want to find?" in visible and 'id="q"' in visible and 'id="go"' in visible
    for hidden_inside in ('id="examples"', 'id="index"', 'id="k"', 'id="allv"'):
        assert hidden_inside not in visible, hidden_inside
        assert hidden_inside in ask
    assert "Options and examples" in ask
    assert 'class="lead"' not in ask and 'class="hint"' not in visible


def test_the_page_uses_the_full_desktop_width_and_leaves_room_under_the_hero():
    css = HTML.split("</style>")[0]
    assert ".shell { max-width: 1760px" in css
    assert "margin: clamp(64px, 9vw, 120px) auto 0" in css


# --- Phase 5: UI polish ---------------------------------------------------------------------------------

def test_result_card_has_flex_row_middle_ellipsis_and_subline():
    css = HTML.split("</style>")[0]
    assert ".hit-top-row" in css and ".hit-subline" in css
    assert "display: flex" in css
    assert "justify-content: space-between" in css
    assert "function middleEllipsis" in HTML
    assert 'class="badge score"' in HTML
    assert "hit-subline" in HTML


def test_dynamic_index_selector_requires_server_discovery():
    assert "FRIENDLY_INDEX_LABELS" in HTML
    assert "F2LLM-1.7B Full" in HTML
    assert "F2LLM-0.6B Lite" in HTML
    assert "Click 40-commit History" in HTML
    assert "FALLBACK_INDEXES" not in HTML
    assert "No indexes available; index discovery failed or returned an empty list." in HTML
    assert "not on disk" in HTML


def test_unsupported_controls_conditionally_disabled_with_tooltip():
    assert "VERSIONED_TOOLTIP" in HTML
    assert "Available on versioned indexes like Click History" in HTML
    assert "function syncIndexCapabilities" in HTML
    assert "loadCategories" in HTML


def test_backend_error_translation_messages():
    assert "function friendlyErrorMessage" in HTML
    assert "Model/index unavailable -- start server or choose an index on disk" in HTML
    assert "Document or version not found" in HTML
    assert "problemBox(r.status" in HTML
