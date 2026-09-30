"""Query routing, the structural index and call graph, categories, the agent loop and its evaluation.

All mocked: the hashing encoder for anything dense, and a small fixture package on disk for anything
structural. No model, no network.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.indexing.categories import ast_family, cluster_labels, tag_rows
from src.indexing.optimizations import analyze_source
from src.indexing.structural import build_from_folder, build_from_sources, parse_module
from src.retrieval.query_router import classify, extract_pair, extract_subject, looks_like_code

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "textkit"

MOD_A = '''"""Module A."""
import os
from .helpers import shout as yell

CONSTANT = "the bluetooth settings deeplink"


def alpha(items):
    """Add up the items."""
    total = 0
    for item in items:
        total += beta(item)
    return total


def beta(value):
    """Double a value."""
    return value * 2


class Runner:
    def run(self, items):
        yell(CONSTANT)
        return alpha(items)
'''

MOD_B = '''"""Module B."""
from a import alpha


def shout(text):
    """Upper-case some text."""
    return text.upper()


def gamma(xs):
    """Call alpha then shout."""
    result = alpha(xs)
    shout("the bluetooth settings deeplink")
    return result
'''


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    d = tmp_path_factory.mktemp("pkg")
    (d / "a.py").write_text(MOD_A, encoding="utf-8")
    (d / "helpers.py").write_text(MOD_B, encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def sindex(tree):
    return build_from_folder(tree)


# --- D: structural facts ---------------------------------------------------------------------------

def test_definitions_are_found_with_qualnames_and_lines():
    mod = parse_module(MOD_A, "a.py")
    by_qual = {d["qualname"]: d for d in mod["defs"]}
    assert {"alpha", "beta", "Runner", "Runner.run"} <= set(by_qual)
    assert by_qual["Runner"]["kind"] == "class"
    assert by_qual["Runner.run"]["kind"] == "method"
    assert by_qual["alpha"]["docstring"] == "Add up the items."
    assert by_qual["alpha"]["start_line"] < by_qual["alpha"]["end_line"]
    assert by_qual["beta"]["args"] == ["value"]


def test_calls_are_recorded_in_source_order_with_their_caller():
    mod = parse_module(MOD_A, "a.py")
    lines = [c["line"] for c in mod["calls"]]
    assert lines == sorted(lines), "calls must be in source order"
    beta_call = next(c for c in mod["calls"] if c["callee"] == "beta")
    assert beta_call["caller"] == "alpha"
    run_call = next(c for c in mod["calls"] if c["callee"] == "yell")
    assert run_call["caller"] == "Runner.run"


def test_imports_record_module_name_and_alias():
    mod = parse_module(MOD_A, "a.py")
    by_alias = {i["alias"]: i for i in mod["imports"]}
    assert by_alias["os"]["kind"] == "import"
    assert by_alias["yell"]["kind"] == "from"
    assert by_alias["yell"]["name"] == "shout"
    assert by_alias["yell"]["module"] == ".helpers"


def test_string_literals_and_name_references_are_captured():
    mod = parse_module(MOD_A, "a.py")
    assert any("bluetooth" in s["value"] for s in mod["strings"])
    assert any(n["name"] == "CONSTANT" and n["store"] for n in mod["names"])
    assert any(n["name"] == "CONSTANT" and not n["store"] for n in mod["names"])


def test_short_strings_are_skipped():
    mod = parse_module('x = "ab"\ny = "long enough here"\n', "s.py")
    values = [s["value"] for s in mod["strings"]]
    assert "ab" not in values and "long enough here" in values


# --- D: the call graph ------------------------------------------------------------------------------

def test_who_calls_crosses_files(sindex):
    callers = sindex.who_calls("alpha")
    files = {e["file"] for e in callers}
    assert files == {"a.py", "helpers.py"}, f"got {files}"
    assert all(e["line"] > 0 for e in callers)


def test_what_calls_lists_callees_in_order(sindex):
    callees = sindex.what_calls("gamma")
    names = [e["callee_text"] for e in callees]
    assert names == ["alpha", "shout"], names


def test_an_import_alias_resolves_to_the_real_definition(sindex):
    """`from .helpers import shout as yell` then `yell(...)` must resolve to shout."""
    callers = sindex.who_calls("shout")
    lines = {(e["file"], e["callee_text"]) for e in callers}
    assert ("a.py", "yell") in lines, f"alias not resolved: {lines}"


def test_where_imported_finds_both_forms(sindex):
    assert {i["file"] for i in sindex.where_imported("shout")} == {"a.py"}
    assert {i["file"] for i in sindex.where_imported("alpha")} == {"helpers.py"}
    assert {i["file"] for i in sindex.where_imported("os")} == {"a.py"}


def test_where_used_finds_a_string_across_files(sindex):
    hits = sindex.where_used("bluetooth settings deeplink")
    assert {h["file"] for h in hits} == {"a.py", "helpers.py"}
    assert all(h["kind"] == "string" for h in hits)


def test_where_used_matches_identifiers_exactly(sindex):
    hits = sindex.where_used("CONSTANT")
    assert hits and all(h["value"] == "CONSTANT" for h in hits)
    assert {h["kind"] for h in hits} <= {"reference", "assignment"}
    assert sindex.where_used("CONST") == [] or all(
        h["kind"] == "string" for h in sindex.where_used("CONST"))


def test_call_order_is_lexical_and_says_so(sindex):
    """gamma() calls alpha before shout, so helpers.py qualifies; the reverse order does not."""
    forward = sindex.files_calling_in_order("alpha", "shout")
    assert "helpers.py" in {r["file"] for r in forward}
    for r in forward:
        assert r["first_line"] < r["second_line"]


def test_ambiguous_calls_are_flagged_not_guessed():
    """Two files defining run(): a call must not silently pick one."""
    sources = {"x.py": "def run():\n    return 1\n",
               "y.py": "def run():\n    return 2\n",
               "z.py": "def go():\n    return run()\n"}
    idx = build_from_sources(sources)
    edges = [e for e in idx.unresolved if e["callee"] == "run"]
    assert edges and edges[0]["ambiguous"] is True
    assert "callee_key" not in edges[0]


def test_an_external_call_is_marked_external():
    idx = build_from_sources({"x.py": "import json\n\ndef f():\n    return json.loads('{}')\n"})
    external = [e for e in idx.unresolved if e["callee"] == "loads"]
    assert external and external[0]["external"] is True


def test_unparseable_files_are_reported_not_crashed(tmp_path):
    (tmp_path / "ok.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def f(:\n", encoding="utf-8")
    idx = build_from_folder(tmp_path)
    assert len(idx.failures) == 1 and idx.failures[0]["file"] == "bad.py"
    assert idx.find_definitions("f")


def test_stats_are_consistent(sindex):
    s = sindex.stats()
    assert s["files"] == 2 and s["failures"] == 0
    assert s["defs"] >= 6 and s["calls"] >= 4 and s["imports"] >= 3


def test_structural_index_works_on_the_example_package():
    idx = build_from_folder(EXAMPLES)
    assert idx.failures == []
    assert {e["file"] for e in idx.who_calls("tokenize")} >= {"search.py", "stats.py"}


# --- C: the router -----------------------------------------------------------------------------------

@pytest.mark.parametrize("query,kind", [
    ("who calls tokenize", "structural"),
    ("callers of invoke", "structural"),
    ("where is stats imported", "structural"),
    ("who imports click", "structural"),
    ('where is the string "deeplink" used', "structural"),
    ("what does Command.invoke call", "structural"),
    ("which files call open before close", "structural"),
    ("def merge(a, b):\n    return sorted(a + b)", "code_query"),
    ("for item in items:\n    total += item", "code_query"),
    ("binary search over a sorted array", "nl_intent"),
    ("remove accents from text", "nl_intent"),
])
def test_router_classifies_each_kind(query, kind):
    assert classify(query)["kind"] == kind, classify(query)


def test_an_apps_style_statement_is_recognised():
    q = ("Given an array of n integers, find the maximum subarray sum.\n\n"
         "Input\nThe first line contains n.\n\nOutput\nPrint one integer.\n\n"
         "Constraints\n1 <= n <= 100000\n")
    out = classify(q)
    assert out["kind"] == "problem_statement" and out["route"] == "dense"
    assert "sections" in out["reason"]


def test_routes_map_to_the_right_retriever():
    assert classify("who calls f")["route"] == "structural"
    assert classify("def f():\n    return 1")["route"] == "dense_code"
    assert classify("sort a list")["route"] == "dense"


def test_a_bare_identifier_is_not_treated_as_code():
    """'invoke' is far more likely a name someone is asking about than a code query."""
    assert looks_like_code("invoke") is False
    assert classify("invoke")["kind"] == "nl_intent"


def test_a_plain_sentence_is_not_code():
    assert looks_like_code("find the maximum value in a list") is False
    assert looks_like_code("'just a string'") is False


def test_subject_extraction_prefers_quotes_and_skips_stopwords():
    assert extract_subject('where is "my constant" used') == "my constant"
    assert extract_subject("who calls tokenize") == "tokenize"
    assert extract_subject("what does Command.invoke call") == "Command.invoke"


def test_call_order_pair_extraction():
    assert extract_pair("which files call open before close") == ("open", "close")
    assert extract_pair("who calls x") == (None, None)


def test_router_always_explains_itself():
    for q in ("who calls f", "def f(): pass", "sort a list", ""):
        out = classify(q)
        assert out["reason"] and isinstance(out["reason"], str)
        assert out["route"] in ("dense", "dense_code", "structural")


def test_llm_fallback_is_off_by_default_and_never_overrides_a_confident_rule():
    calls = []

    def fake(prompt, **kw):
        calls.append(prompt)
        return "structural"

    assert classify("who calls f", allow_llm=False, call=fake)["source"] == "rule"
    assert calls == [], "the LLM must not be consulted when it is not enabled"
    # a confident rule (code) is not second-guessed even with the fallback enabled
    assert classify("def f():\n    return 1", allow_llm=True, call=fake)["source"] == "rule"
    assert calls == []


def test_llm_fallback_is_consulted_only_when_rules_are_unsure():
    def fake(prompt, **kw):
        return "code_query"

    out = classify("something ambiguous here", allow_llm=True, call=fake)
    assert out["source"] == "llm" and out["kind"] == "code_query"


def test_a_failing_llm_degrades_to_the_rule_result():
    def boom(prompt, **kw):
        raise RuntimeError("no key")

    out = classify("something ambiguous here", allow_llm=True, call=boom)
    assert out["route"] == "dense" and out["source"] == "llm-failed"
    assert "unavailable" in out["reason"]


# --- E: categories -------------------------------------------------------------------------------

@pytest.mark.parametrize("source,family", [
    ("def f(n):\n    if n < 2:\n        return n\n    return f(n-1) + f(n-2)\n", "recursion"),
    ("import heapq\ndef f(xs):\n    heapq.heappush(xs, 1)\n", "heap_priority"),
    ("def f(xs):\n    return sorted(xs)\n", "sorting"),
    ("def f(g, start):\n    visited = set()\n    queue = [start]\n    return visited\n", "graph_search"),
    ("def f(n):\n    return gcd(n, 10) + factorial(n)\n", "math_number_theory"),
])
def test_ast_family_rules(source, family):
    assert ast_family(source)[0] == family


def test_ast_family_always_gives_a_reason():
    for source in ("def f():\n    pass\n", "not python at all $$$", ""):
        fam, why = ast_family(source)
        assert fam and why and isinstance(why, str)


def test_unparseable_source_is_other():
    assert ast_family("def f(:")[0] == "other"


def test_cluster_labels_are_readable_and_deterministic():
    texts = ["def parse_token(s):\n    return tokenize(s)"] * 4 + \
            ["def sort_rows(rows):\n    return sorted(rows)"] * 4
    rng = np.random.RandomState(0)
    emb = np.vstack([np.tile(np.array([1.0, 0.0], dtype=np.float32), (4, 1)),
                     np.tile(np.array([0.0, 1.0], dtype=np.float32), (4, 1))])
    emb = emb + rng.randn(8, 2).astype(np.float32) * 0.01
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assign1, labels1 = cluster_labels(emb, texts, n_clusters=2, seed=0)
    assign2, labels2 = cluster_labels(emb, texts, n_clusters=2, seed=0)
    assert assign1 == assign2 and labels1 == labels2, "clustering must be reproducible"
    assert len(set(assign1)) == 2
    assert all(lab and not lab.startswith("cluster ") for lab in labels1)
    joined = " ".join(labels1)
    assert "token" in joined or "parse" in joined


def test_tag_rows_without_embeddings_still_gives_families():
    tags = tag_rows(["def f(xs):\n    return sorted(xs)\n"], embeddings=None)
    assert tags[0]["ast_family"] == "sorting" and "cluster" not in tags[0]


def test_cluster_count_is_capped_by_row_count():
    emb = np.eye(3, dtype=np.float32)
    assign, labels = cluster_labels(emb, ["a b c", "d e f", "g h i"], n_clusters=10)
    assert len(labels) == 3 and max(assign) < 3


# --- optimisation suggestions (labelled an extra) ------------------------------------------------

BAD = '''
def process(items, banned):
    banned = list(banned)
    out = ""
    for a in items:
        for b in items:
            out += str(b)
        if a in banned:
            out += "!"
        total = expensive(banned)
    return out, total
'''


def test_each_check_fires_on_code_that_deserves_it():
    checks = {f["check"] for f in analyze_source(BAD, "bad.py")}
    assert checks == {"nested_loop_same_collection", "membership_test_on_list",
                      "string_concat_in_loop", "loop_invariant_call"}


def test_findings_carry_file_and_line_and_a_suggestion():
    for f in analyze_source(BAD, "bad.py", start_line=100):
        assert f["file"] == "bad.py" and f["line"] >= 100
        assert f["location"] == f"bad.py:{f['line']}"
        assert f["detail"] and f["suggestion"] and f["severity"] in ("low", "medium", "high")


def test_clean_code_produces_nothing():
    good = ("def process(items, banned):\n"
            "    banned = set(banned)\n"
            "    parts = []\n"
            "    for a in items:\n"
            "        if a in banned:\n"
            "            parts.append(str(a))\n"
            "    return ''.join(parts)\n")
    assert analyze_source(good, "good.py") == []


def test_membership_on_a_set_is_not_flagged():
    src = ("def f(items):\n    seen = set()\n    for x in items:\n"
           "        if x in seen:\n            pass\n")
    assert [f for f in analyze_source(src, "s.py")
            if f["check"] == "membership_test_on_list"] == []


def test_unparseable_source_yields_no_findings():
    assert analyze_source("def f(:", "x.py") == []


# --- the agent loop ----------------------------------------------------------------------------------

class FakeService:
    """A SearchService stand-in: returns fixed hits, records what it was asked."""

    def __init__(self, hits=None):
        self.hits = hits if hits is not None else []
        self.queries = []

    def search(self, query, k=10, **kw):
        self.queries.append(query)
        return {"hits": self.hits[:k]}

    def describe(self):
        return {"source_root": None}


def agent_for(tree, hits=None, **kw):
    from src.agent.code_agent import CodeAgent
    return CodeAgent(FakeService(hits), structural=build_from_folder(tree), **kw)


def test_agent_answers_a_structural_question_in_one_step(tree):
    out = agent_for(tree).run("who calls alpha")
    assert out["route"]["kind"] == "structural"
    assert out["trace"][0]["tool"] == "who_calls"
    assert out["answers"] and all("location" in a for a in out["answers"])
    assert out["used_llm"] is False


def test_agent_never_exceeds_the_step_cap(tree):
    hits = [{"doc_id": f"d{i}", "location": f"a.py:{i}", "qualname": "alpha", "score": 0.5,
             "preview": ""} for i in range(3)]
    out = agent_for(tree, hits, max_steps=3).run("explain how the runner works")
    assert out["steps_run"] <= 3
    assert len([s for s in out["trace"] if not s["note"].startswith("skipped")]) <= 3


def test_loop_detection_refuses_a_repeated_step(tree):
    agent = agent_for(tree)
    # a step that would repeat is skipped and recorded, and costs no budget
    out = agent.run("who calls alpha")
    done = {(s["tool"], str(s["argument"])) for s in out["trace"]}
    assert len(done) == len([s for s in out["trace"] if not s["note"]])


def test_every_step_records_a_reason_and_a_duration(tree):
    out = agent_for(tree).run("who calls alpha")
    for step in out["trace"]:
        assert step["reason"] and isinstance(step["seconds"], float)
        assert step["step"] >= 1 and "tool" in step


def test_agent_stops_and_says_why(tree):
    out = agent_for(tree).run("who calls alpha")
    assert out["stop_reason"]
    assert any(word in out["stop_reason"]
               for word in ("evidence", "cap", "nothing further", "plan exhausted"))


def test_agent_works_with_no_structural_index():
    from src.agent.code_agent import CodeAgent
    hits = [{"doc_id": "d1", "location": "x.py:1-3", "qualname": "f", "score": 0.9, "preview": "def f"}]
    out = CodeAgent(FakeService(hits), structural=None).run("who calls f")
    assert out["steps_run"] >= 1 and out["answers"]


def test_a_failing_tool_does_not_kill_the_run(tree, monkeypatch):
    agent = agent_for(tree)

    def boom(tool, argument):
        raise RuntimeError("tool exploded")

    monkeypatch.setattr(agent, "_run_tool", boom)
    out = agent.run("who calls alpha")
    assert out["trace"] and "tool failed" in out["trace"][0]["note"]


def test_the_llm_planner_is_optional_and_off_by_default(tree):
    called = []

    def fake(prompt, **kw):
        called.append(prompt)
        return "one thing\nanother thing"

    agent_for(tree, call=fake).run("who calls alpha")
    assert called == []
    agent_for(tree, use_llm=True, call=fake).run("explain the runner")
    assert called, "with use_llm=True the planner should consult the wrapper"


def test_read_opens_a_file_slice(tree):
    agent = agent_for(tree)
    out = agent._read("a.py:8-12")
    assert out and out[0]["kind"] == "source"
    assert "def alpha" in out[0]["preview"]
    assert agent._read("nope.py:1-2") == []


# --- evaluation labels ---------------------------------------------------------------------------

def test_labels_are_built_without_the_ast(tree):
    from src.eval.agent_questions import build_questions, grep_calls, label_summary
    questions = build_questions(tree, max_questions=30)
    assert questions
    summary = label_summary(questions)
    assert set(summary["by_label_method"]) <= {"grep", "docstring", "manual"}
    assert summary["by_kind"]
    for q in questions:
        assert q["expected"] and q["label_note"]
        assert all(":" in e for e in q["expected"])
    # grep and the AST are genuinely different methods: both find alpha's call sites
    assert {c.split(":")[0] for c in grep_calls(tree, "alpha")} == {"a.py", "helpers.py"}


def test_grep_labels_ignore_the_definition_line_and_comments(tmp_path):
    from src.eval.agent_questions import grep_calls
    (tmp_path / "m.py").write_text(
        "def target():\n    return 1\n\n# target() in a comment\n\ndef c():\n    return target()\n",
        encoding="utf-8")
    hits = grep_calls(tmp_path, "target")
    assert hits == ["m.py:7"], hits


def test_docstring_questions_use_the_authors_words(tree):
    from src.eval.agent_questions import docstring_questions
    items = docstring_questions(tree)
    assert items
    questions = {i["question"] for i in items}
    assert "Call alpha then shout." in questions          # the docstring, verbatim
    for i in items:
        assert ":" in i["answer"] and i["question"]


def test_very_short_docstrings_are_filtered_out(tree):
    """A three-word docstring makes a hopeless query; the length floor is deliberate."""
    from src.eval.agent_questions import docstring_questions
    default = {i["question"] for i in docstring_questions(tree)}
    relaxed = {i["question"] for i in docstring_questions(tree, min_len=10)}
    assert "Add up the items." not in default              # 17 characters, below the floor
    assert "Add up the items." in relaxed


def test_scoring_helpers():
    from src.bench_agent import locations_of, score
    preds = locations_of([{"location": "a.py:10"}, {"location": "b.py:2"}], exact_lines=False)
    assert preds == {"a.py", "b.py"}
    got = score(["a.py", "z.py"], ["a.py:10"], k=2, exact_lines=False)
    assert got["precision_at_k"] == 0.5 and got["recall"] == 1.0 and got["hit"] is True
    miss = score(["z.py"], ["a.py:10"], k=2, exact_lines=False)
    assert miss["hit"] is False and miss["recall"] == 0.0


# --- end to end through the CLI and API ------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_index(tmp_path_factory):
    out = tmp_path_factory.mktemp("ci") / "idx"
    proc = subprocess.run([sys.executable, "src/build_index.py", "--source", str(EXAMPLES),
                           "--mock-encoder", "--out", str(out)], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    tagged = subprocess.run([sys.executable, "src/build_categories.py", "--index", str(out),
                             "--clusters", "4"], cwd=ROOT, capture_output=True, text=True)
    assert tagged.returncode == 0, tagged.stdout + tagged.stderr
    return out


def run_cli(*args):
    proc = subprocess.run([sys.executable, "src/cli.py", *[str(a) for a in args]], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_cli_answers_a_structural_question_exactly(demo_index):
    out = run_cli("who calls tokenize", "--index", demo_index, "--mock-encoder",
                  "--structural-root", str(EXAMPLES))
    assert "route   : structural (who_calls)" in out
    assert "search.py:" in out and "stats.py:" in out


def test_cli_shows_the_route_on_a_dense_search(demo_index):
    out = run_cli("remove accents from text", "-k", 2, "--index", demo_index, "--mock-encoder")
    assert "route   : nl_intent -> dense" in out


def test_cli_no_router_skips_classification(demo_index):
    out = run_cli("who calls tokenize", "-k", 2, "--index", demo_index, "--mock-encoder",
                  "--no-router", "--structural-root", str(EXAMPLES))
    assert "route" not in out.split("query")[1][:200]


def test_cli_agent_prints_a_trace(demo_index):
    out = run_cli("who calls tokenize", "--agent", "--index", demo_index, "--mock-encoder",
                  "--structural-root", str(EXAMPLES))
    assert "planner : deterministic (no LLM)" in out
    assert "trace   :" in out and "stopped :" in out
    assert "why:" in out


def test_cli_category_filter(demo_index):
    payload = json.loads(run_cli("count words", "-k", 3, "--index", demo_index, "--mock-encoder",
                                 "--category", "string_processing", "--json"))
    res = payload["result"]
    assert res["category_filter"] == "string_processing"
    assert res["hits"] and all(h["ast_family"] == "string_processing" for h in res["hits"])


def test_hits_carry_category_tags(demo_index):
    payload = json.loads(run_cli("tokenize text", "-k", 2, "--index", demo_index,
                                 "--mock-encoder", "--json"))
    for hit in payload["result"]["hits"]:
        assert hit["ast_family"] and hit["ast_reason"]
        assert "cluster_label" in hit


def test_api_exposes_route_structural_agent_and_categories(demo_index, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import src.api as api
    monkeypatch.setenv("INDEX_DIR", str(demo_index))
    monkeypatch.setenv("MOCK_ENCODER", "1")
    monkeypatch.setattr(api, "_service", None)
    monkeypatch.setattr(api, "_error", None)
    monkeypatch.setattr(api, "_services", {})
    with TestClient(api.app) as client:
        route = client.get("/route", params={"q": "who calls tokenize"}).json()
        assert route["kind"] == "structural" and route["intent"] == "who_calls"

        cats = client.get("/categories").json()
        assert cats["families"] and sum(cats["families"].values()) == 19

        res = client.post("/search", json={"query": "count words", "k": 3,
                                           "category": "string_processing",
                                           "suggestions": True}).json()
        assert res["hits"] and "route" in res and "suggestions" in res
        assert all(h["ast_family"] == "string_processing" for h in res["hits"])

        st = client.get("/structural", params={"intent": "who_calls", "subject": "tokenize",
                                               "structural_root": str(EXAMPLES)}).json()
        assert st["results"] and {r["file"] for r in st["results"]} >= {"search.py"}

        bad = client.get("/structural", params={"intent": "nonsense", "subject": "x",
                                                "structural_root": str(EXAMPLES)})
        assert bad.status_code == 400

        ag = client.post("/agent", json={"question": "who calls tokenize", "k": 3,
                                         "structural_root": str(EXAMPLES)}).json()
        assert ag["trace"] and ag["steps_run"] >= 1 and ag["stop_reason"]
        assert ag["used_llm"] is False


def test_the_page_still_has_no_external_dependencies(demo_index, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import src.api as api
    monkeypatch.setenv("INDEX_DIR", str(demo_index))
    monkeypatch.setenv("MOCK_ENCODER", "1")
    monkeypatch.setattr(api, "_service", None)
    monkeypatch.setattr(api, "_error", None)
    monkeypatch.setattr(api, "_services", {})
    with TestClient(api.app) as client:
        body = client.get("/").text
    for forbidden in ("http://", "https://", "cdn.", "<script src="):
        assert forbidden not in body
    for needed in ('id="cat"', 'id="agent"', 'id="sugg"', "renderAgent", "TAG_PARTS",
                   "HIT_SECTIONS", "BANNERS"):
        assert needed in body, needed


def test_agent_benchmark_runs_and_reports_by_label_method(demo_index, tmp_path):
    out = tmp_path / "agent.json"
    proc = subprocess.run([sys.executable, "src/bench_agent.py", "--index", str(demo_index),
                           "--source", str(EXAMPLES), "--mock-encoder", "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["label_summary"]["n"] >= 10
    assert set(data["by_label_method"]) <= {"grep", "docstring", "manual"}
    for side in ("dense", "agent"):
        assert 0.0 <= data["overall"][side]["precision_at_k"] <= 1.0
        assert 0.0 <= data["overall"][side]["recall"] <= 1.0
    assert data["overall"]["agent"]["median_steps"] <= 6
