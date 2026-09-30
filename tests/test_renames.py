"""Rename and move tracking in the git-history ingester (--track-renames).

Each scenario builds a real local git repository in a temp directory (no network, no model) and checks
the lineage bookkeeping, the recorded link (kind, similarity) and how many links were made."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.versioning.git_history import ingest
from src.versioning.renames import (find_links, git_file_renames, jaccard, match_functions,
                                    normalized_tokens)

ROOT = Path(__file__).resolve().parents[1]

ADD = '''def add(values, offset):
    """Add an offset to every value."""
    result = []
    for value in values:
        result.append(value + offset)
    return result
'''

SCALE = '''def scale(values, factor):
    """Multiply every value by a factor."""
    result = []
    for value in values:
        result.append(value * factor)
    return result
'''

TOTAL = '''def total_price(items, tax):
    """Sum the item prices and apply a tax rate."""
    subtotal = 0
    for item in items:
        subtotal += item["price"] * item["quantity"]
    return subtotal * (1 + tax)
'''

CLAMP = '''def clamp(value, lowest, highest):
    """Constrain a value to a range."""
    if value < lowest:
        return lowest
    if value > highest:
        return highest
    return value
'''

OTHER = '''def describe(name, count):
    """Say how many of something there are."""
    label = name if count == 1 else name + "s"
    return str(count) + " " + label
'''


def git(repo, *args):
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 0, f"git {args}: {proc.stderr}"
    return proc.stdout


def new_repo(tmp_path, name="r"):
    repo = tmp_path / name
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "T")
    return repo


def commit(repo, message, **files):
    """Write files (path -> text; None deletes) and commit."""
    for path, text in files.items():
        p = repo / path
        if text is None:
            p.unlink()
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def by_lineage(rows):
    out = {}
    for r in rows:
        out.setdefault(r["lineage_id"], []).append(r)
    return out


# --- the primitives ---------------------------------------------------------------------------------

def test_normalised_tokens_ignore_comments_layout_and_the_functions_own_name():
    a = normalized_tokens("def foo(x):\n    # note\n    return foo(x - 1) if x else 0\n", "foo")
    b = normalized_tokens("def bar(x):\n    return bar(x - 1) if x else 0  # other note\n", "bar")
    assert a == b and "<self>" in a and "foo" not in a


def test_jaccard_of_identical_and_disjoint_and_partial_token_lists():
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0
    assert jaccard(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2 / 4)
    assert jaccard([], []) == 0.0


def test_matching_is_one_to_one_and_respects_the_threshold_and_kind():
    def chunk(text, name, kind="function"):
        return {"text": text, "name": name, "kind": kind, "file": "m.py", "qualname": name}
    removed = {"m.py::old": chunk(TOTAL, "total_price")}
    renamed = TOTAL.replace("total_price", "compute_total")
    added = {"m.py::new": chunk(renamed, "compute_total"),
             "m.py::twin": chunk(renamed.replace("compute_total", "twin"), "twin")}
    matches, skipped = match_functions(removed, added)
    assert skipped is False and len(matches) == 1 and matches[0][0] == "m.py::old"
    assert matches[0][2] == 1.0
    # a different kind is never linked, and a dissimilar body is not linked
    assert match_functions(removed, {"m.py::c": chunk(renamed, "compute_total", "class")})[0] == []
    assert match_functions(removed, {"m.py::d": chunk(CLAMP, "clamp")})[0] == []


def test_short_bodies_are_never_matched():
    tiny = {"text": "def f(): pass\n", "name": "f", "kind": "function", "file": "a.py",
            "qualname": "f"}
    other = dict(tiny, name="g", text="def g(): pass\n", qualname="g")
    assert match_functions({"a.py::f": tiny}, {"a.py::g": other})[0] == []


# --- scenario 1: a file is renamed ---------------------------------------------------------------------

def build_file_rename(tmp_path):
    repo = new_repo(tmp_path)
    commit(repo, "add module", **{"pkg/old.py": ADD + "\n\n" + SCALE})
    git(repo, "mv", "pkg/old.py", "pkg/new.py")
    git(repo, "commit", "-q", "-m", "rename module")
    commit(repo, "edit scale", **{"pkg/new.py": ADD + "\n\n" + SCALE.replace("value * factor",
                                                                             "value * factor * 1")})
    return repo


def test_git_reports_the_file_rename(tmp_path):
    repo = build_file_rename(tmp_path)
    shas = git(repo, "rev-list", "--reverse", "HEAD").split()
    assert git_file_renames(repo, shas[0], shas[1]) == [("pkg/old.py", "pkg/new.py", 1.0)]


def test_without_tracking_a_renamed_file_starts_new_lineages(tmp_path):
    rows, _, stats = ingest(build_file_rename(tmp_path), progress=False)
    assert stats["lineages"] == 4 and "track_renames" not in stats
    assert all("link_kind" not in r for r in rows)


def test_with_tracking_a_renamed_file_keeps_its_lineages(tmp_path):
    rows, per_commit, stats = ingest(build_file_rename(tmp_path), progress=False, track_renames=True)
    assert stats["lineages"] == 2 and stats["rows"] == 6
    assert stats["track_renames"]["links"] == 2 and stats["track_renames"]["by_kind"] == {"rename": 2, "move": 0}
    lineages = by_lineage(rows)
    assert set(lineages) == {"pkg/old.py::add", "pkg/old.py::scale"}
    add = lineages["pkg/old.py::add"]
    assert [r["version"] for r in add] == [1, 2, 3]
    assert add[1]["file"] == "pkg/new.py" and add[1]["change"] == "unchanged"      # same text, new home
    assert (add[1]["link_kind"], add[1]["link_scope"], add[1]["link_similarity"]) == ("rename", "file", 1.0)
    assert add[1]["link_from"] == "pkg/old.py::add"
    assert "link_kind" not in add[0] and "link_kind" not in add[2]
    scale = lineages["pkg/old.py::scale"]
    assert scale[2]["change"] == "modified" and scale[2]["file"] == "pkg/new.py"
    assert [c["links"] for c in per_commit] == [0, 2, 0]
    assert all(c["removed"] == 0 for c in per_commit)          # nothing was actually removed


# --- scenario 2: a function is renamed in place ---------------------------------------------------------

def build_function_rename(tmp_path):
    repo = new_repo(tmp_path)
    commit(repo, "add", **{"tools.py": TOTAL + "\n\n" + OTHER})
    commit(repo, "rename", **{"tools.py": TOTAL.replace("total_price", "compute_total") + "\n\n" + OTHER})
    # a genuinely different replacement must NOT be linked
    commit(repo, "rewrite", **{"tools.py": CLAMP + "\n\n" + OTHER})
    return repo


def test_a_renamed_function_keeps_its_lineage_and_records_the_link(tmp_path):
    rows, per_commit, stats = ingest(build_function_rename(tmp_path), progress=False, track_renames=True)
    lineages = by_lineage(rows)
    chain = lineages["tools.py::total_price"]
    assert [r["qualname"] for r in chain] == ["total_price", "compute_total"]
    assert chain[1]["change"] == "modified"                       # the text (its name) did change
    assert (chain[1]["link_kind"], chain[1]["link_scope"]) == ("rename", "function")
    assert chain[1]["link_similarity"] == 1.0 and chain[1]["link_from"] == "tools.py::total_price"
    # the untouched neighbour is not affected
    assert [r["version"] for r in lineages["tools.py::describe"]] == [1, 2, 3]
    # the rewrite is a new lineage, not a link: exactly ONE link was made in the whole history
    assert "tools.py::clamp" in lineages and len(lineages["tools.py::clamp"]) == 1
    assert stats["track_renames"]["links"] == 1 and stats["track_renames"]["by_kind"]["rename"] == 1
    assert per_commit[2]["removed"] == 1


def test_a_lightly_edited_rename_links_but_a_heavily_edited_one_does_not(tmp_path):
    light = new_repo(tmp_path, "light")
    commit(light, "add", **{"t.py": TOTAL})
    commit(light, "rename+tweak", **{"t.py": TOTAL.replace("total_price", "grand_total")
                                     .replace("(1 + tax)", "(1.0 + tax)")})
    _, _, stats = ingest(light, progress=False, track_renames=True)
    assert stats["track_renames"]["links"] == 1
    heavy = new_repo(tmp_path, "heavy")
    commit(heavy, "add", **{"t.py": TOTAL})
    commit(heavy, "rename+rewrite", **{"t.py": '''def grand_total(basket, rate):
    values = [entry["price"] for entry in basket]
    values.sort()
    return sum(values[:3]) * rate
'''})
    _, _, stats = ingest(heavy, progress=False, track_renames=True)
    assert stats["track_renames"]["links"] == 0


def test_the_threshold_is_adjustable(tmp_path):
    repo = new_repo(tmp_path)
    commit(repo, "add", **{"t.py": TOTAL})
    commit(repo, "rename+tweak", **{"t.py": TOTAL.replace("total_price", "grand_total")
                                    .replace("(1 + tax)", "(1.0 + tax)")})
    _, _, strict = ingest(repo, progress=False, track_renames=True, rename_threshold=0.99)
    assert strict["track_renames"]["links"] == 0


# --- scenario 3: a function moves to another file -------------------------------------------------------

def build_move(tmp_path):
    repo = new_repo(tmp_path)
    commit(repo, "add", **{"a.py": CLAMP + "\n\n" + OTHER, "b.py": ADD})
    commit(repo, "move clamp", **{"a.py": OTHER, "b.py": ADD + "\n\n" + CLAMP})
    return repo


def test_without_tracking_a_move_is_a_removal_plus_an_addition(tmp_path):
    rows, per_commit, _ = ingest(build_move(tmp_path), progress=False)
    assert per_commit[1]["removed"] == 1 and per_commit[1]["added"] == 1
    assert "a.py::clamp" in by_lineage(rows) and "b.py::clamp" in by_lineage(rows)


def test_a_function_moved_across_files_keeps_its_lineage_as_a_move(tmp_path):
    rows, per_commit, stats = ingest(build_move(tmp_path), progress=False, track_renames=True)
    chain = by_lineage(rows)["a.py::clamp"]
    assert [r["file"] for r in chain] == ["a.py", "b.py"]
    assert (chain[1]["link_kind"], chain[1]["link_scope"]) == ("move", "function")
    assert chain[1]["link_similarity"] == 1.0 and chain[1]["link_from"] == "a.py::clamp"
    assert "b.py::clamp" not in by_lineage(rows)
    assert stats["track_renames"]["by_kind"] == {"rename": 0, "move": 1}
    assert (per_commit[1]["added"], per_commit[1]["removed"]) == (0, 0)


def test_a_copy_that_leaves_the_original_is_not_a_move(tmp_path):
    repo = new_repo(tmp_path)
    commit(repo, "add", **{"a.py": CLAMP, "b.py": ADD})
    commit(repo, "copy clamp", **{"b.py": ADD + "\n\n" + CLAMP})
    _, _, stats = ingest(repo, progress=False, track_renames=True)
    assert stats["track_renames"]["links"] == 0


def test_a_file_rename_plus_a_function_rename_in_one_commit_is_a_rename(tmp_path):
    repo = new_repo(tmp_path)
    commit(repo, "add", **{"old.py": TOTAL + "\n\n" + OTHER})
    git(repo, "mv", "old.py", "new.py")
    (repo / "new.py").write_text(TOTAL.replace("total_price", "compute_total") + "\n\n" + OTHER)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "rename both")
    rows, _, stats = ingest(repo, progress=False, track_renames=True)
    assert stats["lineages"] == 2
    chain = by_lineage(rows)["old.py::total_price"]
    assert chain[1]["qualname"] == "compute_total" and chain[1]["link_kind"] == "rename"
    assert stats["track_renames"]["by_kind"]["move"] == 0


def test_find_links_needs_both_a_removal_and_an_addition(tmp_path):
    repo = build_move(tmp_path)
    assert find_links(repo, "a", "b", {"x": {}}, {"x": {}}, lambda f, q: f"{f}::{q}") == ([], False)


# --- the index and the CLI --------------------------------------------------------------------------------

def test_links_reach_the_versioned_index(tmp_path):
    from src.runtime_index import HashingQueryEncoder, RuntimeIndex
    from src.versioning.version_index import build_versioned_index, embeddings_for
    rows, _, _ = ingest(build_move(tmp_path), progress=False, track_renames=True)
    enc = HashingQueryEncoder(32)
    emb, _ = embeddings_for(rows, enc.encode_docs, cache={})
    build_versioned_index(tmp_path / "idx", rows, emb, {"model": "mock/hashing-encoder"})
    idx = RuntimeIndex.load(tmp_path / "idx")
    linked = [v for v in idx.versions if v.get("link_kind")]
    assert len(linked) == 1 and linked[0]["link_kind"] == "move" and linked[0]["snippet_id"] == "a.py::clamp"


def test_build_history_index_takes_the_flag_and_reports_the_links(tmp_path):
    repo = build_move(tmp_path)

    def run(*extra):
        proc = subprocess.run([sys.executable, "src/build_history_index.py", "--repo", str(repo),
                               "--mock-encoder", "--out", str(tmp_path / "hidx"),
                               "--queries-out", str(tmp_path / "q.json"),
                               "--stats-out", str(tmp_path / "s.json"), *extra],
                              cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc.stdout
    assert "rename tracking" not in run()
    assert json.loads((tmp_path / "s.json").read_text())["stats"].get("track_renames") is None
    out = run("--track-renames")
    assert "rename tracking: 1 link(s)" in out
    assert json.loads((tmp_path / "s.json").read_text())["stats"]["track_renames"]["links"] == 1
