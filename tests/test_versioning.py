"""P1 + Bonus: content hashing, the version fixture, incremental rebuilds and lineage handling.

All mocked (hashing encoder). The point of most of these is that the *counts* are right: an incremental
rebuild that reused more than it should would silently serve stale embeddings.
"""
import numpy as np
import pytest

from src.runtime_index import HashingQueryEncoder, RuntimeIndex
from src.versioning.content_hash import content_hash, normalize_code, strip_comments
from src.versioning.fixture import build_fixture, fixture_stats
from src.versioning.version_index import (build_versioned_index, collapse_lineages, doc_id_for,
                                          duplication_at_k, embeddings_for, history, parse_doc_id,
                                          rows_for, version_rows)

CODE = "def f(x):\n    return x + 1\n"


# --- content hash --------------------------------------------------------------------------------

def test_comments_and_blank_lines_do_not_change_the_hash():
    other = "def f(x):\n\n    # add one to x\n    return x + 1   \n\n"
    assert content_hash(CODE) == content_hash(other)


def test_line_endings_and_tabs_do_not_change_the_hash():
    assert content_hash(CODE) == content_hash("def f(x):\r\n\treturn x + 1\r\n")


def test_a_rename_changes_the_hash():
    assert content_hash(CODE) != content_hash("def f(y):\n    return y + 1\n")


def test_a_changed_operator_or_constant_changes_the_hash():
    assert content_hash(CODE) != content_hash("def f(x):\n    return x - 1\n")
    assert content_hash(CODE) != content_hash("def f(x):\n    return x + 2\n")


def test_indentation_is_significant():
    """Python indentation carries meaning, so normalisation must not erase it."""
    a = "def f(x):\n    if x:\n        return 1\n    return 0\n"
    b = "def f(x):\n    if x:\n        return 1\n        return 0\n"
    assert content_hash(a) != content_hash(b)


def test_a_hash_inside_a_string_literal_is_not_treated_as_a_comment():
    assert strip_comments('s = "a # b"  # real comment') == 's = "a # b"  '
    assert content_hash('s = "a # b"\n') != content_hash('s = "a"\n')


def test_escaped_quote_inside_a_literal_does_not_end_it():
    assert strip_comments('s = "a\\" # b"') == 's = "a\\" # b"'


def test_normalize_collapses_inner_spaces_but_keeps_indent():
    assert normalize_code("    a  =   1\n") == "    a = 1"


# --- fixture -------------------------------------------------------------------------------------

def test_fixture_is_deterministic_for_a_seed():
    a = build_fixture(12, 4, seed=7)
    b = build_fixture(12, 4, seed=7)
    assert a == b
    assert build_fixture(12, 4, seed=8) != a


def test_fixture_shape_and_version_numbering():
    fx = build_fixture(20, 5, seed=1)
    assert len(fx["snippets"]) == 20
    for s in fx["snippets"]:
        assert [v["version"] for v in s["versions"]] == [1, 2, 3, 4, 5]
        assert s["versions"][0]["mutation"] == "initial"
        for v in s["versions"]:
            assert v["content_hash"] == content_hash(v["text"])


def test_unchanged_versions_repeat_the_text_and_the_hash():
    fx = build_fixture(60, 4, seed=2, unchanged_rate=0.5)
    checked = 0
    for s in fx["snippets"]:
        for prev, cur in zip(s["versions"], s["versions"][1:]):
            if cur["mutation"] == "unchanged":
                assert cur["text"] == prev["text"]
                assert cur["content_hash"] == prev["content_hash"]
                checked += 1
            else:
                assert cur["content_hash"] != prev["content_hash"], cur["mutation"]
    assert checked > 0, "the fixture produced no unchanged versions to check"


def test_every_mutation_kind_actually_changes_the_text():
    """A mutation that silently no-ops would inflate the measured reuse rate."""
    fx = build_fixture(120, 5, seed=3)
    seen = set()
    for s in fx["snippets"]:
        for prev, cur in zip(s["versions"], s["versions"][1:]):
            seen.add(cur["mutation"])
            if cur["mutation"] != "unchanged":
                assert cur["text"] != prev["text"], cur["mutation"]
    assert {"rename", "logic", "add_lines", "remove_lines", "unchanged"} <= seen


def test_fixture_stats_match_the_fixture():
    fx = build_fixture(40, 4, seed=5, unchanged_rate=0.3)
    st = fixture_stats(fx)
    assert st["later_versions"] == 40 * 3
    assert st["unchanged"] == sum(1 for s in fx["snippets"] for v in s["versions"][1:]
                                  if v["mutation"] == "unchanged")
    assert sum(st["per_mutation"].values()) == st["later_versions"]


def test_version_specific_queries_name_a_token_the_target_version_contains():
    fx = build_fixture(40, 4, seed=6)
    specific = [q for q in fx["queries"] if q["kind"] == "version_specific"]
    assert specific, "no version-specific queries generated"
    by_id = {s["snippet_id"]: s for s in fx["snippets"]}
    for q in specific:
        versions = by_id[q["target_snippet"]]["versions"]
        target = next(v for v in versions if v["version"] == q["target_version"])
        assert q["token"] in target["text"]
        assert q["token"] in q["text"]
        earlier = [v for v in versions if v["version"] < q["target_version"]]
        assert all(q["token"] not in v["text"] for v in earlier)      # it really is new at that version
        # the acceptable set is exactly the versions that still contain it
        assert q["acceptable_versions"] == [v["version"] for v in versions
                                            if q["token"] in v["text"]]


# --- incremental rebuild -------------------------------------------------------------------------

def encoder():
    enc = HashingQueryEncoder(dim=32)
    calls = []

    def encode(texts, batch_size=32):
        calls.append(list(texts))
        return enc.encode_docs(texts, batch_size=batch_size)
    return encode, calls


def test_rows_for_flattens_every_version_in_order():
    fx = build_fixture(5, 3, seed=0)
    rows = rows_for(fx)
    assert len(rows) == 15
    assert rows[0]["doc_id"] == doc_id_for("snip0000", 1)
    assert [r["version"] for r in rows[:3]] == [1, 2, 3]
    assert parse_doc_id(rows[1]["doc_id"]) == ("snip0000", 2)


def test_rows_for_can_select_versions():
    rows = rows_for(build_fixture(5, 3, seed=0), versions=[2])
    assert len(rows) == 5 and {r["version"] for r in rows} == {2}


def test_parse_doc_id_rejects_a_plain_id():
    with pytest.raises(ValueError, match="not a versioned document id"):
        parse_doc_id("doc123")


def test_embeddings_for_encodes_each_distinct_hash_once():
    fx = build_fixture(30, 4, seed=4, unchanged_rate=0.5)
    rows = rows_for(fx)
    encode, calls = encoder()
    emb, stats = embeddings_for(rows, encode)
    distinct = len({r["content_hash"] for r in rows})
    assert emb.shape == (len(rows), 32)
    assert stats["recomputed"] == distinct
    assert stats["reused"] == len(rows) - distinct
    assert sum(len(c) for c in calls) == distinct        # the encoder saw each text exactly once
    assert stats["distinct_hashes"] == distinct


def test_identical_rows_get_identical_embeddings():
    fx = build_fixture(20, 4, seed=9, unchanged_rate=0.6)
    rows = rows_for(fx)
    encode, _ = encoder()
    emb, _ = embeddings_for(rows, encode)
    by_hash = {}
    for i, r in enumerate(rows):
        by_hash.setdefault(r["content_hash"], []).append(i)
    repeated = [idxs for idxs in by_hash.values() if len(idxs) > 1]
    assert repeated, "fixture had no repeated hashes to check"
    for idxs in repeated:
        for j in idxs[1:]:
            assert np.array_equal(emb[idxs[0]], emb[j])


def test_a_persistent_cache_makes_the_second_build_free():
    rows = rows_for(build_fixture(15, 3, seed=11))
    encode, calls = encoder()
    cache = {}
    embeddings_for(rows, encode, cache=cache)
    first_calls = sum(len(c) for c in calls)
    emb2, stats2 = embeddings_for(rows, encode, cache=cache)
    assert stats2["recomputed"] == 0
    assert stats2["reused"] == len(rows)
    assert sum(len(c) for c in calls) == first_calls        # nothing was encoded the second time
    assert emb2.shape[0] == len(rows)


def test_incremental_reuse_is_at_least_the_unchanged_count():
    """The fixture knows how many later versions are byte-identical; incremental reuse must cover them
    (it may exceed them, when an edit happens to reproduce an earlier version's text)."""
    from src.bench_versions import expected_unchanged, snapshot_rows
    fx = build_fixture(50, 4, seed=13, unchanged_rate=0.4)
    encode, _ = encoder()
    cache = {}
    for v in range(1, 5):
        rows = snapshot_rows(fx, v)
        assert len(rows) == 50
        _, stats = embeddings_for(rows, encode, cache=cache)
        if v > 1:
            assert stats["reused"] >= expected_unchanged(fx, v)


def test_embeddings_for_rejects_a_bad_encoder():
    rows = rows_for(build_fixture(3, 2, seed=0))

    def short(texts, batch_size=32):
        return np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="vectors for"):
        embeddings_for(rows, short)


# --- lineage / evolutionary retrieval ------------------------------------------------------------

def built_index(tmp_path, n_snippets=8, n_versions=3, seed=0):
    fx = build_fixture(n_snippets, n_versions, seed=seed)
    rows = rows_for(fx)
    enc = HashingQueryEncoder(dim=32)
    emb, _ = embeddings_for(rows, enc.encode_docs)
    build_versioned_index(tmp_path, rows, emb, {"preset": "mock", "model": "mock/hashing-encoder",
                                                "revision": "n/a", "query_prefix": "", "dim": 32})
    return fx, RuntimeIndex.load(tmp_path)


def test_versioned_index_carries_lineage_metadata(tmp_path):
    fx, idx = built_index(tmp_path)
    assert idx.versions is not None and len(idx.versions) == len(idx.doc_ids)
    assert idx.manifest["kind"] == "versioned"
    assert {v["snippet_id"] for v in idx.versions} == {s["snippet_id"] for s in fx["snippets"]}
    assert all("content_hash" in v for v in idx.versions)


def test_version_rows_selects_one_version(tmp_path):
    fx, idx = built_index(tmp_path, n_snippets=8, n_versions=3)
    rows = version_rows(idx, 2)
    assert len(rows) == 8
    assert all(idx.versions[r]["version"] == 2 for r in rows)


def test_version_rows_needs_a_versioned_index(tmp_path):
    from src.runtime_index import write_index
    write_index(tmp_path, np.zeros((2, 4), dtype=np.float32), ["a", "b"], ["x", "y"],
                {"model": "mock/hashing-encoder"})
    with pytest.raises(ValueError, match="not a versioned index"):
        version_rows(RuntimeIndex.load(tmp_path), 1)


def test_collapse_keeps_the_best_version_per_lineage_in_score_order(tmp_path):
    _, idx = built_index(tmp_path)
    hits = [(idx.doc_ids[i], 1.0 - 0.01 * i, i) for i in range(len(idx.doc_ids))]
    collapsed = collapse_lineages(hits, idx)
    lineages = [idx.versions[row]["snippet_id"] for _, _, row in collapsed]
    assert len(lineages) == len(set(lineages))                       # one entry per lineage
    scores = [s for _, s, _ in collapsed]
    assert scores == sorted(scores, reverse=True)
    # the kept row for each lineage is its highest-scoring one
    best = {}
    for doc, score, row in hits:
        sid = idx.versions[row]["snippet_id"]
        if sid not in best or score > best[sid][1]:
            best[sid] = (doc, score, row)
    assert {c[0] for c in collapsed} == {v[0] for v in best.values()}


def test_collapse_respects_k(tmp_path):
    _, idx = built_index(tmp_path)
    hits = [(idx.doc_ids[i], 1.0 - 0.01 * i, i) for i in range(len(idx.doc_ids))]
    assert len(collapse_lineages(hits, idx, k=3)) == 3


def test_duplication_counts_repeat_versions_of_one_lineage(tmp_path):
    _, idx = built_index(tmp_path, n_snippets=8, n_versions=3)
    rows_same_lineage = [i for i, v in enumerate(idx.versions)
                         if v["snippet_id"] == idx.versions[0]["snippet_id"]]
    hits = [(idx.doc_ids[r], 1.0, r) for r in rows_same_lineage]
    d = duplication_at_k(hits, idx, k=3)
    assert d["distinct_lineages"] == 1
    assert d["duplicate_slots"] == len(rows_same_lineage) - 1
    assert d["duplicate_pct"] > 0
    # and collapsing removes exactly those slots
    assert len(collapse_lineages(hits, idx)) == 1


def test_history_is_oldest_first_and_complete(tmp_path):
    fx, idx = built_index(tmp_path, n_snippets=6, n_versions=4)
    sid = fx["snippets"][2]["snippet_id"]
    rows = history(idx, sid)
    assert [r["version"] for r in rows] == [1, 2, 3, 4]
    assert all(r["snippet_id"] == sid for r in rows)
    assert [idx.doc_texts[r["row"]] for r in rows] == [v["text"] for v in fx["snippets"][2]["versions"]]


def test_history_of_an_unknown_snippet_raises(tmp_path):
    _, idx = built_index(tmp_path)
    with pytest.raises(KeyError, match="no snippet"):
        history(idx, "snip9999")
