"""The served index: writing, loading, hash verification, search, and the mock encoder.

All mocked: no model, no download. These cover the file format the official run exports and every
consumer (CLI, API, benchmarks) reads.
"""
import json

import numpy as np
import pytest

from src.runtime_index import (CORPUS_TEXTS, EMBEDDINGS, MANIFEST, HashingQueryEncoder, RuntimeIndex,
                               make_doc_encoder, physical_cores, set_cpu_threads, sha256_file,
                               write_index)

META = {"preset": "mock", "model": "mock/hashing-encoder", "revision": "n/a", "query_prefix": "Q: ",
        "doc_prefix": "", "dense_variant": "registry+full", "max_seq_length": 0, "dtype": "fp32",
        "trust_remote_code": False, "expect_eos": False}


def tiny(tmp_path, n=6, dim=8, versions=None):
    rng = np.random.RandomState(0)
    emb = rng.randn(n, dim).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = [f"d{i}" for i in range(n)]
    texts = [f"text number {i}" for i in range(n)]
    manifest = write_index(tmp_path, emb, ids, texts, dict(META), versions=versions)
    return emb, ids, texts, manifest


def test_write_then_load_roundtrip(tmp_path):
    emb, ids, texts, manifest = tiny(tmp_path)
    assert manifest["n_docs"] == 6 and manifest["dim"] == 8 and manifest["kind"] == "flat"
    assert manifest["stored_dtype"] == "float16"
    idx = RuntimeIndex.load(tmp_path)
    assert idx.doc_ids == ids and idx.doc_texts == texts and idx.versions is None
    # stored as fp16, so expect fp16 precision rather than exact equality
    assert np.allclose(idx.E, emb, atol=1e-3)
    assert idx.row_of("d3") == 3 and idx.row_of("nope") is None


def test_manifest_records_a_correct_hash_per_file(tmp_path):
    _, _, _, manifest = tiny(tmp_path)
    assert set(manifest["files"]) == {EMBEDDINGS, "doc_ids.json", CORPUS_TEXTS}
    for name, rec in manifest["files"].items():
        assert rec["sha256"] == sha256_file(tmp_path / name)
        assert rec["bytes"] == (tmp_path / name).stat().st_size
    assert RuntimeIndex.load(tmp_path).verify_files() == []


def test_verify_files_catches_a_changed_file(tmp_path):
    tiny(tmp_path)
    idx = RuntimeIndex.load(tmp_path)
    (tmp_path / CORPUS_TEXTS).write_text(json.dumps(["tampered"] * 6), encoding="utf-8")
    problems = idx.verify_files()
    assert len(problems) == 1 and CORPUS_TEXTS in problems[0] and "sha256" in problems[0]


def test_verify_files_catches_a_missing_file(tmp_path):
    tiny(tmp_path)
    idx = RuntimeIndex.load(tmp_path)
    (tmp_path / CORPUS_TEXTS).unlink()
    assert any("missing" in p for p in idx.verify_files())


def test_load_without_a_manifest_names_the_build_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        RuntimeIndex.load(tmp_path)


def test_write_index_rejects_mismatched_lengths(tmp_path):
    emb = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="do not match"):
        write_index(tmp_path, emb, ["a", "b"], ["x", "y", "z"], dict(META))
    with pytest.raises(ValueError, match="one entry per row"):
        write_index(tmp_path, emb, ["a", "b", "c"], ["x", "y", "z"], dict(META),
                    versions=[{"snippet_id": "a", "version": 1, "content_hash": "h"}])


def test_search_returns_best_first_and_honours_k(tmp_path):
    emb, ids, _, _ = tiny(tmp_path, n=10, dim=8)
    idx = RuntimeIndex.load(tmp_path)
    hits = idx.search(emb[4], k=3)
    assert len(hits) == 3
    assert hits[0][0] == "d4"                                 # a row is its own nearest neighbour
    assert [h[1] for h in hits] == sorted([h[1] for h in hits], reverse=True)
    assert [h[2] for h in hits][0] == 4


def test_search_restricted_to_rows_only_returns_those_rows(tmp_path):
    emb, _, _, _ = tiny(tmp_path, n=10, dim=8)
    idx = RuntimeIndex.load(tmp_path)
    hits = idx.search(emb[4], k=5, rows=np.array([1, 3, 7]))
    assert {h[2] for h in hits} == {1, 3, 7}                  # 4 is excluded even though it is best
    assert len(hits) == 3                                     # k is capped by the restriction


def test_search_k_larger_than_corpus_is_capped(tmp_path):
    emb, _, _, _ = tiny(tmp_path, n=4, dim=8)
    idx = RuntimeIndex.load(tmp_path)
    assert len(idx.search(emb[0], k=99)) == 4


def test_hashing_encoder_is_deterministic_and_normalised():
    enc = HashingQueryEncoder(dim=32)
    a, b = enc.encode_query("binary search over a sorted array"), enc.encode_query("binary search over a sorted array")
    assert np.array_equal(a, b)
    assert a.shape == (32,)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5
    assert not np.array_equal(a, enc.encode_query("something completely different"))
    docs = enc.encode_docs(["one", "two", "three"])
    assert docs.shape == (3, 32)
    assert enc.encode_docs([]).shape == (0, 32)


def test_hashing_encoder_query_matches_its_own_document():
    """The mock has to be self-consistent or the smoke tests prove nothing: the same text encoded as a
    query and as a document must be the same vector, so retrieval on a mock index is meaningful."""
    enc = HashingQueryEncoder(dim=64)
    text = "sum the prices whose value is < 40"
    assert np.allclose(enc.encode_query(text), enc.encode_docs([text])[0])


def test_make_doc_encoder_mock_needs_no_model():
    enc, meta = make_doc_encoder("f2llm-v2-1.7b", device="cpu", mock=True, mock_dim=16)
    assert meta["model"] == "mock/hashing-encoder" and "meaningless" in meta["warning"]
    assert enc.encode_docs(["a", "b"]).shape == (2, 16)


def test_make_doc_encoder_rejects_a_text_rewriting_variant():
    with pytest.raises(ValueError, match="rewrites the query text"):
        make_doc_encoder("f2llm-v2-1.7b", mock=True, variant="registry+narrative")


def test_manifest_query_prefix_is_exactly_what_the_pipeline_prepends():
    """A served query must be formatted the way the corpus was encoded. The manifest stores
    format_query("", variant) as the prefix, so that has to equal the real prompt minus the query text --
    if the template ever gains a suffix after {query}, this test fails instead of the served index
    quietly disagreeing with the run that produced it."""
    from src.retrieval.query_variants import format_query
    prefix = format_query("", "registry+full")
    assert prefix.startswith("Instruct: ") and prefix.endswith("Query: ")
    assert format_query("abc", "registry+full") == prefix + "abc"


def test_thread_pinning_reports_physical_cores(monkeypatch):
    n, source = physical_cores()
    assert n >= 1 and isinstance(source, str)
    info = set_cpu_threads(3)
    assert info["threads"] == 3 and info["source"] == "explicit --threads"
    import os
    assert os.environ["OMP_NUM_THREADS"] == "3"


def test_manifest_is_valid_json_with_the_index_contract(tmp_path):
    tiny(tmp_path)
    m = json.loads((tmp_path / MANIFEST).read_text(encoding="utf-8"))
    for key in ("model", "revision", "query_prefix", "dim", "n_docs", "kind", "files", "index_version"):
        assert key in m, key
