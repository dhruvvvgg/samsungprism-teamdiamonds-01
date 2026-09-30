"""The submission artefacts: the per-config reference, the rankings/index export, and the verifier.

Mocked throughout -- the pieces under test are the plumbing around `mteb.evaluate`, not the model.
"""
import json
import math

import numpy as np
import pytest

from src.eval.run_official import index_meta, reference_for
from src.retrieval.pipeline import merge_config
from src.verify_submission import check_hashes, mrr_at_k, ndcg_at_k, score_rankings


# --- per-config reference (Part 1, item 1) --------------------------------------------------------

def test_config_reference_wins_over_the_published_default():
    val, label = reference_for({"ndcg_at_10": 0.9371, "label": "dev slice, reranked"}, "f2llm-v2-1.7b")
    assert val == 0.9371 and label == "dev slice, reranked"


def test_reference_falls_back_to_the_published_score_for_the_preset():
    val, label = reference_for(None, "f2llm-v2-1.7b")
    assert val == 0.93692 and "published" in label
    val06, _ = reference_for(None, "f2llm-v2-0.6b")
    assert val06 == 0.90446
    # the point of the fix: two different configs no longer share one hard-coded number
    assert val != val06


def test_reference_is_none_when_nothing_is_known():
    val, label = reference_for(None, "some-unregistered-preset")
    assert val is None and "no published score" in label


def test_reference_ignores_a_malformed_config_entry():
    val, label = reference_for({"label": "no number here"}, "f2llm-v2-1.7b")
    assert val == 0.93692 and "published" in label


def test_the_shipped_official_config_carries_its_own_reference():
    from pathlib import Path
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs"
                      / "official_f2llm17b_noreranker.json").read_text(encoding="utf-8"))
    val, label = reference_for(cfg.get("reference"), cfg["preset"])
    assert val == 0.93692 and "F2LLM-v2-1.7B" in label
    # and the reference key must not leak into the pipeline config, which rejects unknown keys
    raw = {k: v for k, v in cfg.items() if k not in ("_comment", "preset", "reference")}
    assert merge_config(raw)["rerank"]["enabled"] is False


def test_reference_key_would_be_rejected_as_a_pipeline_key():
    """Guards the popping order in run_official: if 'reference' ever stopped being popped, this is the
    error the run would die with -- better to know it here."""
    with pytest.raises(KeyError, match="unknown pipeline config key"):
        merge_config({"reference": {"ndcg_at_10": 0.9}})


# --- runtime index metadata (Part 1, item 2) ------------------------------------------------------

PRESET = {"model": "codefuse-ai/F2LLM-v2-1.7B", "revision": "abc123", "max_seq_length": 8192,
          "dtype": "fp16", "trust_remote_code": False, "expect_eos": True, "doc_prefix": ""}


def test_index_meta_stores_the_exact_query_prefix():
    from src.retrieval.query_variants import format_query
    meta = index_meta(merge_config({}), "f2llm-v2-1.7b", PRESET, "registry+full")
    assert meta["query_prefix"] == format_query("", "registry+full")
    assert meta["revision"] == "abc123" and meta["split"] == "test"


def test_index_meta_refuses_averaged_variants():
    cfg = merge_config({"dense_variants": ["registry+full", "contest+full"]})
    reason = index_meta(cfg, "f2llm-v2-1.7b", PRESET, "registry+full")
    assert isinstance(reason, str) and "averaged 2 query variants" in reason


def test_index_meta_refuses_a_text_rewriting_variant():
    cfg = merge_config({"dense_variants": ["registry+narrative"]})
    reason = index_meta(cfg, "f2llm-v2-1.7b", PRESET, "registry+narrative")
    assert isinstance(reason, str) and "rewrites the query text" in reason


# --- the search model records rankings and can export the index -----------------------------------

class FakeDense:
    """Embeds text as a deterministic 4-d vector; no model involved."""

    def __init__(self):
        self.released = False
        self.fell_back_to_fp32 = False

    def embed(self, texts, batch_size=8, show_progress_bar=False):
        out = []
        for t in texts:
            h = abs(hash(t.strip()[-40:])) % 1000
            v = np.array([1.0, h % 7, h % 11, h % 13], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    def release(self):
        self.released = True


def build_model(cfg=None, depth=100):
    from src.retrieval.search_model import HybridSearchModel
    return HybridSearchModel(merge_config(cfg or {}), FakeDense(), rankings_depth=depth)


def test_search_records_ordered_rankings_matching_the_returned_scores():
    model = build_model(depth=3)
    docs = {"id": [f"d{i}" for i in range(6)], "text": [f"code sample {i}" for i in range(6)]}
    model.index(docs, encode_kwargs={"batch_size": 4})
    out = model.search({"id": ["q1", "q2"], "text": ["find sample 2", "find sample 5"]},
                       top_k=6, encode_kwargs={"batch_size": 4})
    assert set(model.last_rankings) == {"q1", "q2"}
    for qid, ids in model.last_rankings.items():
        assert len(ids) == 3                                   # truncated to rankings_depth
        assert len(set(ids)) == 3 and set(ids) <= set(docs["id"])
        # the stored order must agree with the scores MTEB was given
        scored = sorted(out[qid].items(), key=lambda kv: -kv[1])
        assert ids == [d for d, _ in scored][:3]


def test_rankings_depth_does_not_truncate_what_mteb_is_scored_on():
    model = build_model(depth=2)
    docs = {"id": [f"d{i}" for i in range(5)], "text": [f"code {i}" for i in range(5)]}
    model.index(docs)
    out = model.search({"id": ["q"], "text": ["code 3"]}, top_k=5)
    assert len(out["q"]) == 5                                  # full ranking still goes to MTEB
    assert len(model.last_rankings["q"]) == 2                  # only the export is shortened


def test_export_runtime_index_reuses_the_corpus_embeddings(tmp_path):
    from src.runtime_index import RuntimeIndex
    model = build_model()
    docs = {"id": ["a", "b", "c"], "text": ["def a(): pass", "def b(): pass", "def c(): pass"]}
    model.index(docs)
    manifest = model.export_runtime_index(tmp_path / "idx", {"model": "fake", "revision": "r",
                                                             "query_prefix": "P: "})
    assert manifest["n_docs"] == 3 and manifest["dim"] == 4
    idx = RuntimeIndex.load(tmp_path / "idx")
    assert idx.doc_ids == docs["id"] and idx.doc_texts == docs["text"]
    assert np.allclose(idx.E, model.D, atol=1e-3)              # same matrix, nothing re-encoded
    assert idx.verify_files() == []


def test_export_before_index_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="index\\(\\) must be called"):
        build_model().export_runtime_index(tmp_path / "idx", {})


# --- the verifier's own metric implementation -----------------------------------------------------

def test_ndcg_and_mrr_on_a_hand_computed_case():
    ranked = ["d5", "d1", "d3"]
    assert ndcg_at_k(ranked, {"d1": 1}, 10) == pytest.approx(1 / math.log2(3))
    assert mrr_at_k(ranked, {"d1": 1}, 10) == pytest.approx(0.5)
    assert ndcg_at_k(["d1"], {"d1": 1}, 10) == pytest.approx(1.0)
    assert mrr_at_k(["d1"], {"d1": 1}, 10) == pytest.approx(1.0)


def test_a_relevant_doc_beyond_k_scores_zero():
    ranked = [f"x{i}" for i in range(10)] + ["d1"]
    assert ndcg_at_k(ranked, {"d1": 1}, 10) == 0.0
    assert mrr_at_k(ranked, {"d1": 1}, 10) == 0.0


def test_graded_relevance_uses_the_ideal_ordering():
    """Two relevant docs, the better one second: NDCG must be below 1 but above the single-hit value."""
    got = ndcg_at_k(["a", "b"], {"a": 1, "b": 3}, 10)
    ideal = (3 / math.log2(2)) + (1 / math.log2(3))
    assert got == pytest.approx(((1 / math.log2(2)) + (3 / math.log2(3))) / ideal)
    assert 0 < got < 1


def test_score_rankings_averages_over_judged_queries():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    rankings = {"q1": ["d5", "d1"], "q2": ["d2"]}
    got = score_rankings(rankings, qrels, 10)
    assert got["ndcg_at_10"] == pytest.approx((1 / math.log2(3) + 1.0) / 2)
    assert got["mrr_at_10"] == pytest.approx((0.5 + 1.0) / 2)
    assert got["n_queries_scored"] == 2 and got["missing_from_rankings"] == []


def test_a_query_with_no_ranking_scores_zero_and_is_reported():
    got = score_rankings({"q1": ["d1"]}, {"q1": {"d1": 1}, "q2": {"d9": 1}}, 10)
    assert got["missing_from_rankings"] == ["q2"]
    assert got["ndcg_at_10"] == pytest.approx(0.5)             # 1.0 and 0.0 over two queries


def test_extra_unjudged_queries_in_the_rankings_are_ignored():
    got = score_rankings({"q1": ["d1"], "qX": ["d7"]}, {"q1": {"d1": 1}}, 10)
    assert got["n_queries_scored"] == 1 and got["ndcg_at_10"] == pytest.approx(1.0)


# --- checksum verification -------------------------------------------------------------------------

def test_check_hashes_passes_then_fails_after_a_change(tmp_path):
    from src.runtime_index import sha256_file
    from src.utils_io import write_json_atomic
    res = write_json_atomic(tmp_path / "results.json", {"a": 1})
    rank = write_json_atomic(tmp_path / "rankings.json", {"rankings": {}})
    checksums = {"results": {"path": str(res), "sha256": sha256_file(res)},
                 "rankings": {"path": str(rank), "sha256": sha256_file(rank)}}
    problems, notes = [], []
    check_hashes(checksums, problems, notes)
    assert problems == [] and len(notes) == 2

    write_json_atomic(tmp_path / "results.json", {"a": 2})     # the file changed after the run
    problems, notes = [], []
    check_hashes(checksums, problems, notes)
    assert len(problems) == 1 and "MISMATCH" in problems[0]


def test_check_hashes_reports_a_missing_file_and_a_bad_index(tmp_path):
    checksums = {"results": {"path": str(tmp_path / "gone.json"), "sha256": "0" * 64},
                 "runtime_index": {"dir": str(tmp_path), "files": {"embeddings.npy": {"sha256": "0" * 64}}}}
    problems, notes = [], []
    check_hashes(checksums, problems, notes)
    assert any("is missing" in p for p in problems)
    assert any("embeddings.npy: missing" in p for p in problems)
