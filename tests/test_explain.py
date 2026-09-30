"""'Why this result': matched terms, score gaps and the route, with no model call."""
import pytest

from src.retrieval.explain import explain_hit, identifiers_of, matched_terms, query_terms
from src.runtime_index import HashingQueryEncoder, write_index

CODE = '''def count_primes(limit):
    """Count the primes below limit."""
    sieve = [True] * limit
    total = 0
    for candidate in range(2, limit):
        if sieve[candidate]:
            total += 1
    return total


class PrimeCounter:
    def isPrime(self, n):
        return n > 1
'''


def test_query_terms_drop_stopwords_short_words_and_repeats():
    assert query_terms("Count the primes below n, count them") == ["count", "primes", "below"]
    assert query_terms("") == [] and query_terms(None) == []


def test_identifiers_come_from_the_ast_not_from_strings_or_comments():
    ids = identifiers_of('def f(x):\n    # secret_comment\n    return "not_a_name" + x\n')
    assert "f" in ids and "x" in ids
    assert "secret_comment" not in ids and "not_a_name" not in ids


def test_unparsable_code_falls_back_to_identifier_shaped_tokens():
    ids = identifiers_of("print total_sum\nfor i in range(3): pass")
    assert "total_sum" in ids and "print" in ids and "for" not in ids


def test_terms_match_whole_names_snake_case_and_camel_case_parts():
    found = {m["term"]: m["identifiers"] for m in matched_terms("count the primes below limit", CODE)}
    assert found["count"] == ["count_primes"]
    assert "count_primes" in found["primes"] and "PrimeCounter" in found["primes"]   # plural ignored
    assert found["limit"] == ["limit"]
    assert "below" not in found                                                       # nothing has it


def test_camel_case_is_split():
    assert [m["term"] for m in matched_terms("is prime number", CODE)] == ["prime"]


def test_no_shared_words_means_no_matched_terms():
    assert matched_terms("sort the words alphabetically", CODE) == []
    assert matched_terms("anything", "") == []


def test_explain_hit_reports_gap_route_and_terms():
    route = {"kind": "nl_intent", "route": "dense", "reason": "short question", "confidence": "high"}
    why = explain_hit("count primes", CODE, 0.91, 0.88, route)
    assert why["score_gap_to_next"] == pytest.approx(0.03)
    assert why["ranked_by"] == "dense embedding similarity"
    assert why["route"] == {"kind": "nl_intent", "route": "dense", "reason": "short question"}
    assert {m["term"] for m in why["matched_terms"]} == {"count", "primes"} and why["n_query_terms"] == 2
    assert explain_hit("x", CODE, 0.5, None)["score_gap_to_next"] is None


@pytest.fixture()
def flat_index(tmp_path):
    d = tmp_path / "flat"
    texts = [CODE, "def add(a, b):\n    return a + b\n", "def scale(values, factor):\n    return values\n"]
    enc = HashingQueryEncoder(64)
    write_index(d, enc.encode_docs(texts), ["d0", "d1", "d2"], texts, {"model": "mock/hashing-encoder"})
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
    return TestClient(api.app)


def test_explain_is_opt_in_and_uses_the_full_text(flat_index, monkeypatch):
    client = client_for(flat_index, monkeypatch)
    plain = client.post("/search", json={"query": "count primes limit", "k": 3}).json()
    assert all("why" not in h for h in plain["hits"])
    body = client.post("/search", json={"query": "count primes limit", "k": 3, "explain": True}).json()
    hits = body["hits"]
    assert all("why" in h for h in hits)
    top = hits[0]
    assert top["doc_id"] == "d0"
    # matched from the FULL document even with a 5-character preview
    tiny = client.post("/search", json={"query": "count primes limit", "k": 3, "explain": True,
                                        "preview_chars": 5}).json()["hits"][0]
    assert tiny["why"]["matched_terms"] == top["why"]["matched_terms"] and len(tiny["preview"]) == 5
    assert top["why"]["route"]["route"] in ("dense", "dense_code", "structural")
    gaps = [h["why"]["score_gap_to_next"] for h in hits]
    assert gaps[-1] is None
    for h, nxt in zip(hits, hits[1:]):
        assert h["why"]["score_gap_to_next"] == pytest.approx(h["score"] - nxt["score"], abs=1e-5)
        assert h["why"]["score_gap_to_next"] >= 0


def test_explain_does_not_encode_again(flat_index, monkeypatch):
    client = client_for(flat_index, monkeypatch)
    import src.api as api
    svc = api.get_service()
    calls = []
    real = svc.encoder.encode_query
    svc.encoder.encode_query = lambda t: calls.append(t) or real(t)
    client.post("/search", json={"query": "count primes", "k": 2, "explain": True})
    assert len(calls) == 1


def test_grouped_results_measure_the_gap_to_the_next_lineage(tmp_path, monkeypatch):
    from src.versioning.fixture import build_fixture
    from src.versioning.version_index import build_versioned_index, embeddings_for, rows_for
    rows = rows_for(build_fixture(4, 3, seed=3))
    enc = HashingQueryEncoder(64)
    emb, _ = embeddings_for(rows, enc.encode_docs, cache={})
    build_versioned_index(tmp_path / "v", rows, emb, {"model": "mock/hashing-encoder"})
    client = client_for(tmp_path / "v", monkeypatch)
    body = client.post("/search", json={"query": "sum the scores", "k": 9, "all_versions": True,
                                        "explain": True}).json()
    groups = body["groups"]
    for g, nxt in zip(groups, groups[1:]):
        why = g["best"]["why"]
        assert why["gap_basis"] == "next lineage"
        assert why["score_gap_to_next"] == pytest.approx(g["best"]["score"] - nxt["best"]["score"], abs=1e-5)
    assert groups[-1]["best"]["why"]["score_gap_to_next"] is None


def test_the_page_labels_it_matched_terms_and_disclaims_causality():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "src" / "static" / "index.html").read_text(encoding="utf-8")
    assert "Why this result" in html and "Matched terms" in html
    assert "do not explain why the embedding model scored it" in html
    assert "explain: true" in html
