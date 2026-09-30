"""Tiny local sanity check: a handful of docs, no full corpus / MTEB run."""
import pandas  # noqa: Windows DLL load-order workaround (pyarrow vs torch)
import pytest
np = pytest.importorskip("numpy")
pytest.importorskip("sentence_transformers")
from src.retrieval.dense_encoder import DenseEncoder, make_mteb_encoder

DOCS = ["def add(a, b): return a + b",
        "import os\nprint(os.listdir('.'))",
        "s = input()\nprint(s[::-1])  # reverse a string",
        "for i in range(10): print(i)"]


@pytest.mark.model
def test_rank_and_mteb_adapter():
    enc = DenseEncoder()
    emb = enc.embed(DOCS)
    assert emb.shape[0] == 4 and abs(np.linalg.norm(emb[0]) - 1) < 1e-3
    idx, sc = enc.rank("reverse a string", emb, k=2)
    assert idx[0] == 2 and sc[0] >= sc[1]
    m = make_mteb_encoder()
    out = m.encode([{"text": DOCS[:2]}, {"text": DOCS[2:]}])
    assert out.shape[0] == 4 and np.allclose(out, emb, atol=1e-4)


@pytest.mark.model
def test_query_prefix_applied_only_to_queries():
    m = make_mteb_encoder(query_prefix="QQQ: ")
    plain = DenseEncoder()
    q = m.encode([{"text": ["reverse a string"]}], prompt_type="query")
    d = m.encode([{"text": ["reverse a string"]}], prompt_type="document")
    assert np.allclose(q, plain.embed(["QQQ: reverse a string"]), atol=1e-4)
    assert np.allclose(d, plain.embed(["reverse a string"]), atol=1e-4)


def _fake_st(exc):
    def f(*a, **k):
        raise exc
    return f


@pytest.mark.parametrize("exc,needle", [
    (ImportError("cannot import name 'find_pruneable_heads_and_indices' from 'transformers.pytorch_utils'"),
     "version mismatch"),
    (ModuleNotFoundError("No module named 'einops'", name="einops"), "pip install einops"),
])
def test_load_failures_are_explained(monkeypatch, exc, needle):
    import sentence_transformers
    from src.retrieval.dense_encoder import ModelLoadError
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _fake_st(exc))
    with pytest.raises(ModelLoadError, match=needle) as ei:
        DenseEncoder("some/model", trust_remote_code=True)
    assert "transformers=" in str(ei.value) and ei.value.__cause__ is exc
