"""Query variants, embedding cache, BM25, RRF, reranker maths/prompt. Fake models only."""
import numpy as np
import pytest
import torch

from src.retrieval import query_variants as qv
from src.retrieval.bm25_index import FastBM25, split_identifier, tokenize_code
from src.retrieval.embedding_cache import get_or_build, texts_hash
from src.retrieval.hybrid_encoder import hybrid_rank, rrf_fuse, top_n
from src.retrieval import reranker as rk
from src.retrieval.reranker import RERANKERS

DASH = ("Chef has an array of N numbers and wants the largest sum of two of them. " * 4 +
        "\n\n-----Input-----\nThe first line contains N.\n\n-----Output-----\nPrint the answer.\n\n"
        "-----Examples-----\nInput\n3\n1 2 3\n\nOutput\n5\n")
MARKDOWN = ("Convert a string to a new string where each character is ( if unique and ) otherwise. " * 3 +
            "\n\n## Examples\n```\n\"din\" => \"(((\"\n```\n\n**Notes**\nAssertions may be unclear.")
LEETCODE = ("Given an integer array arr and an integer k, modify the array by repeating it k times. " * 3 +
            "\n \nExample 1:\nInput: arr = [1,2], k = 3\nOutput: 9\n\nConstraints:\n1 <= arr.length <= 10^5")
PLAIN = ("There are N people in the group, find the extra one. " * 4 +
         "\nInput : \n3\nOutput : \nPrint it.\nConstraints : \n1 <= N <= 5\nSample Input : \n3\nSample Output : \n3")


# ---------- query variants ----------
def test_registry_instruction_matches_installed_mteb():
    import pandas  # noqa: F401
    from mteb.models.model_implementations import codefuse_models as c
    assert qv.INSTRUCTIONS["registry"] == c.f2llmv2_prompts_dict["AppsRetrieval"]
    assert qv.INSTRUCTIONS["contest"] == c.c2llm_prompts_dict["AppsRetrieval"]["query"].strip()


def test_format_query_matches_f2llm_template():
    out = qv.format_query("solve it", "registry+full")
    assert out == ("Instruct: Retrieve the most relevant code snippet for the given query.\nQuery: solve it")


@pytest.mark.parametrize("text,kept,dropped_no_examples,dropped_narrative", [
    (DASH, "largest sum", "-----Examples-----", "-----Input-----"),
    (MARKDOWN, "each character", "## Examples", "## Examples"),
    (LEETCODE, "repeating it k times", "Example 1", "Example 1"),
    (PLAIN, "find the extra one", "Sample Input", "Input :"),
])
def test_no_examples_and_narrative_cut_real_formats(text, kept, dropped_no_examples, dropped_narrative):
    for tf, dropped in ((qv.transform_no_examples, dropped_no_examples),
                        (qv.transform_narrative, dropped_narrative)):
        out = tf(text)
        assert kept in out and dropped not in out and len(out) < len(text)


def test_narrative_is_at_least_as_short_as_no_examples_on_dash_format():
    assert len(qv.transform_narrative(DASH)) <= len(qv.transform_no_examples(DASH)) < len(DASH)
    assert "Print the answer" in qv.transform_no_examples(DASH)         # I/O spec kept
    assert "Print the answer" not in qv.transform_narrative(DASH)


def test_transforms_fall_back_to_full_text():
    short = "Short.\n\n-----Examples-----\nInput\n1"                    # remainder < min_chars
    assert qv.transform_no_examples(short) == short
    plain = "No headers here at all, just a paragraph about sums. " * 5
    assert qv.transform_narrative(plain) == plain and qv.transform_no_examples(plain) == plain
    assert qv.transform_full(DASH) == DASH


def test_variant_parsing_and_average():
    assert qv.parse_variant("solution+narrative") == ("solution", "narrative")
    for bad in ("nope+full", "registry+nope", "registry"):
        with pytest.raises(ValueError):
            qv.parse_variant(bad)
    a = np.array([[3.0, 0.0], [0.0, 2.0]])
    b = np.array([[0.0, 5.0], [0.0, 1.0]])
    avg = qv.average_embeddings([a, b])
    assert np.allclose(np.linalg.norm(avg, axis=1), 1.0)
    assert np.allclose(avg[1], [0.0, 1.0])
    assert np.allclose(qv.average_embeddings([a, b]), qv.average_embeddings([b, a]))     # order-free
    assert np.allclose(qv.average_embeddings([a]), a / np.linalg.norm(a, axis=1, keepdims=True))


# ---------- embedding cache ----------
def test_cache_hit_miss_and_invalidation(tmp_path):
    calls = []

    def enc(texts):
        calls.append(len(texts))
        return np.array([[len(t), 1.0] for t in texts]) / 10.0

    e1, m1 = get_or_build("docs", ["a", "bb"], enc, "modelA", tmp_path)
    e2, m2 = get_or_build("docs", ["a", "bb"], enc, "modelA", tmp_path)
    assert calls == [2] and not m1["cache_hit"] and m2["cache_hit"]
    assert np.array_equal(e1, e2) and e1.dtype == np.float32
    assert any(p.suffix == ".npy" for p in tmp_path.iterdir())
    assert np.load(next(p for p in tmp_path.iterdir() if p.suffix == ".npy")).dtype == np.float16
    get_or_build("docs", ["a", "bbb"], enc, "modelA", tmp_path)          # changed text -> miss
    get_or_build("docs", ["a", "bb"], enc, "modelB", tmp_path)           # changed model -> miss
    get_or_build("other", ["a", "bb"], enc, "modelA", tmp_path)          # changed name -> miss
    assert calls == [2, 2, 2, 2]
    assert texts_hash(["a", "b"]) != texts_hash(["ab"])


# ---------- BM25 ----------
def test_identifier_splitting():
    assert split_identifier("maxSubArray_v2") == ["max", "sub", "array", "v", "2"]
    assert split_identifier("HTTPServer") == ["http", "server"]
    assert split_identifier("snake_case_name") == ["snake", "case", "name"]
    toks = tokenize_code("def getMaxValue(arr): return max_val", keep_whole=True)
    assert {"get", "max", "value", "getmaxvalue", "max_val", "val", "arr"} <= set(toks)
    assert "getmaxvalue" not in tokenize_code("getMaxValue", keep_whole=False)
    assert tokenize_code("x = 12", keep_whole=True) == ["x", "12"]


def test_fast_bm25_equals_rank_bm25():
    docs = ["def sum_two(a, b): return a + b", "for i in range(n): print(i)",
            "count = len(items) # count items", "import heapq; heapq.heappush(heap, x)",
            "n = int(input()); total = sum(map(int, input().split()))", "print(count count count)"]
    fast = FastBM25([tokenize_code(d) for d in docs], k1=1.5, b=0.75)
    queries = ["count the items and print count count", "sum of two numbers n", "unknownword zzz", "n n n input"]
    tq = [tokenize_code(q) for q in queries]
    got = fast.scores(tq)
    for i, q in enumerate(tq):
        assert np.allclose(got[i], fast.bm25.get_scores(q), atol=1e-4), queries[i]
    assert (got[2] == 0).all()                                   # all-OOV query scores nothing
    assert fast.scores([]).shape == (0, len(docs))


# ---------- RRF ----------
def test_rrf_fuse_hand_computed():
    fused = rrf_fuse([[1, 2, 3], [3, 2, 9]], [1.0, 1.0], k=60)
    sc = dict(fused)
    assert sc[2] == pytest.approx(1 / 62 + 1 / 62)
    assert sc[1] == pytest.approx(1 / 61) and sc[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[0][0] == 3 and {d for d, _ in fused} == {1, 2, 3, 9}
    w = dict(rrf_fuse([[1], [2]], [1.0, 0.5], k=10))
    assert w[1] == pytest.approx(1 / 11) and w[2] == pytest.approx(0.5 / 11)
    assert rrf_fuse([[1, 2], [2, 1]], [1, 1], 60)[0][0] == 1     # exact tie -> first list's order wins
    assert rrf_fuse([], [], 60) == []


def test_top_n_and_hybrid_rank():
    row = np.array([0.1, 0.9, 0.5, 0.9, 0.2])
    assert top_n(row, 3) == [1, 3, 2]                              # ties by lower index
    assert all(isinstance(i, int) for i in top_n(row, 3))          # plain Python ints, not np.int64
    assert len(top_n(row, 99)) == 5
    dense = np.array([0.9, 0.1, 0.2, 0.3])
    bm = np.array([0.0, 5.0, 0.0, 4.0])
    out = [d for d, _ in hybrid_rank(dense, bm, k=60, depth=2)]
    assert set(out) == {0, 3, 1}                                  # top-2 of each list only
    assert [d for d, _ in hybrid_rank(dense, bm, w_bm25=0.0, depth=2)][:1] == [0]


# ---------- reranker maths / prompt ----------
def test_qwen3_pair_format_matches_mteb_wrapper():
    import pandas  # noqa: F401
    from mteb.models.model_implementations.qwen3_reranker import Qwen3RerankerWrapper
    for ins in (rk.QWEN3_INSTRUCTIONS["card"], rk.QWEN3_INSTRUCTIONS["apps"]):
        assert rk.format_qwen3_pair(ins, "Q?", "code()") == Qwen3RerankerWrapper.format_instruction(ins, "Q?", "code()")
    assert rk.QWEN3_INSTRUCTIONS["card"] == "Given a web search query, retrieve relevant passages that answer the query"
    assert rk.QWEN3_PREFIX.startswith("<|im_start|>system\nJudge whether the Document meets the requirements "
                                      "based on the Query and the Instruct provided.")
    assert rk.QWEN3_SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class FakeTok:
    """Char-level fake: prefix/suffix tokenised into fixed ids, bodies truncated to max_length."""
    def encode(self, s, add_special_tokens=False):
        return [1000 + i for i in range(3 if "system" in s else 2)]

    def __call__(self, pairs, padding=False, truncation=None, return_attention_mask=False, max_length=None):
        return {"input_ids": [[ord(c) for c in p][:max_length] for p in pairs]}

    def pad(self, enc, padding=True, return_tensors=None, max_length=None):
        return enc


def test_build_qwen3_inputs_wraps_and_budgets_length():
    enc = rk.build_qwen3_inputs(FakeTok(), ["abcdefghij", "xy"], max_length=9)
    a, b = enc["input_ids"]
    assert a[:3] == [1000, 1001, 1002] and a[-2:] == [1000, 1001]      # prefix ... suffix
    assert len(a) == 9 and len(b) == 3 + 2 + 2                         # body budget = 9 - 3 - 2 = 4


class _FakeModel:
    def __init__(self):
        self.floated = False

    def float(self):
        self.floated = True


class FakeScorer(rk._TorchScorer):
    """score = -length of doc (so ordering by score is checkable); optionally NaN in 'half' mode."""
    def __init__(self, nan_first=False):
        self.name, self.model, self.fell_back_to_fp32 = "fake", _FakeModel(), False
        self.nan_first, self.batches = nan_first, []

    def _score_batch(self, batch):
        self.batches.append([len(q) + len(d) for q, d in batch])
        if self.nan_first and not self.model.floated:
            return np.full(len(batch), np.nan, dtype=np.float32)
        return np.array([-float(len(d)) for _, d in batch], dtype=np.float32)


def test_score_pairs_sorts_by_length_but_returns_input_order():
    pairs = [("q", "aaaa"), ("q", "a"), ("qqqqqq", "aaaaaaaa"), ("q", "aa")]
    s = FakeScorer()
    out = s.score_pairs(pairs, batch_size=2)
    assert out.tolist() == [-4.0, -1.0, -8.0, -2.0]
    flat = [x for b in s.batches for x in b]
    assert flat == sorted(flat, reverse=True)


def test_score_pairs_nan_falls_back_to_fp32_once():
    s = FakeScorer(nan_first=True)
    out = s.score_pairs([("q", "ab"), ("q", "abc")], batch_size=8)
    assert s.model.floated and s.fell_back_to_fp32 and np.isfinite(out).all()


def test_rerank_scores_splits_per_query():
    scorer = FakeScorer()
    docs = ["a", "bb", "ccc", "dddd"]
    out, secs = rk.rerank_scores(scorer, ["q1", "q2"], [[0, 1], [3, 2, 0]], docs, batch_size=4)
    assert [o.tolist() for o in out] == [[-1.0, -2.0], [-4.0, -3.0, -1.0]] and secs >= 0


def test_interpolation_and_reorder():
    idx, first, rr = [10, 11, 12], np.array([0.9, 0.5, 0.1]), np.array([0.0, 0.1, 0.9])
    assert rk.rerank_order(idx, first, rr, 1.0)[0] == [10, 11, 12]         # alpha=1: first stage only
    assert rk.rerank_order(idx, first, rr, 0.0)[0] == [12, 11, 10]         # alpha=0: pure reranker
    mid, sc = rk.rerank_order(idx, first, rr, 0.5)
    assert sorted(mid) == idx and all(sc[i] >= sc[i + 1] for i in range(2))
    assert np.allclose(rk.zscore([1, 1, 1]), 0)                            # no spread -> zeros, no NaN
    tie, _ = rk.rerank_order([1, 2], np.array([1.0, 1.0]), np.array([1.0, 1.0]), 0.5)
    assert tie == [1, 2]                                                   # stable on ties


# ---------- cache-key builders must survive numpy int64 doc indices (regression: dev_rerank.py TypeError) ----------
def test_top_n_returns_plain_python_ints_not_numpy():
    row = np.array([0.3, 0.1, 0.9, 0.5])
    for i in top_n(row, 4):
        assert type(i) is int                          # not numpy.int64 -- must survive json.dumps


# ---------- Ettin reranker: must load via sentence_transformers.CrossEncoder, never AutoModelForSequenceClassification ----------
class FakeCrossEncoderModel:
    """Stands in for sentence_transformers.CrossEncoder (which is itself an nn.Module/Sequential)."""
    instances = []

    def __init__(self, model_name, max_length=None, **kwargs):
        self.model_name, self.max_length, self.kwargs = model_name, max_length, kwargs
        self._dtype = torch.float32
        self.calls = []
        FakeCrossEncoderModel.instances.append(self)

    def to(self, arg):
        if isinstance(arg, torch.dtype):
            self._dtype = arg
        return self

    def eval(self):
        return self

    def float(self):
        return self.to(torch.float32)

    def parameters(self):
        yield torch.zeros(1, dtype=self._dtype)

    def predict(self, pairs, batch_size=16, show_progress_bar=False, convert_to_numpy=True):
        self.calls.append(list(pairs))
        # score = doc length, so ordering is checkable, like FakeScorer elsewhere in this file
        return np.array([float(len(d)) for _, d in pairs], dtype=np.float32)


def test_ettin_reranker_uses_crossencoder_not_automodel(monkeypatch):
    import sentence_transformers
    import transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoderModel)
    FakeCrossEncoderModel.instances = []

    def boom(*a, **k):
        raise AssertionError("AutoModelForSequenceClassification must NOT be used for an Ettin checkpoint")
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", boom)

    scorer = rk.load_reranker("ettin-reranker-150m", device="cuda", dtype="fp16")
    assert isinstance(scorer, rk.EttinReranker)
    fake = FakeCrossEncoderModel.instances[0]
    assert fake.model_name == "cross-encoder/ettin-reranker-150m-v1"
    assert fake._dtype == torch.float16                                    # _cast() applied fp16 on cuda

    out = scorer.score_pairs([("q1", "abc"), ("q2", "de")], batch_size=8)
    assert out.tolist() == [3.0, 2.0]
    assert RERANKERS["ettin-reranker-150m"] == {"kind": "ettin", "model": "cross-encoder/ettin-reranker-150m-v1"}
    assert RERANKERS["ettin-reranker-68m"] == {"kind": "ettin", "model": "cross-encoder/ettin-reranker-68m-v1"}


def test_ettin_reranker_reuses_shared_nan_fallback(monkeypatch):
    import sentence_transformers

    class NanFirst(FakeCrossEncoderModel):
        def predict(self, pairs, **kw):
            if self._dtype != torch.float32:
                return np.full(len(pairs), np.nan, dtype=np.float32)
            return super().predict(pairs, **kw)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", NanFirst)
    scorer = rk.load_reranker("ettin-reranker-68m", device="cuda", dtype="fp16")
    out = scorer.score_pairs([("q", "abcd")], batch_size=4)
    assert scorer.fell_back_to_fp32 and np.isfinite(out).all()


# ---------- dense+dense fusion: score averaging alongside the existing RRF ----------
def test_normalize_rows_zscore_and_minmax():
    from src.retrieval.hybrid_encoder import normalize_rows

    S = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])   # same shape, wildly different scale
    z = normalize_rows(S, "zscore")
    assert np.allclose(z[0], z[1])                         # scale-invariant: both rows normalise alike
    assert np.allclose(z.mean(axis=1), 0.0) and np.allclose(z.std(axis=1), 1.0)
    mm = normalize_rows(S, "minmax")
    assert np.allclose(mm[0], [0.0, 0.5, 1.0]) and np.allclose(mm[1], [0.0, 0.5, 1.0])
    flat = np.array([[5.0, 5.0, 5.0]])                     # no spread -> no NaN
    assert np.allclose(normalize_rows(flat, "zscore"), 0.0)
    assert np.allclose(normalize_rows(flat, "minmax"), 0.5)
    with pytest.raises(ValueError):
        normalize_rows(S, "nope")


def test_score_average_fuse_weighting_endpoints_and_blend():
    from src.retrieval.hybrid_encoder import normalize_rows, score_average_fuse

    S_a = np.array([[3.0, 2.0, 1.0]])                      # A prefers doc 0
    S_b = np.array([[1.0, 2.0, 3.0]])                      # B prefers doc 2
    assert np.argmax(score_average_fuse(S_a, S_b, w_b=0.0)) == 0      # w_b=0 -> A alone
    assert np.argmax(score_average_fuse(S_a, S_b, w_b=1.0)) == 2      # w_b=1 -> B alone
    assert np.allclose(score_average_fuse(S_a, S_b, w_b=0.0), normalize_rows(S_a))
    blend = score_average_fuse(S_a, S_b, w_b=0.5)
    assert np.allclose(blend, 0.0)                         # symmetric disagreement cancels out
    lean = score_average_fuse(S_a, S_b, w_b=0.25)          # mostly A
    assert np.argmax(lean) == 0


def test_score_average_fuse_is_scale_invariant_across_models():
    """The whole point of normalising first: a model whose cosines happen to live on a different scale
    must not dominate the average just because its numbers are bigger."""
    from src.retrieval.hybrid_encoder import score_average_fuse

    S_a = np.array([[0.9, 0.8, 0.1]])
    S_b_small = np.array([[0.01, 0.02, 0.03]])
    S_b_big = S_b_small * 1000.0                           # same ranking, 1000x the magnitude
    assert np.allclose(score_average_fuse(S_a, S_b_small, 0.5), score_average_fuse(S_a, S_b_big, 0.5))


def test_rrf_fuse_works_unchanged_on_two_dense_lists():
    """Dense+dense reuses the same rrf_fuse as BM25+dense -- no new fusion machinery."""
    from src.retrieval.hybrid_encoder import rrf_fuse, top_n

    S_a = np.array([0.9, 0.5, 0.1, 0.0])
    S_b = np.array([0.0, 0.1, 0.5, 0.9])
    la, lb = list(top_n(S_a, 4)), list(top_n(S_b, 4))
    fused = dict(rrf_fuse([la, lb], [1.0, 1.0], k=60))
    assert fused[0] == pytest.approx(1 / 61 + 1 / 64)      # rank 1 in A, rank 4 in B
    assert fused[1] == pytest.approx(1 / 62 + 1 / 63)
    down_weighted = dict(rrf_fuse([la, lb], [1.0, 0.0], k=60))
    assert down_weighted[0] == pytest.approx(1 / 61)       # w_B=0 contributes nothing
