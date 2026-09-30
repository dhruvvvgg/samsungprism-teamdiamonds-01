"""MTEB `SearchProtocol` model wrapping the whole pipeline (dense [+ query variants] [+ BM25/RRF]
[+ rerank] [+ LLM re-judge]) so it runs through the official `mteb.evaluate` path.

Why a SearchProtocol model and not an AbsEncoder: `mteb.evaluate` ranks encoder embeddings itself
(SearchEncoderWrapper: mteb/models/search_wrappers.py:35) and offers no hook for fusion or a second
stage. A model that implements `index()` / `search()` and `mteb_model_meta`
(mteb/models/models_protocols.py:24-85) is dispatched as-is (mteb/abstasks/retrieval.py:390-403; MTEB's
own BM25 baseline is built this way, models/model_implementations/bm25.py). `search` must return
{query_id: {doc_id: score}} with up to top_k (=1000, retrieval.py:111) documents per query.
"""
import time


from src.retrieval.bm25_index import build_bm25, tokenize_code
from src.retrieval.pipeline import merge_config, run_pipeline
from src.retrieval.query_variants import average_embeddings, format_query


class HybridSearchModel:
    def __init__(self, cfg, dense, scorer_factory=None, judge_call=None, batch_size=16,
                 rankings_depth=100):
        """cfg: pipeline config (merged over defaults). dense: object with .embed(list[str], batch_size)
        -> normalised np.ndarray (a DenseEncoder). scorer_factory: () -> reranker with .score_pairs().

        rankings_depth: how many document ids per query to retain in `last_rankings` for export. MTEB
        only keeps the metrics it computes, so the ordered list is captured here, during the run, rather
        than reconstructed afterwards from a second (re-encoding) pass."""
        from mteb.models.model_meta import ModelMeta
        self.cfg = merge_config(cfg)
        self.dense, self.batch_size = dense, batch_size
        self._scorer_factory, self._scorer, self.judge_call = scorer_factory, None, judge_call
        self.doc_ids = self.doc_texts = self.D = self.bm25 = None
        self.rankings_depth = rankings_depth
        self.last_rankings = {}
        self.timings = {}
        self.mteb_model_meta = ModelMeta.create_empty(
            {"name": "local/hybrid-pipeline", "revision": "local"})

    def _encode(self, texts, bs, label="texts"):
        """Encode with periodic progress. The official run encodes 8,765 documents and then thousands of
        queries; without progress output a long phase is indistinguishable from a hung process, which is
        exactly the ambiguity that makes a killed one-shot run hard to diagnose."""
        import time

        from src.retrieval.dense_encoder import encode_length_sorted
        texts = list(texts)
        n_batches = max(1, (len(texts) + bs - 1) // bs)
        every = max(1, n_batches // 20)                 # ~20 updates over the phase
        t0 = time.time()
        print(f"[search] encoding {len(texts)} {label} in {n_batches} batches of {bs} "
              f"(longest first)", flush=True)

        def on_batch(bi, n, done):
            if bi % every == 0 or done == len(texts):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"[search]   {label}: {done}/{len(texts)} | {rate:.1f}/s | "
                      f"elapsed {el / 60:.1f}m | eta {(len(texts) - done) / max(rate, 1e-9) / 60:.1f}m",
                      flush=True)

        return encode_length_sorted(texts, lambda t: self.dense.embed(t, batch_size=bs), bs,
                                    on_batch=on_batch)

    def index(self, corpus, *, task_metadata=None, hf_split=None, hf_subset=None, encode_kwargs=None,
              num_proc=None):
        bs = (encode_kwargs or {}).get("batch_size") or self.batch_size
        self.doc_ids, self.doc_texts = list(corpus["id"]), list(corpus["text"])
        t0 = time.time()
        self.D = self._encode(self.doc_texts, bs, "documents")         # documents: no instruction
        self.timings["index_dense_s"] = time.time() - t0
        if self.cfg["bm25"]["enabled"]:
            t0 = time.time()
            b = self.cfg["bm25"]
            self.bm25 = build_bm25(self.doc_texts, b["keep_whole"], b["k1"], b["b"])
            self.timings["index_bm25_s"] = time.time() - t0

    def _scorer_or_none(self):
        if not self.cfg["rerank"]["enabled"]:
            return None
        if self._scorer is None and self._scorer_factory is not None:
            self._scorer = self._scorer_factory()
        return self._scorer

    def search(self, queries, *, task_metadata=None, hf_split=None, hf_subset=None, top_k=1000,
               encode_kwargs=None, top_ranked=None, num_proc=None):
        if self.D is None:
            raise ValueError("index() must be called before search()")
        bs = (encode_kwargs or {}).get("batch_size") or self.batch_size
        qids, qtexts = list(queries["id"]), list(queries["text"])
        t0 = time.time()
        embs = [self._encode([format_query(t, v) for t in qtexts], bs, f"queries[{v}]")
                for v in self.cfg["dense_variants"]]
        Q = embs[0] if len(embs) == 1 else average_embeddings(embs)
        S = Q @ self.D.T
        self.timings["encode_queries_s"] = time.time() - t0
        if self.cfg["rerank"]["enabled"] and self.dense is not None:
            # documents (index()) and queries (above) are both encoded; a reranker is about to load a
            # second model onto the same GPU, so free the dense model's memory first (same OOM risk
            # this fixes in the dev scripts). Assumes MTEB calls index()/search() once per subset+split,
            # true for AppsRetrieval's single default subset -- a second search() call after this would
            # raise clearly from DenseEncoder rather than silently misbehave.
            self.dense.release()
            self.dense = None
        Sb = None
        if self.bm25 is not None:
            t0 = time.time()
            Sb = self.bm25.scores([tokenize_code(t, self.cfg["bm25"]["keep_whole"]) for t in qtexts])
            self.timings["bm25_scoring_s"] = time.time() - t0
        cfg = dict(self.cfg)
        cfg["final_depth"] = top_k
        ranked, stage_t = run_pipeline(S, Sb, qtexts, self.doc_texts, cfg, self._scorer_or_none(),
                                       self.judge_call)
        self.timings.update({f"stage_{k}": v for k, v in stage_t.items()})
        out = {}
        for qid, order in zip(qids, ranked):
            n = min(len(order), top_k)
            out[qid] = {self.doc_ids[d]: float(n - r) for r, d in enumerate(order[:n])}   # strictly decreasing
            # the submission ranking: ordered ids, kept verbatim from what MTEB was just scored on
            self.last_rankings[qid] = [self.doc_ids[d] for d in order[:self.rankings_depth]]
        return out

    def export_runtime_index(self, out_dir, meta):
        """Write the corpus embeddings computed by index() as a served runtime index. No re-encoding:
        this is the same matrix MTEB was just evaluated against."""
        from src.runtime_index import write_index
        if self.D is None:
            raise ValueError("index() must be called before export_runtime_index()")
        return write_index(out_dir, self.D, self.doc_ids, self.doc_texts, meta)
