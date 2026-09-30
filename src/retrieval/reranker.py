"""Second-stage rerankers + score interpolation.

Rerankers, all verified from their Hugging Face model cards and/or the installed MTEB registry:

  qwen3-reranker-0.6b   Qwen/Qwen3-Reranker-0.6B  (Qwen3ForCausalLM, 0.6B, transformers>=4.51)
      Pair prompt (model card, mirrored by mteb/models/model_implementations/qwen3_reranker.py):
        <|im_start|>system\\nJudge whether the Document meets the requirements based on the Query and the
        Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\\n<|im_start|>user\\n
        <Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {doc}<|im_end|>\\n<|im_start|>assistant\\n
        <think>\\n\\n</think>\\n\\n
      score = softmax([logit("no"), logit("yes")])[yes] at the last position (left padding).
  bge-reranker-v2-m3    BAAI/bge-reranker-v2-m3  (XLMRobertaForSequenceClassification, one logit per
      (query, passage) pair, raw logit used as the score; the card's sigmoid is monotone so ranking is
      identical; context up to 8192 positions, card examples use max_length=512, we default to 1024).
  ettin-reranker-150m   cross-encoder/ettin-reranker-150m-v1  (ModernBertModel + Dense/LayerNorm/Dense
  ettin-reranker-68m    cross-encoder/ettin-reranker-68m-v1   head; verified via the repo's own file
      listing: 2_Dense/, 3_LayerNorm/, 4_Dense/, modules.json -- this is a Sentence-Transformers
      CrossEncoder checkpoint, NOT a plain *ForSequenceClassification model. Loading it through
      transformers.AutoModelForSequenceClassification (as CrossEncoderReranker does for BGE) would
      either error or silently attach a randomly-initialized head instead of the trained one -- it must
      load through sentence_transformers.CrossEncoder, which the installed sentence-transformers
      confirms is itself an nn.Sequential/Module, so it slots into _TorchScorer's dtype casting as-is.
      Model card default context is 7999 tokens; we default max_length to 1024 like BGE. Raw regression
      logit as the score (card examples show unbounded values e.g. 4.875, 11.625 -- not a probability).
"""
import time

import numpy as np

QWEN3_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
                "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                "<|im_start|>user\n")
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN3_INSTRUCTIONS = {
    # model-card default (used by MTEB when a task supplies none)
    "card": "Given a web search query, retrieve relevant passages that answer the query",
    # ours: the "document" is a candidate Python solution to the "query" problem statement
    "apps": "Given a competitive programming problem statement, judge whether the Python code is a "
            "correct solution to it",
}

RERANKERS = {
    "qwen3-reranker-0.6b": {"kind": "qwen3", "model": "Qwen/Qwen3-Reranker-0.6B"},
    "bge-reranker-v2-m3": {"kind": "cross", "model": "BAAI/bge-reranker-v2-m3"},
    "ettin-reranker-150m": {"kind": "ettin", "model": "cross-encoder/ettin-reranker-150m-v1"},
    "ettin-reranker-68m": {"kind": "ettin", "model": "cross-encoder/ettin-reranker-68m-v1"},
}


def format_qwen3_pair(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def build_qwen3_inputs(tokenizer, pairs, max_length):
    """Token ids for formatted pairs: truncate the pair body so prefix+body+suffix fits max_length, then
    wrap with the fixed prefix / suffix tokens and left-pad. Returns a tokenizer.pad(...) BatchEncoding."""
    prefix_ids = tokenizer.encode(QWEN3_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(QWEN3_SUFFIX, add_special_tokens=False)
    enc = tokenizer(pairs, padding=False, truncation="longest_first", return_attention_mask=False,
                    max_length=max_length - len(prefix_ids) - len(suffix_ids))
    enc["input_ids"] = [prefix_ids + ids + suffix_ids for ids in enc["input_ids"]]
    return tokenizer.pad(enc, padding=True, return_tensors="pt", max_length=max_length)


class _TorchScorer:
    """Shared: length-sorted batching, half-precision with NaN->fp32 fallback, timing."""

    def _cast(self, dtype, device):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        want = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        if want != torch.float32 and not str(self.device).startswith("cuda"):
            print(f"[rerank] WARNING: {dtype} needs CUDA; using fp32", flush=True)
            want = torch.float32
        self.model.to(want)
        self.fell_back_to_fp32 = False
        print(f"[rerank] {self.name} device={self.device} dtype={next(self.model.parameters()).dtype}",
              flush=True)

    def score_pairs(self, pairs, batch_size=16, progress_pairs=2000):
        """pairs: list[(query, doc)] -> np.ndarray float32 aligned with `pairs`. Longest pairs first.

        Logs progress once the job is big enough (`progress_pairs`) that silence would be ambiguous:
        scoring ~113k pairs takes long enough that "no output" otherwise looks identical to a hang."""
        order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        out = np.zeros(len(pairs), dtype=np.float32)
        n_batches = (len(order) + batch_size - 1) // batch_size
        verbose = len(pairs) >= progress_pairs
        every = max(1, n_batches // 20)          # ~20 updates over the whole pass
        t_start = time.time()
        if verbose:
            print(f"[rerank] scoring {len(pairs)} pairs in {n_batches} batches of {batch_size} "
                  f"({self.name}); longest pairs first, so early batches are the slowest", flush=True)
        for bi, s in enumerate(range(0, len(order), batch_size)):
            idx = order[s:s + batch_size]
            scores = self._score_batch([pairs[i] for i in idx])
            if verbose and (bi % every == 0 or bi == n_batches - 1):
                done = min(s + batch_size, len(pairs))
                el = time.time() - t_start
                rate = done / max(el, 1e-9)
                eta = (len(pairs) - done) / max(rate, 1e-9)
                print(f"[rerank]   batch {bi + 1}/{n_batches} | {done}/{len(pairs)} pairs | "
                      f"{rate:.1f} pairs/s | elapsed {el / 60:.1f}m | eta {eta / 60:.1f}m", flush=True)
            if not np.isfinite(scores).all():
                if self.fell_back_to_fp32:
                    raise FloatingPointError("reranker produced non-finite scores in fp32")
                print("[rerank] WARNING: non-finite scores in half precision; switching to fp32", flush=True)
                self.model.float()
                self.fell_back_to_fp32 = True
                scores = self._score_batch([pairs[i] for i in idx])
            out[idx] = scores
        return out


class Qwen3Reranker(_TorchScorer):
    def __init__(self, model="Qwen/Qwen3-Reranker-0.6B", device=None, dtype="fp16", max_length=2048,
                 instruction="apps", revision=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.max_length = model, max_length
        self.instruction = QWEN3_INSTRUCTIONS.get(instruction, instruction)
        kw = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left", **kw)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self._cast(dtype, device)

    def _score_batch(self, batch):
        import torch
        texts = [format_qwen3_pair(self.instruction, q, d) for q, d in batch]
        inputs = build_qwen3_inputs(self.tokenizer, texts, self.max_length).to(self.device)
        with torch.inference_mode():
            # logits_to_keep=1: ask the lm_head for only the LAST position instead of materializing a
            # [batch, seq_len, vocab=151936] tensor and immediately discarding every row but the last --
            # at max_length=2048 that discarded tensor alone can be several GB per batch in fp16. Falls
            # back for transformers versions that don't support the kwarg on this model.
            try:
                out = self.model(**inputs, logits_to_keep=1)
            except TypeError:
                out = self.model(**inputs)
            logits = out.logits[:, -1, :]
            two = torch.stack([logits[:, self.false_id], logits[:, self.true_id]], dim=1).float()
            return torch.log_softmax(two, dim=1)[:, 1].exp().cpu().numpy()


class CrossEncoderReranker(_TorchScorer):
    def __init__(self, model="BAAI/bge-reranker-v2-m3", device=None, dtype="fp16", max_length=1024,
                 revision=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name, self.max_length = model, max_length
        kw = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kw)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, **kw)
        self._cast(dtype, device)

    def _score_batch(self, batch):
        import torch
        inputs = self.tokenizer([q for q, _ in batch], [d for _, d in batch], padding=True,
                                truncation="longest_first", max_length=self.max_length,
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return self.model(**inputs).logits.view(-1).float().cpu().numpy()


class EttinReranker(_TorchScorer):
    """cross-encoder/ettin-reranker-*-v1: loads via sentence_transformers.CrossEncoder (see module
    docstring for why -- AutoModelForSequenceClassification does not fit this checkpoint's format)."""

    def __init__(self, model="cross-encoder/ettin-reranker-150m-v1", device=None, dtype="fp16",
                 max_length=1024, revision=None):
        from sentence_transformers import CrossEncoder
        self.name, self.max_length = model, max_length
        kw = {"revision": revision} if revision else {}
        self.model = CrossEncoder(model, max_length=max_length, **kw)
        self._cast(dtype, device)

    def _score_batch(self, batch):
        return np.asarray(self.model.predict(list(batch), batch_size=len(batch),
                                              show_progress_bar=False, convert_to_numpy=True),
                          dtype=np.float32)


def load_reranker(name, device=None, dtype="fp16", max_length=None, instruction="apps"):
    spec = RERANKERS[name]
    if spec["kind"] == "qwen3":
        return Qwen3Reranker(spec["model"], device, dtype, max_length or 2048, instruction)
    if spec["kind"] == "ettin":
        return EttinReranker(spec["model"], device, dtype, max_length or 1024)
    return CrossEncoderReranker(spec["model"], device, dtype, max_length or 1024)


def rerank_scores(scorer, query_texts, cand_lists, doc_texts, batch_size=16):
    """Score every (query, candidate) pair in one length-sorted pass across ALL queries.
    Returns (list of np.ndarray aligned with cand_lists, seconds)."""
    pairs, spans = [], []
    for q, cands in zip(query_texts, cand_lists):
        spans.append((len(pairs), len(pairs) + len(cands)))
        pairs += [(q, doc_texts[d]) for d in cands]
    t0 = time.time()
    flat = scorer.score_pairs(pairs, batch_size=batch_size)
    return [flat[a:b] for a, b in spans], time.time() - t0


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def interpolate(first_scores, rr_scores, alpha):
    """alpha * z(first-stage) + (1 - alpha) * z(reranker), z-scored within the candidate list.
    alpha=0 is a pure rerank, alpha=1 keeps the first-stage order."""
    return alpha * zscore(first_scores) + (1.0 - alpha) * zscore(rr_scores)


def rerank_order(cand_idx, first_scores, rr_scores, alpha):
    """Reorder `cand_idx` by the interpolated score (stable: ties keep first-stage order).
    Returns (new_idx list, new_scores array aligned with new_idx)."""
    final = interpolate(first_scores, rr_scores, alpha)
    order = sorted(range(len(cand_idx)), key=lambda i: (-final[i], i))
    return [cand_idx[i] for i in order], final[order]
