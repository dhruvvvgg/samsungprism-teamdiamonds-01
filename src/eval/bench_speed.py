"""Encoding-speed benchmark on a small real sample: fp32 vs fp16, MTEB-style in-order batches vs
length-sorted. Prints texts/s for each config. fp16 rows are skipped when no CUDA GPU is present.

    !python src/eval/bench_speed.py --model nomic-ai/CodeRankEmbed --trust-remote-code \
        --device cuda --max-seq-length 1024 --n 256
"""
import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def timed(fn):
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--n", type=int, default=128, help="texts (half queries, half docs)")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    import torch
    from src.eval.dev_data import load_dev
    from src.retrieval.dense_encoder import DenseEncoder

    d = load_dev()   # TRAIN-split texts only; the test split is reserved for the official run
    corpus, queries = dict(zip(d["doc_ids"], d["doc_texts"])), dict(zip(d["query_ids"], d["query_texts"]))
    random.seed(1)
    texts = (random.sample(sorted(queries.values()), a.n // 2) +
             random.sample(sorted(corpus.values()), a.n - a.n // 2))
    random.shuffle(texts)  # MTEB feeds queries / docs in dataset order, i.e. unsorted
    enc = DenseEncoder(a.model, a.device, a.max_seq_length, trust_remote_code=a.trust_remote_code)
    enc.embed(texts[:4], batch_size=4)  # warm-up

    def in_order():  # what the adapter does today: one embed() call per dataloader batch
        for i in range(0, len(texts), a.batch_size):
            enc.embed(texts[i:i + a.batch_size], batch_size=a.batch_size)

    def sorted_all():  # one embed() call: sentence-transformers sorts by length internally
        enc.embed(texts, batch_size=a.batch_size)

    rows = []
    for dtype in ("fp32", "fp16"):
        if dtype == "fp16":
            if not str(enc.model.device).startswith("cuda"):
                print("fp16 skipped: model is not on a CUDA device (no GPU here)")
                break
            enc.model.half()
        for name, fn in (("in-order batches (current)", in_order), ("length-sorted", sorted_all)):
            dt = timed(fn)
            rows.append((dtype, name, dt, len(texts) / dt))
            print(f"{dtype:5s} {name:28s} {dt:7.1f}s  {len(texts) / dt:7.2f} texts/s", flush=True)
    if torch.cuda.is_available():
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
