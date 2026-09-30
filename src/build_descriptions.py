"""F (part 1). Write a one-line description of every snippet with a small local code LLM, offline.

    python src/build_descriptions.py --device cuda                  # the full corpus
    python src/build_descriptions.py --limit 200 --device cuda      # a sample first
    python src/build_descriptions.py --mock                         # no model, for tests and CI

Why: APPS queries are problem statements in English and the documents are bare Python solutions. The
embedder has to bridge that gap on its own. Giving each snippet a one-line English description produces
a second, prose-shaped view of the same document, which can be embedded and fused with the code view.

Qwen2.5-Coder-1.5B-Instruct by default: it is a code model, it is small enough to run beside nothing
else on a T4, and it needs **no API key** -- the whole point is that this stays reproducible offline.

**Resumable by construction.** Descriptions are appended to a JSONL file keyed by the SHA-256 of the
snippet, flushed every `--checkpoint-every` items. A killed Kaggle session loses at most that many
generations: rerun the same command and it skips everything already in the file. The key is the content
hash, not the row index, so a corpus that changes order does not invalidate the cache.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
PROMPT = ("Describe what this Python code does in ONE short English sentence. "
          "Say what it computes, not how. No code, no preamble.\n\n```python\n{code}\n```")
MAX_CODE_CHARS = 3000


def snippet_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def load_done(path):
    """{key: description} already generated. Tolerates a truncated last line from a killed run."""
    out = {}
    path = Path(path)
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a half-written final line is simply redone
            if rec.get("key") and rec.get("description"):
                out[rec["key"]] = rec["description"]
    return out


def mock_description(text):
    """Deterministic stand-in: the first identifiers, as a sentence. No model, no download."""
    import re
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)[:8]
    return "code that " + " ".join(w.lower() for w in words if len(w) > 2)[:120]


def clean(line):
    """One sentence, no markdown, no leading 'This function ...' boilerplate."""
    line = (line or "").strip().split("\n")[0].strip().strip("`").strip()
    for prefix in ("This function ", "This code ", "The function ", "The code ", "It "):
        if line.startswith(prefix):
            line = line[len(prefix):]
            break
    return line[:220].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="apps", choices=["apps", "mock"],
                    help="apps: the 8,765-document corpus via the dev loader (train qrels only)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents (0 = all)")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--mock", action="store_true", help="no model; deterministic placeholder text")
    ap.add_argument("--out", default=str(ROOT / "data" / "cache" / "descriptions.jsonl"))
    a = ap.parse_args()

    if a.source == "mock":
        from src.versioning.fixture import build_fixture
        texts = [s["versions"][0]["text"] for s in build_fixture(a.limit or 20, 1, seed=0)["snippets"]]
    else:
        from src.eval.dev_data import load_dev
        texts = load_dev()["doc_texts"]
    if a.limit:
        texts = texts[:a.limit]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [(snippet_key(t), t) for t in texts if snippet_key(t) not in done]
    print(f"[desc] {len(texts)} documents, {len(done)} already described, {len(todo)} to do")
    if not todo:
        print("[desc] nothing to do -- the cache already covers this corpus")
        return 0

    if a.mock:
        generate = lambda batch: [mock_description(t) for _, t in batch]  # noqa: E731
        print("[desc] MOCK: placeholder descriptions, no model loaded", flush=True)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[desc] loading {a.model} on {a.device} ...", flush=True)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.float16 if a.device.startswith("cuda") else torch.float32)
        model.to(a.device).eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        print(f"[desc] loaded in {time.time() - t0:.1f}s", flush=True)

        def generate(batch):
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(code=t[:MAX_CODE_CHARS])}],
                tokenize=False, add_generation_prompt=True) for _, t in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(a.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            return [clean(s) for s in tok.batch_decode(new, skip_special_tokens=True)]

    written, t_start = 0, time.time()
    with open(out_path, "a", encoding="utf-8") as sink:
        for start in range(0, len(todo), a.batch_size):
            batch = todo[start:start + a.batch_size]
            try:
                descriptions = generate(batch)
            except Exception as exc:  # noqa: BLE001  one bad batch must not lose the whole run
                print(f"[desc] batch at {start} failed ({type(exc).__name__}: {exc}); skipping",
                      flush=True)
                continue
            for (key, text), description in zip(batch, descriptions):
                sink.write(json.dumps({"key": key, "description": description or "unknown code",
                                       "chars": len(text)}) + "\n")
                written += 1
            if written % a.checkpoint_every < a.batch_size:
                sink.flush()
                done_n = len(done) + written
                rate = written / max(time.time() - t_start, 1e-9)
                left = (len(todo) - written) / max(rate, 1e-9)
                print(f"[desc] {done_n}/{len(texts)} | {rate:.1f} docs/s | "
                      f"eta {left / 60:.1f} min", flush=True)
    elapsed = time.time() - t_start
    print(f"[desc] wrote {written} descriptions in {elapsed / 60:.1f} min "
          f"({written / max(elapsed, 1e-9):.2f} docs/s) -> {out_path}")
    sample = load_done(out_path)
    for key, text in list(zip([snippet_key(t) for t in texts], texts))[:3]:
        print(f"\n  {text.strip().splitlines()[0][:70]}\n    -> {sample.get(key, '(missing)')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
