"""Shared plumbing for the dev experiments (train-split protocol; never touches test qrels).

    ctx = DevContext(args)              # data + lazy F2LLM encoder + fp16 embedding cache
    ctx.query_scores(variants, idx)     # (n, D) cosine scores for (an average of) query variants
    record(...)                         # append one dev-table row (metrics, improved/worsened, adopt flag)

Selection rule: experiments are compared on the TUNE set (4,000 queries) only. The 1,000-query HOLDOUT
is never used here; `--use-holdout` exists only for an explicit, single confirmation of a chosen config.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval import dev_metrics as M  # noqa: E402
from src.eval.dev_data import load_dev  # noqa: E402
from src.retrieval.embedding_cache import get_or_build  # noqa: E402
from src.retrieval.query_variants import average_embeddings, format_query  # noqa: E402

DEV_DIR = ROOT / "outputs" / "dev"
RESULTS_JSONL = DEV_DIR / "results.jsonl"
CHOSEN = DEV_DIR / "chosen.json"
TABLE_MD = ROOT / "results" / "dev_table.md"
TEST_NDCG_REFERENCE = {                # published test NDCG@10 per preset (MTEB results repo), keyed
    "f2llm-v2-0.6b": 0.90446,          # so the dev-vs-test contamination check compares against the
    "f2llm-v2-1.7b": 0.93692,          # right model instead of silently reusing the 0.6B's number.
    "f2llm-v2-4b": 0.96102,
}
BASELINE_VARIANT = "registry+full"


def add_common_args(ap):
    ap.add_argument("--preset", default="f2llm-v2-0.6b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-queries", type=int, default=0,
                    help="subsample the tune set (seeded, fixed) for expensive experiments; 0 = all")
    ap.add_argument("--use-holdout", action="store_true",
                    help="evaluate on the reserved 1,000-query holdout instead of the tune set "
                         "(one final confirmation only; do not tune on it)")
    ap.add_argument("--config", default=str(CHOSEN), help="base pipeline config (JSON) to build on")
    return ap


class DevContext:
    def __init__(self, args):
        self.args = args
        d = load_dev()
        self.data = d
        self.doc_texts, self.query_texts = d["doc_texts"], d["query_texts"]
        self.rel = np.array(d["rel_idx"])
        pool = d["holdout_idx"] if getattr(args, "use_holdout", False) else d["tune_idx"]
        if getattr(args, "n_queries", 0):
            pool = sorted(random.Random(7).sample(pool, min(args.n_queries, len(pool))))
        self.idx = np.array(pool)
        self.split_name = "holdout" if getattr(args, "use_holdout", False) else "tune"
        from src.retrieval.model_presets import PRESETS
        self.preset = PRESETS[args.preset]
        self._enc = None

    # ---- model / cache -------------------------------------------------------------------------
    @property
    def model_key(self):
        p = self.preset
        return f"{p['model']}@{p['revision']}|msl{p['max_seq_length']}|{p['dtype']}"

    @property
    def encoder(self):
        if self._enc is None:
            from src.retrieval.dense_encoder import DenseEncoder
            p = self.preset
            self._enc = DenseEncoder(p["model"], self.args.device, p["max_seq_length"],
                                     trust_remote_code=p["trust_remote_code"], dtype=p["dtype"],
                                     revision=p["revision"], expect_eos=p["expect_eos"])
        return self._enc

    def _encode(self, texts):
        from src.retrieval.dense_encoder import encode_length_sorted
        bs = self.args.batch_size
        return encode_length_sorted(list(texts), lambda t: self.encoder.embed(t, batch_size=bs), bs)

    def release_encoder(self):
        """Free the dense (F2LLM) model's GPU memory once its embeddings are computed and cached. Call
        this before loading a second model (a reranker) on the same GPU -- with both resident a T4's
        14.56 GB fills up fast. `.encoder` lazily reloads a fresh model if this DevContext embeds text
        again afterward (a fresh load is idempotent: same preset/revision/dtype every time)."""
        if self._enc is not None:
            self._enc.release()
            self._enc = None

    def doc_emb(self):
        emb, meta = get_or_build("docs", self.doc_texts, self._encode, self.model_key)
        return emb, meta

    def query_emb(self, variant):
        """Embeddings for ALL 5,000 train queries under `variant` (cached; holdout needs no re-encode)."""
        texts = [format_query(t, variant) for t in self.query_texts]
        return get_or_build(f"q_{variant}", texts, self._encode, self.model_key)

    def query_scores(self, variants, idx=None):
        """(len(idx), D) cosine scores; several variants => embeddings averaged. Returns (S, added_ms/query)
        where added_ms is the encode cost beyond the baseline variant, amortised over the cached build."""
        idx = self.idx if idx is None else idx
        D, _ = self.doc_emb()
        embs, secs = [], 0.0
        for v in variants:
            e, meta = self.query_emb(v)
            embs.append(e)
            secs += meta.get("encode_seconds", 0.0)
        Q = embs[0] if len(embs) == 1 else average_embeddings(embs)
        base_secs = self.query_emb(BASELINE_VARIANT)[1].get("encode_seconds", 0.0)
        added_ms = 1000.0 * (secs - base_secs) / max(len(self.query_texts), 1)
        return Q[idx] @ D.T, added_ms

    # ---- ranks ---------------------------------------------------------------------------------
    def rel_of(self, idx=None):
        return self.rel[self.idx if idx is None else idx]

    def baseline_ranks(self):
        S, _ = self.query_scores([BASELINE_VARIANT])
        return M.ranks_from_scores(S, self.rel_of())

    def ranks_from_lists(self, lists, idx=None):
        rel = self.rel_of(idx)
        return np.array([M.rank_from_list(l, t) for l, t in zip(lists, rel)])


def print_code_version(script):
    """Print the commit the running code came from. A stale checkout (e.g. a Kaggle session cloned
    before the flags/behaviour you expect were pushed) otherwise shows up only as a confusing argparse
    error or, worse, as silently old behaviour producing results you then compare against new ones."""
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha, dirty = "", ""
    label = sha or "unknown (not a git checkout?)"
    print(f"[{script}] code version: {label}{' +local-changes' if dirty else ''} "
          f"-- if this is not the commit you expect, run: git pull", flush=True)
    return sha


class CacheMissError(RuntimeError):
    """A preset's embeddings are not on disk. Message names the exact command that builds them."""


def model_key_for(preset_name):
    """The embedding-cache model key for any registered preset (not just this ctx's own)."""
    from src.retrieval.model_presets import PRESETS
    p = PRESETS[preset_name]
    return f"{p['model']}@{p['revision']}|msl{p['max_seq_length']}|{p['dtype']}"


def cached_embeddings(preset_name, doc_texts, query_texts, variant=BASELINE_VARIANT):
    """(doc_emb, query_emb) float32 arrays for `preset_name`, from the on-disk cache ONLY -- never loads
    a model.

    Raises CacheMissError naming the command that would build the missing half, so a fusion experiment
    can never silently turn into a multi-GB GPU encoding run."""
    from src.retrieval.embedding_cache import load_cached
    key = model_key_for(preset_name)
    docs = load_cached("docs", doc_texts, key)
    queries = load_cached(f"q_{variant}", [format_query(t, variant) for t in query_texts], key)
    missing = [n for n, v in (("docs", docs), (f"q_{variant}", queries)) if v is None]
    if missing:
        raise CacheMissError(
            f"preset {preset_name!r} has no cached embeddings for: {', '.join(missing)}.\n"
            f"  Build them first (GPU, one pass over the corpus):\n"
            f"    python src/eval/dev_split.py --preset {preset_name} --device cuda\n"
            f"  Note the cache lives in ./data (gitignored) -- a fresh clone or a new Kaggle session "
            f"starts empty, so it has to be rebuilt there.")
    return docs[0], queries[0]              # load_cached returns (emb, meta); callers want the arrays


def load_base_config(path):
    from src.retrieval.pipeline import merge_config
    p = Path(path)
    return merge_config(json.loads(p.read_text()) if p.exists() else {})


def row_preset(row_or_ctx):
    """The first-stage preset a row/ctx belongs to. Rows written before multi-preset support (all the
    0.6B sweeps already on Kaggle) carry no 'preset' field at all -- for those, and for the rare ctx
    with no 'preset' attribute (e.g. a bare Namespace in a test), fall back to 'f2llm-v2-0.6b', the only
    preset that existed then, so old rows keep comparing against each other exactly as before."""
    if isinstance(row_or_ctx, dict):
        return row_or_ctx.get("preset") or "f2llm-v2-0.6b"
    return getattr(getattr(row_or_ctx, "args", None), "preset", None) or "f2llm-v2-0.6b"


def record(ctx, exp, group, ranks, base_ranks, config_patch, latency_ms=None, notes="", extra=None,
           ref_ranks=None):
    """Compute the dev-table row for one experiment and append it to results.jsonl.

    base_ranks: the CURRENT base pipeline (previous stages already adopted); improved/worsened/CI and the
                ADOPT flag are judged against it, so a stage is credited only for what it adds.
    ref_ranks:  the raw F2LLM baseline (registry+full); its delta is stored as vs_f2llm_* for the table.

    Every row is tagged with the first-stage preset (ctx.args.preset) so results from different F2LLM
    sizes (e.g. 0.6B vs 1.7B) never collide in results.jsonl/dev_table.md or get compared against each
    other by dev_select.py."""
    row = {"exp": exp, "group": group, "split": ctx.split_name, "preset": row_preset(ctx),
           "config_patch": config_patch,
           **M.summarize(ranks), "added_ms_per_query": latency_ms, "notes": notes, **(extra or {})}
    if base_ranks is not None:
        cmp = M.compare(ranks, base_ranks)
        row.update(cmp)
        row["adopt"] = bool(M.adopt(cmp))
    if ref_ranks is not None:
        ref = M.compare(ranks, ref_ranks)
        row.update({"vs_f2llm_delta": ref["delta_ndcg@10"], "vs_f2llm_improved": ref["improved"],
                    "vs_f2llm_worsened": ref["worsened"]})
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(format_row(row), flush=True)
    return row


def format_row(r):
    lat = "-" if r.get("added_ms_per_query") is None else f"{r['added_ms_per_query']:.1f}"
    imp = "-" if "improved" not in r else f"{r['improved']}/{r['worsened']}"
    d = "-" if "delta_ndcg@10" not in r else f"{r['delta_ndcg@10']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"
    if "vs_f2llm_delta" in r:
        d += f" (vs F2LLM {r['vs_f2llm_delta']:+.4f}, {r['vs_f2llm_improved']}/{r['vs_f2llm_worsened']})"
    return (f"| {row_preset(r)} | {r['group']} | {r['exp']} | {r['ndcg@10']:.4f} | {r['mrr@10']:.4f} | "
            f"{r['recall@1']:.3f} | {r['recall@10']:.3f} | {r['recall@100']:.3f} | {imp} | {d} | {lat} | "
            f"{'ADOPT' if r.get('adopt') else ''} | {r['split']} n={r['n']} |")


def render_table():
    """Rebuild results/dev_table.md from every row in results.jsonl (latest row per preset+exp+split wins).
    (preset, group, exp, split, n) is the dedup key so different F2LLM sizes' rows never overwrite each
    other -- rows from before multi-preset support (no 'preset' field) key under 'f2llm-v2-0.6b', see
    row_preset()."""
    if not RESULTS_JSONL.exists():
        return
    rows = {}
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[(row_preset(r), r["group"], r["exp"], r["split"], r["n"])] = r
    head = ("| preset | group | experiment | NDCG@10 | MRR@10 | R@1 | R@10 | R@100 | improved/worsened | "
            "dNDCG@10 [95% CI] | +ms/query (T4) | adopt | set |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    body = "\n".join(format_row(r) for r in rows.values())
    TABLE_MD.parent.mkdir(parents=True, exist_ok=True)
    TABLE_MD.write_text(
        "# Dev table (CoIR-APPS train queries vs full corpus; sweeps on the 4,000-query tune set)\n\n"
        "improved/worsened = queries whose relevant doc moved to a strictly better/worse rank vs the F2LLM "
        "baseline on the same queries. ADOPT = dNDCG@10 >= +0.005, 95% paired-bootstrap CI > 0, and "
        "worsened <= 0.5 x improved. +ms/query = added latency over the baseline on the T4 (see each "
        "script for how it is measured).\n\n" + head + "\n" + body + "\n", encoding="utf-8")
    print("wrote", TABLE_MD)


def ns(**kw):
    return argparse.Namespace(**kw)
