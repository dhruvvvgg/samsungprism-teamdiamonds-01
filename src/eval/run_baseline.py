"""Phase 1: real MTEB AppsRetrieval evaluation with the dense baseline. Run on Kaggle/Colab.

    !python src/eval/run_baseline.py --preset f2llm-v2-0.6b --device cuda --batch-size 16 \
        --out outputs/baseline_f2llm_v2_0.6b.json
    !python src/eval/run_baseline.py --model NAME [--device cuda] [--dtype fp16] [--batch-size 32] ...

--preset fills model / revision / prompts / max length / dtype from src/retrieval/model_presets.py;
any flag you pass explicitly still wins over the preset.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _peak_rss_mb():
    try:
        import resource  # linux/colab/kaggle
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        import psutil
        return psutil.Process().memory_info().rss / 1e6


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="named model config (see src/retrieval/model_presets.py)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None, help="pin a Hugging Face model revision")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cpu (default) or cuda; use cuda on Kaggle GPU")
    ap.add_argument("--dtype", default=None, choices=["fp32", "fp16", "bf16"],
                    help="model precision (fp16/bf16 need CUDA; a T4 has no native bf16 -> use fp16)")
    ap.add_argument("--fp16", action="store_true", help="shorthand for --dtype fp16")
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--doc-prefix", default="")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--expect-eos", action="store_true",
                    help="fail fast unless the tokenizer appends exactly one EOS token")
    ap.add_argument("--add-eos-token", action="store_true",
                    help="force tokenizer add_eos_token=True (only if --expect-eos reports it missing)")
    ap.add_argument("--no-sort", action="store_true",
                    help="legacy: encode in MTEB dataset order instead of length-sorted batches")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "baseline_results.json"))
    return ap


def parse_args(argv=None):
    from src.retrieval.model_presets import apply_preset
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.preset:
        a = apply_preset(a, vars(ap.parse_args([])), a.preset)
    a.query_prefix = a.query_prefix.replace("\\n", "\n")  # allow a literal \n in notebook args
    a.doc_prefix = a.doc_prefix.replace("\\n", "\n")
    return a


def _versions():
    import importlib.metadata as md
    out = {}
    for p in ("torch", "transformers", "sentence-transformers", "mteb"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def main():
    a = parse_args()

    import mteb
    import torch
    from src.retrieval.dense_encoder import make_mteb_encoder

    if a.device is None and torch.cuda.is_available():
        print("NOTE: a CUDA GPU is available but --device is not set; running on CPU. "
              "Add --device cuda for a much faster run.")
    model = make_mteb_encoder(
        a.model, a.device, a.max_seq_length, sort_batches=not a.no_sort, batch_size=a.batch_size,
        query_prefix=a.query_prefix, doc_prefix=a.doc_prefix, trust_remote_code=a.trust_remote_code,
        fp16=a.fp16, dtype=a.dtype, revision=a.revision, expect_eos=a.expect_eos,
        tokenizer_kwargs={"add_eos_token": True} if a.add_eos_token else None)
    task = mteb.get_task("AppsRetrieval")
    t0 = time.time()
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": a.batch_size}, cache=None)
    elapsed = time.time() - t0
    d = list(result.task_results)[0].to_dict()
    dense = model.dense
    d["_run_info"] = {
        "model": dense.model_name, "preset": a.preset, "revision": a.revision,
        "total_eval_seconds": round(elapsed, 1), "peak_rss_mb": round(_peak_rss_mb(), 1),
        "batch_size": a.batch_size, "device": a.device or "cpu",
        "final_dtype": str(next(dense.model.parameters()).dtype),
        "requested_dtype": a.dtype or ("fp16" if a.fp16 else None),
        "fell_back_to_fp32": dense.fell_back_to_fp32,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_seq_length": a.max_seq_length, "length_sorted_batches": not a.no_sort,
        "query_prefix": a.query_prefix, "doc_prefix": a.doc_prefix, "versions": _versions(),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=2, default=str))
    scores = d.get("scores", {}).get("test", [{}])[0]
    print("ndcg_at_10:", scores.get("ndcg_at_10"), "| mrr_at_10:", scores.get("mrr_at_10"))
    print("run_info:", d["_run_info"])


if __name__ == "__main__":
    main()
