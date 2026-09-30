"""B. Contrastive fine-tuning of F2LLM-v2-0.6B on the tune split, evaluated on the reserved holdout.

    # 1. mine hard negatives from the BASE model (tune queries only) -- ~10 min on a T4
    python src/train/finetune_lite.py --stage mine --device cuda

    # 2. train the LoRA adapter (resumable) -- ~60-90 min on a T4
    python src/train/finetune_lite.py --stage train --device cuda

    # 3. evaluate on the 1,000-query HOLDOUT, never seen in training or mining. Base 0.6B and the tuned
    #    model are scored on the SAME queries in one run, with per-query scores and a paired bootstrap
    python src/train/finetune_lite.py --stage eval --device cuda
    #    optional, once: per-query 1.7B ranks, so the tuned model can also be paired against the 1.7B
    python src/train/finetune_lite.py --stage eval --base-only --preset f2llm-v2-1.7b --device cuda

The split discipline is the whole point, so it is enforced in code rather than by care:

  * training pairs come from the 4,000-query **tune** partition;
  * hard negatives are mined with the **base** model over **tune** queries only;
  * the 1,000-query **holdout** (seed 13) is touched by `--stage eval` and nothing else. The earlier
    1,000-query dev slice cannot be used here at all -- it is drawn from the tune queries the model is
    trained on, so a number from it would be measuring memorisation.

Base references on the holdout: 0.6B 0.9054 / 0.8874, 1.7B 0.9306 / 0.9169 (NDCG@10 / MRR@10). The
fine-tuned model is a P0 candidate **only** if it clears the adoption rule against the 1.7B on the
holdout; clearing it against the 0.6B makes it the lite model, which is the realistic outcome. The rule
is the project's usual one: NDCG@10 gain >= 0.005, 95% paired-bootstrap CI excluding zero, and worsened
queries <= half of the improved ones.

LoRA, not full fine-tuning: on a T4 the 0.6B needs ~1.2 GB for fp16 weights, but full fine-tuning needs
optimiser state for every parameter (~7 GB in Adam moments alone) plus activations, which does not leave
room for a useful batch. LoRA trains ~0.5% of the parameters, so the optimiser state is negligible and
the batch can be large enough for in-batch negatives to matter -- and in-batch negatives are most of
where contrastive learning gets its signal.

Every stage checkpoints to disk and resumes: a killed Kaggle session re-runs the same command.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "finetune_lite"
NEGATIVES = OUT_DIR / "hard_negatives.json"
ADAPTER = OUT_DIR / "adapter"
STATE = OUT_DIR / "train_state.json"
HOLDOUT_REFERENCE = {"f2llm-v2-0.6b": {"ndcg@10": 0.9054, "mrr@10": 0.8874},
                     "f2llm-v2-1.7b": {"ndcg@10": 0.9306, "mrr@10": 0.9169}}


def load_split(preset):
    """(data, tune_idx, holdout_idx) with the fixed seed-13 partition."""
    from src.eval.dev_data import load_dev
    d = load_dev()
    return d, list(d["tune_idx"]), list(d["holdout_idx"])


def encoder_for(preset, device, adapter=None):
    from src.retrieval.dense_encoder import DenseEncoder
    from src.retrieval.model_presets import PRESETS
    p = PRESETS[preset]
    enc = DenseEncoder(p["model"], device, p["max_seq_length"],
                       trust_remote_code=p["trust_remote_code"], dtype=p["dtype"],
                       revision=p["revision"], expect_eos=p["expect_eos"])
    if adapter:
        attach_adapter(enc, adapter)
    return enc, p


def attach_adapter(encoder, adapter_dir):
    """Load a trained LoRA adapter onto a DenseEncoder's transformer, in place."""
    from peft import PeftModel
    inner = encoder.model[0].auto_model
    encoder.model[0].auto_model = PeftModel.from_pretrained(inner, str(adapter_dir))
    print(f"[ft] adapter loaded from {adapter_dir}", flush=True)
    return encoder


# --- stage 1: mine hard negatives -------------------------------------------------------------------

def stage_mine(args):
    from src.retrieval.dense_encoder import encode_length_sorted
    from src.retrieval.query_variants import format_query
    d, tune, holdout = load_split(args.preset)
    if NEGATIVES.exists() and not args.force:
        print(f"[ft] {NEGATIVES} already exists; pass --force to re-mine")
        return 0
    enc, _ = encoder_for(args.preset, args.device)
    doc_texts = d["doc_texts"]
    print(f"[ft] encoding {len(doc_texts)} documents with the BASE model ...", flush=True)
    D = encode_length_sorted(doc_texts, lambda t: enc.embed(t, batch_size=args.batch_size),
                             args.batch_size)
    q_texts = [format_query(d["query_texts"][i], "registry+full") for i in tune]
    print(f"[ft] encoding {len(q_texts)} TUNE queries (holdout excluded) ...", flush=True)
    Q = encode_length_sorted(q_texts, lambda t: enc.embed(t, batch_size=args.batch_size),
                             args.batch_size)
    rel = [d["rel_idx"][i] for i in tune]
    out = {}
    for row, (qi, positive) in enumerate(zip(tune, rel)):
        scores = Q[row] @ D.T
        scores[positive] = -1e9                       # never mine the positive as its own negative
        top = np.argpartition(-scores, args.n_negatives)[:args.n_negatives]
        out[str(qi)] = [int(x) for x in top[np.argsort(-scores[top])]]
        if (row + 1) % 500 == 0:
            print(f"[ft]   mined {row + 1}/{len(tune)}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from src.utils_io import write_json_atomic
    write_json_atomic(NEGATIVES, {"preset": args.preset, "n_negatives": args.n_negatives,
                                  "split": "tune", "negatives": out})
    print(f"[ft] wrote {len(out)} negative lists -> {NEGATIVES}")
    enc.release()
    return 0


# --- stage 2: train ------------------------------------------------------------------------------------

def stage_train(args):
    import torch
    from torch.utils.data import DataLoader
    from src.retrieval.query_variants import format_query

    if not NEGATIVES.exists():
        raise SystemExit(f"Mine hard negatives first: --stage mine (expected {NEGATIVES})")
    mined = json.loads(NEGATIVES.read_text(encoding="utf-8"))
    d, tune, holdout = load_split(args.preset)
    assert not (set(tune) & set(holdout)), "tune and holdout overlap -- the split is broken"

    enc, preset = encoder_for(args.preset, args.device)
    model = enc.model
    inner = model[0].auto_model
    tokenizer = model.tokenizer

    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model[0].auto_model = get_peft_model(inner, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[ft] LoRA r={args.lora_r}: training {trainable:,} of {total:,} parameters "
          f"({100 * trainable / total:.2f}%)", flush=True)
    model.to(args.device)
    if args.gradient_checkpointing:
        try:
            model[0].auto_model.gradient_checkpointing_enable()
        except Exception as exc:  # noqa: BLE001
            print(f"[ft] gradient checkpointing unavailable ({exc})", flush=True)

    pairs = [(format_query(d["query_texts"][i], "registry+full"), d["doc_texts"][d["rel_idx"][i]],
              mined["negatives"].get(str(i), [])) for i in tune]
    start_epoch, start_step = 0, 0
    if STATE.exists() and not args.force:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        start_epoch, start_step = state.get("epoch", 0), state.get("step", 0)
        if ADAPTER.exists():
            from peft import PeftModel
            model[0].auto_model = PeftModel.from_pretrained(inner, str(ADAPTER), is_trainable=True)
            model.to(args.device)
            print(f"[ft] resumed from {ADAPTER} at epoch {start_epoch} step {start_step}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))

    def embed(texts):
        enc_in = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=args.max_len).to(args.device)
        out = model[0].auto_model(**enc_in).last_hidden_state
        mask = enc_in["attention_mask"]
        last = mask.sum(dim=1) - 1                        # last-token pooling, as F2LLM does
        pooled = out[torch.arange(out.size(0), device=out.device), last]
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    loader = DataLoader(list(range(len(pairs))), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from src.utils_io import write_json_atomic
    t_start, losses = time.time(), []
    for epoch in range(start_epoch, args.epochs):
        for step, batch in enumerate(loader):
            if epoch == start_epoch and step < start_step:
                continue
            idxs = [int(x) for x in batch]
            queries = [pairs[i][0] for i in idxs]
            docs = [pairs[i][1] for i in idxs]
            for i in idxs[:args.hard_per_batch]:          # a few mined negatives per batch
                negs = pairs[i][2]
                if negs:
                    docs.append(d["doc_texts"][negs[step % len(negs)]])
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=args.device.startswith("cuda")):
                qe, de = embed(queries), embed(docs)
                logits = (qe @ de.T) / args.temperature   # in-batch + hard negatives
                target = torch.arange(len(queries), device=logits.device)
                loss = torch.nn.functional.cross_entropy(logits, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
            if (step + 1) % args.log_every == 0:
                print(f"[ft] epoch {epoch} step {step + 1}/{len(loader)} "
                      f"loss {np.mean(losses[-args.log_every:]):.4f} "
                      f"({(time.time() - t_start) / 60:.1f} min)", flush=True)
            if (step + 1) % args.checkpoint_every == 0:
                model[0].auto_model.save_pretrained(str(ADAPTER))
                write_json_atomic(STATE, {"epoch": epoch, "step": step + 1,
                                          "loss": float(np.mean(losses[-100:])),
                                          "preset": args.preset})
                print(f"[ft] checkpointed to {ADAPTER}", flush=True)
        start_step = 0
    model[0].auto_model.save_pretrained(str(ADAPTER))
    revision = {"base_model": preset["model"], "base_revision": preset["revision"],
                "adapter_dir": str(ADAPTER), "lora_r": args.lora_r, "epochs": args.epochs,
                "lr": args.lr, "batch_size": args.batch_size, "temperature": args.temperature,
                "trained_on": "tune (4,000 queries)", "holdout_untouched": True,
                "final_loss": float(np.mean(losses[-100:])) if losses else None,
                "minutes": round((time.time() - t_start) / 60, 1),
                "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json_atomic(OUT_DIR / "adapter_revision.json", revision, indent=2)
    write_json_atomic(STATE, {"epoch": args.epochs, "step": 0, "done": True})
    print(f"[ft] done in {revision['minutes']} min -> {ADAPTER}")
    print(json.dumps(revision, indent=2))
    return 0


# --- stage 3: evaluate on the holdout -------------------------------------------------------------------

RESULTS_JSON = ROOT / "results" / "finetune_lite.json"
PER_QUERY = OUT_DIR / "holdout_per_query.json"
REFERENCE_PRESET = "f2llm-v2-1.7b"
N_BOOT = 2000


def reference_ranks_path(preset):
    return OUT_DIR / f"holdout_ranks_{preset}.json"


def holdout_ranks(d, holdout, preset, device, batch_size, adapter=None):
    """Rank of the relevant document for every holdout query, from one model (base, or base + adapter).
    The encoder is released before returning so two models are never resident at once."""
    from src.eval import dev_metrics as M
    from src.retrieval.dense_encoder import encode_length_sorted
    from src.retrieval.query_variants import format_query
    enc, _ = encoder_for(preset, device, adapter=adapter)
    try:
        D = encode_length_sorted(d["doc_texts"], lambda t: enc.embed(t, batch_size=batch_size),
                                 batch_size)
        q_texts = [format_query(d["query_texts"][i], "registry+full") for i in holdout]
        Q = encode_length_sorted(q_texts, lambda t: enc.embed(t, batch_size=batch_size), batch_size)
    finally:
        enc.release()
    rel = np.array([d["rel_idx"][i] for i in holdout])
    return M.ranks_from_scores(Q @ D.T, rel)


def per_query_scores(ranks):
    """Per-query NDCG@10 and MRR@10 (plus the raw rank) -- what a paired comparison is made of."""
    from src.eval import dev_metrics as M
    r = np.asarray(ranks)
    return {"rank": [int(x) for x in r], "ndcg@10": [float(x) for x in M.ndcg10_per_query(r)],
            "mrr@10": [float(x) for x in np.where(r <= 10, 1.0 / r, 0.0)]}


def _mrr_delta_ci(ranks, base_ranks, n_boot, seed):
    r, b = np.asarray(ranks), np.asarray(base_ranks)
    d = np.where(r <= 10, 1.0 / r, 0.0) - np.where(b <= 10, 1.0 / b, 0.0)
    idx = np.random.RandomState(seed).randint(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {"delta_mrr@10": float(d.mean()), "mrr_ci_low": float(np.percentile(boots, 2.5)),
            "mrr_ci_high": float(np.percentile(boots, 97.5))}


def paired_comparison(ranks, base_ranks, n_boot=N_BOOT, seed=0):
    """`ranks` vs `base_ranks` on the SAME queries: NDCG@10 and MRR@10 deltas with 95% paired-bootstrap
    CIs, improved / worsened / same counts, and the adoption verdict under the project's rule
    (NDCG@10 gain >= 0.005, CI excludes zero, worsened <= half of improved)."""
    from src.eval import dev_metrics as M
    cmp = M.compare(ranks, base_ranks, n_boot=n_boot, seed=seed)
    cmp.update(_mrr_delta_ci(ranks, base_ranks, n_boot, seed))
    cmp["adopt"] = bool(M.adopt(cmp))
    cmp["rule"] = "dNDCG@10 >= +0.005, 95% paired CI excludes 0, worsened <= 0.5 x improved"
    return cmp


def paired_report(tuned_ranks, base_ranks, reference_ranks=None, n_boot=N_BOOT):
    """Everything the eval stage reports: summaries, per-query scores, and the paired comparisons of
    the tuned model against the base 0.6B and, if its ranks were loaded, against the 1.7B."""
    from src.eval import dev_metrics as M
    out = {"tuned": M.summarize(tuned_ranks), "base_0.6b": M.summarize(base_ranks),
           "vs_base_0.6b": paired_comparison(tuned_ranks, base_ranks, n_boot),
           "per_query": {"tuned": per_query_scores(tuned_ranks), "base_0.6b": per_query_scores(base_ranks)}}
    if reference_ranks is not None:
        out["reference_1.7b"] = M.summarize(reference_ranks)
        out["vs_1.7b"] = paired_comparison(tuned_ranks, reference_ranks, n_boot)
        out["per_query"]["reference_1.7b"] = per_query_scores(reference_ranks)
    return out


def load_reference_ranks(path, holdout):
    """Ranks saved by `--stage eval --base-only --preset f2llm-v2-1.7b`, or None if there are none.
    Refuses a file made on a different holdout: pairing across different queries would be meaningless."""
    path = Path(path)
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("holdout_idx") != [int(i) for i in holdout]:
        raise SystemExit(f"{path} was made on a different holdout; re-run "
                         f"`--stage eval --base-only --preset {REFERENCE_PRESET}`")
    return np.array(saved["ranks"])


def stage_eval(args):
    """The ONLY stage that reads the holdout. Evaluates the base 0.6B and the tuned model on the same
    holdout queries in one run, so their per-query scores pair exactly."""
    from src.eval import dev_metrics as M
    from src.utils_io import write_json_atomic

    d, tune, holdout = load_split(args.preset)
    n_boot = getattr(args, "n_boot", N_BOOT)

    if getattr(args, "base_only", False):
        # e.g. `--preset f2llm-v2-1.7b`: save the base model's per-query ranks so a later tuned-model
        # eval can be paired against it. Still the eval stage, still the holdout, no adapter involved.
        ranks = holdout_ranks(d, holdout, args.preset, args.device, args.batch_size)
        write_json_atomic(reference_ranks_path(args.preset),
                          {"preset": args.preset, "holdout_idx": [int(i) for i in holdout],
                           "ranks": [int(x) for x in ranks], "summary": M.summarize(ranks),
                           "when": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"[ft] {args.preset} on the holdout: {M.summarize(ranks)}")
        print(f"[ft] per-query ranks saved -> {reference_ranks_path(args.preset)}")
        return 0

    adapter = ADAPTER if (args.adapter is None) else Path(args.adapter)
    if not adapter.exists():
        raise SystemExit(f"No adapter at {adapter}; train one first (--stage train)")
    ref_path = getattr(args, "ref_ranks", None) or reference_ranks_path(REFERENCE_PRESET)
    reference = load_reference_ranks(ref_path, holdout)
    print(f"[ft] evaluating BASE and TUNED {args.preset} on the same {len(holdout)}-query HOLDOUT "
          f"(never trained or mined on)", flush=True)
    base_ranks = holdout_ranks(d, holdout, args.preset, args.device, args.batch_size)
    tuned_ranks = holdout_ranks(d, holdout, args.preset, args.device, args.batch_size, adapter=adapter)
    rep = paired_report(tuned_ranks, base_ranks, reference, n_boot=n_boot)

    def line(name, s):
        return f"  {name:<14}: NDCG@10 {s['ndcg@10']:.4f}  MRR@10 {s['mrr@10']:.4f}"

    def verdict(label, c):
        lo, hi = c["ci_low"], c["ci_high"]
        return (f"  {label}: dNDCG@10 {c['delta_ndcg@10']:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]  "
                f"dMRR@10 {c['delta_mrr@10']:+.4f}  improved {c['improved']} / worsened {c['worsened']} "
                f"/ same {c['same']}  -> {'ADOPT' if c['adopt'] else 'do not adopt'}")
    print("\n" + "=" * 78)
    print(f"[ft] PAIRED HOLDOUT EVALUATION, {len(holdout)} queries (same queries for every model)")
    print("=" * 78)
    print(line("tuned", rep["tuned"]))
    print(line("base 0.6B", rep["base_0.6b"]))
    if reference is not None:
        print(line("1.7B", rep["reference_1.7b"]))
    print()
    print(verdict("tuned vs base 0.6B", rep["vs_base_0.6b"]))
    print("     -> becomes the lite model" if rep["vs_base_0.6b"]["adopt"] else "     -> stays a candidate only")
    if reference is not None:
        print(verdict("tuned vs 1.7B     ", rep["vs_1.7b"]))
        print("     -> P0 candidate" if rep["vs_1.7b"]["adopt"] else "     -> not a P0 candidate")
    else:
        print(f"  tuned vs 1.7B     : not available -- no per-query 1.7B ranks at {ref_path}. Run "
              f"`--stage eval --base-only --preset {REFERENCE_PRESET}` once to create them.")
    print(f"\n  rule: {rep['vs_base_0.6b']['rule']}")
    measured, recorded = rep["base_0.6b"]["ndcg@10"], HOLDOUT_REFERENCE["f2llm-v2-0.6b"]["ndcg@10"]
    if abs(measured - recorded) > 0.002:
        print(f"  WARNING: the base 0.6B measured {measured:.4f} here vs {recorded:.4f} recorded earlier; "
              f"the paired comparison uses the number measured in this run.")

    per_query = rep.pop("per_query")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(PER_QUERY, {"preset": args.preset, "holdout_idx": [int(i) for i in holdout],
                                  **per_query})
    write_json_atomic(RESULTS_JSON,
                      {"preset": args.preset, "adapter": str(adapter), "n_holdout": len(holdout),
                       "finetuned": rep["tuned"], "base_0.6b": rep["base_0.6b"],
                       "base_1.7b": rep.get("reference_1.7b", HOLDOUT_REFERENCE["f2llm-v2-1.7b"]),
                       "paired": {k: v for k, v in rep.items() if k.startswith("vs_")},
                       "per_query_file": str(PER_QUERY),
                       "beats_base_0.6b": bool(rep["vs_base_0.6b"]["adopt"]),
                       "beats_base_1.7b": bool(rep.get("vs_1.7b", {}).get("adopt", False)),
                       "vs_1.7b_paired": reference is not None,
                       "when": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["mine", "train", "eval"])
    ap.add_argument("--preset", default="f2llm-v2-0.6b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-negatives", type=int, default=8)
    ap.add_argument("--hard-per-batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=512,
                    help="training sequence length; shorter than inference to fit a T4")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-only", action="store_true",
                    help="eval stage: evaluate the base --preset alone and save its per-query holdout "
                         "ranks (run once with --preset f2llm-v2-1.7b to pair the tuned model with it)")
    ap.add_argument("--ref-ranks", default=None,
                    help="eval stage: per-query 1.7B ranks file (default: the one --base-only wrote)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    return {"mine": stage_mine, "train": stage_train, "eval": stage_eval}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
