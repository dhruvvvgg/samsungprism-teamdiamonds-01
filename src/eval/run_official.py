"""Official run: the chosen pipeline through `mteb.evaluate` on AppsRetrieval (test split), writing
`appsretrieval_results.json` in MTEB's format (`task_result.to_dict()`, same as the reference pattern).

THIS IS THE ONLY SCRIPT THAT TOUCHES THE TEST SPLIT. To keep the "one chosen candidate" rule honest it
refuses to run without --confirm-test and refuses a second run (a lock file records the first) unless
--allow-rerun is passed.

    !python src/eval/run_official.py --config configs/official_f2llm17b_noreranker.json \
        --device cuda --confirm-test

One run produces everything the submission needs, out of work it has already paid for -- no second
encoding pass and no second evaluation:

    outputs/appsretrieval_results.json     MTEB's own result dict (+ _run_info)
    outputs/appsretrieval_rankings.json    ordered top-100 document ids per test query
    outputs/submission_checksums.json      sha256 of the two files above + the index files
    runtime_index/                         corpus embeddings, ids, texts, manifest -- the served index

Components that cannot run through mteb.evaluate as an AbsEncoder (BM25 fusion, reranking, LLM re-judge)
run inside a SearchProtocol model, which mteb.evaluate does accept (see src/retrieval/search_model.py).
A config with everything disabled reproduces the plain F2LLM baseline.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LOCK = ROOT / "outputs" / "dev" / "OFFICIAL_RUN_DONE.json"

# Keys a config file may carry that are NOT pipeline settings, and so must be removed before
# merge_config() (which rejects unknown keys, deliberately, to catch typos). Anything added to a config
# without being added here would abort the run.
NON_PIPELINE_KEYS = ("_comment", "preset", "reference")


def reference_for(raw_reference, preset_name):
    """(value, label) to compare this run's NDCG@10 against, or (None, reason).

    Per config, NOT one global constant: the candidate configurations differ from each other by several
    times the size of the gap being checked, so a single hard-coded number would raise a false alarm on
    one config while staying silent on a real problem in another. A config states its own expectation in
    a "reference" key; without one, the published MTEB score for its first-stage model is used."""
    if isinstance(raw_reference, dict) and isinstance(raw_reference.get("ndcg_at_10"), (int, float)):
        return float(raw_reference["ndcg_at_10"]), raw_reference.get("label") or "config reference"
    from src.eval.dev_lib import TEST_NDCG_REFERENCE
    published = TEST_NDCG_REFERENCE.get(preset_name)
    if isinstance(published, (int, float)):
        return float(published), f"published MTEB test score for {preset_name} (no 'reference' in config)"
    return None, (f"config has no 'reference' key and {preset_name!r} has no published score in "
                  f"TEST_NDCG_REFERENCE")


def index_meta(cfg, preset_name, preset, variant):
    """Manifest fields for the exported runtime index, or a string saying why it cannot be exported.

    The manifest has to pin down how a served query must be formatted. That is only well defined when
    the run used exactly one dense variant over the untouched query text; anything else (averaged
    variants, a text transform) cannot be reproduced from a single stored prefix, and exporting a
    manifest that quietly disagrees with the run would be worse than exporting nothing."""
    from src.retrieval.query_variants import format_query, parse_variant
    variants = cfg["dense_variants"]
    if len(variants) != 1:
        return (f"the run averaged {len(variants)} query variants ({variants}); a runtime index stores "
                f"one query prefix, so it cannot represent this configuration")
    _, transform = parse_variant(variant)
    if transform != "full":
        return (f"variant {variant!r} rewrites the query text (transform={transform!r}); the runtime "
                f"index only stores a prefix, so serving it would not match this run")
    return {
        "preset": preset_name,
        "model": preset["model"],
        "revision": preset["revision"],
        "query_prefix": format_query("", variant),     # exactly what search() prepended, minus the text
        "doc_prefix": preset.get("doc_prefix", ""),
        "dense_variant": variant,
        "max_seq_length": preset["max_seq_length"],
        "dtype": preset["dtype"],
        "trust_remote_code": preset["trust_remote_code"],
        "expect_eos": preset.get("expect_eos", False),
        "task": "AppsRetrieval",
        "split": "test",
        "source": "exported from the official run (no re-encoding)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="pipeline config JSON. Prefer a version-controlled file under configs/ over "
                         "outputs/dev/chosen*.json, which dev_select.py rewrites (mutable Kaggle state)")
    ap.add_argument("--preset", default=None,
                    help="first-stage embedder preset. May instead be pinned by a 'preset' key in "
                         "--config, which keeps the model and pipeline from being mismatched")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "appsretrieval_results.json"))
    ap.add_argument("--rankings-out", default=str(ROOT / "outputs" / "appsretrieval_rankings.json"))
    ap.add_argument("--checksums-out", default=str(ROOT / "outputs" / "submission_checksums.json"))
    ap.add_argument("--rankings-depth", type=int, default=100,
                    help="document ids kept per query in the rankings file (default 100)")
    ap.add_argument("--index-dir", default=str(ROOT / "runtime_index"),
                    help="where to export the served index built from this run's corpus embeddings")
    ap.add_argument("--no-index-export", action="store_true",
                    help="skip the runtime_index/ export (the evaluation itself is unaffected)")
    ap.add_argument("--confirm-test", action="store_true", help="required: acknowledges this touches the test split")
    ap.add_argument("--allow-rerun", action="store_true")
    a = ap.parse_args()
    if not a.confirm_test:
        raise SystemExit("Refusing to run: this evaluates on the TEST split. Pass --confirm-test once the "
                         "candidate was chosen on dev (outputs/dev/chosen.json).")
    if LOCK.exists() and not a.allow_rerun:
        raise SystemExit(f"Refusing to run again: an official run already happened ({LOCK}). "
                         "Pass --allow-rerun only if you accept spending another test-split evaluation.")

    import mteb

    from src.retrieval.dense_encoder import DenseEncoder
    from src.retrieval.model_presets import PRESETS
    from src.retrieval.pipeline import merge_config
    from src.retrieval.search_model import HybridSearchModel
    from src.runtime_index import sha256_file
    from src.utils_io import write_json_atomic

    # A missing/typo'd config path must NOT silently fall back to defaults: the defaults are a plain
    # unreranked 0.6B dense run, which would burn the one-shot test-split evaluation on the wrong system.
    cfg_path = Path(a.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}\n  Refusing to fall back to defaults -- those are "
                         f"an unreranked dense run, not the chosen configuration.")
    raw = json.loads(cfg_path.read_text())
    raw.pop("_comment", None)
    cfg_preset = raw.pop("preset", None)          # optional pin, kept out of the pipeline config proper
    raw_reference = raw.pop("reference", None)    # per-config expected score, likewise not a pipeline key
    assert not set(raw) & set(NON_PIPELINE_KEYS), "NON_PIPELINE_KEYS is out of sync with the pops above"
    if a.preset and cfg_preset and a.preset != cfg_preset:
        raise SystemExit(f"--preset {a.preset!r} contradicts the 'preset' pinned in {cfg_path.name} "
                         f"({cfg_preset!r}). Pass one or make them agree.")
    preset_name = a.preset or cfg_preset
    if not preset_name:
        raise SystemExit(f"No first-stage preset: pass --preset, or add a 'preset' key to {cfg_path.name}.")
    cfg = merge_config(raw)

    rc = cfg["rerank"]
    ref_value, ref_label = reference_for(raw_reference, preset_name)
    print("\n" + "=" * 78)
    print(f"[official] TEST-SPLIT RUN -- config {cfg_path}")
    print(f"[official]   first stage : {preset_name} ({PRESETS[preset_name]['model']})")
    print(f"[official]   variants    : {cfg['dense_variants']}")
    print(f"[official]   bm25        : {'ON' if cfg['bm25']['enabled'] else 'off'}")
    print("[official]   reranker    : " + (f"{rc['name']}[{rc['instruction']}] k={rc['k']} "
          f"alpha={rc['alpha']} dtype={rc['dtype']}" if rc["enabled"] else "off"))
    print(f"[official]   llm rejudge : {'ON' if cfg['rejudge']['enabled'] else 'off'}")
    print("[official]   reference   : " + (f"{ref_value:.5f} ({ref_label})" if ref_value is not None
                                           else f"NONE -- {ref_label}"))
    print("=" * 78 + "\n", flush=True)
    print("[official] full pipeline config:\n" + json.dumps(cfg, indent=2), flush=True)
    p = PRESETS[preset_name]
    dense = DenseEncoder(p["model"], a.device, p["max_seq_length"], trust_remote_code=p["trust_remote_code"],
                         dtype=p["dtype"], revision=p["revision"], expect_eos=p["expect_eos"])

    def scorer_factory():
        from src.retrieval.reranker import load_reranker
        rc = cfg["rerank"]
        return load_reranker(rc["name"], a.device, rc["dtype"], rc["max_length"], rc["instruction"])

    model = HybridSearchModel(cfg, dense, scorer_factory, batch_size=a.batch_size,
                              rankings_depth=a.rankings_depth)
    t0 = time.time()
    result = mteb.evaluate(model, [mteb.get_task("AppsRetrieval")], encode_kwargs={"batch_size": a.batch_size},
                           cache=None)
    elapsed = time.time() - t0
    d = list(result.task_results)[0].to_dict()

    # --- artefacts, all from work already done -----------------------------------------------------
    rankings = {qid: list(ids) for qid, ids in model.last_rankings.items()}
    rankings_path = write_json_atomic(a.rankings_out, {
        "task": "AppsRetrieval", "split": "test", "depth": a.rankings_depth,
        "n_queries": len(rankings), "preset": preset_name, "config": str(cfg_path),
        "rankings": rankings})
    rankings_sha = sha256_file(rankings_path)
    print(f"\n[official] rankings: {len(rankings)} queries x up to {a.rankings_depth} ids -> "
          f"{rankings_path} (sha256 {rankings_sha[:16]}...)", flush=True)

    index_manifest = None
    if a.no_index_export:
        print("[official] runtime_index export skipped (--no-index-export)", flush=True)
    else:
        meta = index_meta(cfg, preset_name, p, cfg["dense_variants"][0])
        if isinstance(meta, str):
            print(f"[official] NOT exporting runtime_index/: {meta}", flush=True)
        else:
            meta["config_path"] = str(cfg_path)
            t1 = time.time()
            index_manifest = model.export_runtime_index(a.index_dir, meta)
            print(f"[official] runtime_index -> {a.index_dir}: {index_manifest['n_docs']} docs x "
                  f"dim {index_manifest['dim']} fp16, written in {time.time() - t1:.1f}s "
                  f"(reused this run's corpus embeddings, nothing re-encoded)", flush=True)
            for name, rec in index_manifest["files"].items():
                print(f"[official]   {name:20s} {rec['bytes'] / 1e6:8.2f} MB  "
                      f"sha256 {rec['sha256'][:16]}...", flush=True)

    d["_run_info"] = {"preset": preset_name, "model": p["model"], "revision": p["revision"],
                      "config_path": str(cfg_path), "pipeline_config": cfg,
                      "total_eval_seconds": round(elapsed, 1),
                      "stage_timings_s": {k: round(v, 2) for k, v in model.timings.items()},
                      "device": a.device, "fell_back_to_fp32": dense.fell_back_to_fp32,
                      "reference_ndcg_at_10": ref_value, "reference_label": ref_label,
                      "rankings_file": str(rankings_path), "rankings_sha256": rankings_sha,
                      "rankings_depth": a.rankings_depth,
                      "runtime_index_dir": None if index_manifest is None else str(a.index_dir),
                      "runtime_index_files": None if index_manifest is None else index_manifest["files"]}
    # atomic: a kill mid-write must not leave a truncated results file that looks like a real result
    results_path = write_json_atomic(a.out, d, indent=2, default=str)
    # The results file cannot contain its own hash, so the checksums live in their own file, written
    # after it. That ordering is what lets verify_submission.py notice a truncated results file.
    write_json_atomic(a.checksums_out, {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {"path": str(results_path), "sha256": sha256_file(results_path),
                    "bytes": Path(results_path).stat().st_size},
        "rankings": {"path": str(rankings_path), "sha256": rankings_sha,
                     "bytes": Path(rankings_path).stat().st_size},
        "runtime_index": None if index_manifest is None else {
            "dir": str(a.index_dir), "files": index_manifest["files"]},
    }, indent=2)
    write_json_atomic(LOCK, {"out": a.out, "config": cfg, "preset": preset_name,
                             "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "ndcg_at_10": d.get("scores", {}).get("test", [{}])[0].get("ndcg_at_10"),
                             "completed": True})
    s = d.get("scores", {}).get("test", [{}])[0]
    print("\n" + "=" * 78)
    print(f"[official] OFFICIAL TEST-SPLIT RESULT  ({preset_name}, "
          f"{'rerank ' + rc['name'] if rc['enabled'] else 'dense only'})")
    print("=" * 78)
    print(f"  ndcg_at_10 = {s.get('ndcg_at_10')}      <-- headline metric")
    print(f"  mrr_at_10  = {s.get('mrr_at_10')}")
    for group in ("ndcg", "mrr", "recall", "precision", "map"):
        vals = {k: v for k, v in sorted(s.items()) if k.startswith(group + "_at_")}
        if vals:
            print(f"  {group:10s}: " + "  ".join(f"{k.split('_at_')[1]}={v:.5f}"
                                                 for k, v in vals.items() if isinstance(v, (int, float))))
    other = {k: v for k, v in sorted(s.items())
             if not any(k.startswith(g + "_at_") for g in ("ndcg", "mrr", "recall", "precision", "map"))}
    if other:
        print(f"  other MTEB fields: {json.dumps(other, default=str)}")

    got = s.get("ndcg_at_10")
    if ref_value is None:
        print(f"\n  NO discrepancy check was run: {ref_label}")
    elif isinstance(got, (int, float)):
        gap = got - ref_value
        print(f"\n  reference for THIS config: {ref_value:.5f} ({ref_label}) -> gap {gap:+.4f}")
        if abs(gap) > 0.004:
            print("  " + "!" * 74)
            print(f"  LOUD WARNING: the official test NDCG@10 differs from this config's reference by {gap:+.4f},")
            print("  which is BEYOND the ~0.003-0.004 gaps seen so far. Investigate this")
            print("  before acting on the result: likely suspects are a different first stage than dev")
            print("  used, the reranker silently failing (check for a [pipeline] WARNING above), or a")
            print("  dev/test distribution difference. Do NOT treat this number as final until explained.")
            print("  " + "!" * 74)
        else:
            print("  OK: within the dev-vs-test gap seen throughout this project.")
    print(f"\n  results   : {results_path}")
    print(f"  rankings  : {rankings_path}")
    print(f"  checksums : {a.checksums_out}")
    if index_manifest is not None:
        print(f"  index     : {a.index_dir}")
    print("\n  verify with:  python src/verify_submission.py --confirm-test")
    print(f"  run_info: {json.dumps(d['_run_info'], default=str)}", flush=True)


if __name__ == "__main__":
    main()
