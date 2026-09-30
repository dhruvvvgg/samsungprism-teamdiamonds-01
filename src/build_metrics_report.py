"""Collect every measured number into one file: results/metrics.md.

    python src/build_metrics_report.py

Reads whatever result files exist and writes a single report. **Missing inputs are never an error** --
each one becomes a "not yet measured" row naming the command that would produce it. That matters because
the numbers arrive in several separate Kaggle sessions, and a report that crashes on the first missing
file is useless until the very last run finishes.

No model, no GPU, no network: this only reads JSON that earlier runs wrote.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MISSING = "_not yet measured_"

SOURCES = {
    "official": (ROOT / "outputs" / "appsretrieval_results.json",
                 "python src/eval/run_official.py --config configs/official_f2llm17b_noreranker.json "
                 "--device cuda --confirm-test"),
    "cpu_full": (ROOT / "results" / "cpu_full_capped.json",
                 "python src/bench_cpu.py --index full --thread-sweep"),
    "cpu_lite": (ROOT / "results" / "cpu_lite_capped.json",
                 "python src/bench_cpu.py --index lite --thread-sweep"),
    "precision": (ROOT / "results" / "cpu_precision.json",
                  "python src/check_cpu_precision.py --index full --modes fp32 int8"),
    "history": (ROOT / "results" / "real_history_benchmark.json",
                "python src/bench_real_history.py --device cuda --max-commits 40"),
    "history_ingest": (ROOT / "results" / "history_ingest.json",
                       "python src/build_history_index.py --device cuda --max-commits 40"),
    "synthetic_p1": (ROOT / "results" / "version_rebuild_benchmark.json",
                     "python src/bench_versions.py --device cuda"),
    "bonus": (ROOT / "results" / "evolution_benchmark.json",
              "python src/bench_evolution.py --device cpu"),
    "agent": (ROOT / "results" / "agent_benchmark.json",
              "python src/bench_agent.py --index history --source data/repos/click/src/click"),
    "gated": (ROOT / "results" / "gated_rerank.json",
              "python src/eval/dev_gated_rerank.py --preset f2llm-v2-1.7b --device cuda"),
    "descriptions": (ROOT / "results" / "description_fusion.json",
                     "python src/eval/dev_descriptions.py --preset f2llm-v2-1.7b --device cuda"),
    "finetune": (ROOT / "results" / "finetune_lite.json",
                 "python src/train/finetune_lite.py --stage eval --device cuda"),
    "categories": (ROOT / "results" / "category_tiebreak.json",
                   "python src/eval/dev_categories.py --preset f2llm-v2-1.7b --device cuda"),
}


def load(name):
    path, _ = SOURCES[name]
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def fmt(value, digits=4, suffix=""):
    if value is None:
        return MISSING
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"
    return str(value)


def missing_row(name, label):
    _, command = SOURCES[name]
    return f"| {label} | {MISSING} | `{command}` |"


def section_official(out):
    data = load("official")
    out.append("## Official test-split result\n")
    if not data:
        out += ["| Metric | Value | How to produce it |", "|---|---|---|",
                missing_row("official", "everything below"), ""]
        return
    scores = (data.get("scores") or {}).get("test") or [{}]
    s, info = scores[0], data.get("_run_info", {})
    out.append(f"Model **{info.get('model', '?')}** at revision `{str(info.get('revision', ''))[:12]}`, "
               f"config `{Path(str(info.get('config_path', '?'))).name}`, "
               f"{fmt(info.get('total_eval_seconds'), 1)} s on {info.get('device', '?')}.\n")
    out += ["| k | NDCG@k | MRR@k | Recall@k | Precision@k | MAP@k |", "|---|---|---|---|---|---|"]
    for k in (1, 3, 5, 10, 20, 100, 1000):
        row = [f"{k}"]
        found = False
        for group in ("ndcg", "mrr", "recall", "precision", "map"):
            v = s.get(f"{group}_at_{k}")
            found = found or v is not None
            row.append(fmt(v, 4) if v is not None else "-")
        if found:
            out.append("| " + " | ".join(row) + " |")
    out.append("")
    if info.get("fell_back_to_fp32"):
        out.append("> fp16 fell back to fp32 during this run.\n")


def section_latency(out):
    out.append("## Latency and serving cost\n")
    out += ["| Setting | p50 | p95 | Peak RSS | Threads |", "|---|---|---|---|---|"]
    any_row = False
    for name, label in (("cpu_full", "CPU, 1.7B (full)"), ("cpu_lite", "CPU, 0.6B (lite)")):
        data = load(name)
        if not data:
            continue
        any_row = True
        lat = data.get("latency_ms", {})
        out.append(f"| {label} | {fmt(lat.get('p50'), 0, ' ms')} | {fmt(lat.get('p95'), 0, ' ms')} | "
                   f"{fmt(data.get('peak_rss_mb'), 0, ' MB')} | {data.get('threads', '?')} |")
    if not any_row:
        out.append(f"| CPU latency | {MISSING} | | | |")
    out.append("")
    for name, label in (("cpu_full", "full"), ("cpu_lite", "lite")):
        data = load(name)
        if data and data.get("thread_sweep"):
            out.append(f"Thread scaling ({label}): " + ", ".join(
                f"{r['threads']} thread(s) {r['latency_ms']['p50']:.0f} ms"
                for r in data["thread_sweep"]) + "\n")
    prec = load("precision")
    if prec:
        out += ["| CPU precision | ms/query | Peak RSS | cosine to fp32 | top-10 overlap |",
                "|---|---|---|---|---|"]
        for mode in prec.get("modes", []):
            out.append(f"| {mode['mode']} | {fmt(mode.get('encode_ms_p50'), 0)} | "
                       f"{fmt(mode.get('peak_rss_mb'), 0, ' MB')} | "
                       f"{fmt(mode.get('cosine_to_fp32_mean'), 6)} | "
                       f"{fmt(mode.get('top10_overlap_mean'), 3)} |")
        out.append("")
    else:
        out.append(f"CPU precision (fp32 / bf16 / int8): {MISSING} — "
                   f"`{SOURCES['precision'][1]}`\n")


def section_indexing(out):
    out.append("## Indexing cost\n")
    out += ["| Corpus | Documents | Encode time | Index size | Notes |", "|---|---|---|---|---|"]
    official = load("official")
    if official:
        files = (official.get("_run_info") or {}).get("runtime_index_files") or {}
        size = sum(f.get("bytes", 0) for f in files.values()) / 1e6 if files else None
        timings = (official.get("_run_info") or {}).get("stage_timings_s") or {}
        out.append(f"| APPS (full, 1.7B) | 8,765 | {fmt(timings.get('index_dense_s'), 0, ' s')} | "
                   f"{fmt(size, 1, ' MB')} | exported by the official run |")
    else:
        out.append(f"| APPS (full, 1.7B) | 8,765 | {MISSING} | {MISSING} | `{SOURCES['official'][1]}` |")
    ing = load("history_ingest")
    if ing:
        st = ing.get("stats", {})
        emb = ing.get("embed_stats", {})
        out.append(f"| click history | {fmt(st.get('rows'), 0)} rows / {fmt(st.get('lineages'), 0)} "
                   f"lineages | {fmt(emb.get('seconds'), 0, ' s')} | — | "
                   f"{fmt(st.get('commits'), 0)} commits, "
                   f"{fmt(emb.get('reuse_pct'), 1, '%')} embeddings reused |")
    else:
        out.append(f"| click history | {MISSING} | | | `{SOURCES['history_ingest'][1]}` |")
    out.append("")


def section_versions(out):
    out.append("## P1 (incremental rebuild) and Bonus (evolutionary retrieval)\n")
    hist = load("history")
    if hist:
        totals, retr = hist.get("totals", {}), hist.get("retrieval") or {}
        out.append(f"**Real history ({Path(str(hist.get('repo', ''))).name}, "
                   f"{hist.get('ingest', {}).get('commits', '?')} commits)**\n")
        out += ["| Measure | Value |", "|---|---|",
                f"| embeddings, full rebuild | {fmt(totals.get('embeddings_full'), 0)} |",
                f"| embeddings, incremental | {fmt(totals.get('embeddings_incremental'), 0)} |",
                f"| saved | {fmt(totals.get('saved_pct'), 1, '%')} |",
                f"| speed-up | {fmt(totals.get('speedup_x'), 2, 'x')} |"]
        if retr:
            p1, bonus = retr.get("p1_version_targeted", {}), retr.get("bonus_all_versions", {})
            out += [f"| P1: correct lineage at rank 1 when targeting a version | "
                    f"{fmt(p1.get('top1_correct_lineage'), 3)} |",
                    f"| Bonus: top-10 duplicate slots, all versions | "
                    f"{fmt(bonus.get('duplicate_pct_at_10_all_versions'), 1, '%')} |",
                    f"| Bonus: top-10 duplicate slots, collapsed | "
                    f"{fmt(bonus.get('duplicate_pct_at_10_collapsed'), 1, '%')} |",
                    f"| Bonus: lineage recall@10, all versions -> collapsed | "
                    f"{fmt(bonus.get('lineage_recall_at_10_all_versions'), 3)} -> "
                    f"{fmt(bonus.get('lineage_recall_at_10_collapsed'), 3)} |"]
        out.append("")
        if hist.get("warnings"):
            out.append("> Warnings: " + "; ".join(hist["warnings"]) + "\n")
    else:
        out.append(f"Real history: {MISSING} — `{SOURCES['history'][1]}`\n")
    syn = load("synthetic_p1")
    if syn:
        t = syn.get("totals", {})
        out.append(f"**Synthetic fixture** (known-by-construction mutations): "
                   f"{fmt(t.get('embeddings_full'), 0)} -> {fmt(t.get('embeddings_incremental'), 0)} "
                   f"embeddings, {fmt(t.get('saved_pct'), 1, '%')} saved.\n")
    else:
        out.append(f"Synthetic fixture: {MISSING} — `{SOURCES['synthetic_p1'][1]}`\n")


def section_agent(out):
    out.append("## Agent vs plain dense search\n")
    data = load("agent")
    if not data:
        out.append(f"{MISSING} — `{SOURCES['agent'][1]}`\n")
        return
    labels = data.get("label_summary", {})
    out.append(f"{labels.get('n', '?')} labelled questions ({labels.get('by_kind', {})}), "
               f"ground truth by {labels.get('by_label_method', {})} — never by the parser under test.\n")
    out += ["| Group | P@k dense | P@k agent | Recall dense | Recall agent | ms dense | ms agent |",
            "|---|---|---|---|---|---|---|"]
    groups = [("all", data.get("overall", {}))] + sorted((data.get("by_kind") or {}).items())
    for name, b in groups:
        if not b:
            continue
        out.append(f"| {name} (n={b.get('n', '?')}) | {fmt(b['dense']['precision_at_k'], 3)} | "
                   f"{fmt(b['agent']['precision_at_k'], 3)} | {fmt(b['dense']['recall'], 3)} | "
                   f"{fmt(b['agent']['recall'], 3)} | {fmt(b['dense']['median_ms'], 0)} | "
                   f"{fmt(b['agent']['median_ms'], 0)} |")
    out.append("")


def section_experiments(out):
    out.append("## Dev experiments (adoption rule: >= +0.005 NDCG@10, CI excludes zero, "
               "worsened <= half improved)\n")
    out += ["| Experiment | Best setting | NDCG@10 | Delta | Verdict |", "|---|---|---|---|---|"]
    gated = load("gated")
    if gated:
        best = gated.get("best_gate")
        if best:
            out.append(f"| A. Confidence-gated rerank | threshold {best['threshold']} "
                       f"({best['pct_reranked']}% reranked) | {fmt(best['ndcg@10'])} | "
                       f"{fmt(best['delta_vs_never'], 4)} vs never | "
                       f"{'ADOPT' if best['adopt_vs_never'] else 'reject'} |")
        else:
            out.append("| A. Confidence-gated rerank | no gate passed | — | — | reject |")
    else:
        out.append(f"| A. Confidence-gated rerank | {MISSING} | | | |")
    desc = load("descriptions")
    if desc:
        best = desc.get("best")
        out.append(f"| F. Code-to-description fusion | w={best['weight'] if best else '?'} | "
                   f"{fmt(best['ndcg@10'] if best else None)} | "
                   f"{fmt(best['delta_ndcg@10'] if best else None, 4)} | "
                   f"{'ADOPT' if best and best['adopt'] else 'reject'} |")
    else:
        out.append(f"| F. Code-to-description fusion | {MISSING} | | | |")
    ft = load("finetune")
    if ft:
        out.append(f"| B. Fine-tuned 0.6B (holdout) | LoRA | {fmt(ft['finetuned']['ndcg@10'])} | "
                   f"{fmt(ft['finetuned']['ndcg@10'] - ft['base_0.6b']['ndcg@10'], 4)} vs base 0.6B | "
                   f"{'lite model' if ft.get('beats_base_0.6b') else 'reject'}"
                   f"{', P0 candidate' if ft.get('beats_base_1.7b') else ''} |")
    else:
        out.append(f"| B. Fine-tuned 0.6B (holdout) | {MISSING} | | | |")
    cat = load("categories")
    if cat and cat.get("rows"):
        best = max(cat["rows"], key=lambda r: r["delta_ndcg@10"])
        out.append(f"| E. Category tiebreaker | w={best['weight']} | {fmt(best['ndcg@10'])} | "
                   f"{fmt(best['delta_ndcg@10'], 4)} | {'ADOPT' if best['adopt'] else 'reject'} |")
    else:
        out.append(f"| E. Category tiebreaker | {MISSING} | | | |")
    out.append("")
    for name in ("gated", "descriptions", "finetune", "categories"):
        if not load(name):
            out.append(f"- {name}: `{SOURCES[name][1]}`")
    out.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "metrics.md"))
    a = ap.parse_args()

    out = ["# Metrics report", "",
           f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `src/build_metrics_report.py`._", "",
           f"Rows marked {MISSING} have no result file yet; the command that produces each one is "
           f"given beside it.", ""]
    section_official(out)
    section_latency(out)
    section_indexing(out)
    section_versions(out)
    section_agent(out)
    section_experiments(out)

    present = [n for n in SOURCES if load(n) is not None]
    absent = [n for n in SOURCES if load(n) is None]
    out += ["## Coverage", "",
            f"- measured: {len(present)}/{len(SOURCES)} — {', '.join(present) or 'none'}",
            f"- not yet measured: {', '.join(absent) or 'none'}", ""]

    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"[metrics] wrote {path} ({len(present)}/{len(SOURCES)} sources present)")
    for name in absent:
        print(f"[metrics]   not yet measured: {name}  ->  {SOURCES[name][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
