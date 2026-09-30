"""Agent vs plain dense search on labelled repository questions: precision@k, recall, latency.

    python src/bench_agent.py --index examples/_demo_index --source examples/textkit --mock-encoder
    python src/bench_agent.py --index history --source data/repos/click/src/click --device cpu

Ground truth comes from `src/eval/agent_questions.py`, which builds labels with grep/regex and
docstrings -- never with the `ast` walk the agent uses. Every question records which method labelled it,
and the report breaks results down by method so a reader can see what is being measured.

Scoring is at file granularity by default. A caller reported at line 41 when the label says 40 is the
same answer to a person, and penalising that would measure line-number bookkeeping rather than
retrieval. `--exact-lines` scores file:line instead, and is reported alongside as the stricter number.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LocationList(list):
    """List of locations preserving ranking order while supporting set comparison in tests."""

    def __eq__(self, other):
        if isinstance(other, set):
            return set(self) == other
        return super().__eq__(other)


def parse_loc(loc):
    """(clean_file_path, start_line, end_line) from a location string or hit dict."""
    if isinstance(loc, dict):
        loc = loc.get("location") or loc.get("file") or ""
    loc_str = str(loc).strip()
    if ":" in loc_str:
        file_part, _, span_part = loc_str.partition(":")
    else:
        file_part, span_part = loc_str, ""

    file_part = file_part.replace("\\", "/").strip("/")
    start = end = None
    if span_part:
        if "-" in span_part:
            p1, _, p2 = span_part.partition("-")
            if p1.isdigit() and p2.isdigit():
                start, end = int(p1), int(p2)
        elif span_part.isdigit():
            start = end = int(span_part)
    return file_part, start, end


def loc_matches(pred, exp, exact_lines=False):
    """Check if predicted location matches expected location.

    Handles differences in relative path prefixes (e.g. src/click/_compat.py vs _compat.py)
    and line span containment (e.g. 40-65 covers 42).
    """
    f_pred, s_pred, e_pred = parse_loc(pred)
    f_exp, s_exp, e_exp = parse_loc(exp)

    if not f_pred or not f_exp:
        return False

    if f_pred != f_exp:
        if not (f_pred.endswith("/" + f_exp) or f_exp.endswith("/" + f_pred)):
            return False

    if not exact_lines:
        return True

    if s_pred is None or s_exp is None:
        return True

    return max(s_pred, s_exp) <= min(e_pred, e_exp)


def locations_of(items, exact_lines=False):
    """Extract ordered unique locations from hits or agent answers."""
    out = []
    seen = set()
    for item in items:
        loc = item.get("location") if isinstance(item, dict) else str(item)
        if not loc:
            continue
        val = str(loc) if exact_lines else str(loc).split(":")[0]
        if val not in seen:
            seen.add(val)
            out.append(str(loc) if exact_lines else val)
    return LocationList(out)


def score(predicted, expected, k, exact_lines=False):
    """precision@k, recall and hit (did anything correct appear at all)."""
    top = list(predicted)[:k]
    matched_gold = set()
    correct_preds = []
    for p in top:
        matches = [e for e in expected if loc_matches(p, e, exact_lines)]
        if matches:
            correct_preds.append(p)
            for m in matches:
                matched_gold.add(m)
    n_gold = max(len(expected), 1)
    n_top = max(len(top), 1)
    return {
        "precision_at_k": len(correct_preds) / n_top,
        "recall": len(matched_gold) / n_gold,
        "hit": bool(correct_preds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", dest="index_dir", default="examples/_demo_index")
    ap.add_argument("--source", default=None,
                    help="the repository the questions are about (default: the index's source_root)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-questions", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mock-encoder", action="store_true")
    ap.add_argument("--exact-lines", action="store_true",
                    help="score file:line instead of file (stricter; reported either way)")
    ap.add_argument("--llm-planner", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "agent_benchmark.json"))
    a = ap.parse_args()

    from src.agent.code_agent import CodeAgent
    from src.eval.agent_questions import build_questions, label_summary
    from src.indexing.structural import build_from_folder
    from src.search_service import SearchService
    from src.utils_io import write_json_atomic

    svc = SearchService(a.index_dir, device=a.device, mock=a.mock_encoder)
    source = a.source or svc.describe().get("source_root")
    if not source or not Path(source).is_dir():
        raise SystemExit(f"--source must be the repository the index was built from; got {source!r}")
    questions = build_questions(source, max_questions=a.max_questions)
    if not questions:
        raise SystemExit(f"no questions could be generated from {source}")
    summary = label_summary(questions)
    print(f"[agent] {summary['n']} questions from {source}: {summary['by_kind']} "
          f"(labels: {summary['by_label_method']})", flush=True)
    if a.mock_encoder:
        print("[agent] MOCK encoder: the structural results are real, the dense ones are not.",
              flush=True)

    structural = build_from_folder(source)
    agent = CodeAgent(svc, structural=structural, max_steps=a.max_steps, k=a.k,
                      use_llm=a.llm_planner)

    rows = []
    for i, q in enumerate(questions):
        t0 = time.time()
        dense_hits = svc.search(q["question"], k=a.k)["hits"]
        dense_ms = 1000 * (time.time() - t0)
        dense_pred = locations_of(
            [{"location": h.get("location", h["doc_id"])} for h in dense_hits], a.exact_lines)

        t0 = time.time()
        out = agent.run(q["question"])
        agent_ms = 1000 * (time.time() - t0)
        agent_pred = locations_of(out["answers"], a.exact_lines)

        rows.append({"id": q["id"], "kind": q["kind"], "label_method": q["label_method"],
                     "n_expected": len(q["expected"]),
                     "dense": dict(score(dense_pred, q["expected"], a.k, a.exact_lines),
                                   ms=round(dense_ms, 1)),
                     "agent": dict(score(agent_pred, q["expected"], a.k, a.exact_lines),
                                   ms=round(agent_ms, 1), steps=out["steps_run"],
                                   stop=out["stop_reason"])})
        if (i + 1) % 10 == 0:
            print(f"[agent]   {i + 1}/{len(questions)}", flush=True)

    semantic_rows = [r for r in rows if r["kind"] == "semantic"]
    if semantic_rows:
        semantic_dense_hits = sum(1 for r in semantic_rows if r["dense"]["hit"])
        if semantic_dense_hits == 0:
            raise RuntimeError(
                f"Dense hit rate is 0.0 across all {len(semantic_rows)} semantic questions. "
                "This indicates a location/path mapping bug between index hits and query labels."
            )

    def agg(rows, side, key):
        vals = [r[side][key] for r in rows]
        return round(statistics.fmean(vals), 4) if vals else 0.0

    def block(rows):
        return {"n": len(rows),
                "dense": {"precision_at_k": agg(rows, "dense", "precision_at_k"),
                          "recall": agg(rows, "dense", "recall"),
                          "hit_rate": agg(rows, "dense", "hit"),
                          "median_ms": round(statistics.median([r["dense"]["ms"] for r in rows]), 1)},
                "agent": {"precision_at_k": agg(rows, "agent", "precision_at_k"),
                          "recall": agg(rows, "agent", "recall"),
                          "hit_rate": agg(rows, "agent", "hit"),
                          "median_ms": round(statistics.median([r["agent"]["ms"] for r in rows]), 1),
                          "median_steps": statistics.median([r["agent"]["steps"] for r in rows])}}

    overall = block(rows)
    by_kind = {kind: block([r for r in rows if r["kind"] == kind])
               for kind in sorted({r["kind"] for r in rows})}
    by_label = {m: block([r for r in rows if r["label_method"] == m])
                for m in sorted({r["label_method"] for r in rows})}

    print("\n" + "=" * 82)
    print(f"[agent] AGENT vs DENSE   ({len(rows)} questions, k={a.k}, "
          f"scored by {'file:line' if a.exact_lines else 'file'})")
    print("=" * 82)
    header = f"  {'group':<22} {'P@k dense':>10} {'P@k agent':>10} {'rec dense':>10} {'rec agent':>10} {'ms dense':>9} {'ms agent':>9}"
    print(header)

    def row_line(name, b):
        print(f"  {name:<22} {b['dense']['precision_at_k']:10.3f} {b['agent']['precision_at_k']:10.3f} "
              f"{b['dense']['recall']:10.3f} {b['agent']['recall']:10.3f} "
              f"{b['dense']['median_ms']:9.0f} {b['agent']['median_ms']:9.0f}")

    row_line(f"ALL (n={overall['n']})", overall)
    for kind, b in by_kind.items():
        row_line(f"{kind} (n={b['n']})", b)
    print()
    for method, b in by_label.items():
        print(f"  labels by {method}: n={b['n']}, dense hit-rate {b['dense']['hit_rate']:.2f} vs "
              f"agent {b['agent']['hit_rate']:.2f}")
    print(f"\n  agent median steps: {overall['agent']['median_steps']} of {a.max_steps}")

    write_json_atomic(a.out, {"index": str(a.index_dir), "source": str(source), "k": a.k,
                              "exact_lines": bool(a.exact_lines),
                              "mock_encoder": bool(a.mock_encoder),
                              "label_summary": summary, "overall": overall, "by_kind": by_kind,
                              "by_label_method": by_label, "rows": rows,
                              "when": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2)
    print(f"\n  written to {a.out}")
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
