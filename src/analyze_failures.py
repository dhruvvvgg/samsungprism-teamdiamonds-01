"""Where does the official run actually fail, and why?

    python src/analyze_failures.py --confirm-test

NDCG@10 0.9376 means roughly one test query in sixteen does not get its answer into the top 10. This
reads the exported rankings and says which ones, groups them by a likely cause, and writes 3-5 worked
examples -- query, expected snippet, top-3 retrieved -- ready to put on a slide.

CPU only, no model, no encoding: everything comes from the rankings file the official run already wrote
plus the corpus and query texts.

The groups are diagnoses offered with their evidence, not verdicts:

    very_long_query        the query is far longer than the median, so the signal is diluted and the
                           1024-token serving cap would truncate it
    near_duplicate_corpus  the retrieved top-3 are highly similar to the expected document by token
                           overlap -- the system found the right *kind* of code, and the corpus has
                           several near-identical solutions, which is an APPS property rather than a
                           retrieval error
    generic_wording        the query is short and has few distinctive terms, so many documents are
                           plausible
    other                  none of the above; these are the genuinely interesting ones

Similarity here is Jaccard over identifier tokens, deliberately NOT the embedding: using the model's own
notion of similarity to explain the model's mistakes would be circular.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
NEAR_DUPLICATE_JACCARD = 0.6
GENERIC_TOKEN_COUNT = 25


def tokens(text):
    return set(t.lower() for t in TOKEN.findall(text or ""))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def classify(query_text, expected_text, top_texts, median_chars):
    """(group, evidence) for one failed query."""
    q_tokens = tokens(query_text)
    if len(query_text) > 3 * median_chars:
        return "very_long_query", (f"{len(query_text)} characters vs a median of {median_chars}; "
                                   f"the 1024-token serving cap would truncate this")
    sims = [jaccard(tokens(expected_text), tokens(t)) for t in top_texts]
    best = max(sims) if sims else 0.0
    if best >= NEAR_DUPLICATE_JACCARD:
        return "near_duplicate_corpus", (f"a retrieved document shares {best:.0%} of its identifiers "
                                         f"with the expected one: the corpus holds near-identical "
                                         f"solutions")
    if len(q_tokens) < GENERIC_TOKEN_COUNT:
        return "generic_wording", (f"only {len(q_tokens)} distinct terms in the query; many documents "
                                   f"are plausible answers")
    return "other", (f"{len(q_tokens)} query terms, best overlap with a retrieved document "
                     f"{best:.0%} -- no obvious cause")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", default=str(ROOT / "outputs" / "appsretrieval_rankings.json"))
    ap.add_argument("--qrels", default=None, help="qrels JSON; default is the official test qrels")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--examples", type=int, default=5, help="worked examples to export (3-5)")
    ap.add_argument("--preview-chars", type=int, default=700)
    ap.add_argument("--confirm-test", action="store_true",
                    help="acknowledge reading the official test qrels and texts (no retrieval is run)")
    ap.add_argument("--out", default=str(ROOT / "results" / "failure_analysis.json"))
    ap.add_argument("--markdown-out", default=str(ROOT / "results" / "failure_examples.md"))
    a = ap.parse_args()

    from src.utils_io import write_json_atomic

    rpath = Path(a.rankings)
    if not rpath.exists():
        raise SystemExit(f"No rankings at {rpath}. It is written by src/eval/run_official.py.")
    payload = json.loads(rpath.read_text(encoding="utf-8"))
    rankings = payload["rankings"] if isinstance(payload, dict) and "rankings" in payload else payload

    if a.qrels:
        raw = json.loads(Path(a.qrels).read_text(encoding="utf-8"))
        qrels = {str(q): {str(d): int(s) for d, s in rels.items()} for q, rels in raw.items()}
        corpus, queries = {}, {}
        print("[fail] using a local qrels file; corpus and query texts unavailable, so examples will "
              "be ids only", flush=True)
    else:
        if not a.confirm_test:
            raise SystemExit("Reading the official test qrels and texts needs --confirm-test "
                             "(no retrieval is run; this only reads the finished ranking).")
        from src.eval.load_data import load_apps
        corpus, queries, qrels = load_apps(allow_test=True)

    lengths = [len(queries.get(q, "")) for q in qrels if queries.get(q)]
    median_chars = int(sorted(lengths)[len(lengths) // 2]) if lengths else 0
    print(f"[fail] {len(qrels)} judged queries, median query length {median_chars} characters",
          flush=True)

    failures, ranks_found = [], 0
    for qid, rels in qrels.items():
        gold = {d for d, s in rels.items() if s > 0}
        ranked = rankings.get(qid, [])
        position = next((i + 1 for i, d in enumerate(ranked) if d in gold), None)
        if position is not None and position <= a.k:
            ranks_found += 1
            continue
        expected_id = sorted(gold)[0] if gold else None
        top_ids = ranked[:3]
        group, evidence = classify(queries.get(qid, ""), corpus.get(expected_id, ""),
                                   [corpus.get(t, "") for t in top_ids], median_chars)
        failures.append({"query_id": qid, "rank_of_relevant": position, "group": group,
                         "evidence": evidence, "expected_doc": expected_id, "top_docs": top_ids,
                         "query_chars": len(queries.get(qid, ""))})

    total = len(qrels)
    groups = Counter(f["group"] for f in failures)
    print(f"[fail] {len(failures)} of {total} queries miss the top {a.k} "
          f"({100 * len(failures) / max(total, 1):.1f}%)")
    for group, n in groups.most_common():
        print(f"[fail]   {group:<22} {n:5d}  ({100 * n / max(len(failures), 1):.1f}% of failures)")
    beyond = [f for f in failures if f["rank_of_relevant"] is None]
    print(f"[fail]   of those, {len(beyond)} have the relevant document outside the whole "
          f"{len(next(iter(rankings.values()), []))}-deep ranking")

    # --- worked examples for a slide -----------------------------------------------------------------
    chosen, seen_groups = [], set()
    for group, _ in groups.most_common():
        for f in failures:
            if f["group"] == group and group not in seen_groups:
                seen_groups.add(group)
                chosen.append(f)
                break
        if len(chosen) >= a.examples:
            break
    for f in failures:                                  # top up to --examples with the worst misses
        if len(chosen) >= a.examples:
            break
        if f not in chosen:
            chosen.append(f)

    lines = ["# Failure analysis: worked examples", "",
             f"From the official run: {len(failures)} of {total} test queries "
             f"({100 * len(failures) / max(total, 1):.1f}%) do not place the relevant document in the "
             f"top {a.k}.", "",
             "| Group | Queries | Share of failures |", "|---|---|---|"]
    for group, n in groups.most_common():
        lines.append(f"| {group} | {n} | {100 * n / max(len(failures), 1):.1f}% |")
    lines.append("")
    for i, f in enumerate(chosen, start=1):
        rank = f["rank_of_relevant"] or "not retrieved"
        lines += [f"## Example {i}: {f['group']} (relevant document at rank {rank})", "",
                  f"**Why it is grouped here:** {f['evidence']}", "", "### Query", "",
                  "```", (queries.get(f["query_id"], "(text unavailable)")[:a.preview_chars]).strip(),
                  "```", "", f"### Expected document (`{f['expected_doc']}`)", "", "```python",
                  (corpus.get(f["expected_doc"], "(text unavailable)")[:a.preview_chars]).strip(),
                  "```", ""]
        for j, doc_id in enumerate(f["top_docs"], start=1):
            lines += [f"### Retrieved #{j} (`{doc_id}`)", "", "```python",
                      (corpus.get(doc_id, "(text unavailable)")[:a.preview_chars]).strip(), "```", ""]

    Path(a.markdown_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.markdown_out).write_text("\n".join(lines), encoding="utf-8")
    write_json_atomic(a.out, {"k": a.k, "n_queries": total, "n_in_top_k": ranks_found,
                              "n_failures": len(failures),
                              "failure_rate": round(len(failures) / max(total, 1), 4),
                              "groups": dict(groups), "median_query_chars": median_chars,
                              "examples": [f["query_id"] for f in chosen],
                              "failures": failures}, indent=2)
    print(f"\n  {len(chosen)} worked examples -> {a.markdown_out}")
    print(f"  full breakdown            -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
