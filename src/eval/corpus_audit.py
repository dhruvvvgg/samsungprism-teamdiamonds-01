"""Are corpus items standalone? (Justifies skipping Phase 6 context expansion.) Also profiles the
query format for the query-side variants.

Only the *train* partition rows are read (docs and queries carry a `partition` field); no qrels, no
test-partition text. Pure `ast` / regex work: no model, no inference.

    python src/eval/corpus_audit.py [--partition train] [--out results/corpus_audit.md]
"""
import argparse
import ast
import builtins
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUILTINS = set(dir(builtins))
SECTION_RE = re.compile(r"^-{3,}\s*([A-Za-z ]+?)\s*-{3,}\s*$", re.M)


def audit_doc(src):
    """Return dict of structural facts about one solution file."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError):
        return {"parses": False}
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    imports, relative, defined = [], False, set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports += [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                relative = True
            elif n.module:
                imports.append(n.module.split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            defined.add(n.id)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, ast.alias):
            defined.add((n.asname or n.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
    # calls to plain names that are neither builtin, defined/assigned/imported, nor a parameter
    free_calls = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id not in BUILTINS and n.func.id not in defined}
    return {
        "parses": True,
        "has_def": any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)),
        "has_class": any(isinstance(n, ast.ClassDef) for n in ast.walk(tree)),
        "imports": sorted(set(imports)),
        "non_stdlib_imports": sorted({m for m in imports if m not in stdlib}),
        "relative_import": relative,
        "free_calls": sorted(free_calls),
        "reads_stdin": "input(" in src or "sys.stdin" in src,
        "n_lines": src.count("\n") + 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="train")
    ap.add_argument("--out", default=str(ROOT / "results" / "corpus_audit.md"))
    a = ap.parse_args()

    from datasets import load_dataset
    corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")
    queries = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
    docs = [r for r in corpus if r["partition"] == a.partition]
    qs = [r for r in queries if r["partition"] == a.partition]
    print(f"partition={a.partition}: {len(docs)} docs, {len(qs)} queries "
          f"(of {len(corpus)} / {len(queries)} total rows; other partitions not analysed)")

    facts = [audit_doc(d["text"]) for d in docs]
    ok = [f for f in facts if f["parses"]]
    n = len(ok)
    pct = lambda k: 100.0 * k / max(len(facts), 1)  # noqa: E731
    non_std = Counter(m for f in ok for m in f["non_stdlib_imports"])
    std = Counter(m for f in ok for m in f["imports"] if m not in non_std)
    free = Counter(c for f in ok for c in f["free_calls"])
    lines_ = sorted(f["n_lines"] for f in ok)
    stats = {
        "docs": len(facts), "parse_ok_pct": pct(n),
        "with_any_import_pct": pct(sum(1 for f in ok if f["imports"])),
        "with_non_stdlib_import_pct": pct(sum(1 for f in ok if f["non_stdlib_imports"])),
        "with_relative_import_pct": pct(sum(1 for f in ok if f["relative_import"])),
        "define_functions_pct": pct(sum(1 for f in ok if f["has_def"])),
        "define_classes_pct": pct(sum(1 for f in ok if f["has_class"])),
        "with_call_to_undefined_name_pct": pct(sum(1 for f in ok if f["free_calls"])),
        "reads_stdin_pct": pct(sum(1 for f in ok if f["reads_stdin"])),
        "median_lines": lines_[len(lines_) // 2] if lines_ else 0,
        "top_non_stdlib_imports": non_std.most_common(8),
        "top_stdlib_imports": std.most_common(8),
        "top_undefined_calls": free.most_common(8),
    }

    marks = Counter()
    for q in qs:
        found = {m.group(1).strip().lower() for m in SECTION_RE.finditer(q["text"])}
        marks.update(found)
    qlen = sorted(len(q["text"]) for q in qs)
    qstats = {"queries": len(qs), "median_chars": qlen[len(qlen) // 2] if qlen else 0,
              "section_markers_pct": {k: round(100.0 * v / len(qs), 1) for k, v in marks.most_common(12)}}
    meta_keys = sorted({k for d in docs[:200] for k in (d.get("meta_information") or {})})

    print(json.dumps({"docs": stats, "queries": qstats, "doc_meta_keys": meta_keys}, indent=1, default=str))
    md = [f"# Corpus audit (partition `{a.partition}` only)\n",
          "Generated by `src/eval/corpus_audit.py` (AST + regex; no model, no labels, no test partition).\n",
          f"## Documents: {stats['docs']} solution files\n",
          "| Property | % of docs |\n|---|---|"]
    for k in ("parse_ok_pct", "with_any_import_pct", "with_non_stdlib_import_pct", "with_relative_import_pct",
              "define_functions_pct", "define_classes_pct", "with_call_to_undefined_name_pct", "reads_stdin_pct"):
        md.append(f"| {k[:-4].replace('_', ' ')} | {stats[k]:.1f} |")
    md += [f"\nMedian length: {stats['median_lines']} lines.",
           f"\nMost common non-stdlib imports: {stats['top_non_stdlib_imports']}",
           f"\nMost common calls to names that are not builtin / defined / imported: {stats['top_undefined_calls']}",
           f"\n## Queries: {qstats['queries']} problem statements\n",
           f"Median length {qstats['median_chars']} chars. Section markers (% of queries containing them): "
           f"{qstats['section_markers_pct']}"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
