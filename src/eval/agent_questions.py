"""Labelled repository questions for the agent evaluation, with ground truth built WITHOUT the parser.

The circularity trap: if the answer to "who calls `invoke`" comes from the same `ast` walk the agent
uses, the evaluation measures nothing -- it proves the parser agrees with itself. So every label here is
produced by an independent method, and each question records which one:

    grep      regex over the raw source text, never the AST. Used for call sites, imports and string
              or constant usage. A regex and an AST walk disagree in real ways (comments, strings,
              attribute chains), which is exactly what makes it an independent check.
    docstring the function's own docstring, written by the library's authors. Used for semantic
              questions: the question is the docstring, the answer is the function it documents.
    manual    hand-written for the demo repo, for questions neither method produces.

Questions are generated from whatever repository is passed in, so this works on `examples/textkit`, on
a click checkout, or on a fixture repo in a test.
"""
import re
from collections import Counter
from pathlib import Path

from src.indexing.code_chunker import iter_python_files

# A call site in raw text: `name(` not preceded by `def `/`.`-chain start, and not inside an obvious
# comment. Deliberately crude -- being a different method from the AST is the point.
CALL_RE = "(?<![\\w.])%s\\s*\\("
DEF_RE = r"^\s*(?:async\s+)?def\s+%s\s*\("
IMPORT_RE = r"^\s*(?:from\s+[\w.]+\s+import\s+.*\b%s\b|import\s+.*\b%s\b)"


def _read(root, rel):
    try:
        return (Path(root) / rel).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _files(root):
    return [p.relative_to(root).as_posix() for p in iter_python_files(Path(root))]


def grep_calls(root, name):
    """file:line of every textual call to `name`, excluding its own definition and comment lines."""
    pattern = re.compile(CALL_RE % re.escape(name))
    defpat = re.compile(DEF_RE % re.escape(name))
    hits = []
    for rel in _files(root):
        for i, line in enumerate(_read(root, rel).split("\n"), start=1):
            if defpat.search(line) or line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                hits.append(f"{rel}:{i}")
    return hits


def grep_imports(root, name):
    pattern = re.compile(IMPORT_RE % (re.escape(name), re.escape(name)))
    hits = []
    for rel in _files(root):
        for i, line in enumerate(_read(root, rel).split("\n"), start=1):
            if pattern.search(line):
                hits.append(f"{rel}:{i}")
    return hits


def grep_text(root, needle):
    """file:line wherever a literal string or constant name appears in the source text."""
    hits = []
    for rel in _files(root):
        for i, line in enumerate(_read(root, rel).split("\n"), start=1):
            if needle in line:
                hits.append(f"{rel}:{i}")
    return hits


def grep_defs(root):
    """{name: [file:line]} for every `def`, found textually."""
    pattern = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
    out = {}
    for rel in _files(root):
        for i, line in enumerate(_read(root, rel).split("\n"), start=1):
            m = pattern.match(line)
            if m:
                out.setdefault(m.group(1), []).append(f"{rel}:{i}")
    return out


def docstring_questions(root, limit=12, min_len=20):
    """(question, answer location) from each function's own docstring.

    The docstring is read with a text scan, not `ast`, keeping even the semantic labels independent of
    the module under evaluation."""
    out = []
    doc_re = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(.*?:\s*$")
    for rel in _files(root):
        lines = _read(root, rel).split("\n")
        for i, line in enumerate(lines):
            m = doc_re.match(line)
            if not m:
                continue
            for j in range(i + 1, min(i + 3, len(lines))):
                stripped = lines[j].strip()
                if stripped.startswith(('"""', "'''")):
                    text = stripped.strip('"\' ')
                    if len(text) >= min_len and not text.lower().startswith(("todo", "fixme")):
                        out.append({"question": text, "answer": f"{rel}:{i + 1}", "name": m.group(1)})
                    break
                if stripped:
                    break
    return out[:limit]


def build_questions(root, max_questions=30, seed=0):
    """~max_questions labelled questions across structural, usage and semantic kinds."""
    root = Path(root)
    defs = grep_defs(root)
    call_counts = Counter()
    for name in defs:
        call_counts[name] = len(grep_calls(root, name))
    popular = [n for n, c in call_counts.most_common() if c >= 1 and not n.startswith("_")]

    questions = []

    # --- structural: who calls X (labels from grep) --------------------------------------------------
    for name in popular[:8]:
        locations = grep_calls(root, name)
        if not locations:
            continue
        questions.append({"id": f"who_calls::{name}", "kind": "structural",
                          "question": f"who calls {name}", "label_method": "grep",
                          "label_note": f"regex {CALL_RE % name!r} over raw source, excluding its own def",
                          "expected": sorted(set(locations)),
                          "expected_files": sorted({p.split(':')[0] for p in locations})})

    # --- structural: where is X defined --------------------------------------------------------------
    for name in list(defs)[:6]:
        if name.startswith("_"):
            continue
        questions.append({"id": f"where_defined::{name}", "kind": "structural",
                          "question": f"where is {name} defined", "label_method": "grep",
                          "label_note": "textual `def NAME(` match",
                          "expected": sorted(defs[name]),
                          "expected_files": sorted({p.split(':')[0] for p in defs[name]})})

    # --- usage: imports and literals ------------------------------------------------------------------
    module_names = sorted({Path(f).stem for f in _files(root)})
    for mod in module_names[:5]:
        locations = grep_imports(root, mod)
        if locations:
            questions.append({"id": f"where_imported::{mod}", "kind": "usage",
                             "question": f"where is {mod} imported", "label_method": "grep",
                             "label_note": "textual import-line match",
                             "expected": sorted(set(locations)),
                             "expected_files": sorted({p.split(':')[0] for p in locations})})
    strings = Counter()
    literal_re = re.compile(r"[\"']([A-Za-z][\w .\-/]{5,40})[\"']")
    for rel in _files(root):
        for match in literal_re.finditer(_read(root, rel)):
            strings[match.group(1)] += 1
    for literal, count in strings.most_common(4):
        if count < 2:
            continue
        locations = grep_text(root, literal)
        questions.append({"id": f"where_used::{literal[:20]}", "kind": "usage",
                          "question": f'where is the string "{literal}" used',
                          "label_method": "grep", "label_note": "literal substring match in source",
                          "expected": sorted(set(locations)),
                          "expected_files": sorted({p.split(':')[0] for p in locations})})

    # --- semantic: the function's own docstring --------------------------------------------------------
    for item in docstring_questions(root, limit=12):
        questions.append({"id": f"semantic::{item['name']}", "kind": "semantic",
                          "question": item["question"], "label_method": "docstring",
                          "label_note": "the question IS the function's docstring, written by the "
                                        "library's authors; the answer is that function",
                          "expected": [item["answer"]],
                          "expected_files": [item["answer"].split(":")[0]]})

    return questions[:max_questions]


def label_summary(questions):
    return {"n": len(questions),
            "by_kind": dict(Counter(q["kind"] for q in questions)),
            "by_label_method": dict(Counter(q["label_method"] for q in questions))}
