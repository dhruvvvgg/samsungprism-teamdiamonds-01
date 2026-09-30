"""Split Python files into retrievable chunks at function and class level, keeping file and line numbers.

    chunks = chunk_folder("my_project")
    # -> [{chunk_id: "pkg/mod.py:12-40", file: "pkg/mod.py", start_line: 12, end_line: 40,
    #      kind: "function", name: "parse", qualname: "parse", text: "def parse(...)..."}]

Chunking policy, chosen deliberately because retrieval quality depends on it:

  * one chunk per top-level function, one per top-level class (the whole class, decorators and
    docstring included);
  * one "module" chunk for everything left at module level -- imports, constants, `if __name__`
    blocks -- but only when it carries enough content to be worth retrieving;
  * methods are NOT separate chunks by default. They live inside their class chunk, so the text of a
    method would otherwise be indexed twice, and duplicate text in a dense index costs a top-10 slot
    for no new information. `--chunk-methods` turns them into their own chunks for codebases of large
    classes, where finding the class is not the same as finding the method.

Line numbers come from the AST and cover decorators (ast gives `decorator_list` line numbers that
precede `node.lineno`), so `file:start-end` points at the whole definition a reader expects to see.

A file that does not parse is reported, not skipped silently: a syntax error in the corpus should be
visible, because it means those functions are missing from every search result.
"""
import ast
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache",
             ".tox", "build", "dist", ".eggs", ".idea", ".vscode"}
MIN_MODULE_CHUNK_CHARS = 40


def chunk_id_for(rel_path, start, end):
    """The document id: `path/to/file.py:12-40`. Readable, unique, and the location itself."""
    return f"{rel_path}:{start}-{end}"


def _span(node, source_lines):
    """(start, end) 1-based inclusive line span of `node`, including any decorators."""
    start = node.lineno
    for dec in getattr(node, "decorator_list", []) or []:
        start = min(start, dec.lineno)
    end = getattr(node, "end_lineno", None) or node.lineno
    return start, min(end, len(source_lines))


def _text(source_lines, start, end):
    return "\n".join(source_lines[start - 1:end])


def chunk_source(source, rel_path, chunk_methods=False):
    """Chunks for one file's `source`. Raises SyntaxError if it does not parse."""
    tree = ast.parse(source)
    lines = source.split("\n")
    chunks, covered = [], set()

    def add(node, kind, qualname):
        start, end = _span(node, lines)
        covered.update(range(start, end + 1))
        chunks.append({"chunk_id": chunk_id_for(rel_path, start, end), "file": rel_path,
                       "start_line": start, "end_line": end, "kind": kind,
                       "name": node.name, "qualname": qualname,
                       "text": _text(lines, start, end)})

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, "function", node.name)
        elif isinstance(node, ast.ClassDef):
            add(node, "class", node.name)
            if chunk_methods:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        add(sub, "method", f"{node.name}.{sub.name}")

    # whatever is left at module level, as one chunk, if it is substantial
    rest = [i for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()]
    if rest:
        start, end = min(rest), max(rest)
        body = "\n".join(lines[i - 1] for i in rest)
        if len(body.strip()) >= MIN_MODULE_CHUNK_CHARS:
            chunks.append({"chunk_id": chunk_id_for(rel_path, start, end), "file": rel_path,
                           "start_line": start, "end_line": end, "kind": "module",
                           "name": Path(rel_path).stem, "qualname": Path(rel_path).stem,
                           "text": body})
    return sorted(chunks, key=lambda c: (c["start_line"], c["end_line"]))


def iter_python_files(root, skip_dirs=SKIP_DIRS):
    """Every .py file under `root`, skipping the usual noise directories. Sorted, so an index built
    twice from the same folder has the same row order."""
    root = Path(root)
    out = []
    for path in sorted(root.rglob("*.py")):
        if any(part in skip_dirs for part in path.relative_to(root).parts[:-1]):
            continue
        out.append(path)
    return out


def chunk_folder(root, chunk_methods=False, max_chunk_chars=20000, skip_dirs=SKIP_DIRS):
    """(chunks, stats) for every Python file under `root`.

    `max_chunk_chars` drops pathological chunks (a generated file as one giant function): they blow up
    encoding time and are useless as a search result. Dropped chunks are counted, not hidden."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    chunks, failed, skipped_big = [], [], 0
    files = iter_python_files(root, skip_dirs)
    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            failed.append({"file": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            file_chunks = chunk_source(source, rel, chunk_methods=chunk_methods)
        except SyntaxError as exc:
            failed.append({"file": rel, "error": f"SyntaxError: {exc.msg} (line {exc.lineno})"})
            continue
        for c in file_chunks:
            if len(c["text"]) > max_chunk_chars:
                skipped_big += 1
                continue
            chunks.append(c)
    stats = {"root": str(root), "files_scanned": len(files), "files_failed": len(failed),
             "failures": failed[:20], "chunks": len(chunks), "skipped_oversize": skipped_big,
             "by_kind": {}}
    for c in chunks:
        stats["by_kind"][c["kind"]] = stats["by_kind"].get(c["kind"], 0) + 1
    return chunks, stats
