"""Structural facts about Python source: definitions, calls in order, imports, strings, references.

Dense retrieval answers "what does this code do". It cannot answer "who calls `invoke`", "where is
`ExitCode` imported", or "which files call `open` before `close`" -- those are questions about the
*structure* of the code, and the answer has to be exact, not ranked by similarity. This module extracts
those facts with `ast` and builds a cross-file call graph.

What is recorded per file (all with file and line):

    defs        functions, methods and classes, with their qualified name and line span
    calls       every call site, IN SOURCE ORDER, with the callee name as written
    imports     `import x`, `from a import b as c`, with the local alias and the origin module
    strings     string literals over a minimum length (for "where is this constant used")
    names       identifier references (Name and Attribute nodes), for usage lookups

Call resolution is deliberately conservative. `foo()` resolves to a definition when exactly one
candidate matches after import aliases are applied; when several files define `foo`, the edge is
recorded as ambiguous rather than guessed. A call graph that quietly picks the wrong `run()` is worse
than one that admits it does not know, because nothing downstream can tell the difference.
"""
import ast
from collections import defaultdict
from pathlib import Path

from src.indexing.code_chunker import SKIP_DIRS, iter_python_files

MIN_STRING_LEN = 4


def _qual(stack, name):
    return ".".join([*stack, name])


class _Visitor(ast.NodeVisitor):
    """Walks one module, recording definitions, calls, imports, strings and name references."""

    def __init__(self, file):
        self.file = file
        self.defs, self.calls, self.imports, self.strings, self.names = [], [], [], [], []
        self._stack = []          # enclosing def/class names, for qualnames
        self._scope = []          # enclosing qualname for attributing a call to its caller

    # --- definitions -----------------------------------------------------------------------------
    def _visit_def(self, node, kind):
        qualname = _qual(self._stack, node.name)
        self.defs.append({"file": self.file, "name": node.name, "qualname": qualname, "kind": kind,
                          "start_line": node.lineno,
                          "end_line": getattr(node, "end_lineno", node.lineno),
                          "docstring": (ast.get_docstring(node) or "").strip().split("\n")[0][:200],
                          "args": [a.arg for a in getattr(getattr(node, "args", None), "args", [])]
                          if kind != "class" else []})
        self._stack.append(node.name)
        self._scope.append(qualname)
        self.generic_visit(node)
        self._scope.pop()
        self._stack.pop()

    def visit_FunctionDef(self, node):
        self._visit_def(node, "method" if len(self._stack) else "function")

    def visit_AsyncFunctionDef(self, node):
        self._visit_def(node, "method" if len(self._stack) else "function")

    def visit_ClassDef(self, node):
        self._visit_def(node, "class")

    # --- calls, in source order ------------------------------------------------------------------
    def visit_Call(self, node):
        name, attr_of = self._call_name(node.func)
        if name:
            self.calls.append({"file": self.file, "line": node.lineno, "callee": name,
                               "base": attr_of, "caller": self._scope[-1] if self._scope else None,
                               "col": node.col_offset})
        self.generic_visit(node)

    @staticmethod
    def _call_name(func):
        """('open', None) for open(), ('f.close', 'f') for f.close(), (None, None) for f()()."""
        if isinstance(func, ast.Name):
            return func.id, None
        if isinstance(func, ast.Attribute):
            parts, cur = [func.attr], func.value
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            base = cur.id if isinstance(cur, ast.Name) else None
            if base:
                parts.append(base)
            return ".".join(reversed(parts)), base
        return None, None

    # --- imports ---------------------------------------------------------------------------------
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({"file": self.file, "line": node.lineno, "module": alias.name,
                                 "name": None, "alias": alias.asname or alias.name.split(".")[0],
                                 "kind": "import"})
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = ("." * (node.level or 0)) + (node.module or "")
        for alias in node.names:
            self.imports.append({"file": self.file, "line": node.lineno, "module": module,
                                 "name": alias.name, "alias": alias.asname or alias.name,
                                 "kind": "from"})
        self.generic_visit(node)

    # --- literals and references -------------------------------------------------------------------
    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) >= MIN_STRING_LEN:
            self.strings.append({"file": self.file, "line": node.lineno, "value": node.value[:300],
                                 "scope": self._scope[-1] if self._scope else None})
        self.generic_visit(node)

    def visit_Name(self, node):
        self.names.append({"file": self.file, "line": node.lineno, "name": node.id,
                           "scope": self._scope[-1] if self._scope else None,
                           "store": isinstance(node.ctx, ast.Store)})
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.names.append({"file": self.file, "line": node.lineno, "name": node.attr,
                           "scope": self._scope[-1] if self._scope else None, "store": False})
        self.generic_visit(node)


def parse_module(source, file):
    """Structural facts for one file. Raises SyntaxError if it does not parse."""
    visitor = _Visitor(file)
    visitor.visit(ast.parse(source))
    return {"file": file, "defs": visitor.defs, "calls": visitor.calls, "imports": visitor.imports,
            "strings": visitor.strings, "names": visitor.names}


class StructuralIndex:
    """Structural facts for a whole tree, plus a resolved cross-file call graph."""

    def __init__(self, modules, root=None, failures=None):
        self.root = str(root) if root else None
        self.modules = {m["file"]: m for m in modules}
        self.failures = failures or []
        self._build()

    # --- construction ------------------------------------------------------------------------------
    def _build(self):
        self.defs_by_name = defaultdict(list)          # short name -> [def]
        self.defs_by_qual = {}                         # "file::qualname" -> def
        for module in self.modules.values():
            for d in module["defs"]:
                self.defs_by_name[d["name"]].append(d)
                self.defs_by_qual[f"{d['file']}::{d['qualname']}"] = d
        self.callers = defaultdict(list)               # callee key -> [edge]
        self.callees = defaultdict(list)               # caller key -> [edge]
        self.unresolved = []
        for module in self.modules.values():
            aliases = self._alias_map(module)
            for call in module["calls"]:
                self._add_edge(module["file"], call, aliases)

    @staticmethod
    def _alias_map(module):
        """local alias -> imported name, so `from x import invoke as run` makes run() resolve."""
        out = {}
        for imp in module["imports"]:
            if imp["kind"] == "from" and imp["name"]:
                out[imp["alias"]] = imp["name"]
        return out

    def _add_edge(self, file, call, aliases):
        short = call["callee"].rsplit(".", 1)[-1]
        target = aliases.get(call["callee"], short)
        candidates = self.defs_by_name.get(target, [])
        local = [d for d in candidates if d["file"] == file]
        chosen = local[0] if len(local) == 1 else (candidates[0] if len(candidates) == 1 else None)
        edge = {"file": file, "line": call["line"], "callee_text": call["callee"],
                "callee": target, "caller": call["caller"],
                "caller_key": f"{file}::{call['caller']}" if call["caller"] else f"{file}::<module>",
                "ambiguous": chosen is None and len(candidates) > 1,
                "external": not candidates}
        if chosen is not None:
            edge["callee_key"] = f"{chosen['file']}::{chosen['qualname']}"
            edge["callee_file"] = chosen["file"]
            edge["callee_line"] = chosen["start_line"]
            self.callers[edge["callee_key"]].append(edge)
            self.callees[edge["caller_key"]].append(edge)
        else:
            self.unresolved.append(edge)
            self.callers[target].append(edge)          # still findable by plain name
            self.callees[edge["caller_key"]].append(edge)

    # --- queries ------------------------------------------------------------------------------------
    def find_definitions(self, name):
        """Definitions matching a short name or a `file::qualname` key."""
        if name in self.defs_by_qual:
            return [self.defs_by_qual[name]]
        short = name.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
        out = list(self.defs_by_name.get(short, []))
        if not out:                                     # qualname match, e.g. "Command.invoke"
            out = [d for d in self.defs_by_qual.values() if d["qualname"].endswith(name)]
        return out

    def who_calls(self, name):
        """Every call site of `name`, resolved or not, with file and line."""
        edges = []
        for d in self.find_definitions(name):
            edges.extend(self.callers.get(f"{d['file']}::{d['qualname']}", []))
        short = name.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
        edges.extend(e for e in self.callers.get(short, []) if e not in edges)
        return sorted(edges, key=lambda e: (e["file"], e["line"]))

    def what_calls(self, name):
        """Everything the named definition calls, in source order."""
        out = []
        for d in self.find_definitions(name):
            out.extend(self.callees.get(f"{d['file']}::{d['qualname']}", []))
        return sorted(out, key=lambda e: (e["file"], e["line"]))

    def where_imported(self, name):
        """Files importing `name`, by module or by imported symbol."""
        out = []
        for module in self.modules.values():
            for imp in module["imports"]:
                if name in (imp["name"], imp["alias"]) or imp["module"] == name \
                        or imp["module"].endswith("." + name):
                    out.append(imp)
        return sorted(out, key=lambda i: (i["file"], i["line"]))

    def where_used(self, text, include_strings=True, include_names=True):
        """Where a constant, string or identifier appears. Substring match for strings, exact for
        identifiers -- a search for 'settings' should not match every name containing it."""
        hits = []
        needle = text.strip().strip("'\"")
        for module in self.modules.values():
            if include_strings:
                for s in module["strings"]:
                    if needle in s["value"]:
                        hits.append({"file": s["file"], "line": s["line"], "kind": "string",
                                     "value": s["value"], "scope": s["scope"]})
            if include_names:
                for n in module["names"]:
                    if n["name"] == needle:
                        hits.append({"file": n["file"], "line": n["line"],
                                     "kind": "assignment" if n["store"] else "reference",
                                     "value": n["name"], "scope": n["scope"]})
        return sorted(hits, key=lambda h: (h["file"], h["line"]))

    def files_calling_in_order(self, first, second):
        """Files where a call to `first` appears on an earlier line than a call to `second`.

        The ordering is lexical (line numbers), not an execution trace: it answers "which files call X
        before Y in the source", which is what someone reading code is asking. Anything stronger would
        need control-flow analysis, and claiming it from line numbers would be wrong."""
        out = []
        for file, module in self.modules.items():
            firsts = [c["line"] for c in module["calls"]
                      if c["callee"].rsplit(".", 1)[-1] == first.rsplit(".", 1)[-1]]
            seconds = [c["line"] for c in module["calls"]
                       if c["callee"].rsplit(".", 1)[-1] == second.rsplit(".", 1)[-1]]
            if firsts and seconds and min(firsts) < max(seconds):
                out.append({"file": file, "first_line": min(firsts), "second_line": max(seconds),
                            "first": first, "second": second})
        return sorted(out, key=lambda r: r["file"])

    def stats(self):
        return {"files": len(self.modules), "failures": len(self.failures),
                "defs": len(self.defs_by_qual),
                "calls": sum(len(m["calls"]) for m in self.modules.values()),
                "imports": sum(len(m["imports"]) for m in self.modules.values()),
                "strings": sum(len(m["strings"]) for m in self.modules.values()),
                "resolved_edges": sum(len(v) for k, v in self.callers.items() if "::" in k),
                "unresolved_edges": len(self.unresolved)}


def build_from_folder(root, skip_dirs=SKIP_DIRS):
    """StructuralIndex over every Python file under `root`."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    modules, failures = [], []
    for path in iter_python_files(root, skip_dirs):
        rel = path.relative_to(root).as_posix()
        try:
            modules.append(parse_module(path.read_text(encoding="utf-8"), rel))
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            failures.append({"file": rel, "error": f"{type(exc).__name__}: {exc}"})
    return StructuralIndex(modules, root=root, failures=failures)


def build_from_sources(sources, root=None):
    """StructuralIndex from {file: source}. Used for a history index, where the code lives in git
    rather than on disk."""
    modules, failures = [], []
    for file, source in sorted(sources.items()):
        try:
            modules.append(parse_module(source, file))
        except SyntaxError as exc:
            failures.append({"file": file, "error": f"SyntaxError: {exc.msg}"})
    return StructuralIndex(modules, root=root, failures=failures)
