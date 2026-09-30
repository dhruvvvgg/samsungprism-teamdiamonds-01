"""Rule-based performance smells in surfaced Python, each with its file:line.

An extra, not part of what this system is scored on: the task is retrieval, and generating code advice
is out of scope. It is here because once a result puts a function in front of someone, the cheap
structural checks are nearly free and occasionally worth the glance.

Four checks, each chosen because it is decidable from the AST alone and is a real cost, not a style
opinion:

    nested_loop_same_collection   two loops over the same iterable, one inside the other -> O(n^2)
    membership_test_on_list       `x in some_list` inside a loop -> O(n) per test; a set is O(1)
    string_concat_in_loop         `s += ...` on a str inside a loop -> repeated reallocation
    loop_invariant_call           a call with loop-independent arguments recomputed every iteration

Every check is conservative: it reports only what it can see, and says what it assumed. `x in y` is
flagged only when `y` is a name bound to a list or tuple literal in the same function, because flagging
every membership test would be noise -- on a set it is already optimal.
"""
import ast

SEVERITY = {"nested_loop_same_collection": "high", "membership_test_on_list": "medium",
            "string_concat_in_loop": "medium", "loop_invariant_call": "low"}


def _iter_name(node):
    """The iterable's name for `for x in NAME:` / `for x in NAME.items():`, else None."""
    it = node.iter
    if isinstance(it, ast.Name):
        return it.id
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) \
            and isinstance(it.func.value, ast.Name):
        return it.func.value.id
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.args \
            and isinstance(it.args[0], ast.Name):
        return it.args[0].id                     # enumerate(items) / sorted(items)
    return None


def _loop_targets(node):
    """Names bound by a loop header, which make anything using them loop-dependent."""
    out = set()
    for child in ast.walk(node.target):
        if isinstance(child, ast.Name):
            out.add(child.id)
    return out


def _list_like_names(func_node):
    """Names assigned a list/tuple literal or a list() call anywhere in this function."""
    out = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            is_listy = isinstance(node.value, (ast.List, ast.Tuple, ast.ListComp)) or (
                isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("list", "tuple", "sorted"))
            if is_listy:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out.add(target.id)
    return out


def _str_names(func_node):
    """Names assigned a string literal, so `+=` on them is string concatenation."""
    out = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.add(target.id)
    return out


def _check_function(func, file, line_offset):
    findings = []
    listy, stringy = _list_like_names(func), _str_names(func)
    loops = [n for n in ast.walk(func) if isinstance(n, (ast.For, ast.While))]

    for loop in loops:
        if not isinstance(loop, ast.For):
            continue
        outer_name = _iter_name(loop)
        for inner in ast.walk(loop):
            if inner is loop or not isinstance(inner, ast.For):
                continue
            if outer_name and _iter_name(inner) == outer_name:
                findings.append({
                    "check": "nested_loop_same_collection", "line": inner.lineno + line_offset,
                    "detail": f"loop over {outer_name!r} nested inside another loop over the same "
                              f"collection: this is quadratic in len({outer_name})",
                    "suggestion": "index it once (a dict or set keyed by what you are matching on) "
                                  "instead of re-scanning the inner collection"})

    for loop in loops:
        targets = _loop_targets(loop) if isinstance(loop, ast.For) else set()
        for node in ast.walk(loop):
            if isinstance(node, ast.Compare) and node.ops and isinstance(node.ops[0], ast.In):
                right = node.comparators[0]
                if isinstance(right, ast.Name) and right.id in listy:
                    findings.append({
                        "check": "membership_test_on_list", "line": node.lineno + line_offset,
                        "detail": f"`in {right.id}` inside a loop, and {right.id} is a list: each test "
                                  f"scans it",
                        "suggestion": f"build `{right.id}_set = set({right.id})` once outside the loop"})
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add) \
                    and isinstance(node.target, ast.Name) and node.target.id in stringy:
                findings.append({
                    "check": "string_concat_in_loop", "line": node.lineno + line_offset,
                    "detail": f"`{node.target.id} +=` inside a loop builds a new string each time",
                    "suggestion": "append to a list and \"\".join(parts) after the loop"})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
                used = {c.id for c in ast.walk(node) if isinstance(c, ast.Name)}
                if targets and not (used & targets) and node.func.id not in ("print", "append"):
                    findings.append({
                        "check": "loop_invariant_call", "line": node.lineno + line_offset,
                        "detail": f"`{node.func.id}(...)` inside a loop uses none of the loop "
                                  f"variables ({', '.join(sorted(targets))}), so it recomputes the "
                                  f"same value every iteration",
                        "suggestion": "hoist it above the loop"})
    for f in findings:
        f.update({"file": file, "severity": SEVERITY[f["check"]],
                  "location": f"{file}:{f['line']}"})
    return findings


def analyze_source(source, file="<snippet>", start_line=1):
    """Findings for a snippet. `start_line` maps snippet lines back to the real file."""
    try:
        tree = ast.parse(source.lstrip("\n"))
    except SyntaxError:
        return []
    offset = start_line - 1
    findings = []
    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for func in functions or [tree]:
        findings.extend(_check_function(func, file, offset))
    seen, unique = set(), []
    for f in sorted(findings, key=lambda f: (f["line"], f["check"])):
        key = (f["check"], f["line"])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def analyze_hit(hit):
    """Findings for one search result, using its real file and line when it has them."""
    text = hit.get("text") or hit.get("preview") or ""
    return analyze_source(text, hit.get("file", hit.get("doc_id", "<result>")),
                          hit.get("start_line", 1))
