"""Offline category tags per snippet: an algorithm family from AST features, plus embedding clusters.

Two independent signals, kept separate because they fail differently:

    ast_family      rule-based, from structural features -- recursion, nested loops, a memo dict, a
                    sort call, a heap import. Precise and explainable ("recursive + memo dict"),
                    but only covers patterns someone wrote a rule for.
    cluster         k-means over the existing embeddings, labelled by the identifiers that are most
                    over-represented in each cluster against the corpus as a whole. Covers everything,
                    and is readable rather than "cluster 7", but the label is a guess.

Tags are shown on results and can filter a search. Using them to *reorder* results is a retrieval change
and so goes through the adoption rule like anything else -- `src/eval/dev_categories.py` measures it and
it stays off unless it passes.

Everything here is offline and deterministic: no model call, no network, fixed seed.
"""
import ast
import math
import re
from collections import Counter

import numpy as np

FAMILIES = ("recursion", "dynamic_programming", "sorting", "graph_search", "heap_priority",
            "hashing_lookup", "string_processing", "matrix_grid", "math_number_theory",
            "iteration", "io_parsing", "other")

GRAPH_HINTS = {"bfs", "dfs", "adjacency", "graph", "visited", "neighbors", "neighbours", "queue"}
DP_HINTS = {"memo", "cache", "dp", "lru_cache", "table"}
STRING_HINTS = {"split", "join", "strip", "replace", "startswith", "endswith", "lower", "upper",
                "format", "encode", "decode"}
MATH_HINTS = {"gcd", "lcm", "prime", "factorial", "modulo", "pow", "sqrt", "isqrt", "comb"}
IO_HINTS = {"input", "readline", "stdin", "stdout", "print", "open", "loads", "dumps"}
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def ast_features(source):
    """Structural features of a snippet. Returns {} when it does not parse."""
    try:
        tree = ast.parse(source.lstrip("\n"))
    except SyntaxError:
        return {}
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    names = {n.name for n in funcs}
    calls, attrs, imports = Counter(), Counter(), set()
    loops = nested = subscripts = 0
    recursive = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            loops += 1
            for inner in ast.walk(node):
                if inner is not node and isinstance(inner, (ast.For, ast.While)):
                    nested += 1
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls[node.func.id] += 1
                if node.func.id in names:
                    recursive = True
            elif isinstance(node.func, ast.Attribute):
                attrs[node.func.attr] += 1
        elif isinstance(node, ast.Subscript):
            subscripts += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in getattr(node, "names", []):
                imports.add(alias.name.split(".")[0])
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    identifiers = {t.lower() for t in IDENT.findall(source)}
    return {"loops": loops, "nested_loops": nested, "recursive": recursive,
            "subscripts": subscripts, "calls": calls, "attrs": attrs, "imports": imports,
            "identifiers": identifiers, "n_functions": len(funcs),
            "dicts": sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Dict, ast.DictComp))),
            "sets": sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Set, ast.SetComp)))}


def ast_family(source):
    """(family, reason). Rules are ordered most-specific first; the reason is always explainable."""
    f = ast_features(source)
    if not f:
        return "other", "does not parse as Python"
    ids, calls, attrs = f["identifiers"], f["calls"], f["attrs"]
    if f["recursive"] and (ids & DP_HINTS or f["dicts"]):
        return "dynamic_programming", "recursive with a memo table"
    if ids & DP_HINTS and f["loops"]:
        return "dynamic_programming", "loop over a dp/memo table"
    if "heapq" in f["imports"] or "heappush" in calls or "heappop" in calls:
        return "heap_priority", "uses a heap"
    if ids & GRAPH_HINTS:
        return "graph_search", f"graph vocabulary ({', '.join(sorted(ids & GRAPH_HINTS)[:3])})"
    if f["recursive"]:
        return "recursion", "calls itself"
    if "sorted" in calls or "sort" in attrs:
        return "sorting", "sorts its input"
    if ids & MATH_HINTS:
        return "math_number_theory", f"number-theory vocabulary ({', '.join(sorted(ids & MATH_HINTS)[:3])})"
    if f["nested_loops"] and f["subscripts"] >= 2:
        return "matrix_grid", "nested loops over indexed data"
    if f["dicts"] or f["sets"]:
        return "hashing_lookup", "builds a dict or set for lookup"
    if len(ids & STRING_HINTS) >= 2 or sum(attrs[a] for a in STRING_HINTS) >= 2:
        return "string_processing", "string methods"
    if len(ids & IO_HINTS) >= 2:
        return "io_parsing", "reads or writes streams"
    if f["loops"]:
        return "iteration", "a plain loop"
    return "other", "no distinctive structure"


def cluster_labels(embeddings, texts, n_clusters=12, seed=0, top_terms=3, max_iter=25):
    """k-means over the embeddings, each cluster labelled by its most over-represented identifiers.

    Plain Lloyd's algorithm on the unit sphere (the embeddings are L2-normalised, so a dot product is
    the cosine). Implemented here rather than pulled from scikit-learn to keep the serving dependency
    list small and the result reproducible from a seed.

    A cluster is labelled by comparing each term's frequency inside it against the whole corpus, which
    is what makes 'parse, token, lexer' come out instead of 'self, return, value'."""
    E = np.asarray(embeddings, dtype=np.float32)
    n = E.shape[0]
    k = max(1, min(int(n_clusters), n))
    rng = np.random.RandomState(seed)
    centers = E[rng.choice(n, size=k, replace=False)].copy()
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        new_assign = np.argmax(E @ centers.T, axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(k):
            members = E[assign == c]
            if len(members):
                v = members.mean(axis=0)
                centers[c] = v / max(float(np.linalg.norm(v)), 1e-12)
    corpus = Counter()
    per_cluster = [Counter() for _ in range(k)]
    for i, text in enumerate(texts):
        terms = set(t.lower() for t in IDENT.findall(text or ""))
        corpus.update(terms)
        per_cluster[assign[i]].update(terms)
    labels = []
    for c in range(k):
        size = max(1, int((assign == c).sum()))
        scored = []
        for term, count in per_cluster[c].items():
            if count < 2:
                continue
            inside = count / size
            overall = corpus[term] / n
            scored.append((inside * math.log((inside + 1e-9) / (overall + 1e-9)), term))
        scored.sort(reverse=True)
        labels.append(", ".join(t for _, t in scored[:top_terms]) or f"cluster {c}")
    return assign.tolist(), labels


def tag_rows(texts, embeddings=None, n_clusters=12, seed=0):
    """[{ast_family, ast_reason, cluster, cluster_label}] per row. Embeddings are optional."""
    families = [ast_family(t or "") for t in texts]
    out = [{"ast_family": fam, "ast_reason": why} for fam, why in families]
    if embeddings is not None and len(texts):
        assign, labels = cluster_labels(embeddings, texts, n_clusters=n_clusters, seed=seed)
        for row, c in zip(out, assign):
            row["cluster"] = int(c)
            row["cluster_label"] = labels[c]
    return out


def family_counts(tags):
    return dict(Counter(t["ast_family"] for t in tags).most_common())
