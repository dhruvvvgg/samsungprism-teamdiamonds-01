"""The signals behind a result that can be read off without another model call.

These are lexical and arithmetic facts about a hit -- which query words also appear among the snippet's
identifiers, how far the score sits above the next result -- and NOT an explanation of why the embedding
model scored it as it did. The UI labels them "matched terms" for that reason: a dense retriever can rank a
snippet first with no shared words at all, and a shared word does not prove that is why it ranked.
"""
import ast
import keyword
import re

WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
STOPWORDS = frozenset("""the and for with that this from are was were into over than then them their there
has have had not but you your can all any each one two given find print output input example return
returns number numbers first second line lines list must may should would could will also only such when
what which where while about after before between per its it's set case cases""".split())
MAX_TERMS = 8
MAX_IDENTS_PER_TERM = 3


def query_terms(query):
    """Distinct lowercase content words of the query, in order of first appearance."""
    seen, out = set(), []
    for w in WORD.findall(query or ""):
        t = w.lower()
        if len(t) < 3 or t in STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def identifiers_of(text):
    """Names the code defines or uses. Parsed with `ast` when the snippet parses (so words in strings and
    comments do not count); a snippet that does not parse falls back to every identifier-shaped token."""
    names = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return {n for n in IDENT.findall(text) if not keyword.iskeyword(n)}
    for node in ast.walk(tree):
        for attr in ("name", "id", "attr", "arg", "asname"):
            v = getattr(node, attr, None)
            if isinstance(v, str):
                names.add(v)
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def _subwords(name):
    return [p.lower() for p in CAMEL.sub("_", name).split("_") if p]


def _stem(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def matched_terms(query, text):
    """[{term, identifiers}] -- query words found among the snippet's identifiers, either as a whole
    name or as one part of a snake_case / camelCase name (plural 's' ignored). Sorted by term."""
    terms = query_terms(query)
    if not terms or not text:
        return []
    idents = identifiers_of(text)
    parts = {n: {_stem(w) for w in _subwords(n)} | {_stem(n.lower())} for n in idents}
    out = []
    for t in terms:
        hit = sorted(n for n, ps in parts.items() if _stem(t) in ps)
        if hit:
            out.append({"term": t, "identifiers": hit[:MAX_IDENTS_PER_TERM]})
    return sorted(out, key=lambda m: m["term"])[:MAX_TERMS]


def explain_hit(query, text, score, next_score=None, route=None):
    """The `why` block for one hit."""
    why = {"ranked_by": "dense embedding similarity",
           "score_gap_to_next": None if next_score is None else round(float(score) - float(next_score), 6),
           "gap_basis": "next result",
           "matched_terms": matched_terms(query, text),
           "n_query_terms": len(query_terms(query))}
    if route:
        why["route"] = {k: route.get(k) for k in ("kind", "route", "reason")}
    return why
