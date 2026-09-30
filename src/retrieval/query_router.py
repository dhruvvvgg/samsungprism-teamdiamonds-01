"""Classify a query, so it can be sent to the retriever that can actually answer it.

Four kinds, and the reason each needs different handling:

    problem_statement   a full APPS-style task ("Given an array of n integers... Input... Output...").
                        Long, prose-heavy, often with worked examples. Plain dense retrieval, which is
                        what the whole P0 system was tuned for.
    nl_intent           a short request ("binary search over a sorted array"). Also dense, but worth
                        distinguishing: it is short enough that the query cap never bites and long
                        enough context tricks would only add noise.
    code_query          the query IS code ("def merge(a, b): return sorted(a + b)"). Embedding it with
                        an instruction prefix written for prose hurts; code-to-code means dropping the
                        instruction and comparing code with code.
    structural          a question about the codebase's structure or usage ("who calls invoke",
                        "where is ExitCode imported", "where is this string used"). Dense retrieval
                        cannot answer these at all -- the answer is an exact set of file:line, which
                        only the structural index has.

Rules first, and the rules are strong here because the four classes are lexically distinct: code parses
as Python, structural questions use a small closed vocabulary of verbs, and APPS statements are long and
full of I/O section headers. The optional LLM fallback goes through `llm_client` and is off by default;
it is consulted only when the rules are genuinely unsure, never to second-guess a confident rule.

Routing decides which retriever runs, not which answer is right; every route degrades to dense, so a
misroute costs quality, never an error.
"""
import ast
import re

ROUTES = {
    "problem_statement": "dense",
    "nl_intent": "dense",
    "code_query": "dense_code",
    "structural": "structural",
}

# "who calls X", "what does X call", "where is X imported", "where is X used", "X before Y"
STRUCTURAL_PATTERNS = [
    (re.compile(r"\bwho\s+(?:calls|call|uses|invokes)\b", re.I), "who_calls"),
    (re.compile(r"\bwhat\s+does\s+\S+\s+call\b", re.I), "what_calls"),
    (re.compile(r"\bcallers?\s+of\b", re.I), "who_calls"),
    (re.compile(r"\bcalle+s?\s+of\b", re.I), "what_calls"),
    (re.compile(r"\bwhere\s+is\s+.+\bimported\b", re.I), "where_imported"),
    (re.compile(r"\bwho\s+imports\b|\bimports?\s+of\b", re.I), "where_imported"),
    (re.compile(r"\bwhere\s+is\s+.+\b(?:used|referenced|defined)\b", re.I), "where_used"),
    (re.compile(r"\bwhere\s+.+\b(?:used|referenced)\b", re.I), "where_used"),
    (re.compile(r"\bwhich\s+files?\b.*\bcalls?\b.*\bbefore\b", re.I), "call_order"),
    (re.compile(r"\bcalls?\b.*\bbefore\b.*", re.I), "call_order"),
    (re.compile(r"\bwhere\s+is\s+.+\bdeclared\b", re.I), "where_used"),
]

CODE_TOKENS = re.compile(r"(^|\n)\s*(def |class |import |from \w+ import |return |for |while |if )")
APPS_SECTIONS = re.compile(r"^\s*(input|output|constraints?|examples?|sample input|sample output|"
                           r"note|explanation)\s*[:\-]?\s*$", re.I | re.M)
LONG_QUERY_CHARS = 400


def looks_like_code(text):
    """True when the query parses as Python, or is unmistakably code by shape.

    `ast.parse` is the strong signal, but a fragment like `for x in items:` is code and does not parse
    on its own, so a shape check backs it up. A bare identifier is NOT code: 'invoke' is far more likely
    to be a name someone is asking about."""
    stripped = text.strip()
    if not stripped or "\n" not in stripped and " " not in stripped:
        return False
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return bool(CODE_TOKENS.search(stripped)) and any(c in stripped for c in "(){}[]:=")
    # a single string or name expression parses but is prose, not code
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        value = tree.body[0].value
        if isinstance(value, (ast.Constant, ast.Name, ast.Attribute)):
            return False
        if isinstance(value, ast.Call):
            return True
        return False
    return bool(tree.body)


def structural_intent(text):
    """(intent, subject) when the query asks a structural question, else (None, None)."""
    for pattern, intent in STRUCTURAL_PATTERNS:
        if pattern.search(text):
            return intent, extract_subject(text, intent)
    return None, None

_QUOTED = re.compile(r"[\"'`]([^\"'`]{2,})[\"'`]")
_IDENTIFIER = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\b")
_STOPWORDS = {"who", "what", "where", "which", "does", "do", "is", "are", "the", "a", "an", "call",
              "calls", "called", "caller", "callers", "callee", "callees", "uses", "used", "use",
              "import", "imports", "imported", "in", "of", "to", "from", "and", "before", "after",
              "file", "files", "function", "method", "class", "this", "that", "it", "defined",
              "referenced", "declared", "invokes", "constant", "string", "value", "code", "where's"}


def extract_subject(text, intent=None):
    """The thing being asked about: a quoted string wins, else the first non-stopword identifier."""
    quoted = _QUOTED.search(text)
    if quoted:
        return quoted.group(1)
    for match in _IDENTIFIER.finditer(text):
        token = match.group(1)
        if token.lower() in _STOPWORDS:
            continue
        return token
    return None


def extract_pair(text):
    """(first, second) for a 'calls X before Y' question."""
    match = re.search(r"\bcalls?\s+(.+?)\s+before\s+(.+?)\s*[?.]?$", text, re.I)
    if not match:
        return None, None
    return (extract_subject(match.group(1)) or None, extract_subject(match.group(2)) or None)


def classify(text, allow_llm=False, call=None):
    """Classify `text`. Returns a dict: kind, route, reason, confidence, and any extracted subject.

    `reason` is always populated and always human-readable -- a router that cannot explain itself is
    impossible to debug from a result list, and this string is shown in the CLI, API and UI."""
    raw = (text or "").strip()
    if not raw:
        return {"kind": "nl_intent", "route": "dense", "reason": "empty query; defaulting to dense",
                "confidence": "low", "subject": None, "source": "rule"}

    intent, subject = structural_intent(raw)
    if intent:
        out = {"kind": "structural", "route": "structural", "intent": intent, "subject": subject,
               "confidence": "high", "source": "rule",
               "reason": f"asks a structural question ({intent.replace('_', ' ')})"
                         + (f" about {subject!r}" if subject else "")}
        if intent == "call_order":
            first, second = extract_pair(raw)
            out.update({"first": first, "second": second})
            if not (first and second):
                out.update({"confidence": "low",
                            "reason": out["reason"] + "; could not read both call names"})
        elif not subject:
            out.update({"confidence": "low", "reason": out["reason"] + "; no subject found"})
        return out

    if looks_like_code(raw):
        return {"kind": "code_query", "route": "dense_code", "subject": None, "confidence": "high",
                "source": "rule",
                "reason": "the query is Python source, so it is embedded as code rather than prose"}

    sections = len(APPS_SECTIONS.findall(raw))
    if sections >= 2 or (len(raw) >= LONG_QUERY_CHARS and sections >= 1):
        return {"kind": "problem_statement", "route": "dense", "subject": None,
                "confidence": "high", "source": "rule",
                "reason": f"long prose with {sections} problem-statement sections "
                          f"(Input/Output/Constraints), like an APPS task"}
    if len(raw) >= LONG_QUERY_CHARS:
        return {"kind": "problem_statement", "route": "dense", "subject": None,
                "confidence": "medium", "source": "rule",
                "reason": f"long prose ({len(raw)} characters) with no structural cue"}

    decided = {"kind": "nl_intent", "route": "dense", "subject": None, "confidence": "medium",
               "source": "rule", "reason": f"short natural-language request ({len(raw)} characters)"}
    if allow_llm:
        guess = _llm_classify(raw, call)
        if guess:
            decided = guess
    return decided


LLM_PROMPT = """Classify this code-search query into exactly one category.

problem_statement - a full programming task with input/output description
nl_intent - a short request in plain English
code_query - the query is itself source code
structural - a question about code structure or usage (who calls X, where is Y imported/used)

Answer with the category name only.

Query:
{query}
"""


def _llm_classify(text, call=None):
    """Optional fallback. Off unless allow_llm=True, and it never overrides a confident rule.

    Routed through the provider-agnostic wrapper, whose default provider is the keyless mock -- so this
    path runs in tests and offline without an API key, and a failure degrades to the rule result."""
    try:
        if call is None:
            from src.agent.llm_client import call_llm as call
        answer = (call(LLM_PROMPT.format(query=text[:2000])) or "").strip().lower()
    except Exception as exc:  # noqa: BLE001  the router must never fail a search
        return {"kind": "nl_intent", "route": "dense", "subject": None, "confidence": "low",
                "source": "llm-failed",
                "reason": f"LLM fallback unavailable ({type(exc).__name__}); kept the rule result"}
    for kind in ROUTES:
        if kind in answer:
            return {"kind": kind, "route": ROUTES[kind], "subject": extract_subject(text),
                    "confidence": "medium", "source": "llm",
                    "reason": f"rules were unsure; the LLM fallback said {kind}"}
    return None
