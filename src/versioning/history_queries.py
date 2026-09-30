"""Auto-generate version-targeted queries with known answers, from real history.

A version-targeted query needs two things: text a person might plausibly type, and an answer that is
known without asking the system being tested. Both come from the diff itself.

  text    the function's own docstring first line where it has one, else a readable phrase built from
          its qualname ("invoke on Command in click/core.py"). Docstrings are written by the library's
          authors, not by us and not by the retriever, so they are independent of what we are measuring.
  answer  the (lineage, version) whose content hash changed at that commit, plus -- when the change
          introduced a new identifier -- that identifier as a version-discriminating token.

The honest caveat, carried in the data rather than hidden: an identifier introduced at version v usually
survives into later versions, so a token identifies the version that INTRODUCED it but does not exclude
its descendants. `acceptable_versions` is therefore computed by scanning the actual texts for the token,
and the benchmark reports exact-version accuracy and token-consistent accuracy separately.

A change that introduces no new identifier yields a `lineage_only` query: it still tests version
targeting (where the version is supplied explicitly), but it cannot discriminate between versions on
text alone, so `query_stats` reports the two kinds separately rather than blending them.
"""
import ast
import re

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
PLACEHOLDER_DOC = re.compile(r"^(todo|fixme|xxx)\b", re.I)


def first_docstring_line(text):
    """The first sentence of the definition's docstring, or None. Parsed, not regexed, so a string that
    merely looks like a docstring does not become a query."""
    try:
        tree = ast.parse(text.lstrip())
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if not doc:
                return None
            line = doc.strip().split("\n")[0].strip()
            if len(line) < 12 or PLACEHOLDER_DOC.match(line):
                return None
            return line
    return None


def readable_phrase(row):
    """A fallback query built from the identifier itself: `Command.invoke` -> 'command invoke'."""
    words = re.sub(r"[_.]+", " ", row["qualname"]).strip()
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", words).lower()
    where = row["file"].rsplit("/", 1)[-1]
    return f"{words} in {where}"


def identifiers(text):
    return set(IDENT.findall(text))


def new_identifiers(before_text, after_text):
    """Identifiers present after the change and absent before -- the change's fingerprint."""
    return identifiers(after_text) - identifiers(before_text)


def build_queries(rows, max_per_lineage=1, min_token_len=4):
    """Version-targeted queries for lineages that actually changed.

    `rows` is the output of git_history.ingest (oldest version first). One query per changed
    (lineage, version), capped at `max_per_lineage` so a hot file does not dominate the evaluation."""
    by_lineage = {}
    for r in rows:
        by_lineage.setdefault(r["lineage_id"], []).append(r)
    for versions in by_lineage.values():
        versions.sort(key=lambda r: r["version"])

    queries, per_lineage = [], {}
    for lid, versions in sorted(by_lineage.items()):
        texts = {v["version"]: v["text"] for v in versions}
        for i, row in enumerate(versions):
            if row["change"] != "modified" or i == 0:
                continue                                   # need a before-state to diff against
            if per_lineage.get(lid, 0) >= max_per_lineage:
                break
            before = versions[i - 1]
            base = first_docstring_line(row["text"])
            source = "docstring" if base else "qualname"
            if not base:
                base = readable_phrase(row)
            tokens = sorted(t for t in new_identifiers(before["text"], row["text"])
                            if len(t) >= min_token_len)
            entry = {"query_id": f"{lid}@v{row['version']}", "lineage_id": lid,
                     "target_version": row["version"], "target_doc_id": row["doc_id"],
                     "commit": row["commit_short"], "file": row["file"],
                     "qualname": row["qualname"], "label_source": source}
            if tokens:
                token = max(tokens, key=len)               # the most distinctive new name
                # A token from new_identifiers() is by construction absent from the PREVIOUS version,
                # so it always discriminates at least that boundary; no "present everywhere" case can
                # occur, and a guard against one would be unreachable code.
                acceptable = [v for v, t in sorted(texts.items()) if token in t]
                entry.update({"text": f"{base} using {token}", "token": token,
                              "acceptable_versions": acceptable, "kind": "version_specific"})
            else:
                entry.update({"text": base, "token": None,
                              "acceptable_versions": [row["version"]], "kind": "lineage_only"})
            queries.append(entry)
            per_lineage[lid] = per_lineage.get(lid, 0) + 1
    return queries


def query_stats(queries):
    kinds, sources = {}, {}
    for q in queries:
        kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
        sources[q["label_source"]] = sources.get(q["label_source"], 0) + 1
    return {"n": len(queries), "by_kind": kinds, "by_label_source": sources,
            "lineages": len({q["lineage_id"] for q in queries})}
