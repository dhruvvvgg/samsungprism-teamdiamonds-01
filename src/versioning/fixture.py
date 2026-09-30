"""A controlled version-history fixture, because APPS has none.

CoIR-APPS is a flat corpus: one text per document, no history, no lineage. P1 ("retrieval across
versions") therefore cannot be measured on it directly, so this module builds a fixture where the
history is known by construction: N snippets, each with several versions produced by named mutations.

    unchanged      byte-identical to the previous version  -> same content hash, embedding reused
    rename         an identifier renamed throughout         -> new hash
    logic          a comparison operator / constant changed -> new hash
    add_lines      a guard clause inserted                  -> new hash
    remove_lines   a guard clause deleted                   -> new hash

Because the mutation is known per version, two things are measurable rather than guessed: how much of
an incremental rebuild is legitimately reusable (the `unchanged` versions, and only those), and whether
a version-specific query retrieves the version it actually describes.

Everything is seeded and deterministic: the same seed gives byte-identical snippets, so a benchmark run
today is comparable with one from last week. `--source corpus` swaps the synthetic bases for real APPS
corpus texts (mutated generically); the default synthetic source needs no dataset download, which is
what lets the tests and CI run it.
"""
import random
import re

from src.versioning.content_hash import content_hash

MUTATIONS = ("unchanged", "rename", "logic", "add_lines", "remove_lines")

_VERBS = ("sum", "collect", "accumulate", "total", "aggregate", "gather", "reduce", "combine")
_NOUNS = ("scores", "weights", "prices", "deltas", "counts", "ratings", "offsets", "samples",
          "distances", "durations")
_OPS = ("<", "<=", ">", ">=", "==", "!=")

TEMPLATE = """def {fn}({arg}, limit={limit}):
    total = 0
    seen = 0
    for value in {arg}:
        if value {op} limit:
            total += value * {mul}
            seen += 1
    if seen == 0:
        return 0
    return total / seen
"""


def _base_snippets(n, rng):
    """n distinct synthetic Python snippets, each with a short description of what it does."""
    out, used = [], set()
    while len(out) < n:
        verb, noun = rng.choice(_VERBS), rng.choice(_NOUNS)
        op, limit, mul = rng.choice(_OPS), rng.randrange(2, 400), rng.randrange(2, 9)
        fn = f"{verb}_{noun}_{len(out)}"
        key = (verb, noun, op, limit, mul)
        if key in used:
            continue
        used.add(key)
        text = TEMPLATE.format(fn=fn, arg=noun, limit=limit, op=op, mul=mul)
        desc = (f"{verb} the {noun} whose value is {op} {limit}, scaling each by {mul} and returning "
                f"the mean")
        out.append({"text": text, "description": desc, "fn": fn, "limit": limit, "mul": mul, "op": op})
    return out


def _rename(text, new_name):
    """Rename the accumulator consistently (word-boundary, so `total` inside another word is safe)."""
    if re.search(r"\btotal\b", text):
        return re.sub(r"\btotal\b", new_name, text), f"identifier renamed to {new_name}"
    ident = re.search(r"\b([a-z_][a-z0-9_]{3,})\b", text)          # generic fallback (real corpus text)
    if not ident:
        return text + f"\n{new_name} = None\n", f"identifier {new_name} introduced"
    old = ident.group(1)
    return re.sub(rf"\b{re.escape(old)}\b", new_name, text), f"identifier {old} renamed to {new_name}"


def _logic(text, rng):
    """Change behaviour: swap the comparison operator, else bump the first integer literal."""
    for op in _OPS:
        pat = f" {op} "
        if pat in text:
            new = rng.choice([o for o in _OPS if o != op])
            return text.replace(pat, f" {new} ", 1), f"comparison changed from {op} to {new}"
    m = re.search(r"\b(\d+)\b", text)
    if m:
        bumped = str(int(m.group(1)) + 1)
        return text[:m.start()] + bumped + text[m.end():], f"constant {m.group(1)} -> {bumped}"
    return text + "\nassert True\n", "assertion appended"


def _add_lines(text, rng):
    """Insert a guard clause just after the signature (or at the top of a non-function snippet)."""
    lines = text.split("\n")
    at = 1 if lines and lines[0].startswith("def ") else 0
    guard = (["    if limit is None:", "        limit = 0"] if at
             else ["STRICT_MODE = True", "MAX_ITEMS = 1000"])
    return "\n".join(lines[:at] + guard + lines[at:]), f"{len(guard)} line(s) added"


def _remove_lines(text, rng):
    """Delete the `if seen == 0: return 0` guard (synthetic bases), else the last code line."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("if seen == 0"):
            return "\n".join(lines[:i] + lines[i + 2:]), "2 line(s) removed (empty-input guard)"
    body = [i for i, line in enumerate(lines) if line.strip()]
    if len(body) <= 2:
        return text, "nothing safe to remove (snippet too short)"
    drop = body[-1]
    return "\n".join(lines[:drop] + lines[drop + 1:]), "1 line removed"


def _mutate(kind, text, rng, marker):
    if kind == "unchanged":
        return text, "unchanged"
    if kind == "rename":
        return _rename(text, marker)
    if kind == "logic":
        return _logic(text, rng)
    if kind == "add_lines":
        return _add_lines(text, rng)
    if kind == "remove_lines":
        return _remove_lines(text, rng)
    raise ValueError(f"unknown mutation {kind!r}")


def build_fixture(n_snippets=500, n_versions=4, seed=0, source_texts=None,
                  unchanged_rate=0.25):
    """Build the fixture.

    Returns {"snippets": [...], "queries": [...], "meta": {...}}, where each snippet is
    {snippet_id, description, versions: [{version, text, mutation, note, content_hash}]} with versions
    numbered 1..n_versions oldest first.

    `unchanged_rate` is the share of *later* versions that repeat the previous one byte for byte. It is
    the fixture's reuse ceiling: an incremental rebuild cannot legitimately skip more than these, so the
    benchmark has a known right answer to be checked against rather than just a number to admire."""
    rng = random.Random(seed)
    if n_versions < 1:
        raise ValueError("n_versions must be >= 1")
    if source_texts:
        bases = [{"text": t, "description": " ".join(t.split()[:20]), "fn": f"snippet_{i}"}
                 for i, t in enumerate(list(source_texts)[:n_snippets])]
    else:
        bases = _base_snippets(n_snippets, rng)
    snippets, queries = [], []
    for i, base in enumerate(bases):
        sid = f"snip{i:04d}"
        versions = [{"version": 1, "text": base["text"], "mutation": "initial", "note": "initial version",
                     "content_hash": content_hash(base["text"])}]
        text = base["text"]
        for v in range(2, n_versions + 1):
            kind = ("unchanged" if rng.random() < unchanged_rate
                    else rng.choice([m for m in MUTATIONS if m != "unchanged"]))
            marker = f"{base['fn']}_acc_v{v}"
            text, note = _mutate(kind, text, rng, marker)
            versions.append({"version": v, "text": text, "mutation": kind, "note": note,
                             "content_hash": content_hash(text)})
        snippets.append({"snippet_id": sid, "description": base["description"], "versions": versions})

        # one query describing the snippet generally (any version may legitimately answer it) ...
        queries.append({"query_id": f"q{i:04d}g", "text": base["description"], "target_snippet": sid,
                        "target_version": n_versions, "kind": "generic"})
        # ... and one that names a token introduced by a specific version.
        #
        # Honest caveat, recorded in the fixture rather than glossed over: a renamed identifier persists
        # into every LATER version too, so the token identifies the version that introduced it but does
        # not exclude its descendants. `target_version` is where it was introduced;
        # `acceptable_versions` is every version that still contains it. The benchmark reports accuracy
        # both ways instead of quietly picking whichever looks better.
        specific = [v for v in versions[1:] if v["mutation"] == "rename"]
        if specific:
            v = specific[-1]
            token = v["note"].split()[-1]
            # read the acceptable set off the actual texts rather than assuming every later version
            # inherits the token: a later remove_lines can delete the line that carried it
            acceptable = [x["version"] for x in versions if token in x["text"]]
            queries.append({"query_id": f"q{i:04d}s", "text": f"{base['description']} using {token}",
                            "target_snippet": sid, "target_version": v["version"],
                            "acceptable_versions": acceptable,
                            "token": token, "kind": "version_specific"})
    return {"snippets": snippets, "queries": queries,
            "meta": {"n_snippets": len(snippets), "n_versions": n_versions, "seed": seed,
                     "unchanged_rate": unchanged_rate,
                     "source": "apps-corpus" if source_texts else "synthetic",
                     "n_queries": len(queries),
                     "n_version_specific": sum(1 for q in queries if q["kind"] == "version_specific")}}


def fixture_stats(fixture):
    """Counts per mutation kind, and how many (snippet, version) pairs are byte-identical repeats."""
    per_kind, unchanged, total = {}, 0, 0
    for s in fixture["snippets"]:
        for v in s["versions"][1:]:
            per_kind[v["mutation"]] = per_kind.get(v["mutation"], 0) + 1
            total += 1
            if v["mutation"] == "unchanged":
                unchanged += 1
    return {"later_versions": total, "unchanged": unchanged, "per_mutation": per_kind,
            "unchanged_share": (unchanged / total) if total else 0.0}
