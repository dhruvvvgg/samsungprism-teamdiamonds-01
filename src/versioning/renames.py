"""Rename and move tracking for the git-history ingester (off by default; `--track-renames`).

Without it a lineage is `file::qualname`, so a renamed file or function ends one lineage and starts
another at the same commit. With it, a link joins the two so the lineage keeps its identity, and the
link is recorded on the row that follows it: its kind (`rename` or `move`), the body similarity and where
it came from.

Two evidence sources, both read within ONE commit (parent -> commit), never across a longer span:

  file renames      git's own rename detection (`git diff -M`). Every function of the old file whose
                    qualname exists in the new file continues its lineage. Kind: `rename`.
  function renames  AST-level body similarity between a function that disappeared and one that appeared
                    in the same commit: Jaccard of the multiset of normalised tokens >= a threshold
                    (default 0.8). Same file (after any file rename): `rename`. Different file: `move`.

Normalised tokens are the Python tokens of the definition with comments and layout dropped and the
function's own name replaced by a placeholder, so renaming a function (and its recursive calls) does not
by itself lower the similarity. Matching is one-to-one and greedy from the best pair down, ties broken by
name, so the result is deterministic.

What this does not do: it never links across more than one commit, never links two functions of different
kinds (function vs class), and ignores bodies shorter than MIN_TOKENS tokens (a `return x` looks like every
other `return x`). A copy that leaves the original in place is not a move.
"""
import io
import tokenize
from collections import Counter

DEFAULT_THRESHOLD = 0.8
MIN_TOKENS = 8
MAX_PAIRS = 200_000           # removed x added candidates per commit; beyond this, matching is skipped
_LAYOUT = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
           tokenize.ENDMARKER}


def normalized_tokens(text, own_name=None):
    """Tokens of `text` without comments/layout; `own_name` becomes the placeholder `<self>`."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in _LAYOUT:
                continue
            out.append("<self>" if (tok.type == tokenize.NAME and tok.string == own_name)
                       else tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass                                   # keep whatever tokenised before the problem
    return out


def jaccard(a, b):
    """Multiset Jaccard of two token lists: sum(min counts) / sum(max counts)."""
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return (sum(min(ca[k], cb[k]) for k in keys) / union) if union else 0.0


def git_file_renames(repo, parent, sha):
    """[(old_path, new_path, similarity 0..1)] for Python files git detects as renamed in parent->sha."""
    from src.versioning.git_history import git
    out = git(["-c", "core.quotepath=off", "diff", "-M", "--name-status", "--diff-filter=R", parent, sha],
              repo, check=False)
    renames = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].startswith("R") and parts[1].endswith(".py") \
                and parts[2].endswith(".py"):
            try:
                sim = int(parts[0][1:]) / 100.0
            except ValueError:
                sim = 1.0
            renames.append((parts[1], parts[2], sim))
    return renames


def match_functions(removed, added, threshold=DEFAULT_THRESHOLD, min_tokens=MIN_TOKENS):
    """One-to-one greedy matching of disappeared to appeared chunks by body similarity.
    `removed` / `added`: {key: chunk}. Returns ([(old_key, new_key, similarity)], skipped: bool)."""
    if len(removed) * len(added) > MAX_PAIRS:
        return [], True
    toks_old = {k: normalized_tokens(c["text"], c.get("name")) for k, c in removed.items()}
    toks_new = {k: normalized_tokens(c["text"], c.get("name")) for k, c in added.items()}
    scored = []
    for ko, to in toks_old.items():
        if len(to) < min_tokens:
            continue
        for kn, tn in toks_new.items():
            if len(tn) < min_tokens or removed[ko]["kind"] != added[kn]["kind"]:
                continue
            sim = jaccard(to, tn)
            if sim >= threshold:
                scored.append((-sim, ko, kn))
    scored.sort()
    used_old, used_new, matches = set(), set(), []
    for neg, ko, kn in scored:
        if ko in used_old or kn in used_new:
            continue
        used_old.add(ko)
        used_new.add(kn)
        matches.append((ko, kn, round(-neg, 4)))
    return matches, False


def find_links(repo, parent, sha, prev_snap, snap, lineage_id_for, threshold=DEFAULT_THRESHOLD,
               min_tokens=MIN_TOKENS):
    """Links between the previous snapshot and this one, as
    [{old_key, new_key, kind, scope, similarity}] plus a `skipped` flag for the matcher's size guard."""
    removed = {k: c for k, c in prev_snap.items() if k not in snap}
    added = {k: c for k, c in snap.items() if k not in prev_snap}
    links, skipped = [], False
    if not removed or not added:
        return links, skipped
    file_map, used_old, used_new = {}, set(), set()
    for old_path, new_path, _sim in git_file_renames(repo, parent, sha):
        file_map[old_path] = new_path
        for old_key, oc in sorted(removed.items()):
            if oc["file"] != old_path:
                continue
            new_key = lineage_id_for(new_path, oc["qualname"])
            if new_key in added and new_key not in used_new:
                sim = jaccard(normalized_tokens(oc["text"], oc.get("name")),
                              normalized_tokens(added[new_key]["text"], added[new_key].get("name")))
                links.append({"old_key": old_key, "new_key": new_key, "kind": "rename",
                              "scope": "file", "similarity": round(sim, 4)})
                used_old.add(old_key)
                used_new.add(new_key)
    rem = {k: c for k, c in removed.items() if k not in used_old}
    add = {k: c for k, c in added.items() if k not in used_new}
    matches, skipped = match_functions(rem, add, threshold, min_tokens)
    for ko, kn, sim in matches:
        same_file = file_map.get(rem[ko]["file"], rem[ko]["file"]) == add[kn]["file"]
        links.append({"old_key": ko, "new_key": kn, "kind": "rename" if same_file else "move",
                      "scope": "function", "similarity": sim})
    return links, skipped
