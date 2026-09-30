"""Compare one query across two versions of a versioned index, and diff one lineage between versions.

Both are pure functions over data the search already produced, so they need no model: the caller encodes
the query once and ranks it against each version (SearchService.compare).

A result is identified by its LINEAGE (`snippet_id`), not its document id -- the same function at v2 and
v5 has two document ids and is one thing. Statuses are relative to the two top-k lists, so
`appeared` means "in B's top-k but not in A's top-k", which is not the same as "did not exist in A":
it may simply have ranked below the cut. The panel says so.
"""
import difflib


def lineage_key(hit):
    return hit.get("snippet_id") or hit["doc_id"]


def compare_hits(hits_a, hits_b):
    """Annotate two ranked hit lists against each other.

    Each B hit gets `status` in {appeared, moved, same}, `rank_in_a` and `rank_change` (positive = it
    ranks higher in B than in A) and `content_changed` (the text differs between the two versions). Each
    A hit gets `status` in {disappeared, kept} and `rank_in_b`. Hits are copied, never mutated."""
    by_a = {lineage_key(h): h for h in hits_a}
    by_b = {lineage_key(h): h for h in hits_b}
    out_b, out_a = [], []
    for h in hits_b:
        other = by_a.get(lineage_key(h))
        row = dict(h)
        if other is None:
            row.update(status="appeared", rank_in_a=None, rank_change=None, content_changed=None)
        else:
            change = other["rank"] - h["rank"]
            row.update(status="same" if change == 0 else "moved", rank_in_a=other["rank"],
                       rank_change=change,
                       content_changed=other.get("content_hash") != h.get("content_hash"))
        out_b.append(row)
    for h in hits_a:
        other = by_b.get(lineage_key(h))
        row = dict(h)
        if other is None:
            row.update(status="disappeared", rank_in_b=None)
        else:
            row.update(status="kept", rank_in_b=other["rank"])
        out_a.append(row)
    summary = {"appeared": sum(1 for r in out_b if r["status"] == "appeared"),
               "disappeared": sum(1 for r in out_a if r["status"] == "disappeared"),
               "moved": sum(1 for r in out_b if r["status"] == "moved"),
               "same": sum(1 for r in out_b if r["status"] == "same"),
               "content_changed": sum(1 for r in out_b if r.get("content_changed"))}
    return out_a, out_b, summary


def unified_diff(text_a, text_b, label_a, label_b, context=3):
    """{diff, identical, added_lines, removed_lines} -- a standard unified diff from difflib."""
    lines = list(difflib.unified_diff(text_a.splitlines(), text_b.splitlines(), fromfile=label_a,
                                      tofile=label_b, lineterm="", n=context))
    return {"diff": "\n".join(lines), "identical": not lines,
            "added_lines": sum(1 for x in lines if x.startswith("+") and not x.startswith("+++")),
            "removed_lines": sum(1 for x in lines if x.startswith("-") and not x.startswith("---"))}
