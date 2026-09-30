"""Group ranked hits by lineage, for display.

With `all_versions` every version of a function competes, so one function can fill several of the top
slots. The ranking itself is left alone (that is what the retriever produced); grouping is a view over it:
one entry per lineage, headed by its best-ranked hit, with the other versions that made the list kept
underneath. A hit with no lineage (a flat index) is its own group.
"""


def group_by_lineage(hits):
    """[{snippet_id, rank, best, others, n_in_results, versions_in_results}], in order of first
    appearance -- which is the best-ranked hit of each lineage, so group order follows the ranking."""
    groups, index = [], {}
    for h in hits:
        key = h.get("snippet_id") or h["doc_id"]
        if key not in index:
            index[key] = len(groups)
            groups.append({"snippet_id": key, "rank": len(groups) + 1, "best": h, "others": []})
        else:
            groups[index[key]]["others"].append(h)
    for g in groups:
        g["others"].sort(key=lambda x: (x.get("version") is None, x.get("version") or 0))
        versions = [g["best"], *g["others"]]
        g["n_in_results"] = len(versions)
        g["versions_in_results"] = sorted({x["version"] for x in versions if x.get("version") is not None})
    return groups
