"""Ingest a real Python repository commit by commit, so P1/Bonus are measured on real edits.

The synthetic fixture (src/versioning/fixture.py) has known-by-construction mutations, which is what
makes its reuse rate checkable -- but its edits are ones we invented. Real history is messier in exactly
the ways that matter: a commit touches a handful of files out of hundreds, renames move code between
files, and most "changes" to a function are whitespace or comments that should NOT cost an embedding.
This module reads that history out of git and turns it into the same row shape the versioned index
already understands, so both benchmarks run on the same machinery.

Lineage identity is `file::qualname` (e.g. `src/click/core.py::Command.invoke`), not a line range: line
numbers move on every edit above a function, so a line-based identity would report a rename of the whole
file as a full rebuild. A function that moves *between* files does start a new lineage by default -- that is a
genuine limitation, recorded here rather than papered over, and it shows up as one lineage ending and
another starting at the same commit. `ingest(track_renames=True)` (`--track-renames`) joins renamed files
and renamed / moved functions instead; see src/versioning/renames.py.

Versions are commit ordinals, oldest first (v1 = the oldest commit ingested), with the sha kept per row.

Everything goes through the `git` CLI via subprocess with list arguments; nothing here needs network
access once a repository exists on disk, which is what lets the tests build a fixture repo in a temp
directory and run the whole pipeline offline.
"""
import subprocess
import time
from pathlib import Path

from src.indexing.code_chunker import SKIP_DIRS, chunk_source
from src.versioning.content_hash import content_hash

# pallets/click is the default demo repository. Why this one:
#   * pure Python, no compiled extensions, so every file is parseable by `ast` on any machine;
#   * a real library with a long, linear-enough history and functions that genuinely evolve
#     (decorators, options, error handling), rather than a toy or a monorepo;
#   * small enough that a few hundred commits index in minutes on a T4, unlike Django or CPython;
#   * permissively licensed (BSD-3-Clause) and widely known, so a reviewer can sanity-check a result
#     ("who calls `Command.invoke`") against their own knowledge of the codebase.
DEFAULT_REPO = "https://github.com/pallets/click.git"
DEFAULT_SUBDIR = "src/click"


def git(args, cwd, check=True, timeout=600):
    """Run a git command with list arguments. Returns stdout; raises RuntimeError with stderr."""
    proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}:\n{proc.stderr.strip()}")
    return proc.stdout


def ensure_repo(url, dest, depth=None):
    """Clone `url` to `dest` if it is not already there; return the path. Network only on first use."""
    dest = Path(dest)
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = ["clone"]
    if depth:
        args += ["--depth", str(depth)]
    args += [url, str(dest)]
    subprocess.run(["git", *args], check=True, capture_output=True, text=True, timeout=1800)
    return dest


def list_commits(repo, max_commits=None, subdir=None, first_parent=True):
    """Commits oldest first: [{sha, short, date, subject}].

    `first_parent` follows the mainline only, so a merge-heavy history yields a linear sequence of
    releases rather than interleaved topic-branch commits -- which is what "version by version" means
    to anyone reading the result."""
    args = ["log", "--reverse", "--format=%H%x09%cI%x09%s"]
    if first_parent:
        args.append("--first-parent")
    if subdir:
        args += ["--", subdir]
    out = git(args, repo)
    commits = []
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            commits.append({"sha": parts[0], "short": parts[0][:10], "date": parts[1],
                            "subject": parts[2]})
    if max_commits and len(commits) > max_commits:
        commits = commits[-max_commits:]        # the most RECENT n, still oldest-first
    return commits


def files_at(repo, sha, subdir=None, skip_dirs=SKIP_DIRS):
    """Python files present at `sha`, as repo-relative posix paths."""
    args = ["ls-tree", "-r", "--name-only", sha]
    if subdir:
        args += ["--", subdir]
    out = git(args, repo)
    files = []
    for name in out.splitlines():
        if not name.endswith(".py"):
            continue
        if any(part in skip_dirs for part in Path(name).parts[:-1]):
            continue
        files.append(name)
    return sorted(files)


def changed_files(repo, sha, parent=None):
    """Python files this commit touched. None means 'unknown, treat everything as changed'."""
    if parent is None:
        return None
    out = git(["diff", "--name-only", parent, sha], repo, check=False)
    return {n for n in out.splitlines() if n.endswith(".py")}


def read_file_at(repo, sha, path):
    """File contents at a commit, or None if it cannot be read as UTF-8 text."""
    proc = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=str(repo),
                          capture_output=True, timeout=120)
    if proc.returncode != 0:
        return None
    try:
        return proc.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def lineage_id_for(file, qualname):
    return f"{file}::{qualname}"


def snapshot_chunks(repo, sha, subdir=None, cache=None, changed=None, previous=None):
    """Every indexable chunk at one commit, keyed by lineage id.

    `cache` maps file -> (blob_sha, chunks) so an unchanged FILE is not re-parsed; `changed` is the set
    of files this commit touched, and `previous` the previous snapshot. Parsing is cheap next to
    embedding, but on a few hundred commits of a few hundred files it is the difference between a
    benchmark that finishes and one that does not.

    Returns ({lineage_id: chunk}, parse_failures)."""
    cache = {} if cache is None else cache
    out, failures = {}, []
    for path in files_at(repo, sha, subdir):
        reusable = (previous is not None and changed is not None and path not in changed
                    and path in cache)
        if reusable:
            file_chunks = cache[path]
        else:
            source = read_file_at(repo, sha, path)
            if source is None:
                failures.append({"file": path, "error": "unreadable or not UTF-8"})
                continue
            try:
                file_chunks = chunk_source(source, path)
            except SyntaxError as exc:
                failures.append({"file": path, "error": f"SyntaxError: {exc.msg} (line {exc.lineno})"})
                continue
            cache[path] = file_chunks
        for c in file_chunks:
            if c["kind"] == "module":
                continue                      # module-level leftovers have no stable lineage identity
            lid = lineage_id_for(path, c["qualname"])
            if lid in out:
                continue                      # duplicate qualname in one file: keep the first
            out[lid] = dict(c, lineage_id=lid, content_hash=c.get("content_hash")
                            or content_hash(c["text"]))
    return out, failures


def ingest(repo, max_commits=None, subdir=None, progress=True, track_renames=False,
           rename_threshold=None):
    """Walk the history and return (rows, commits, stats).

    Rows are the versioned-index rows: one per (lineage, commit), oldest commit first, carrying the
    content hash, the source location and the commit it came from. `change` records what happened to
    that lineage at that commit -- added / modified / unchanged -- which is the ground truth the reuse
    numbers are checked against.

    track_renames (default False -- the output is then exactly what it always was): join a renamed file
    or a renamed / moved function to its earlier lineage (src/versioning/renames.py). The lineage id stays
    the one it was first given, `file` / `qualname` on a row are always the CURRENT ones, and the row at
    the link carries link_kind (`rename` | `move`), link_similarity, link_scope and link_from."""
    from src.versioning.renames import DEFAULT_THRESHOLD, find_links
    threshold = DEFAULT_THRESHOLD if rename_threshold is None else rename_threshold
    repo = Path(repo)
    commits = list_commits(repo, max_commits=max_commits, subdir=subdir)
    if not commits:
        raise ValueError(f"no commits found in {repo}"
                         + (f" touching {subdir}" if subdir else ""))
    rows, per_commit, file_cache, previous = [], [], {}, None
    prev_snap, prev_canon = None, {}
    link_totals = {"rename": 0, "move": 0}
    matcher_skipped = 0
    t_start = time.time()
    for i, commit in enumerate(commits, start=1):
        parent = commits[i - 2]["sha"] if i > 1 else None
        t0 = time.time()
        changed = changed_files(repo, commit["sha"], parent)
        snap, failures = snapshot_chunks(repo, commit["sha"], subdir, file_cache, changed, previous)
        links = {}
        if track_renames and prev_snap is not None:
            found, skipped = find_links(repo, parent, commit["sha"], prev_snap, snap, lineage_id_for,
                                        threshold)
            links = {l["new_key"]: l for l in found}
            matcher_skipped += int(skipped)
        # canonical lineage id per current key: the id it was first given, carried across links
        canon = {}
        for key in snap:
            if prev_snap is not None and key in prev_snap:
                canon[key] = prev_canon[key]
            elif key in links:
                canon[key] = prev_canon[links[key]["old_key"]]
            else:
                canon[key] = key
        current = {canon[key]: chunk for key, chunk in snap.items()}
        added = modified = unchanged = 0
        for key, chunk in snap.items():
            lid = canon[key]
            before = previous.get(lid) if previous else None
            if before is None:
                change = "added"
                added += 1
            elif before["content_hash"] != chunk["content_hash"]:
                change = "modified"
                modified += 1
            else:
                change = "unchanged"
                unchanged += 1
            row = {"doc_id": f"{lid}@v{i}", "snippet_id": lid, "lineage_id": lid, "version": i,
                   "commit": commit["sha"], "commit_short": commit["short"],
                   "commit_date": commit["date"], "commit_subject": commit["subject"],
                   "content_hash": chunk["content_hash"], "mutation": change, "change": change,
                   "file": chunk["file"], "start_line": chunk["start_line"],
                   "end_line": chunk["end_line"], "kind": chunk["kind"],
                   "name": chunk["name"], "qualname": chunk["qualname"],
                   "text": chunk["text"]}
            if key in links:
                link = links[key]
                row.update(link_kind=link["kind"], link_similarity=link["similarity"],
                           link_scope=link["scope"], link_from=link["old_key"])
                link_totals[link["kind"]] += 1
            rows.append(row)
        removed = len(set(previous) - set(current)) if previous else 0
        entry = {"version": i, "sha": commit["sha"], "short": commit["short"],
                 "date": commit["date"], "subject": commit["subject"],
                 "lineages": len(snap), "added": added, "modified": modified,
                 "unchanged": unchanged, "removed": removed,
                 "files_changed": None if changed is None else len(changed),
                 "parse_failures": len(failures),
                 "walk_seconds": round(time.time() - t0, 3)}
        if track_renames:
            entry["links"] = len(links)
        per_commit.append(entry)
        if progress and (i % 10 == 0 or i == len(commits)):
            print(f"[history] commit {i}/{len(commits)} {commit['short']} "
                  f"lineages={len(snap)} +{added} ~{modified} ={unchanged} "
                  f"({time.time() - t_start:.1f}s)", flush=True)
        previous, prev_snap, prev_canon = current, snap, canon
    stats = {"repo": str(repo), "subdir": subdir, "commits": len(commits), "rows": len(rows),
             "lineages": len({r["lineage_id"] for r in rows}),
             "distinct_hashes": len({r["content_hash"] for r in rows}),
             "walk_seconds": round(time.time() - t_start, 2),
             "changed_totals": {k: sum(c[k] for c in per_commit)
                                for k in ("added", "modified", "unchanged", "removed")}}
    if track_renames:
        stats["track_renames"] = {"enabled": True, "threshold": threshold,
                                  "links": link_totals["rename"] + link_totals["move"],
                                  "by_kind": link_totals, "matcher_skipped_commits": matcher_skipped}
    return rows, per_commit, stats
