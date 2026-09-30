"""Atomic file writes.

A long run that gets killed (Kaggle session timeout, OOM-kill, lost connection) can be interrupted
*during* a write. A half-written JSON file is worse than no file: `appsretrieval_results.json`
truncated mid-dump still looks like a result, and a truncated lock file still blocks a retry. Writing
to a temporary file in the same directory and then os.replace()-ing it makes the visible file either
the complete old one or the complete new one, never a partial one (os.replace is atomic on POSIX and
on Windows for same-volume renames).
"""
import json
import os
from pathlib import Path


def write_json_atomic(path, obj, **dump_kwargs):
    """Serialise `obj` to `path` atomically. Returns the Path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())        # survive a host-level kill, not just a process exit
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)  # a failed write leaves no misleading .partial behind
    return path


def save_npy_atomic(path, array):
    """np.save to `path` atomically, so a killed run cannot leave a truncated .npy in a cache."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial.npy")
    try:
        with open(tmp, "wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path
