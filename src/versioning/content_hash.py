"""Content hashing for change detection: SHA-256 of *normalised* code.

The hash is the unit of work in an incremental rebuild -- a snippet whose hash is unchanged is never
re-embedded. So the normalisation decides what counts as "unchanged", and it is deliberately
conservative:

  removed    comments, trailing whitespace, blank lines, a UTF-8 BOM, \\r\\n vs \\n
  collapsed  tabs -> 4 spaces; runs of spaces *inside* a line -> one space
  KEPT       leading indentation (semantic in Python), identifier names, literals, operator choice

Anything that could change behaviour must change the hash: a renamed variable, a flipped comparison or
an added guard all produce a new hash and a fresh embedding. The reverse mistake -- normalising so
aggressively that a real edit looks unchanged -- would silently serve a stale embedding, which is much
worse than re-embedding something needlessly.

Comment stripping is lexical and string-aware (a `#` inside a string literal is not a comment), which
covers Python; a `#` inside an unterminated literal is treated as code, erring towards a new hash.
"""
import hashlib
import re

_SPACES = re.compile(r"(?<=\S) {2,}")


def strip_comments(line):
    """Drop a trailing Python comment, ignoring `#` inside string literals."""
    out, quote, i = [], None, 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":                     # escaped char inside a literal: copy both
                out.append(line[i:i + 2])
                i += 2
                continue
            if c == quote:
                quote = None
            out.append(c)
        elif c in ("'", '"'):
            quote = c
            out.append(c)
        elif c == "#":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def normalize_code(text):
    """Canonical form of a snippet for hashing (see module docstring)."""
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw in text.split("\n"):
        line = strip_comments(raw.replace("\t", "    ")).rstrip()
        if line.strip():
            lines.append(_SPACES.sub(" ", line))
    return "\n".join(lines)


def content_hash(text):
    """SHA-256 hex digest of the normalised code."""
    return hashlib.sha256(normalize_code(text).encode("utf-8")).hexdigest()
