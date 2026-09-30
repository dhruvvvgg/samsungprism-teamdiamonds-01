"""Query-side variants for an instruction-tuned embedder (F2LLM template `Instruct: {i}\\nQuery: {q}`).

A variant = (instruction key, text-transform key). Variants are encoded separately (cached) and can
be averaged: mean of L2-normalised embeddings, re-normalised.
"""
import re

import numpy as np

TEMPLATE = "Instruct: {instruction}\nQuery: {query}"

# Instruction wordings. Sources (all read in the installed MTEB registry, mteb/models/model_implementations):
#   registry      codefuse_models.py f2llmv2_prompts_dict["AppsRetrieval"]  (the wording F2LLM-v2's 0.905 used)
#   contest       codefuse_models.py c2llm_prompts_dict["AppsRetrieval"]["query"]  (C2LLM, sibling model)
#   code_contest  geevec_models.py / seed_1_6_embedding_models_1215.py "AppsRetrieval"
#   solution      ours (no registry precedent): asks for a Python solution explicitly
INSTRUCTIONS = {
    "registry": "Retrieve the most relevant code snippet for the given query.",
    "contest": "Given a problem description from a programming contest, retrieve code examples that "
               "can assist in solving it.",
    "code_contest": "Given a code contest problem description, retrieve relevant code that can help "
                    "solve the problem.",
    "solution": "Given a competitive programming problem statement, retrieve a Python solution "
                "that solves it.",
}

# A header line: optional dashes / markdown '#' / bold '**', a keyword, optional number, optional colon.
_HDR = r"^[ \t#*\-]*(?:{kw})\b[^\n]{{0,25}}$"
_EXAMPLE_KW = (r"examples?|sample\s+(?:input|output|tests?)|example\s+(?:input|output)|"
               r"test\s+cases?|sample\s+cases?")
_IO_KW = r"input(?:\s+format)?|output(?:\s+format)?|constraints?|subtasks?|" + _EXAMPLE_KW
EXAMPLE_HEADER = re.compile(_HDR.format(kw=_EXAMPLE_KW), re.I | re.M)
SPEC_HEADER = re.compile(_HDR.format(kw=_IO_KW), re.I | re.M)


def _cut_at_header(text, header_re, min_chars):
    """Keep the text before the first header line; fall back to the full text if there is no header or
    the remainder would be shorter than min_chars (never return an empty / trivial query)."""
    m = header_re.search(text)
    if not m:
        return text
    head = text[:m.start()].rstrip()
    return head if len(head) >= min_chars else text


def transform_full(text, min_chars=120):
    return text


def transform_no_examples(text, min_chars=120):
    """Drop everything from the first Example/Sample header on."""
    return _cut_at_header(text, EXAMPLE_HEADER, min_chars)


def transform_narrative(text, min_chars=120):
    """Keep only the narrative: drop from the first Input/Output/Constraints/Example header on."""
    return _cut_at_header(text, SPEC_HEADER, min_chars)


TRANSFORMS = {"full": transform_full, "no_examples": transform_no_examples,
              "narrative": transform_narrative}


def variant_name(instruction, transform):
    return f"{instruction}+{transform}"


def parse_variant(name):
    instruction, transform = name.split("+")
    if instruction not in INSTRUCTIONS or transform not in TRANSFORMS:
        raise ValueError(f"unknown variant {name!r}; instructions={sorted(INSTRUCTIONS)}, "
                         f"transforms={sorted(TRANSFORMS)}")
    return instruction, transform


def format_query(text, variant):
    """Full string sent to the encoder for a query under `variant` (e.g. 'registry+full')."""
    instruction, transform = parse_variant(variant)
    return TEMPLATE.format(instruction=INSTRUCTIONS[instruction], query=TRANSFORMS[transform](text))


def average_embeddings(embs):
    """Mean of L2-normalised embeddings, re-normalised. embs: list of (Q, D) arrays."""
    stack = np.stack([e.astype(np.float32) / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
                      for e in embs])
    mean = stack.mean(axis=0)
    return mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
