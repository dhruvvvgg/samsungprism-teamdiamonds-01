"""Phase 6 (LLM re-judge only): re-order the top few candidates with an LLM's "does this code solve this
problem?" score, but ONLY when the retrieval ranking is uncertain.

  trigger : margin_z = (s1 - s2) / std(top-10 scores)  <  margin_z_threshold   (scale-free, so it works
            for cosine, RRF and reranker scores alike)
  judge   : one call_llm() call per candidate in the top `judge_top` -> score in [0, 1]
  fuse    : final = (1 - beta) * minmax(retrieval score) + beta * judge score, within the judged block;
            candidates below the block keep their order.

All LLM traffic goes through src.agent.llm_client.call_llm (provider-agnostic, disk-cached; set
LLM_PROVIDER=mock for keyless dry runs). Context expansion is intentionally NOT implemented: corpus items
are standalone scripts (see results/corpus_audit.md).
"""
import json
import re

import numpy as np

JUDGE_SYSTEM = "You are a strict competitive-programming code reviewer."
JUDGE_PROMPT = (
    "Decide whether the Python code below is a correct solution to the problem. Reply with ONLY a JSON "
    'object {{"score": <number from 0 to 1>, "reason": "<one short sentence>"}}, where 1 means certainly '
    "correct and 0 means certainly wrong or unrelated.\n\n"
    "PROBLEM:\n{problem}\nCODE:\n{code}"
)


def build_prompt(problem, code, max_problem_chars=4000, max_code_chars=4000):
    return JUDGE_PROMPT.format(problem=problem[:max_problem_chars].strip(), code=code[:max_code_chars].strip())


def parse_score(text):
    """Extract a score in [0, 1] from an LLM reply; None if nothing usable."""
    if not text:
        return None
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0)).get("score")
            if isinstance(v, (int, float)):
                return float(min(1.0, max(0.0, v)))
        except (ValueError, AttributeError):
            pass
    m = re.search(r"score\"?\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.I) or re.search(r"\b([01](?:\.\d+)?)\b", text)
    return float(min(1.0, max(0.0, float(m.group(1))))) if m else None


def judge_score(problem, code, call=None):
    """One LLM call -> score in [0,1] or None (unparseable / provider error other than bad config)."""
    if call is None:
        from src.agent.llm_client import LLMConfigError, call_llm
        call = call_llm
    else:
        LLMConfigError = ()  # injected callables: never swallow anything special
    try:
        return parse_score(call(build_prompt(problem, code), system=JUDGE_SYSTEM))
    except LLMConfigError:
        raise                       # missing key / bad provider must be loud, not silently ignored
    except Exception:               # noqa: BLE001  network / rate-limit: degrade to "no opinion"
        return None


def margin_z(scores):
    """(s1 - s2) / std(top-10). 0 when fewer than 2 candidates or no spread."""
    s = np.asarray(scores[:10], dtype=np.float64)
    if len(s) < 2:
        return 0.0
    sd = s.std()
    return float((s[0] - s[1]) / sd) if sd > 1e-12 else 0.0


def _minmax(x):
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 1e-12 else np.full_like(x, 0.5)


def rejudge(problem, ranked_idx, ranked_scores, doc_texts, *, judge_top=3, margin_z_threshold=0.5,
            beta=0.5, call=None):
    """Return (new_idx, new_scores, info). Never raises on LLM failure: falls back to the input ranking."""
    ranked_idx, ranked_scores = list(ranked_idx), np.asarray(ranked_scores, dtype=np.float64)
    mz = margin_z(ranked_scores)
    info = {"margin_z": mz, "triggered": False, "n_calls": 0}
    if len(ranked_idx) < 2 or mz >= margin_z_threshold:
        return ranked_idx, ranked_scores, info
    info["triggered"] = True
    j = min(judge_top, len(ranked_idx))
    judged = [judge_score(problem, doc_texts[d], call) for d in ranked_idx[:j]]
    info["n_calls"] = j
    if all(v is None for v in judged):
        info["fallback"] = True
        return ranked_idx, ranked_scores, info
    retr = _minmax(ranked_scores[:j])
    judge = np.array([retr[i] if v is None else v for i, v in enumerate(judged)])   # None: keep retrieval view
    final = (1.0 - beta) * retr + beta * judge
    order = sorted(range(j), key=lambda i: (-final[i], i))
    new_idx = [ranked_idx[i] for i in order] + ranked_idx[j:]
    info["judge_scores"] = judged
    # only the ORDER changes; positions keep their original (descending) score values, so scores stay monotone
    return new_idx, ranked_scores.copy(), info
