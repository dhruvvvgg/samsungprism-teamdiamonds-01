"""The ONLY module allowed to import a provider SDK. Use call_llm everywhere else."""
import hashlib
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "llm_cache"

DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
}
KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


MOCK_PROVIDER = "mock"   # LLM_PROVIDER=mock: no key, no network; deterministic; still uses the disk cache


class LLMConfigError(RuntimeError):
    """Raised for missing/invalid provider configuration."""


def _resolve(model=None):
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider == MOCK_PROVIDER:
        return provider, model or os.environ.get("LLM_MODEL") or "mock-judge"
    if provider not in KEY_ENV:
        raise LLMConfigError(
            f"Unsupported LLM_PROVIDER={provider!r}; expected one of {sorted(KEY_ENV) + [MOCK_PROVIDER]}")
    model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS[provider]
    return provider, model


def _require_key(provider):
    name = KEY_ENV[provider]
    key = os.environ.get(name)
    if not key:
        raise LLMConfigError(
            f"Missing environment variable {name} (required for LLM_PROVIDER={provider}). "
            f"Set it in the notebook environment or a .env file.")
    return key


def _cache_path(provider, model, prompt, system):
    blob = json.dumps([provider, model, prompt, system], ensure_ascii=False)
    return CACHE_DIR / (hashlib.sha256(blob.encode("utf-8")).hexdigest() + ".json")


def _call_openai(key, model, prompt, system):
    from openai import OpenAI
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    r = OpenAI(api_key=key).chat.completions.create(model=model, messages=msgs)
    return r.choices[0].message.content or ""


def _call_gemini(key, model, prompt, system):
    import google.generativeai as genai
    genai.configure(api_key=key)
    m = genai.GenerativeModel(model, system_instruction=system) if system \
        else genai.GenerativeModel(model)
    return m.generate_content(prompt).text


def _call_anthropic(key, model, prompt, system):
    import anthropic
    kw = {"system": system} if system else {}
    r = anthropic.Anthropic(api_key=key).messages.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content": prompt}], **kw)
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def _call_mock(key, model, prompt, system):
    """Deterministic stand-in. If the prompt has 'PROBLEM:' and 'CODE:' sections (see
    src/agent/llm_rejudge.py) return a JSON score = word overlap (Jaccard) between the two; else echo."""
    import re
    m = re.search(r"PROBLEM:\n(.*?)\nCODE:\n(.*)", prompt, re.S)
    if not m:
        return "mock response"
    words = lambda t: set(re.findall(r"[a-z]{3,}", t.lower()))  # noqa: E731
    a, b = words(m.group(1)), words(m.group(2))
    score = len(a & b) / max(len(a | b), 1)
    return json.dumps({"score": round(min(1.0, 2.0 * score), 3), "reason": "mock"})


_BACKENDS = {"openai": _call_openai, "gemini": _call_gemini, "anthropic": _call_anthropic}


def call_llm(prompt: str, system: str = None, model: str = None) -> str:
    provider, model = _resolve(model)
    path = _cache_path(provider, model, prompt, system)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        except (OSError, ValueError, KeyError):
            pass  # corrupt cache entry: fall through and re-call
    if provider == MOCK_PROVIDER:
        key, backend = "", _call_mock
    else:
        key, backend = _require_key(provider), _BACKENDS[provider]
    text = backend(key, model, prompt, system)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"provider": provider, "model": model, "response": text}),
                    encoding="utf-8")
    return text
