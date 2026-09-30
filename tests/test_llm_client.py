import pytest
from src.agent import llm_client as L


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    for v in ("LLM_PROVIDER", "LLM_MODEL", *L.KEY_ENV.values()):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(L, "CACHE_DIR", tmp_path / "cache")


def patch_backends(monkeypatch):
    calls = []
    def mk(name):
        def f(key, model, prompt, system):
            calls.append((name, key, model, prompt, system))
            return f"mock-{name}"
        return f
    monkeypatch.setattr(L, "_BACKENDS", {p: mk(p) for p in L.KEY_ENV})
    return calls


@pytest.mark.parametrize("prov", ["openai", "gemini", "anthropic"])
def test_routes_to_provider_and_model_from_env(monkeypatch, prov):
    calls = patch_backends(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", prov)
    monkeypatch.setenv("LLM_MODEL", "my-model")
    monkeypatch.setenv(L.KEY_ENV[prov], "k123")
    assert L.call_llm("hi", system="s") == f"mock-{prov}"
    assert calls == [(prov, "k123", "my-model", "hi", "s")]


def test_defaults(monkeypatch):
    calls = patch_backends(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    L.call_llm("x")
    assert calls[0][0] == "openai" and calls[0][2] == L.DEFAULT_MODELS["openai"]


def test_model_arg_overrides_env(monkeypatch):
    calls = patch_backends(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    L.call_llm("x", model="arg-model")
    assert calls[0][2] == "arg-model"


def test_cache_hit_skips_backend(monkeypatch):
    calls = patch_backends(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert L.call_llm("p", "s") == L.call_llm("p", "s")
    assert len(calls) == 1
    L.call_llm("p", "other-system")          # different key -> new call
    assert len(calls) == 2


def test_cache_served_without_key(monkeypatch):
    patch_backends(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    L.call_llm("p")
    monkeypatch.delenv("OPENAI_API_KEY")
    assert L.call_llm("p") == "mock-openai"


@pytest.mark.parametrize("prov", ["openai", "gemini", "anthropic"])
def test_missing_key_names_variable(monkeypatch, prov):
    patch_backends(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", prov)
    with pytest.raises(L.LLMConfigError, match=L.KEY_ENV[prov]):
        L.call_llm("p")


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(L.LLMConfigError, match="bogus"):
        L.call_llm("p")
