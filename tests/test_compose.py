"""docker-compose.yml: serve the API and UI on CPU with an index directory mounted.

No YAML parser is a dependency, and Docker is not available to the test suite, so this reads the file as
text. It pins the properties that make the one-command demo safe: CPU only, the index mounted read-only,
the weights cached in a named volume, and consistency with the Dockerfile it builds.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_it_builds_the_repo_dockerfile_and_publishes_the_api_port():
    assert re.search(r"^\s+build:\s*\.\s*$", COMPOSE, re.M)
    assert '"${PORT:-8000}:8000"' in COMPOSE
    assert "EXPOSE 8000" in DOCKERFILE


def test_the_index_directory_is_mounted_read_only_and_selectable():
    assert "${INDEX_HOST_DIR:-./runtime_index}:/app/runtime_index:ro" in COMPOSE
    assert "INDEX_DIR: /app/runtime_index" in COMPOSE
    assert "INDEX_DIR=/app/runtime_index" in DOCKERFILE           # the same path the image expects


def test_it_is_cpu_only():
    assert "SEARCH_DEVICE: cpu" in COMPOSE
    lowered = COMPOSE.lower()
    for gpu in ("nvidia", "gpus:", "runtime: nvidia", "device_ids", "capabilities"):
        assert gpu not in lowered
    assert "cuda" not in lowered


def test_the_model_cache_is_a_named_volume_that_matches_the_dockerfile():
    assert "hf-cache:/app/hf_cache" in COMPOSE
    assert re.search(r"^volumes:\s*\n\s+hf-cache:", COMPOSE, re.M)
    assert "HF_HOME=/app/hf_cache" in DOCKERFILE
    assert "mkdir -p /app/hf_cache" in DOCKERFILE and "chown -R appuser /app" in DOCKERFILE
    # the directory must be created before the chown, or the volume would inherit root ownership
    assert DOCKERFILE.index("mkdir -p /app/hf_cache") < DOCKERFILE.index("chown -R appuser /app")


def test_serving_defaults_match_the_documented_ones():
    from src.runtime_index import DEFAULT_SERVING_QUERY_TOKENS
    assert f"MAX_QUERY_TOKENS: ${{MAX_QUERY_TOKENS:-{DEFAULT_SERVING_QUERY_TOKENS}}}" in COMPOSE


def test_the_official_configuration_is_not_touched():
    assert "official" not in COMPOSE.lower().replace("official apps", "")
    assert "run_official" not in COMPOSE and "reranker" not in COMPOSE.lower()


def test_the_readme_documents_the_exact_command():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docker compose up --build" in readme
    assert "INDEX_HOST_DIR=./runtime_index_lite docker compose up --build" in readme
    assert "has not been executed" in readme                   # says what was and was not run
