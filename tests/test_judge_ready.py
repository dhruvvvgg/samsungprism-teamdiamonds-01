"""Tests for scripts/fetch_release_assets.py and scripts/serve.py."""
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import threading
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


class SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress console logs during tests


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fetch_release_assets_good_download_and_extract(tmp_path):
    from scripts.fetch_release_assets import fetch_asset

    # Setup a local http server serving from a temp dir
    server_dir = tmp_path / "server_files"
    server_dir.mkdir()

    # Create dummy zip with content
    zip_bytes_io = server_dir / "sample_index.zip"
    with zipfile.ZipFile(zip_bytes_io, "w") as z:
        z.writestr("sample_index/manifest.json", json.dumps({"kind": "test"}))
        z.writestr("sample_index/data.txt", "hello world")

    raw_zip = zip_bytes_io.read_bytes()
    expected_sha = sha256_of_bytes(raw_zip)

    # Start HTTP server on random port
    handler = functools.partial(SilentHandler, directory=str(server_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        dest = tmp_path / "client_dest"
        spec = {
            "sha256": expected_sha,
            "category": "indexes",
            "is_zip": True,
            "target_dir": "sample_index",
        }
        base_url = f"http://127.0.0.1:{port}"

        # 1. Fetch asset
        success = fetch_asset("sample_index.zip", spec, dest, repo="dummy/repo", tag="v1", base_url=base_url)
        assert success is True
        assert (dest / "sample_index.zip").exists()
        assert (dest / "sample_index" / "manifest.json").exists()
        assert (dest / "sample_index" / "data.txt").read_text() == "hello world"

        # 2. Idempotent skip: directory exists and is non-empty
        skip_success = fetch_asset("sample_index.zip", spec, dest, repo="dummy/repo", tag="v1", base_url=base_url)
        assert skip_success is True
    finally:
        server.shutdown()


def test_fetch_release_assets_corrupt_download_fail_closed(tmp_path):
    from scripts.fetch_release_assets import fetch_asset

    server_dir = tmp_path / "server_files"
    server_dir.mkdir()

    file_path = server_dir / "corrupt.json"
    file_path.write_text('{"bad": "data"}', encoding="utf-8")

    handler = functools.partial(SilentHandler, directory=str(server_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        dest = tmp_path / "client_dest"
        # Provide mismatched SHA-256
        spec = {
            "sha256": "0" * 64,
            "category": "results",
            "is_zip": False,
            "target_dir": None,
        }
        base_url = f"http://127.0.0.1:{port}"

        success = fetch_asset("corrupt.json", spec, dest, repo="dummy/repo", tag="v1", base_url=base_url)
        assert success is False
        assert not (dest / "corrupt.json").exists()
        assert not (dest / ".corrupt.json.tmp").exists()
    finally:
        server.shutdown()


def test_release_assets_specs_match_doc():
    from scripts.fetch_release_assets import ASSET_SPECS, load_specs_from_markdown

    doc_path = ROOT / "docs" / "RELEASE_ASSETS.md"
    assert doc_path.exists()
    load_specs_from_markdown(doc_path)

    assert len(ASSET_SPECS) == 8
    assert "runtime_index.zip" in ASSET_SPECS
    assert ASSET_SPECS["runtime_index.zip"]["sha256"] == "7dbd4c6a2b5449707f95c07c66924d6e90258e949ae43dafcac3b6801215f234"
    assert ASSET_SPECS["appsretrieval_results.json"]["sha256"] == "1d323033cfea7a688cfcaf6e885d2a5703dc9ec7e5457ba53ebbbac04cfb41a7"


def test_serve_detect_available_indexes(tmp_path):
    from scripts.serve import check_neural_weights_available, detect_available_indexes

    # None detected in empty folder
    assert detect_available_indexes(tmp_path) == []

    # Create dummy indexes
    (tmp_path / "runtime_index").mkdir()
    (tmp_path / "runtime_index" / "manifest.json").write_text("{}", encoding="utf-8")

    (tmp_path / "history_index").mkdir()
    (tmp_path / "history_index" / "manifest.json").write_text("{}", encoding="utf-8")

    detected = detect_available_indexes(tmp_path)
    names = [name for name, _ in detected]
    assert "full" in names
    assert "history" in names
    assert "lite" not in names

    # Weight check returns boolean without crash
    res = check_neural_weights_available("nonexistent/model")
    assert isinstance(res, bool)
