#!/usr/bin/env python3
"""Judge-ready serving launcher: start the API server and web UI over pre-downloaded indexes.

Usage:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8000
    python scripts/serve.py --mock
"""
import argparse
import os
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def detect_available_indexes(root: Path):
    """Detect which precomputed index directories exist on disk and have valid manifests."""
    candidate_map = {
        "full": root / "runtime_index",
        "lite": root / "runtime_index_lite",
        "history": root / "history_index",
        "versions": root / "version_index",
    }
    available = []
    for name, path in candidate_map.items():
        manifest = path / "manifest.json"
        if path.is_dir() and manifest.is_file():
            try:
                json.loads(manifest.read_text(encoding="utf-8"))
                available.append((name, path))
            except Exception:
                pass
    return available


def check_neural_weights_available(model_id: str, revision=None) -> bool:
    try:
        from huggingface_hub import snapshot_download
        cache = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
        if not (cache / "config.json").is_file():
            return False
        for name in ("model.safetensors", "pytorch_model.bin"):
            if (cache / name).is_file():
                return True
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if (cache / name).is_file():
                shards = json.loads((cache / name).read_text())["weight_map"].values()
                return bool(shards) and all((cache / shard).is_file() for shard in shards)
        return False
    except Exception:  # noqa: BLE001  cache lookup must fail closed
        return False


def main():
    parser = argparse.ArgumentParser(description="Serve the Samsung PRISM Code Intelligence API and Web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--index", choices=["full", "lite", "history", "versions"], default=None,
                        help="Default index to serve (default: lite or first detected)")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock hashing encoder (no model weights, CI/smoke test mode)")
    args = parser.parse_args()

    # Detect available indexes
    available = detect_available_indexes(ROOT)
    available_names = [name for name, _ in available]

    # Set ALLOWED_INDEXES
    if not os.environ.get("ALLOWED_INDEXES"):
        if available_names:
            os.environ["ALLOWED_INDEXES"] = ",".join(available_names)
        else:
            os.environ["ALLOWED_INDEXES"] = "lite,full"

    # Determine default index
    if args.index:
        chosen_index = args.index
    elif os.environ.get("INDEX_DIR"):
        chosen_index = os.environ["INDEX_DIR"]
    elif "lite" in available_names:
        chosen_index = "lite"
    elif "full" in available_names:
        chosen_index = "full"
    elif "history" in available_names:
        chosen_index = "history"
    else:
        chosen_index = "lite"

    os.environ["INDEX_DIR"] = chosen_index
    if not os.environ.get("SEARCH_DEVICE"):
        os.environ["SEARCH_DEVICE"] = "cpu"

    from src.runtime_index import resolve_index_dir
    index_dir = resolve_index_dir(chosen_index)
    required = ("manifest.json", "embeddings.npy", "doc_ids.json", "corpus_texts.json")
    missing = [str(index_dir / name) for name in required if not (index_dir / name).is_file()]
    if missing:
        parser.error("Missing index files: " + ", ".join(missing) +
                     ". Run python scripts/fetch_release_assets.py --assets indexes. "
                     "--mock skips weights only; it still requires an index.")
    manifest = json.loads((index_dir / "manifest.json").read_text())
    target_model = manifest["model"]
    if not args.mock and (target_model == "mock/hashing-encoder" or
                         not check_neural_weights_available(target_model, manifest.get("revision"))):
        parser.error(f"Missing neural weights for {target_model} at revision "
                     f"{manifest.get('revision')!r}. Download the model first, "
                     "or explicitly pass --mock for hashing-only smoke tests.")
    os.environ["MOCK_ENCODER"] = "1" if args.mock else "0"
    os.environ["FAIL_CLOSED_STARTUP"] = "1"
    if not args.mock:
        os.environ["HF_HUB_OFFLINE"] = "1"
    encoder_status = "mock/hashing-encoder" if args.mock else f"Neural Encoder ({target_model})"

    # Print banner
    print()
    print("=" * 72)
    print(" Samsung PRISM GenAI Hackathon - Code Intelligence Service")
    print("=" * 72)
    print(f" Web UI URL:        http://{args.host}:{args.port}/")
    print(f" OpenAPI Docs:      http://{args.host}:{args.port}/docs")
    print(f" Detected Indexes:  {', '.join(available_names) if available_names else 'None (downloading required)'}")
    print(f" Default Index:     {chosen_index}")
    print(f" Allowed Indexes:   {os.environ['ALLOWED_INDEXES']}")
    print(f" Query Encoder:     {encoder_status}")
    print(f" Compute Device:    {os.environ['SEARCH_DEVICE']}")
    print("=" * 72)
    print()

    # Launch uvicorn
    try:
        import uvicorn
        uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)
    except ImportError:
        sys.stderr.write(
            "[ERROR] uvicorn or fastapi is not installed in the current Python environment.\n"
            "        Install serving requirements with:\n"
            "        pip install -r requirements-serve.txt\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
