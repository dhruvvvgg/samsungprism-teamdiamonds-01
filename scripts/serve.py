#!/usr/bin/env python3
"""Judge-ready serving launcher: start the API server and web UI over pre-downloaded indexes.

Usage:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8000
    python scripts/serve.py --mock
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def detect_available_indexes(root: Path):
    """Detect which precomputed index directories exist on disk."""
    candidate_map = {
        "full": root / "runtime_index",
        "lite": root / "runtime_index_lite",
        "history": root / "history_index",
        "versions": root / "version_index",
    }
    available = []
    for name, path in candidate_map.items():
        if (path / "manifest.json").exists():
            available.append((name, path))
    return available


def check_neural_weights_available(model_id: str) -> bool:
    """Check if model weights are present in local HuggingFace cache without triggering a download."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
        cached = try_to_load_from_cache(model_id, "config.json")
        if cached is not None and not isinstance(cached, type(_CACHED_NO_EXIST)):
            return True
        return False
    except Exception:
        return False


def main(argv=None, run_server=True):
    parser = argparse.ArgumentParser(description="Serve the Samsung PRISM Code Intelligence API and Web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--index", choices=["full", "lite", "history", "versions"], default=None,
                        help="Default index to serve (default: full or first detected)")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock hashing encoder (no model weights; for UI development only)")
    args = parser.parse_args(argv)

    # Detect available indexes
    available = detect_available_indexes(ROOT)
    available_names = [name for name, _ in available]

    if not available_names:
        sys.stderr.write(
            "\n[ERROR] No precomputed indexes found in repository root.\n"
            "        Run: python scripts/fetch_release_assets.py --assets indexes\n"
            "        to download the official submission indexes before starting the server.\n\n"
        )
        sys.exit(1)

    # Set ALLOWED_INDEXES
    if not os.environ.get("ALLOWED_INDEXES"):
        os.environ["ALLOWED_INDEXES"] = ",".join(available_names)

    # Determine default index
    if args.index:
        chosen_index = args.index
    elif os.environ.get("INDEX_DIR"):
        chosen_index = os.environ["INDEX_DIR"]
    elif "full" in available_names:
        chosen_index = "full"
    elif "lite" in available_names:
        chosen_index = "lite"
    elif "history" in available_names:
        chosen_index = "history"
    else:
        chosen_index = "full"

    os.environ["INDEX_DIR"] = chosen_index
    if not os.environ.get("SEARCH_DEVICE"):
        os.environ["SEARCH_DEVICE"] = "cpu"

    # Determine model and encoder mode
    target_model = (
        "FreedomIntelligence/F2LLM-v2-1.7B" if chosen_index == "full"
        else "FreedomIntelligence/F2LLM-v2-0.6B"
    )
    if args.mock:
        os.environ["MOCK_ENCODER"] = "1"
        print("=" * 72)
        print("*** MOCK ENCODER: results are not real ***")
        print("=" * 72)
        encoder_status = "Mock Hashing Query Encoder (--mock enabled; UI development only)"
    else:
        if os.environ.get("MOCK_ENCODER") == "1":
            os.environ.pop("MOCK_ENCODER", None)
        has_weights = check_neural_weights_available(target_model)
        if has_weights:
            encoder_status = f"Neural Encoder ({target_model} from local HF cache)"
        else:
            encoder_status = f"Neural Encoder ({target_model})"

    # Print banner
    print()
    print("=" * 72)
    print(" Samsung PRISM GenAI Hackathon - Code Intelligence Service")
    print("=" * 72)
    print(f" Web UI URL:        http://{args.host}:{args.port}/")
    detected_str = ', '.join(available_names) if available_names else 'None (downloading required)'
    print(f" Detected Indexes:  {detected_str}")
    print(f" Default Index:     {chosen_index}")
    print(f" Allowed Indexes:   {os.environ['ALLOWED_INDEXES']}")
    print(f" Query Encoder:     {encoder_status}")
    print(f" Compute Device:    {os.environ['SEARCH_DEVICE']}")
    print("=" * 72)
    print()

    # Launch uvicorn
    if run_server:
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
