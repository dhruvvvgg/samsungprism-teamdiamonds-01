#!/usr/bin/env python3
"""Fetch and verify release assets from GitHub Releases for Samsung PRISM Hackathon submission.

Usage:
    python scripts/fetch_release_assets.py --assets all
    python scripts/fetch_release_assets.py --assets indexes
    python scripts/fetch_release_assets.py --assets results
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
import zipfile

DEFAULT_TAG = "PRISM_GENAI_HACKATHON_Y2026"
DEFAULT_REPO = "dhruvvvgg/samsungprism-teamdiamonds-01"

# Release assets specification with expected SHA-256 checksums
ASSET_SPECS = {
    "appsretrieval_results.json": {
        "sha256": "1d323033cfea7a688cfcaf6e885d2a5703dc9ec7e5457ba53ebbbac04cfb41a7",
        "category": "results",
        "is_zip": False,
        "target_dir": None,
    },
    "appsretrieval_rankings.json": {
        "sha256": "41480c0168d9b018a24d1d308beff33cdf9ae1de053f318260679e43ebd2c9bf",
        "category": "results",
        "is_zip": False,
        "target_dir": None,
    },
    "submission_checksums.json": {
        "sha256": "5f1c35f8e7035eb7424e5148f161359f590a78e94903b3980bdeee877541bc27",
        "category": "results",
        "is_zip": False,
        "target_dir": None,
    },
    "runtime_index.zip": {
        "sha256": "7dbd4c6a2b5449707f95c07c66924d6e90258e949ae43dafcac3b6801215f234",
        "category": "indexes",
        "is_zip": True,
        "target_dir": "runtime_index",
    },
    "runtime_index_lite.zip": {
        "sha256": "d6b2c794fc978e4dc84be23404e15ef63983ed060889a485be23646fd269b16a",
        "category": "indexes",
        "is_zip": True,
        "target_dir": "runtime_index_lite",
    },
    "history_index.zip": {
        "sha256": "56c5354d524f6a275b3c09095e4148581351ae17fff9585f2c7fa070e075f4e6",
        "category": "indexes",
        "is_zip": True,
        "target_dir": "history_index",
    },
    "results_final.zip": {
        "sha256": "d7c4e8156e662c1ca4414f1c00bdf8ee8302830faf3be7daa539cf79cb5b3eb7",
        "category": "results",
        "is_zip": True,
        "target_dir": "results",
    },
    "run_logs.zip": {
        "sha256": "30ecdcb47221e1c72e52103bd8f1eea67cee450c66e8a373a402bc80d41a3988",
        "category": "results",
        "is_zip": True,
        "target_dir": "run_logs",
    },
}


def load_specs_from_markdown(md_path: Path):
    """Optionally load or verify asset hashes against docs/RELEASE_ASSETS.md."""
    if not md_path.exists():
        return
    text = md_path.read_text(encoding="utf-8")
    for match in re.finditer(r"\|\s*`([^`]+)`\s*\|\s*[\d,]+\s*\|\s*`([a-f0-9]{64})`", text):
        name, sha = match.group(1), match.group(2)
        if name in ASSET_SPECS:
            ASSET_SPECS[name]["sha256"] = sha


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def is_directory_non_empty(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.iterdir())


def fetch_asset(name: str, spec: dict, dest: Path, repo: str, tag: str, base_url: str = None) -> bool:
    expected_sha = spec["sha256"]
    is_zip = spec["is_zip"]
    target_dir_name = spec.get("target_dir")
    unzip_target = (dest / target_dir_name) if target_dir_name else None
    dest_file = dest / name

    # Idempotency check 1: Unpacked directory already exists and is non-empty
    if is_zip and unzip_target and is_directory_non_empty(unzip_target):
        print(f"[SKIP] {name}: target directory '{unzip_target}' already exists and is non-empty.")
        return True

    # Idempotency check 2: Archive or file already exists with matching checksum
    if dest_file.exists():
        actual_sha = sha256_of_file(dest_file)
        if actual_sha.lower() == expected_sha.lower():
            print(f"[SKIP] {name}: file already exists and matches expected SHA-256.")
            if is_zip and unzip_target and not is_directory_non_empty(unzip_target):
                print(f"[EXTRACT] Extracting {name} to {dest}...")
                extract_zip(dest_file, dest, unzip_target)
            return True
        else:
            print(f"[WARN] {name} exists but has mismatched checksum ({actual_sha[:12]}... != {expected_sha[:12]}...). Re-downloading.")

    # Determine URL
    if base_url:
        url = f"{base_url.rstrip('/')}/{name}"
    else:
        url = f"https://github.com/{repo}/releases/download/{tag}/{name}"

    print(f"[FETCH] Downloading {name} from {url}...")
    tmp_file = dest / f".{name}.tmp"
    dest.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    bytes_downloaded = 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SamsungPRISM-AssetFetcher/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_file, "wb") as out:
            total_header = resp.headers.get("Content-Length")
            total_size = int(total_header) if total_header and total_header.isdigit() else None
            while chunk := resp.read(65536):
                hasher.update(chunk)
                out.write(chunk)
                bytes_downloaded += len(chunk)
                if total_size:
                    pct = (bytes_downloaded / total_size) * 100
                    print(f"\r  Downloaded {bytes_downloaded:,} / {total_size:,} bytes ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\r  Downloaded {bytes_downloaded:,} bytes", end="", flush=True)
            print()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)
        sys.stderr.write(f"\n[ERROR] Failed to download {name} from {url}: {exc}\n")
        return False

    actual_sha = hasher.hexdigest().lower()
    if actual_sha != expected_sha.lower():
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)
        sys.stderr.write(
            f"\n[ERROR] Checksum failure for {name}:\n"
            f"  Expected: {expected_sha}\n"
            f"  Got:      {actual_sha}\n"
        )
        return False

    if dest_file.exists():
        dest_file.unlink(missing_ok=True)
    tmp_file.rename(dest_file)
    print(f"[VERIFIED] {name} SHA-256 matches expected: {expected_sha[:12]}...")

    if is_zip:
        extract_zip(dest_file, dest, unzip_target)

    return True


def extract_zip(zip_path: Path, dest: Path, default_target: Path = None):
    """Safely extract zip archive to dest or subfolder."""
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        # If all members share a root directory (e.g. 'runtime_index/'), extract directly to dest
        has_common_prefix = len(names) > 0 and all("/" in name for name in names if not name.endswith("/"))
        if has_common_prefix:
            extract_dir = dest
        else:
            extract_dir = default_target if default_target else dest
        extract_dir.mkdir(parents=True, exist_ok=True)
        z.extractall(extract_dir)
        print(f"[EXTRACTED] {zip_path.name} -> {extract_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fetch and verify release assets for Samsung PRISM Hackathon.")
    parser.add_argument("--assets", choices=["all", "indexes", "results"], default="all",
                        help="Which subset of assets to download (default: all)")
    parser.add_argument("--dest", type=Path, default=Path("."),
                        help="Destination directory (default: repo root '.')")
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help=f"GitHub Release tag (default: {DEFAULT_TAG})")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"GitHub repository owner/repo (default: {DEFAULT_REPO})")
    parser.add_argument("--base-url", default=None,
                        help=argparse.SUPPRESS)  # hidden for testing with local http server

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    load_specs_from_markdown(repo_root / "docs" / "RELEASE_ASSETS.md")

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    assets_to_fetch = []
    for name, spec in ASSET_SPECS.items():
        if args.assets == "all" or spec["category"] == args.assets:
            assets_to_fetch.append((name, spec))

    print(f"=== Fetching {len(assets_to_fetch)} assets (mode: {args.assets}) to {dest} ===")
    failed = []
    for name, spec in assets_to_fetch:
        success = fetch_asset(name, spec, dest, args.repo, args.tag, args.base_url)
        if not success:
            failed.append(name)

    if failed:
        sys.stderr.write(f"\n[FAILED] {len(failed)} asset(s) failed download or verification: {', '.join(failed)}\n")
        sys.exit(1)

    print("\n[SUCCESS] All requested assets are verified and ready.")


if __name__ == "__main__":
    main()
