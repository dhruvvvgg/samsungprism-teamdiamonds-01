# CELL 4 - collect the release files (path, size, SHA-256), zip the indexes, refresh the metrics report and,
# if PUSH is True, commit ONLY small text/JSON results to branch results/final-run (never main or develop).
import sys
from pathlib import Path

REPO = Path("/kaggle/working/prism-test")
PUSH = True
TARGET_REPO = "dhruvvvgg/samsungprism-teamdiamonds-01"
try:
    sys.path.insert(0, str(REPO / "scripts"))
    from nb_follow import follow
    args = ["--only", "C", "--target-repo", TARGET_REPO] + (["--push-results"] if PUSH else [])
    follow(REPO, args, tag="collect")
    print("\nDownload from /kaggle/working/release/ : runtime_index.zip, runtime_index_lite.zip, "
          "results_final.zip (and results_partial.zip in /kaggle/working/).")
except Exception as exc:
    print(f"CELL 4 hit an error, and it is safe to re-run: {type(exc).__name__}: {exc}")
