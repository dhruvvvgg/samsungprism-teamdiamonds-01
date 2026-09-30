# CELL 3 - OPTIONAL Tier 3 experiments (installs requirements-experiments.txt). Off unless you run it.
# A = gated rerank, F = description fusion, B = fine-tune with paired eval, E = category tiebreaker.
# Delete letters from ONLY to run fewer. F stops after a 200-document sample: read the three descriptions
# it prints, then set CONTINUE_F = True and re-run this cell to describe the whole corpus and fuse.
import sys
from pathlib import Path

REPO = Path("/kaggle/working/prism-test")
ONLY = ["A", "F", "B", "E"]
CONTINUE_F = False
try:
    sys.path.insert(0, str(REPO / "scripts"))
    from nb_follow import follow
    args = ["--only", *ONLY, "--resume", "--max-hours", "10"] + (["--continue-f"] if CONTINUE_F else [])
    follow(REPO, args, tag="tier3")
except Exception as exc:
    print(f"CELL 3 hit an error, and it is safe to re-run: {type(exc).__name__}: {exc}")
