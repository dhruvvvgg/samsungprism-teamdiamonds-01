# CELL 2 - Tier 1 (official run, indexes, analysis, CPU checks) then Tier 2 (API smoke test, real history).
# Runs detached: if the browser disconnects or you re-run this cell it reattaches, and --resume skips any
# phase that already passed. A failing phase never stops the ones after it.
import sys
from pathlib import Path

REPO = Path("/kaggle/working/prism-test")
try:
    sys.path.insert(0, str(REPO / "scripts"))
    from nb_follow import follow
    follow(REPO, ["--tier", "2", "--resume", "--max-hours", "9.5"], tag="tier12")
except Exception as exc:
    print(f"CELL 2 hit an error, and it is safe to re-run: {type(exc).__name__}: {exc}")
