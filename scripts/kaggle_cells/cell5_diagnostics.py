# CELL 5 - DIAGNOSTICS: run only if something failed. Prints every phase's status and the tail of each
# failing phase's log (P1's is always shown).
import subprocess, sys
from pathlib import Path

REPO = Path("/kaggle/working/prism-test")
try:
    p = subprocess.run([sys.executable, "scripts/run_all_checks.py", "--diagnose"], cwd=REPO,
                       capture_output=True, text=True)
    print(p.stdout[-30000:], p.stderr[-3000:])
except Exception as exc:
    print(f"CELL 5 hit an error: {type(exc).__name__}: {exc}")
