"""Notebook helper: run the orchestrator in the BACKGROUND and follow its output.

Running it detached means a dropped browser connection or a re-run cell does not kill hours of work: a
second call finds the running process (via a pid file) and simply reattaches to its output.

    from nb_follow import follow
    follow("/kaggle/working/prism-test", ["--tier", "2", "--resume"], tag="tier12")
"""
import os
import subprocess
import sys
import time
from pathlib import Path


def _alive(pid):
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)      # our own child: reap it if it has finished
        return done == 0
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def follow(repo, argv, tag="run", poll=4.0):
    repo = Path(repo)
    script = repo / "scripts" / "run_all_checks.py"
    if not script.exists():
        print(f"{script} not found: run cell 1 first.")
        return 2
    res = repo / "results"
    res.mkdir(parents=True, exist_ok=True)
    out, pidfile = res / f"orchestrator_{tag}.out", res / f"orchestrator_{tag}.pid"
    pid, offset = None, 0
    if pidfile.exists():
        try:
            old = int(pidfile.read_text().strip())
            if _alive(old):
                pid = old
                offset = max(0, out.stat().st_size - 3000) if out.exists() else 0
                print(f"[follow] '{tag}' is already running (pid {pid}); reattaching to its output.\n")
        except (ValueError, OSError):
            pass
    if pid is None:
        with open(out, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen([sys.executable, str(script), *argv], cwd=str(repo), stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        pid = proc.pid
        pidfile.write_text(str(pid))
        print(f"[follow] started {' '.join(argv)} (pid {pid}); output also in {out}\n")
    try:
        while True:
            running = _alive(pid)
            if out.exists():
                with open(out, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offset)
                    chunk = fh.read()
                    offset = fh.tell()
                if chunk:
                    print(chunk, end="", flush=True)
            if not running:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\n[follow] stopped following. The run continues in the background; re-run this cell to "
              "reattach.")
        return 130
    pidfile.unlink(missing_ok=True)
    summary = res / "FINAL_SUMMARY.md"
    if summary.exists():
        print("\n" + "=" * 78 + f"\n{summary}\n" + "=" * 78)
        print(summary.read_text(encoding="utf-8"))
    return 0
