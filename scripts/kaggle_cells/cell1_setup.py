# CELL 1 - setup: read the token, clone, install, check the environment.  Safe to re-run: on a re-run it
# refreshes the code and keeps results/ (and so the run state).
import os, subprocess, sys
from pathlib import Path

BRANCH = "develop"                                   # or the branch you want to test
REPO_HOST = "github.com/dhruvvvgg/samsungprism-teamdiamonds-01.git"
REPO = Path("/kaggle/working/prism-test")

try:
    try:                                             # Kaggle
        from kaggle_secrets import UserSecretsClient
        TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:                                # Colab or a plain environment
        TOKEN = os.environ.get("GITHUB_TOKEN", "")
    os.environ["GITHUB_TOKEN"] = TOKEN or ""
    print("token:", "found" if TOKEN else "not provided (repo is public; token only needed for --push-results)")

    def sh(cmd, cwd=None):
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        text = p.stdout + p.stderr
        print((text.replace(TOKEN, "***") if TOKEN else text)[-2500:])
        return p.returncode

    url = f"https://x-access-token:{TOKEN}@{REPO_HOST}" if TOKEN else f"https://{REPO_HOST}"
    if (REPO / ".git").exists():                     # re-run: refresh tracked files, keep results/
        sh(["git", "-C", str(REPO), "fetch", "--quiet", url, BRANCH])
        sh(["git", "-C", str(REPO), "checkout", "--quiet", "-f", "-B", BRANCH, "FETCH_HEAD"])
    else:
        if sh(["git", "clone", "--quiet", "--branch", BRANCH, "--single-branch", url, str(REPO)]) != 0:
            raise SystemExit("clone failed: check GITHUB_TOKEN and the branch name")
    sh(["git", "-C", str(REPO), "remote", "set-url", "origin", f"https://{REPO_HOST}"])   # no token in .git
    sh(["git", "-C", str(REPO), "log", "-1", "--oneline"])
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")])
    sys.path.insert(0, str(REPO / "scripts"))
    from nb_follow import follow
    follow(REPO, ["--only", "P0"], tag="p0")         # environment check: GPU, cores, RAM, disk, versions
except SystemExit:
    raise
except Exception as exc:
    print(f"CELL 1 hit an error, and it is safe to re-run: {type(exc).__name__}: {exc}")
