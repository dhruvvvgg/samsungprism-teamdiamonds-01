#!/usr/bin/env python
"""Run every real-model check in one go, resiliently, and say exactly what to paste back.

Written for ONE fresh Kaggle notebook (GPU T4, internet on, no input datasets): it builds every index it
needs and reads nothing that was prepared earlier.

    python scripts/run_all_checks.py --tier 2 --resume --max-hours 9          # tiers 1 + 2
    python scripts/run_all_checks.py --only C                                 # collect release files
    python scripts/run_all_checks.py --diagnose                               # tail of every phase log
    python scripts/run_all_checks.py --tier 2 --dry-run                       # the plan, nothing run

Phases
  Tier 1  P0 environment | P1 official 1.7B run + verification | P2 lite index + hash checks
          P3 failure analysis + metrics report | P4 CPU precision and latency
  Tier 2  P5 API smoke test over HTTP | P6 real history (click), delta vectors, rename tracking, agent
  Final   C collect release files

Resilience
  * results/run_state.json records each finished phase; with --resume a finished phase is skipped.
  * one log per phase in results/logs/; a phase runs in try/except with its own time budget, so a failure or
    a timeout never stops the phases after it.
  * after every phase results/ is re-zipped to /kaggle/working/results_partial.zip.
  * results/FINAL_SUMMARY.md is rewritten after every run: a PASS/FAIL/SKIPPED table with timings and,
    per phase, the exact lines and numbers to paste back.

This script never changes retrieval behaviour: it only invokes the repository's own scripts.
"""
import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
STATE = RESULTS / "run_state.json"
SUMMARY = RESULTS / "FINAL_SUMMARY.md"
RELEASE_JSON = RESULTS / "release_files.json"
KAGGLE = Path("/kaggle/working")
PARTIAL_ZIP = (KAGGLE if KAGGLE.exists() else ROOT) / "results_partial.zip"
RELEASE_DIR = (KAGGLE if KAGGLE.exists() else ROOT) / "release"
LOCK = ROOT / "outputs" / "dev" / "OFFICIAL_RUN_DONE.json"
OFFICIAL_CONFIG = "configs/official_f2llm17b_noreranker.json"
TARGET_NDCG, TARGET_MRR, TOLERANCE = 0.9376, 0.9238, 0.002
PY = sys.executable


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------

def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path, block=1 << 20):
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def grep_lines(text, patterns, limit=12):
    """Lines of `text` matching any regex in `patterns`, stripped, de-duplicated, in order."""
    out, seen = [], set()
    for line in text.splitlines():
        s = line.strip()
        if s and s not in seen and any(re.search(p, s) for p in patterns):
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                break
    return out


def extract_block(text, start_pat, end_pat=None, max_lines=30):
    """The first block of `text` that starts at a line matching start_pat, up to and including the first
    later line matching end_pat (or max_lines). Separator rows and blank lines are dropped."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(start_pat, line):
            block = []
            for j in range(i, min(i + max_lines, len(lines))):
                block.append(lines[j].rstrip())
                if end_pat and j > i and re.search(end_pat, lines[j]):
                    break
            return [b for b in block if b.strip() and not re.fullmatch(r"\s*=+\s*", b)]
    return []


def sysinfo():
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "logical_cores": os.cpu_count(), "physical_cores": None, "ram_total_gb": None,
            "ram_available_gb": None}
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 1)
        info["ram_available_gb"] = round(vm.available / 1e9, 1)
    except Exception:  # noqa: BLE001  psutil is optional here
        try:
            mem = {k: int(v.split()[0]) for k, v in
                   (ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)}
            info["ram_total_gb"] = round(mem["MemTotal"] * 1024 / 1e9, 1)
            info["ram_available_gb"] = round(mem["MemAvailable"] * 1024 / 1e9, 1)
        except Exception:  # noqa: BLE001
            pass
    try:
        du = shutil.disk_usage(str(KAGGLE if KAGGLE.exists() else ROOT))
        info["disk_free_gb"] = round(du.free / 1e9, 1)
    except OSError:
        info["disk_free_gb"] = None
    return info


def ram_available_gb():
    return sysinfo().get("ram_available_gb") or 0.0


def git_head():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
        br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout
        return sha.strip(), br.strip()
    except OSError:
        return "unknown", "unknown"


# --------------------------------------------------------------------------------------------------
# running commands
# --------------------------------------------------------------------------------------------------

class PhaseTimeout(Exception):
    """The phase's own time budget ran out."""


class SkipPhase(Exception):
    """Raised by a phase that decides it should not run (with the reason)."""


class Ctx:
    """What a phase gets: a logger, a command runner, and the lists of key lines and failures."""

    def __init__(self, pid, args, log_path, budget_s, deadline_s, token=None):
        self.pid, self.args, self.log_path = pid, args, Path(log_path)
        self.budget_s, self.deadline = budget_s, deadline_s
        self.phase_start = time.time()
        self.keys, self.fails, self.notes = [], [], []
        self.done = True
        self.data = {}
        self.token = token
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log.write(f"\n{'=' * 78}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] phase {pid} start\n{'=' * 78}\n")

    # --- bookkeeping ---
    def close(self):
        try:
            self._log.close()
        except OSError:
            pass

    def log(self, msg):
        self._log.write(msg + "\n")
        self._log.flush()

    def key(self, line):
        self.keys.append(str(line))
        self.log(f"KEY: {line}")

    def keys_from(self, lines):
        for line in lines:
            self.key(line)

    def fail(self, msg):
        self.fails.append(str(msg))
        self.log(f"FAIL: {msg}")
        self.key(f"FAIL: {msg}")

    def note(self, msg):
        self.notes.append(str(msg))
        self.log(f"NOTE: {msg}")

    def remaining_s(self):
        return min(self.phase_start + self.budget_s, self.deadline) - time.time()

    def _mask(self, text):
        return text.replace(self.token, "***") if self.token else text

    # --- commands ---
    def sh(self, cmd, label=None, cpu=False, timeout_min=None, env=None, cwd=None):
        """Run `cmd` (a list), stream it to the phase log, and return SimpleNamespace(rc, out, seconds,
        timed_out). Never raises for a non-zero exit; raises PhaseTimeout only if the phase budget is
        already spent before the command starts. A command is killed (whole process group) when its own
        timeout or the phase budget ends, whichever is first."""
        label = label or " ".join(str(c) for c in cmd)[:90]
        rem = self.remaining_s()
        if rem <= 5:
            raise PhaseTimeout(f"phase time budget spent before: {label}")
        limit = min(rem, timeout_min * 60) if timeout_min else rem
        full_env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
        if cpu:
            full_env["CUDA_VISIBLE_DEVICES"] = ""
        if env:
            full_env.update(env)
        say(f"  [{self.pid}] {label}" + ("   (CPU only)" if cpu else ""))
        self.log(f"$ {self._mask(' '.join(str(c) for c in cmd))}   [cpu_only={cpu}, limit {limit / 60:.1f} min]")
        t0 = time.time()
        try:
            proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd or ROOT), env=full_env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    errors="replace", start_new_session=True)
        except OSError as exc:
            self.log(f"could not start: {exc}")
            return SimpleNamespace(rc=127, out=str(exc), seconds=0.0, timed_out=False)
        state = {"timed_out": False}

        def kill():
            state["timed_out"] = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

        timer = threading.Timer(limit, kill)
        timer.daemon = True
        timer.start()
        chunks, size, last_beat, last_line = [], 0, time.time(), ""
        try:
            for line in proc.stdout:
                line = self._mask(line.rstrip("\n"))
                self._log.write(line + "\n")
                if line.strip():
                    last_line = line
                if size < 4_000_000:                       # bound memory; the log file has everything
                    chunks.append(line)
                    size += len(line) + 1
                if time.time() - last_beat > 120:
                    say(f"  [{self.pid}] ... {label[:40]} running {(time.time() - t0) / 60:.0f} min | "
                        f"{last_line[:90]}")
                    last_beat = time.time()
            proc.wait()
        finally:
            timer.cancel()
            self._log.flush()
        seconds = time.time() - t0
        out = "\n".join(chunks)
        if state["timed_out"]:
            self.log(f"TIMEOUT after {seconds / 60:.1f} min")
            say(f"  [{self.pid}] TIMEOUT: {label[:60]} after {seconds / 60:.1f} min")
        self.log(f"-> exit {proc.returncode} in {seconds:.1f}s")
        return SimpleNamespace(rc=proc.returncode, out=out, seconds=seconds, timed_out=state["timed_out"])


# --------------------------------------------------------------------------------------------------
# phase registry
# --------------------------------------------------------------------------------------------------

PHASES = {}


def phase(pid, title, tier, budget_min, cpu_only=False, requires=()):
    def deco(fn):
        PHASES[pid] = SimpleNamespace(id=pid, title=title, tier=tier, budget_min=budget_min,
                                      cpu_only=cpu_only, requires=list(requires), fn=fn)
        return fn
    return deco


def record_cpu_env(ctx):
    si = sysinfo()
    ctx.key(f"CPU-only phase: physical cores {si['physical_cores']}, logical {si['logical_cores']}, "
            f"RAM {si['ram_total_gb']} GB total / {si['ram_available_gb']} GB available")


# --- P0 ---------------------------------------------------------------------------------------------

@phase("P0", "Environment check", 1, 5)
def p0_env(ctx):
    si = sysinfo()
    sha, branch = git_head()
    smi = ctx.sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                 "nvidia-smi")
    gpu = smi.out.strip().splitlines()[0] if smi.rc == 0 and smi.out.strip() else None
    torch_probe = ctx.sh([PY, "-c", "import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda, "
                          "'available', torch.cuda.is_available())"], "torch probe")
    packages = {}
    try:
        from importlib import metadata
        for name in ("torch", "transformers", "sentence-transformers", "mteb", "datasets", "numpy", "scipy",
                     "pandas", "fastapi", "uvicorn", "psutil", "peft", "accelerate"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
    except Exception as exc:  # noqa: BLE001
        ctx.note(f"could not read package versions: {exc}")
    info = {"system": si, "gpu": gpu, "torch": torch_probe.out.strip(), "packages": packages,
            "git_commit": sha, "git_branch": branch, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(RESULTS / "environment.json", info)
    ctx.key(f"GPU: {gpu or 'NONE FOUND'}")
    ctx.key(f"{torch_probe.out.strip()}")
    ctx.key(f"CPU cores: physical {si['physical_cores']}, logical {si['logical_cores']}; RAM {si['ram_total_gb']} GB "
            f"({si['ram_available_gb']} GB available); disk free {si.get('disk_free_gb')} GB")
    ctx.key("packages: " + ", ".join(f"{k} {v}" for k, v in packages.items() if v))
    ctx.key(f"git: {branch} @ {sha[:12]}")
    if gpu is None or "available True" not in torch_probe.out:
        ctx.fail("no usable CUDA GPU: P1, P2, P6 need one "
                 "(Settings -> Accelerator -> GPU T4, then restart)")
    if (si.get("ram_total_gb") or 99) < 20:
        ctx.note("under 20 GB RAM: the full-index CPU checks (P4, P5) may run out of memory")


# --- P1 ---------------------------------------------------------------------------------------------

SUSPECTS = (
    "First suspects (both changed in PR #1 / demo-polish):\n"
    "  1. PresetQueryEncoder.encode_docs in src/runtime_index.py -- documents are now encoded at the preset's\n"
    "     full length even when a query cap is set;\n"
    "  2. service_for in src/api.py -- the default index now reuses the startup service.\n"
    "  Neither is imported by run_official.py (tests/test_official_path_guard.py asserts that), so if the score\n"
    "  is off also compare library versions in P0 (transformers / sentence-transformers / mteb), check\n"
    "  _run_info.fell_back_to_fp32 in outputs/appsretrieval_results.json, and diff against the commit that\n"
    "  produced 0.9376.")


def check_official_scores(results_path):
    """(ndcg, mrr, run_info, problems). problems is empty when both are within TOLERANCE of the targets."""
    d = read_json(results_path)
    if not d:
        return None, None, {}, [f"{results_path} is missing or unreadable"]
    try:
        s = d["scores"]["test"][0]
        ndcg, mrr = float(s["ndcg_at_10"]), float(s["mrr_at_10"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None, None, d.get("_run_info", {}), ["no scores.test[0].ndcg_at_10 / mrr_at_10 in the results file"]
    problems = []
    if abs(ndcg - TARGET_NDCG) > TOLERANCE:
        problems.append(f"NDCG@10 {ndcg:.4f} is more than {TOLERANCE} from {TARGET_NDCG}")
    if abs(mrr - TARGET_MRR) > TOLERANCE:
        problems.append(f"MRR@10 {mrr:.4f} is more than {TOLERANCE} from {TARGET_MRR}")
    return ndcg, mrr, d.get("_run_info", {}), problems


@phase("P1", "Official 1.7B run, rankings, runtime index, verification", 1, 100)
def p1_official(ctx):
    cmd = [PY, "src/eval/run_official.py", "--config", OFFICIAL_CONFIG, "--device", "cuda", "--confirm-test"]
    if LOCK.exists():
        cmd.append("--allow-rerun")
        ctx.note("a previous official-run lock exists, so --allow-rerun was added")
    run = ctx.sh(cmd, "official run (1.7B, T4)")
    if run.rc != 0:
        ctx.fail(f"run_official.py exited {run.rc}" + (" (timed out)" if run.timed_out else ""))
    results = ROOT / "outputs" / "appsretrieval_results.json"
    ndcg, mrr, info, problems = check_official_scores(results)
    if ndcg is not None:
        ctx.key(f"NDCG@10 = {ndcg:.4f}   target {TARGET_NDCG} +/- {TOLERANCE}   "
                f"{'PASS' if abs(ndcg - TARGET_NDCG) <= TOLERANCE else 'FAIL'}")
        ctx.key(f"MRR@10  = {mrr:.4f}   target {TARGET_MRR} +/- {TOLERANCE}   "
                f"{'PASS' if abs(mrr - TARGET_MRR) <= TOLERANCE else 'FAIL'}")
        s = read_json(results)["scores"]["test"][0]
        extra = "  ".join(f"{k}={s[k]:.5f}" for k in ("recall_at_10", "recall_at_100") if k in s)
        if extra:
            ctx.key(extra)
        ctx.key(f"total_eval_seconds={info.get('total_eval_seconds')}  stage_timings_s={info.get('stage_timings_s')}  "
                f"fell_back_to_fp32={info.get('fell_back_to_fp32')}")
    for p in problems:
        ctx.fail(p)
    if problems:
        for line in SUSPECTS.splitlines():
            ctx.key(line)
            print(line, flush=True)
    ctx.keys_from(grep_lines(run.out, [r"OK: within the dev-vs-test gap", r"LOUD WARNING", r"^WARNING"], 4))

    # verification, then the hash the release will quote
    ver = ctx.sh([PY, "src/verify_submission.py", "--confirm-test"], "verify_submission", cpu=True, timeout_min=15)
    passed = ver.rc == 0 and "PASSED" in ver.out
    ctx.key(f"verify_submission: {'PASSED' if passed else 'FAILED'}")
    if not passed:
        ctx.fail("verify_submission.py did not report PASSED")
        ctx.keys_from(grep_lines(ver.out, [r"^\s*- ", r"FAILED"], 6))
    if results.exists():
        ctx.key(f"sha256(appsretrieval_results.json) = {sha256_file(results)}")
    rankings = ROOT / "outputs" / "appsretrieval_rankings.json"
    if rankings.exists():
        ctx.key(f"sha256(appsretrieval_rankings.json) = {sha256_file(rankings)}  ({rankings.stat().st_size / 1e6:.1f} MB)")
    man = read_json(ROOT / "runtime_index" / "manifest.json")
    if man:
        mb = sum(f.get("bytes", 0) for f in man.get("files", {}).values()) / 1e6
        ctx.key(f"runtime_index (full): {man.get('n_docs')} x {man.get('dim')}, {mb:.1f} MB, model {man.get('model')}")
    else:
        ctx.fail("runtime_index/manifest.json was not exported; P2 will rebuild the full index itself")


# --- P2 ---------------------------------------------------------------------------------------------

VERIFY_SNIPPET = r'''
from src.runtime_index import INDEX_NAMES, RuntimeIndex
for name in ("full", "lite"):
    try:
        ix = RuntimeIndex.load(INDEX_NAMES[name])
    except Exception as e:
        print(f"{name:5s} FAIL: {type(e).__name__}: {str(e).splitlines()[0]}"); continue
    bad = ix.verify_files()
    m = ix.manifest
    print(f"{name:5s} rows={m['n_docs']:6d} dim={m['dim']:5d} kind={m.get('kind')} model={m.get('model')} "
          f"hashes={'OK' if not bad else bad}")
'''


@phase("P2", "Lite (0.6B) index, then hash-verify both indexes", 1, 45)
def p2_indexes(ctx):
    if not (ROOT / "runtime_index" / "manifest.json").exists():
        ctx.note("runtime_index/ is missing (P1 did not export it): building the full index here instead")
        r = ctx.sh([PY, "src/build_index.py", "--preset", "f2llm-v2-1.7b", "--device", "cuda", "--out", "full"],
                   "build full index (fallback)", timeout_min=30)
        if r.rc != 0:
            ctx.fail("building the full index failed")
    r = ctx.sh([PY, "src/build_index.py", "--preset", "f2llm-v2-0.6b", "--device", "cuda", "--out", "lite"],
               "build lite index (0.6B)", timeout_min=25)
    if r.rc != 0:
        ctx.fail(f"build_index.py (lite) exited {r.rc}")
    ctx.keys_from(grep_lines(r.out, [r"\[build\] encoded", r"\[build\] wrote"], 3))
    v = ctx.sh([PY, "-c", VERIFY_SNIPPET], "verify index hashes", cpu=True, timeout_min=10)
    lines = [ln for ln in v.out.splitlines() if re.match(r"(full|lite)\s", ln)]
    ctx.keys_from(lines)
    for name in ("full", "lite"):
        line = next((ln for ln in lines if ln.startswith(name)), "")
        if "hashes=OK" not in line or "rows=  8765" not in line:
            ctx.fail(f"{name} index did not verify with 8765 rows and matching hashes: {line or 'no output'}")


# --- P3 ---------------------------------------------------------------------------------------------

@phase("P3", "Failure analysis and metrics report", 1, 15, cpu_only=True,
       requires=["outputs/appsretrieval_rankings.json"])
def p3_analysis(ctx):
    record_cpu_env(ctx)
    r = ctx.sh([PY, "src/analyze_failures.py", "--confirm-test"], "analyze_failures", cpu=True, timeout_min=10)
    if r.rc != 0:
        ctx.fail(f"analyze_failures.py exited {r.rc}")
    ctx.keys_from(grep_lines(r.out, [r"^\[fail\]"], 10))
    m = ctx.sh([PY, "src/build_metrics_report.py"], "build_metrics_report", cpu=True, timeout_min=5)
    if m.rc != 0:
        ctx.fail(f"build_metrics_report.py exited {m.rc}")
    text = (RESULTS / "metrics.md").read_text(encoding="utf-8") if (RESULTS / "metrics.md").exists() else ""
    ctx.keys_from(grep_lines(text, [r"^- measured:", r"^- not yet measured:"], 2))


# --- P4 ---------------------------------------------------------------------------------------------

def latency_summary(name, path):
    j = read_json(path)
    if not j:
        return f"{name}: no result file"
    sweep = ", ".join(f"{r['threads']}t p50 {r['latency_ms']['p50']:.0f} ms" for r in j.get("thread_sweep") or [])
    tok = j.get("query_tokens") or {}
    return (f"{name}: p50 {j['latency_ms']['p50']:.0f} / p95 {j['latency_ms']['p95']:.0f} ms "
            f"(encode p50 {j['query_encode_ms']['p50']:.0f}, search p50 {j['search_ms']['p50']:.2f}) | "
            f"cap {j.get('max_query_tokens')} | query tokens p50 {tok.get('p50')} | peak RSS {j.get('peak_rss_mb')} MB | "
            f"threads {j.get('threads')} of {j.get('physical_cores')} physical"
            + (f" | sweep: {sweep}" if sweep else ""))


def precision_summary(path):
    j = read_json(path)
    if not j:
        return []
    out = []
    for m in j.get("modes", []):
        out.append(f"{m['mode']}: {m['encode_ms_p50']:.0f} ms/query, RSS {m['peak_rss_mb']} MB, "
                   f"cos {m['cosine_to_fp32_mean']:.6f}, top-10 overlap {m.get('top10_overlap_mean')}, "
                   f"rank-1 changed {m.get('rank1_changed')}, identical lists "
                   f"{m.get('top10_identical_and_in_order', 'n/a')}/{j['n_queries']}, int8 applied "
                   f"{m['applied']['int8_applied']}")
    modes = {m["mode"]: m for m in j.get("modes", [])}
    if "int8" in modes and "fp32" in modes:
        i8, f32 = modes["int8"], modes["fp32"]
        bug = (i8["cosine_to_fp32_mean"] >= 0.9999995 and i8.get("top10_identical_and_in_order") == j["n_queries"]
               and (i8["peak_rss_mb"] or 0) > 1.3 * (f32["peak_rss_mb"] or 1))
        out.append("int8 verdict: " + ("BUG STILL PRESENT (cosine exactly 1, identical lists, RSS well above fp32)"
                                        if bug else "quantisation took effect or failed loudly (not the old bug)"))
    return out


@phase("P4", "CPU checks: int8 vs fp32, capped vs uncapped latency, thread sweep", 1, 75, cpu_only=True,
       requires=["runtime_index_lite/manifest.json"])
def p4_cpu(ctx):
    record_cpu_env(ctx)
    for idx in ("full", "lite"):
        if idx == "full" and (not (ROOT / "runtime_index" / "manifest.json").exists()
                              or ram_available_gb() < 13):
            ctx.fail("full-index CPU checks skipped: no runtime_index/ or under 13 GB RAM available")
            continue
        out = RESULTS / ("cpu_precision.json" if idx == "full" else "cpu_precision_lite.json")
        r = ctx.sh([PY, "src/check_cpu_precision.py", "--index", idx, "--modes", "fp32", "int8",
                    "--n-queries", "20", "--out", str(out)], f"precision {idx}", cpu=True, timeout_min=25)
        ctx.keys_from(grep_lines(r.out, [r"int8 census"], 1))
        if r.rc != 0 and "did not take" in r.out:
            ctx.key(f"{idx}: int8 quantisation failed LOUDLY (a correct outcome): "
                    + (grep_lines(r.out, [r"did not take"], 1) or [""])[0][:200])
        elif r.rc != 0:
            ctx.fail(f"check_cpu_precision.py ({idx}) exited {r.rc}")
        else:
            ctx.keys_from([f"{idx} " + s for s in precision_summary(out)])
        for arm, extra in (("capped", []), ("uncapped", ["--max-query-tokens", "0"])):
            path = RESULTS / f"cpu_{idx}_{arm}.json"
            b = ctx.sh([PY, "src/bench_cpu.py", "--index", idx, "--thread-sweep", "--n-queries", "20",
                        *extra, "--out", str(path)], f"latency {idx} {arm}", cpu=True, timeout_min=25)
            if b.rc != 0:
                ctx.fail(f"bench_cpu.py ({idx} {arm}) exited {b.rc}")
            else:
                ctx.key(latency_summary(f"{idx} {arm}", path))


# --- shared: click history builds ------------------------------------------------------------------

def ensure_history(ctx, n, name="history", extra=(), timeout_min=30):
    """Build the click history index for `n` commits unless one with that many commits already exists."""
    d = ROOT / ("history_index" if name == "history" else name)
    man = read_json(d / "manifest.json")
    if man and man.get("commits") == n and (d / "versions.json").exists():
        ctx.note(f"reusing {d.name} ({n} commits)")
        return d
    cmd = [PY, "src/build_history_index.py", "--device", "cuda", "--max-commits", str(n), "--out", str(d)]
    if name != "history":
        cmd += ["--queries-out", str(RESULTS / f"history_queries_{n}.json"),
                "--stats-out", str(RESULTS / f"history_ingest_{n}.json")]
    r = ctx.sh(cmd + list(extra), f"build history index ({n} commits)", timeout_min=timeout_min)
    ctx.keys_from(grep_lines(r.out, [r"\[history\] \d+ commits", r"\[history\] embedded"], 2))
    if r.rc != 0 or not (d / "manifest.json").exists():
        ctx.fail(f"build_history_index.py ({n} commits) failed (exit {r.rc})")
        return None
    return d


# --- P5 ---------------------------------------------------------------------------------------------

def http(method, url, body=None, timeout=600):
    import urllib.error
    import urllib.request
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except ValueError:
            return exc.code, {}


class Checks:
    """PASS/FAIL per HTTP check; a check that raises is a FAIL, never a crash."""

    def __init__(self, ctx):
        self.ctx, self.n_pass, self.n_fail = ctx, 0, 0

    def run(self, name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        self.n_pass += int(bool(ok))
        self.n_fail += int(not ok)
        line = f"{'PASS' if ok else 'FAIL'}  {name} -- {detail}"
        self.ctx.log(line)
        self.ctx.keys.append(line)
        say("    " + line[:150])
        if not ok:
            self.ctx.fails.append(f"{name}: {detail}")
        return ok


def snapshot_index(idx_dir):
    import numpy as np
    texts = json.loads((Path(idx_dir) / "corpus_texts.json").read_text(encoding="utf-8"))
    emb = np.load(Path(idx_dir) / "embeddings.npy")
    return {t: emb[i].copy() for i, t in enumerate(texts)}


@phase("P5", "API smoke test over HTTP", 2, 70, requires=["runtime_index_lite/manifest.json"])
def p5_api(ctx):
    import numpy as np
    work = ROOT / "work"
    src_copy, demo_idx = work / "textkit", work / "demo_idx"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(ROOT / "examples" / "textkit", src_copy)
    b = ctx.sh([PY, "src/build_index.py", "--source", str(src_copy), "--preset", "f2llm-v2-0.6b", "--device",
                "cuda", "--out", str(demo_idx)], "build folder index (textkit copy)", timeout_min=15)
    if b.rc != 0:
        ctx.fail("could not build the folder index; folder-index checks will fail")
    ctx.sh([PY, "src/build_categories.py", "--index", str(demo_idx), "--clusters", "8"], "tag categories",
           cpu=True, timeout_min=5)
    hist = ensure_history(ctx, 40)

    port = 8765
    allowed = ",".join(["full", "lite", "history", str(demo_idx)])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", SEARCH_DEVICE="cpu", INDEX_DIR="lite", ALLOWED_INDEXES=allowed,
               PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    server_log = open(LOGS / "P5_server.log", "w", encoding="utf-8")
    say(f"  [P5] starting the API on 127.0.0.1:{port} (CPU only, default index lite)")
    server = subprocess.Popen([PY, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", str(port)],
                              cwd=str(ROOT), env=env, stdout=server_log, stderr=subprocess.STDOUT,
                              start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        ready = False
        for _ in range(120):
            if server.poll() is not None:
                break
            try:
                st, h = http("GET", base + "/health", timeout=5)
                if st == 200 and h.get("loaded"):
                    ready = True
                    break
            except Exception:  # noqa: BLE001  not up yet
                pass
            time.sleep(5)
        if not ready:
            ctx.fail("the API did not become healthy within 10 minutes (see results/logs/P5_server.log)")
            return
        checks = Checks(ctx)
        q = "count the number of primes below n"
        di = str(demo_idx)
        from urllib.parse import quote as enc

        checks.run("GET /health (default index)", lambda: (
            lambda h: (h.get("loaded") is True and h["index"]["n_docs"] == 8765,
                       f"n_docs {h['index']['n_docs']}, model {h['index']['model']}, device {h['index']['device']}")
        )(http("GET", base + "/health")[1]))
        if hist:
            checks.run("GET /health?index=history (unloaded: its own manifest)", lambda: (
                lambda h: (h.get("loaded") is False and h.get("available") is True and h["facts"]["n_docs"] != 8765,
                           f"loaded {h.get('loaded')}, n_docs {h.get('facts', {}).get('n_docs')}")
            )(http("GET", base + "/health?index=history")[1]))

        def search_plain():
            st, r = http("POST", base + "/search", {"query": q, "k": 5, "index": "lite"})
            return (st == 200 and len(r["hits"]) == 5 and "performance" in r,
                    f"top1 {r['hits'][0]['doc_id']}, similarity {r['hits'][0]['score']:.4f}, encode "
                    f"{r['timings_ms']['encode_query_ms']:.0f} ms, search {r['timings_ms']['search_ms']:.1f} ms")
        checks.run("POST /search (lite)", search_plain)

        def search_explain():
            st, r = http("POST", base + "/search", {"query": q + " below a limit", "k": 3, "explain": True,
                                                    "index": "lite"})
            w = r["hits"][0].get("why", {})
            return (st == 200 and "matched_terms" in w and "route" in w and "score_gap_to_next" in w,
                    f"gap {w.get('score_gap_to_next')}, matched {[m['term'] for m in w.get('matched_terms', [])]}")
        checks.run("POST /search explain=true", search_explain)

        def cache():
            probe = f"binary search over a sorted list of integers {int(time.time())}"
            t0 = time.time()
            _, a = http("POST", base + "/search", {"query": probe, "k": 3, "index": "lite"})
            w1 = 1000 * (time.time() - t0)
            t0 = time.time()
            _, c = http("POST", base + "/search", {"query": probe, "k": 3, "index": "lite"})
            w2 = 1000 * (time.time() - t0)
            e1, e2 = a["timings_ms"]["encode_query_ms"], c["timings_ms"]["encode_query_ms"]
            ok = (a["timings_ms"]["cached"] is False and c["timings_ms"]["cached"] is True and e2 < e1
                  and [h["doc_id"] for h in a["hits"]] == [h["doc_id"] for h in c["hits"]])
            return ok, (f"miss cached={a['timings_ms']['cached']} encode {e1:.1f} ms (wall {w1:.0f} ms); "
                        f"hit cached={c['timings_ms']['cached']} encode {e2:.2f} ms (wall {w2:.0f} ms); same top-3")
        checks.run("exact-query cache: miss then hit", cache)

        def multiline():
            text = "Given n and k count the pairs.\n\nInput\n5 3\n1 2 3 4 5\n\nOutput\n4"
            st, r = http("POST", base + "/search", {"query": text, "k": 3, "index": "lite"})
            return st == 200 and r["query"] == text, f"newlines preserved: {r.get('query') == text}, route " \
                                                    f"{r.get('route', {}).get('kind')}"
        checks.run("multiline query round-trips", multiline)

        checks.run("GET /route", lambda: (lambda r: (r.get("route") in ("structural", "dense", "dense_code"),
                                                     f"{r.get('kind')} -> {r.get('route')}"))(
            http("GET", base + "/route?q=" + enc("who calls tokenize"))[1]))

        st_ok = [False]

        def structural():
            st, r = http("GET", base + f"/structural?intent=who_calls&subject=tokenize&index={enc(di)}")
            st_ok[0] = st == 200 and len(r.get("results", [])) > 0
            return st_ok[0], f"{len(r.get('results', []))} exact call site(s)"
        checks.run("GET /structural who_calls tokenize", structural)

        checks.run("GET /categories (tagged folder index)", lambda: (lambda r: (
            len(r.get("families", {})) > 0, f"families {list(r.get('families', {}))[:5]}"))(
            http("GET", base + "/categories?index=" + enc(di))[1]))

        def agent():
            st, r = http("POST", base + "/agent", {"question": "who calls tokenize", "index": di})
            return (st == 200 and r.get("steps_run", 0) >= 1 and len(r.get("answers", [])) > 0,
                    f"{r.get('steps_run')} step(s), {len(r.get('answers', []))} answer(s), stopped: "
                    f"{r.get('stop_reason')}")
        checks.run("POST /agent", agent)

        def doc():
            _, r = http("POST", base + "/search", {"query": "split text into sentences", "k": 1, "index": di})
            did = r["hits"][0]["doc_id"]
            st, d = http("GET", base + f"/doc?id={enc(did)}&index={enc(di)}")
            return (st == 200 and d.get("has_source_file") is True and d["n_lines"] >= 1 and d.get("text"),
                    f"{d.get('location')}, {d.get('n_lines')} lines, first line {d.get('first_line')}")
        checks.run("GET /doc (folder index: file:lines)", doc)

        def doc_apps():
            _, r = http("POST", base + "/search", {"query": q, "k": 1, "index": "lite"})
            st, d = http("GET", base + f"/doc?id={enc(r['hits'][0]['doc_id'])}&index=lite")
            return st == 200 and d.get("has_source_file") is False and "file" not in d, \
                f"APPS doc {d.get('doc_id')}: no invented path"
        checks.run("GET /doc (APPS: id only, no file path)", doc_apps)

        if hist:
            hq = "run a command and invoke its callback"

            def versions():
                _, r = http("GET", base + "/versions?index=history")
                vs = r.get("versions", [])
                checks.versions = vs
                return r.get("versioned") is True and len(vs) >= 2, f"{len(vs)} versions, {vs[:1]}..{vs[-1:]}"
            checks.run("GET /versions (history)", versions)
            vs = getattr(checks, "versions", [1, 2])

            def compare():
                st, r = http("GET", base + f"/compare?q={enc(hq)}&a={vs[0]}&b={vs[-1]}&k=8&index=history")
                s = r.get("summary", {})
                return st == 200 and len(r.get("b", [])) > 0, f"v{vs[0]} vs v{vs[-1]}: {s}"
            checks.run("GET /compare", compare)

            def diff_and_history():
                _, r = http("POST", base + "/search", {"query": hq, "k": 12, "all_versions": True,
                                                       "index": "history", "explain": True})
                groups = r.get("groups", [])

                # Pick a lineage with at least two distinct versions: check /compare content_changed first,
                # then search groups.
                candidates = []
                st_c, cmp_res = http("GET", base + f"/compare?q={enc(hq)}&a={vs[0]}&b={vs[-1]}&k=12&index=history")
                if st_c == 200:
                    for hb in cmp_res.get("b", []):
                        if hb.get("content_changed") and hb.get("snippet_id"):
                            candidates.append(hb["snippet_id"])

                for g in sorted(groups, key=lambda x: len(x.get("others", [])), reverse=True):
                    sid = g.get("snippet_id")
                    if sid and sid not in candidates:
                        candidates.append(sid)

                chosen_sid = None
                chosen_pair = None
                chosen_h = None

                for sid in candidates:
                    st, h = http("GET", base + f"/history/{enc(sid, safe='')}?index=history")
                    if st != 200:
                        continue
                    hashes = {}
                    for v in h.get("versions", []):
                        hashes.setdefault(v["content_hash"], v["version"])
                    if len(hashes) >= 2:
                        chosen_sid = sid
                        chosen_pair = sorted(hashes.values())[:2]
                        chosen_h = h
                        break

                if not chosen_sid or not chosen_pair:
                    return False, "no lineage with at least two distinct versions could be found in history; try another query"

                st2, d = http("GET", base + f"/diff?snippet_id={enc(chosen_sid)}&a={chosen_pair[0]}&b={chosen_pair[1]}&index=history")
                return (st2 == 200 and d.get("identical") is False and len(groups) > 0,
                        f"{len(groups)} lineage groups; history of {chosen_sid}: {chosen_h.get('n_versions')} versions; diff "
                        f"v{chosen_pair[0]}->v{chosen_pair[1]} +{d.get('added_lines')} -{d.get('removed_lines')}")
            checks.run("groups + GET /history/<id> + GET /diff", diff_and_history)

            def history_doc():
                _, r = http("POST", base + "/search", {"query": hq, "k": 1, "index": "history"})
                did = r["hits"][0]["doc_id"]
                st, d = http("GET", base + f"/doc?id={enc(did)}&index=history")
                return st == 200 and d.get("location") and d.get("commit_short"), \
                    f"{d.get('location')} @ {d.get('commit_short')}"
            checks.run("GET /doc (history: file:lines + commit)", history_doc)
        else:
            ctx.fail("history checks skipped: the history index could not be built")

        # --- reindex on the folder index --------------------------------------------------------------
        def reindex_flow():
            before = snapshot_index(demo_idx)
            tok = src_copy / "tokenizing.py"
            tok.write_text(tok.read_text(encoding="utf-8")
                           + '\n\ndef shout(text):\n    """Upper-case the text and end it with an exclamation mark."""\n'
                             '    return text.upper() + "!"\n', encoding="utf-8")
            _, dry = http("POST", base + "/reindex", {"index": di, "dry_run": True})
            n_before = dry.get("n_docs_before")
            _, real = http("POST", base + "/reindex", {"index": di})
            _, s = http("POST", base + "/search", {"query": "make text upper case and add an exclamation mark",
                                                   "k": 3, "index": di})
            _, again = http("POST", base + "/reindex", {"index": di})
            after = snapshot_index(demo_idx)
            kept = [t for t in before if t in after]
            identical = sum(bool(np.array_equal(before[t], after[t])) for t in kept)
            ok = (dry.get("added") == 1 and dry.get("written") is False and dry.get("embeddings_recomputed") == 1
                  and real.get("written") is True and real.get("added") == 1
                  and real.get("embeddings_recomputed") == 1 and real.get("embeddings_reused") == n_before
                  and any("shout" in h["preview"] for h in s["hits"][:1])
                  and again.get("written") is False and again.get("embeddings_recomputed") == 0
                  and identical == len(kept) == n_before)
            return ok, (f"dry run +{dry.get('added')} (would recompute {dry.get('embeddings_recomputed')}); real: "
                        f"+{real.get('added')} ~{real.get('modified')} ={real.get('unchanged')} -{real.get('removed')}, "
                        f"reused {real.get('embeddings_reused')} recomputed {real.get('embeddings_recomputed')}, "
                        f"{real.get('elapsed_seconds')} s (encode {real.get('encode_seconds')} s); top hit has "
                        f"'shout': {'shout' in s['hits'][0]['preview']}; second run written={again.get('written')} "
                        f"in {again.get('elapsed_seconds')} s; unchanged vectors byte-identical {identical}/{len(kept)}")
        checks.run("POST /reindex: edit, dry run, real run, next search, no-op run, byte-identical vectors",
                   reindex_flow)
        checks.run("POST /reindex refuses the official index", lambda: (
            lambda r: (r[0] == 400, f"HTTP {r[0]}: {str(r[1].get('detail'))[:80]}"))(
            http("POST", base + "/reindex", {"index": "full"})))

        # --- the full index last, only if memory allows ------------------------------------------------
        avail = ram_available_gb()
        if (ROOT / "runtime_index" / "manifest.json").exists() and avail >= 14:
            def full_search():
                t0 = time.time()
                st, r = http("POST", base + "/search", {"query": q, "k": 3, "index": "full"}, timeout=1200)
                _, h = http("GET", base + "/health?index=full")
                return (st == 200 and len(r["hits"]) == 3 and h.get("loaded") is True,
                        f"first search incl. model load {time.time() - t0:.0f} s; top1 {r['hits'][0]['doc_id']}; "
                        f"model {h['index']['model']}")
            checks.run("POST /search index=full (1.7B on CPU)", full_search)
        else:
            ctx.key(f"SKIPPED  full-index HTTP check -- {avail:.0f} GB RAM available (needs 14 GB) "
                    f"or no runtime_index/")
        try:
            import psutil
            rss = psutil.Process(server.pid).memory_info().rss / 1e9
            ctx.key(f"API process RSS after all checks: {rss:.1f} GB")
        except Exception:  # noqa: BLE001
            pass
        ctx.key(f"HTTP checks: {checks.n_pass} passed, {checks.n_fail} failed")
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except OSError:
                pass
        server_log.close()


# --- P6 ---------------------------------------------------------------------------------------------

def history_summary(path):
    j = read_json(path)
    if not j:
        return [f"{Path(path).name}: not written"]
    ing, t = j.get("ingest", {}), j.get("totals", {})
    lines = [f"{ing.get('commits')} commits, {ing.get('rows')} rows, {ing.get('lineages')} lineages, "
             f"{ing.get('distinct_hashes')} distinct contents",
             f"P1 rebuild: full {t.get('embeddings_full')} embeddings ({t.get('seconds_full')} s) -> incremental "
             f"{t.get('embeddings_incremental')} ({t.get('seconds_incremental')} s); saved {t.get('saved_pct')}%, "
             f"x{t.get('speedup_x')}; warnings {len(j.get('warnings', []))}"]
    r = j.get("retrieval")
    if r:
        p1, bonus = r["p1_version_targeted"], r["bonus_all_versions"]
        k = j.get("k", 10)
        lines.append(f"P1 targeting: correct lineage at rank 1 = {p1['top1_correct_lineage']} "
                     f"({p1['median_latency_ms']} ms median), n={r['n_queries']}")
        lines.append(f"Bonus: top-{k} duplicate slots {bonus.get(f'duplicate_pct_at_{k}_all_versions')}% -> "
                     f"{bonus.get(f'duplicate_pct_at_{k}_collapsed')}% collapsed; lineage recall@{k} "
                     f"{bonus.get(f'lineage_recall_at_{k}_all_versions')} -> "
                     f"{bonus.get(f'lineage_recall_at_{k}_collapsed')}; top-1 exact version "
                     f"{bonus.get('top1_exact_version')}, token-consistent {bonus.get('top1_token_consistent_version')}")
    d = j.get("delta_vectors")
    if d:
        lines.append(f"Delta vectors (lineage_k {d['lineage_k']}): index {d['index_bytes_delta']:,} B vs "
                     f"{d['index_bytes_current']:,} B (x{d['size_ratio_delta_over_current']}); top-1 exact version "
                     f"{d['top1_exact_version']['delta']} vs {d['top1_exact_version']['current']} current; lineage "
                     f"{d['top1_lineage']['delta']} vs {d['top1_lineage']['current']}; agree "
                     f"{d['top1_agreement_delta_vs_current']}")
    return lines


@phase("P6", "Real history (click): P1/Bonus, delta vectors, rename tracking, agent vs dense", 2, 130,
       requires=["runtime_index_lite/manifest.json"])
def p6_history(ctx):
    hist = ensure_history(ctx, 40)
    if hist:
        for label, extra, out in (("delta vectors", ["--delta-vectors"], "real_history_benchmark.json"),
                                  ("delta vectors, lineage_k=10", ["--delta-vectors", "--delta-lineage-k", "10"],
                                   "delta_k10.json")):
            path = RESULTS / out
            r = ctx.sh([PY, "src/bench_real_history.py", "--device", "cuda", "--max-commits", "40", "--index",
                        "history", *extra, "--out", str(path)], f"bench_real_history 40 ({label})", timeout_min=30)
            if r.rc != 0:
                ctx.fail(f"bench_real_history.py 40 commits ({label}) exited {r.rc} "
                         f"({len(read_json(path, {}).get('warnings', []))} warnings)")
            ctx.keys_from([f"[40 commits, {label}] " + s for s in history_summary(path)])
    else:
        ctx.fail("no 40-commit history index: the benchmarks and the agent evaluation cannot run")

    # rename tracking: an ingest-level measurement, so the (unused) index is built with the mock encoder
    rn = ctx.sh([PY, "src/build_history_index.py", "--device", "cuda", "--max-commits", "40", "--mock-encoder",
                 "--track-renames", "--out", "work/history_renames_ingest_only",
                 "--queries-out", str(RESULTS / "history_queries_renames.json"),
                 "--stats-out", str(RESULTS / "history_ingest_renames.json")],
                "rename-tracking ingest (40 commits, ingest only)", timeout_min=10)
    tr = (read_json(RESULTS / "history_ingest_renames.json", {}).get("stats") or {})
    plain = (read_json(RESULTS / "history_ingest.json", {}).get("stats") or {})
    if rn.rc == 0 and tr.get("track_renames"):
        t = tr["track_renames"]
        ctx.key(f"track-renames (40 commits): {t['links']} link(s) {t['by_kind']}; lineages {plain.get('lineages')} "
                f"(off) -> {tr.get('lineages')} (on); threshold {t['threshold']}")
    else:
        ctx.fail(f"--track-renames ingest failed (exit {rn.rc})")

    # agent vs dense
    if hist and (ROOT / "data" / "repos" / "click" / "src" / "click").exists():
        a = ctx.sh([PY, "src/bench_agent.py", "--index", "history", "--source", "data/repos/click/src/click",
                    "--device", "cuda", "--max-questions", "30"], "agent vs dense", timeout_min=25)
        if a.rc != 0:
            ctx.fail(f"bench_agent.py exited {a.rc}")
        ctx.keys_from(extract_block(a.out, r"AGENT vs DENSE", r"agent median steps", 24))
    else:
        ctx.fail("agent evaluation skipped: needs the history index and data/repos/click")

    # the 150-commit run, only if there is time left
    if ctx.remaining_s() > 45 * 60:
        h150 = ensure_history(ctx, 150, "history_index_150", timeout_min=40)
        if h150:
            path = RESULTS / "real_history_benchmark_150.json"
            r = ctx.sh([PY, "src/bench_real_history.py", "--device", "cuda", "--max-commits", "150", "--index",
                        str(h150), "--out", str(path)], "bench_real_history 150", timeout_min=40)
            if r.rc != 0:
                ctx.fail(f"bench_real_history.py 150 commits exited {r.rc}")
            ctx.keys_from(["[150 commits] " + s for s in history_summary(path)])
    else:
        ctx.key("SKIPPED  150-commit run: under 45 minutes left in this phase's budget")

    # synthetic P1/Bonus: fills two more rows of the metrics report; small and cheap
    if ctx.remaining_s() > 12 * 60:
        s1 = ctx.sh([PY, "src/build_version_index.py", "--preset", "f2llm-v2-0.6b", "--device", "cuda",
                     "--n-snippets", "300"], "synthetic version index", timeout_min=10)
        b1 = ctx.sh([PY, "src/bench_versions.py", "--preset", "f2llm-v2-0.6b", "--device", "cuda",
                     "--n-snippets", "300"], "bench_versions (synthetic P1)", timeout_min=10)
        b2 = ctx.sh([PY, "src/bench_evolution.py", "--device", "cpu"], "bench_evolution (synthetic Bonus)",
                    cpu=True, timeout_min=10)
        for name, r in (("build_version_index", s1), ("bench_versions", b1), ("bench_evolution", b2)):
            if r.rc != 0:
                ctx.fail(f"{name}.py exited {r.rc}")
        ctx.keys_from(grep_lines(b1.out, [r"P1 |saved|speedup|reuse"], 3))


# --- C: collect --------------------------------------------------------------------------------------

RELEASE_CANDIDATES = [
    ("outputs/appsretrieval_results.json", "the official MTEB result"),
    ("outputs/appsretrieval_rankings.json", "top-100 rankings for every test query"),
    ("outputs/submission_checksums.json", "checksums written by the official run"),
    ("results/metrics.md", "collected metrics report"),
    ("results/failure_analysis.json", "failure analysis"),
    ("results/failure_examples.md", "worked failure examples"),
]


def zip_dir(src, dest, exclude_suffixes=()):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(src).rglob("*")):
            if p.is_file() and p.suffix not in exclude_suffixes:
                z.write(p, p.relative_to(Path(src).parent))
    return dest


@phase("C", "Collect release files", 0, 25)
def pc_collect(ctx):
    record_cpu_env(ctx)
    ctx.sh([PY, "src/build_metrics_report.py"], "refresh metrics report", cpu=True, timeout_min=5)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    listing = []

    def add(path, why):
        path = Path(path)
        if path.exists():
            listing.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "what": why})

    for rel, why in RELEASE_CANDIDATES:
        add(ROOT / rel, why)
    for name, d in (("runtime_index.zip", ROOT / "runtime_index"), ("runtime_index_lite.zip", ROOT / "runtime_index_lite")):
        if (d / "manifest.json").exists():
            z = RELEASE_DIR / name
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(d.iterdir()):
                    if f.is_file():
                        zf.write(f, f"{d.name}/{f.name}")
            add(z, f"served index ({d.name})")
        else:
            ctx.fail(f"{d.name}/ is missing, so {name} was not created")
    for name, d in (("history_index.zip", ROOT / "history_index"),):
        if (d / "manifest.json").exists():
            z = RELEASE_DIR / name
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(d.iterdir()):
                    if f.is_file():
                        zf.write(f, f"{d.name}/{f.name}")
            add(z, "click history index (demo)")
    zres = zip_dir(RESULTS, RELEASE_DIR / "results_final.zip")
    add(zres, "all results/ and logs")
    write_json(RELEASE_JSON, listing)
    for item in listing:
        ctx.key(f"{item['path']} | {item['bytes'] / 1e6:.2f} MB | sha256 {item['sha256']}")


# --------------------------------------------------------------------------------------------------
# state, zip, summary
# --------------------------------------------------------------------------------------------------

def load_state(fresh=False):
    """The saved state, unless --fresh. Phases that are re-run overwrite their own entry; the rest stay."""
    st = None if fresh else read_json(STATE)
    if not st or st.get("version") != 1:
        st = {"version": 1, "phases": {}, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    return st


def save_state(state):
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(STATE, state)


def zip_partial():
    try:
        dest = PARTIAL_ZIP
        tmp = dest.with_name(dest.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(RESULTS.rglob("*")):
                if p.is_file() and p.suffix != ".tmp":
                    z.write(p, p.relative_to(RESULTS.parent))
            for rel in ("outputs/appsretrieval_results.json", "outputs/submission_checksums.json"):
                if (ROOT / rel).exists():
                    z.write(ROOT / rel, rel)
        os.replace(tmp, dest)
    except Exception as exc:  # noqa: BLE001  a backup must never stop the run
        say(f"could not write {PARTIAL_ZIP}: {exc}")


def _rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def fmt_seconds(s):
    if s is None:
        return "-"
    return f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m" if s >= 3600 else f"{int(s // 60)}m{int(s % 60):02d}s"


def write_summary(state, args=None):
    sha, branch = git_head()
    si = sysinfo()
    order = list(PHASES)
    rows, sections = [], []
    total = 0.0
    for pid in order:
        e = state["phases"].get(pid)
        if not e:
            continue
        total += e.get("seconds") or 0
        rows.append(f"| {pid} | {PHASES[pid].title} | **{e['status']}** | {fmt_seconds(e.get('seconds'))} | "
                    f"{(e.get('reason') or '').replace('|', '/')[:100]} |")
        body = e.get("keys") or []
        if e["status"] == "SKIPPED":
            body = [f"SKIPPED: {e.get('reason')}"]
        if e["status"] == "FAIL" and not body:
            body = ["FAIL: " + "; ".join(e.get("fails", [])[:3])]
        sections.append(f"### {pid} - {PHASES[pid].title} - {e['status']}\n\n```\n" + "\n".join(body) + "\n```\n")
    counts = {k: sum(1 for e in state["phases"].values() if e["status"] == k) for k in ("PASS", "FAIL", "SKIPPED")}
    text = "\n".join([
        "# Final run summary", "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `scripts/run_all_checks.py`._", "",
        f"- commit: `{branch} @ {sha[:12]}`",
        f"- host: {si['physical_cores']} physical / {si['logical_cores']} logical cores, {si['ram_total_gb']} GB RAM",
        f"- phases: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['SKIPPED']} SKIPPED; "
        f"time in phases {fmt_seconds(total)}", "",
        "| Phase | What | Status | Time | Note |", "|---|---|---|---|---|", *rows, "",
        "## Paste these back (nothing else)", "",
        "If a phase FAILED, its lines below say why; the full log is `results/logs/<phase>.log` "
        "(notebook cell 5 prints the tails).", "", *sections])
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------------------------------
# selection and the main loop
# --------------------------------------------------------------------------------------------------

def split_ids(values):
    out = []
    for v in values or []:
        out += [x.strip().upper() for x in re.split(r"[,\s]+", v) if x.strip()]
    return out


def select_phases(args):
    only, skip = split_ids(args.only), set(split_ids(args.skip))
    unknown = [p for p in only + list(skip) if p not in PHASES]
    if unknown:
        raise SystemExit(f"unknown phase(s) {unknown}; choose from {list(PHASES)}")
    if only:
        chosen = [p for p in PHASES if p in only]
    else:
        chosen = [p for p, ph in PHASES.items() if 1 <= ph.tier <= args.tier]
    return [p for p in chosen if p not in skip]


def run_phase(pid, args, state, deadline):
    ph = PHASES[pid]
    prev = state["phases"].get(pid)
    if (args.resume and prev and prev.get("status") == "PASS" and prev.get("done", True)
            and pid not in split_ids(args.rerun)):
        say(f"{pid}: already PASS, skipping (--resume)")
        return
    reason = None
    missing = [r for r in ph.requires if not (ROOT / r).exists()]
    remaining = deadline - time.time()
    if missing:
        reason = f"needs {', '.join(missing)} (an earlier phase did not produce it)"
    elif remaining < 60 * min(ph.budget_min, 10):
        reason = f"time: only {remaining / 60:.0f} min left of --max-hours"
    if reason:
        say(f"{pid}: SKIPPED - {reason}")
        state["phases"][pid] = {"status": "SKIPPED", "reason": reason, "seconds": 0, "keys": [], "fails": [],
                                "done": False}
        save_state(state)
        return
    budget_s = min(ph.budget_min * 60, remaining)
    say(f"{'=' * 20} {pid}: {ph.title}  (budget {budget_s / 60:.0f} min) {'=' * 20}")
    token = os.environ.get(args.token_env) or None
    ctx = Ctx(pid, args, LOGS / f"{pid}.log", budget_s, deadline, token=token)
    t0 = time.time()
    status, why = "PASS", ""
    try:
        ph.fn(ctx)
    except PhaseTimeout as exc:
        ctx.fails.append(str(exc))
        ctx.key(f"TIMEOUT: {exc}")
        why = "time budget"
    except SkipPhase as exc:
        status, why = "SKIPPED", str(exc)
    except Exception as exc:  # noqa: BLE001  a phase must never stop the run
        tb = traceback.format_exc()
        ctx.log(tb)
        ctx.fails.append(f"{type(exc).__name__}: {exc}")
        ctx.key(f"EXCEPTION {type(exc).__name__}: {exc}")
        why = "exception (see log)"
    finally:
        ctx.close()
    if status != "SKIPPED" and ctx.fails:
        status = "FAIL"
        why = why or "; ".join(ctx.fails)[:120]
    seconds = time.time() - t0
    state["phases"][pid] = {"status": status, "seconds": round(seconds, 1), "reason": why or "; ".join(ctx.notes)[:120],
                            "keys": ctx.keys, "fails": ctx.fails, "done": ctx.done and status == "PASS",
                            "log": _rel(ctx.log_path), "finished": time.strftime("%H:%M:%S")}
    save_state(state)
    zip_partial()
    write_summary(state)
    say(f"{pid}: {status} in {fmt_seconds(seconds)}" + (f" - {why}" if why and status != 'PASS' else ""))


def diagnose(tail=60):
    state = read_json(STATE, {"phases": {}})
    print("=" * 78)
    for pid, e in state.get("phases", {}).items():
        print(f"{pid}: {e['status']}  {e.get('reason', '')}")
    print("=" * 78)
    for log in sorted(LOGS.glob("*.log")):
        pid = log.stem
        e = state.get("phases", {}).get(pid, {})
        if e.get("status") == "PASS" and pid != "P1":
            continue
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"\n----- {log.name} (last {min(tail, len(lines))} of {len(lines)} lines; status {e.get('status')}) -----")
        print("\n".join(lines[-tail:]))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, choices=[1, 2], default=1, help="run every phase up to this tier")
    ap.add_argument("--only", nargs="*", default=[], help="phase ids, e.g. P1 P4 or P1,P4 (overrides --tier)")
    ap.add_argument("--skip", nargs="*", default=[], help="phase ids to leave out")
    ap.add_argument("--resume", action="store_true", help="skip phases already PASS in results/run_state.json")
    ap.add_argument("--fresh", action="store_true", help="forget results/run_state.json before starting")
    ap.add_argument("--rerun", nargs="*", default=[], help="with --resume, phases to run again anyway")
    ap.add_argument("--max-hours", type=float, default=10.0, help="stop starting phases when this is spent")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--diagnose", action="store_true", help="print the tail of every phase log and exit")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if args.diagnose:
        return diagnose()
    chosen = select_phases(args)
    if args.dry_run:
        print(f"plan: {len(chosen)} phase(s), --max-hours {args.max_hours}")
        for pid in chosen:
            ph = PHASES[pid]
            print(f"  {pid:2s} tier {ph.tier}  budget {ph.budget_min:>3} min  {'CPU-only ' if ph.cpu_only else '         '}{ph.title}")
        return 0
    state = load_state(args.fresh)
    state["args"] = vars(args)
    deadline = time.time() + args.max_hours * 3600
    say(f"running {chosen} | resume={args.resume} | max {args.max_hours} h | results in {RESULTS}")
    try:
        for pid in chosen:
            run_phase(pid, args, state, deadline)
    except KeyboardInterrupt:
        say("interrupted: state and summary saved; re-run with --resume to continue")
    finally:
        save_state(state)
        zip_partial()
        write_summary(state)
    counts = {k: sum(1 for e in state["phases"].values() if e["status"] == k) for k in ("PASS", "FAIL", "SKIPPED")}
    say(f"done: {counts}. Summary: {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
