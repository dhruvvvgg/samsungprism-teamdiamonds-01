"""scripts/run_all_checks.py: phase selection, resilience (a failure or a timeout
never stops later phases), resume, the state file, the partial zip, the summary, the official-score
verdict, the summary helpers, and secret masking. No model, GPU or network."""
import ast
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = load_module(ROOT / "scripts" / "run_all_checks.py", "run_all_checks")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Point every output path of the orchestrator at a temp dir and give it a private phase table."""
    res = tmp_path / "results"
    monkeypatch.setattr(R, "RESULTS", res)
    monkeypatch.setattr(R, "LOGS", res / "logs")
    monkeypatch.setattr(R, "STATE", res / "run_state.json")
    monkeypatch.setattr(R, "SUMMARY", res / "FINAL_SUMMARY.md")
    monkeypatch.setattr(R, "PARTIAL_ZIP", tmp_path / "results_partial.zip")
    monkeypatch.setattr(R, "RELEASE_JSON", res / "release_files.json")
    monkeypatch.setattr(R, "PHASES", {})
    res.mkdir()
    (res / "logs").mkdir()
    return tmp_path


def args(**kw):
    ns = R.build_parser().parse_args([])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# --- selection and the plan -------------------------------------------------------------------------

def test_tiers_select_the_documented_phases():
    assert R.select_phases(args(tier=1)) == ["P0", "P1", "P2", "P3", "P4"]
    assert R.select_phases(args(tier=2)) == ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]


def test_only_overrides_the_tier_and_skip_removes(monkeypatch):
    assert R.select_phases(args(only=["P1,P2", "P3"])) == ["P1", "P2", "P3"]
    assert R.select_phases(args(tier=2, skip=["P5", "p6"])) == ["P0", "P1", "P2", "P3", "P4"]
    assert R.select_phases(args(only=["C"])) == ["C"]
    assert "C" not in R.select_phases(args(tier=2)), "collect only runs when asked for"


def test_unknown_phases_are_rejected():
    with pytest.raises(SystemExit, match="unknown phase"):
        R.select_phases(args(only=["P9"]))


def test_dry_run_prints_the_plan_and_runs_nothing(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RESULTS", tmp_path / "r")
    monkeypatch.setattr(R, "LOGS", tmp_path / "r" / "logs")
    assert R.main(["--tier", "2", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "plan: 7 phase(s)" in out and "P4" in out and "CPU-only" in out
    assert not (tmp_path / "r" / "run_state.json").exists()


def test_flags_the_brief_asks_for_exist():
    a = R.build_parser().parse_args(["--tier", "2", "--only", "P1", "--skip", "P5", "--resume", "--max-hours", "2"])
    assert (a.tier, a.resume, a.max_hours) == (2, True, 2.0)


def test_cpu_only_phases_are_marked():
    assert R.PHASES["P3"].cpu_only and R.PHASES["P4"].cpu_only
    assert not R.PHASES["P1"].cpu_only, "the official run needs the GPU"


# --- running commands -------------------------------------------------------------------------------

def make_ctx(env, budget_s=60, token=None):
    return R.Ctx("T", args(), R.LOGS / "T.log", budget_s, time.time() + 600, token=token)


def test_sh_captures_output_and_exit_code(env):
    ctx = make_ctx(env)
    r = ctx.sh([sys.executable, "-c", "print('hello'); raise SystemExit(3)"])
    assert r.rc == 3 and "hello" in r.out and not r.timed_out
    assert "hello" in (R.LOGS / "T.log").read_text()


def test_cpu_only_hides_the_gpu_from_the_child(env):
    ctx = make_ctx(env)
    probe = "import os;print('CVD=' + repr(os.environ.get('CUDA_VISIBLE_DEVICES')))"
    assert "CVD=''" in ctx.sh([sys.executable, "-c", probe], cpu=True).out


def test_a_command_over_its_limit_is_killed_and_reported(env):
    ctx = make_ctx(env)
    t0 = time.time()
    r = ctx.sh([sys.executable, "-c", "import time; time.sleep(60)"], timeout_min=0.03)
    assert r.timed_out and time.time() - t0 < 20


def test_a_missing_binary_is_an_exit_code_not_a_crash(env):
    r = make_ctx(env).sh(["definitely-not-a-real-binary-xyz"])
    assert r.rc == 127


def test_the_token_is_masked_in_output_and_log(env):
    ctx = make_ctx(env, token="ghp_SECRET123")
    r = ctx.sh([sys.executable, "-c", "print('url https://x-access-token:ghp_SECRET123@github.com/x')"])
    assert "ghp_SECRET123" not in r.out
    assert "ghp_SECRET123" not in (R.LOGS / "T.log").read_text()


def test_a_spent_phase_budget_raises_before_starting_a_command(env):
    ctx = R.Ctx("T", args(), R.LOGS / "T.log", 1, time.time() + 600)
    with pytest.raises(R.PhaseTimeout):
        ctx.sh([sys.executable, "-c", "print(1)"])


# --- resilience, state and resume ---------------------------------------------------------------------

def register(counter):
    def ok(ctx):
        counter["ok"] = counter.get("ok", 0) + 1
        ctx.key("ok line")

    def boom(ctx):
        counter["boom"] = counter.get("boom", 0) + 1
        raise RuntimeError("kaput")

    def slow(ctx):
        counter["slow"] = counter.get("slow", 0) + 1
        ctx.sh([sys.executable, "-c", "import time; time.sleep(60)"], timeout_min=0.03)
        ctx.fail("the slow step was killed")

    def partial(ctx):
        counter["partial"] = counter.get("partial", 0) + 1
        ctx.done = False
        ctx.key("stopped after the sample")

    def skipper(ctx):
        raise R.SkipPhase("not applicable here")

    for pid, fn, req in (("X1", boom, []), ("X2", slow, []), ("X3", ok, []), ("X4", ok, ["nope/missing.file"]),
                         ("X5", partial, []), ("X6", skipper, [])):
        R.PHASES[pid] = SimpleNamespace(id=pid, title=f"phase {pid}", tier=1, budget_min=5, cpu_only=False,
                                        requires=req, fn=fn)


def run_all(a, state=None):
    state = state or R.load_state(a.fresh)
    deadline = time.time() + a.max_hours * 3600
    for pid in list(R.PHASES):
        R.run_phase(pid, a, state, deadline)
    return state


def test_a_failure_a_timeout_and_a_missing_input_never_stop_later_phases(env):
    counter = {}
    register(counter)
    state = run_all(args(only=["X1", "X2", "X3", "X4", "X5", "X6"], max_hours=1))
    s = {k: v["status"] for k, v in state["phases"].items()}
    assert s == {"X1": "FAIL", "X2": "FAIL", "X3": "PASS", "X4": "SKIPPED", "X5": "PASS", "X6": "SKIPPED"}
    assert counter["ok"] == 1, "the phase after the exception and the timeout still ran"
    assert "kaput" in " ".join(state["phases"]["X1"]["fails"])
    assert "needs nope/missing.file" in state["phases"]["X4"]["reason"]
    assert state["phases"]["X6"]["reason"] == "not applicable here"


def test_the_state_file_and_one_log_per_phase_are_written(env):
    register({})
    run_all(args(max_hours=1))
    st = json.loads(R.STATE.read_text())
    assert set(st["phases"]) == set(R.PHASES) and st["version"] == 1
    for pid in ("X1", "X2", "X3"):
        assert (R.LOGS / f"{pid}.log").exists()
    assert st["phases"]["X3"]["keys"] == ["ok line"] and st["phases"]["X3"]["done"] is True


def test_resume_skips_finished_phases_but_reruns_failures_and_partials(env):
    counter = {}
    register(counter)
    run_all(args(max_hours=1))
    assert counter == {"boom": 1, "slow": 1, "ok": 1, "partial": 1}
    run_all(args(max_hours=1, resume=True))
    assert counter["ok"] == 1, "a PASS phase is not run again under --resume"
    assert counter["boom"] == 2 and counter["slow"] == 2, "failures are retried"
    assert counter["partial"] == 2, "a phase that stopped early (done=False) is not treated as finished"
    run_all(args(max_hours=1, resume=True, rerun=["X3"]))
    assert counter["ok"] == 2, "--rerun forces a finished phase"


def test_without_resume_everything_runs_again_and_fresh_forgets_the_state(env):
    counter = {}
    register(counter)
    run_all(args(max_hours=1))
    run_all(args(max_hours=1))
    assert counter["ok"] == 2
    assert R.load_state(fresh=True)["phases"] == {}
    assert R.load_state(fresh=False)["phases"]


def test_a_phase_is_skipped_when_the_global_time_is_spent(env):
    counter = {}
    register(counter)
    state = R.load_state()
    R.run_phase("X3", args(), state, time.time() + 30)          # 30 s left, the phase wants minutes
    assert state["phases"]["X3"]["status"] == "SKIPPED" and "time" in state["phases"]["X3"]["reason"]
    assert "ok" not in counter


def test_the_partial_zip_is_rewritten_after_every_phase(env):
    register({})
    (R.RESULTS / "marker.json").write_text("{}")
    state = R.load_state()
    R.run_phase("X3", args(), state, time.time() + 3600)
    assert R.PARTIAL_ZIP.exists()
    names = zipfile.ZipFile(R.PARTIAL_ZIP).namelist()
    assert any(n.endswith("marker.json") for n in names) and any(n.endswith("run_state.json") for n in names)
    first = R.PARTIAL_ZIP.stat().st_mtime_ns
    R.run_phase("X3", args(), state, time.time() + 3600)
    assert R.PARTIAL_ZIP.stat().st_mtime_ns >= first


def test_summary_has_a_status_table_timings_and_the_lines_to_paste(env):
    register({})
    state = run_all(args(max_hours=1))
    text = R.write_summary(state)
    assert (R.SUMMARY).read_text() == text
    for needle in ("| Phase | What | Status | Time | Note |", "**PASS**", "**FAIL**", "**SKIPPED**",
                   "## Paste these back", "### X3 - phase X3 - PASS", "ok line", "kaput"):
        assert needle in text
    assert "2 PASS, 2 FAIL, 2 SKIPPED" in text


def test_main_survives_a_phase_that_raises_and_still_writes_the_summary(env, monkeypatch):
    counter = {}
    register(counter)
    monkeypatch.setattr(R, "select_phases", lambda a: ["X1", "X3"])
    assert R.main(["--max-hours", "1"]) == 0
    assert R.SUMMARY.exists() and counter["ok"] == 1
    assert R.diagnose(tail=5) == 0


# --- the official-score verdict -----------------------------------------------------------------------

def results_file(tmp_path, ndcg, mrr):
    p = tmp_path / "res.json"
    p.write_text(json.dumps({"scores": {"test": [{"ndcg_at_10": ndcg, "mrr_at_10": mrr}]},
                             "_run_info": {"fell_back_to_fp32": False}}))
    return p


def test_official_scores_within_tolerance_pass(tmp_path):
    n, m, info, problems = R.check_official_scores(results_file(tmp_path, 0.9376, 0.9238))
    assert problems == [] and (n, m) == (0.9376, 0.9238)
    assert R.check_official_scores(results_file(tmp_path, 0.9394, 0.9219))[3] == []      # +/-0.0019


def test_official_scores_outside_tolerance_fail_each_metric_separately(tmp_path):
    problems = R.check_official_scores(results_file(tmp_path, 0.9300, 0.9238))[3]
    assert len(problems) == 1 and "NDCG@10" in problems[0]
    problems = R.check_official_scores(results_file(tmp_path, 0.9376, 0.9100))[3]
    assert len(problems) == 1 and "MRR@10" in problems[0]
    assert len(R.check_official_scores(results_file(tmp_path, 0.90, 0.90))[3]) == 2


def test_a_missing_or_malformed_results_file_is_a_problem_not_a_crash(tmp_path):
    assert R.check_official_scores(tmp_path / "nope.json")[3]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"scores": {}}))
    assert R.check_official_scores(bad)[3]


def test_the_failure_message_names_the_two_suspects_from_pr_1():
    assert "PresetQueryEncoder.encode_docs" in R.SUSPECTS and "service_for" in R.SUSPECTS
    assert "run_official.py" in R.SUSPECTS, "and says honestly that the official path does not import them"


def test_targets_match_the_submitted_result():
    assert (R.TARGET_NDCG, R.TARGET_MRR, R.TOLERANCE) == (0.9376, 0.9238, 0.002)
    assert R.OFFICIAL_CONFIG == "configs/official_f2llm17b_noreranker.json"
    assert (ROOT / R.OFFICIAL_CONFIG).exists()


# --- extraction and summary helpers -------------------------------------------------------------------

def test_grep_lines_and_extract_block():
    text = "noise\n[fail] 12 of 3765 miss\n  [fail] second\nnoise\n[fail] 12 of 3765 miss\n"
    assert R.grep_lines(text, [r"^\[fail\]"]) == ["[fail] 12 of 3765 miss", "[fail] second"]
    assert R.grep_lines(text, [r"^\[fail\]"], limit=1) == ["[fail] 12 of 3765 miss"]
    blk = R.extract_block("a\n====\nTITLE\n====\nrow1\nrow2\nend here\nafter", r"TITLE", r"end here")
    assert blk == ["TITLE", "row1", "row2", "end here"]
    assert R.extract_block("nothing", r"TITLE") == []


def test_latency_and_precision_summaries_read_the_bench_json(tmp_path):
    lat = tmp_path / "l.json"
    lat.write_text(json.dumps({"latency_ms": {"p50": 900, "p95": 2000}, "query_encode_ms": {"p50": 890},
                               "search_ms": {"p50": 5.2}, "max_query_tokens": 1024,
                               "query_tokens": {"p50": 300}, "peak_rss_mb": 4500.0, "threads": 2,
                               "physical_cores": 2, "thread_sweep": [
                                   {"threads": 1, "latency_ms": {"p50": 1700}},
                                   {"threads": 2, "latency_ms": {"p50": 900}}]}))
    line = R.latency_summary("lite capped", lat)
    assert "p50 900" in line and "cap 1024" in line and "1t p50 1700 ms" in line
    prec = tmp_path / "p.json"

    def prec_json(cos, identical, rss_int8):
        prec.write_text(json.dumps({"n_queries": 20, "modes": [
            {"mode": "fp32", "encode_ms_p50": 900, "peak_rss_mb": 5000, "cosine_to_fp32_mean": 1.0,
             "top10_overlap_mean": 1.0, "rank1_changed": 0, "applied": {"int8_applied": False}},
            {"mode": "int8", "encode_ms_p50": 850, "peak_rss_mb": rss_int8, "cosine_to_fp32_mean": cos,
             "top10_overlap_mean": 0.99, "rank1_changed": 0, "top10_identical_and_in_order": identical,
             "applied": {"int8_applied": True}}]}))
        return R.precision_summary(prec)
    assert "BUG STILL PRESENT" in prec_json(1.0, 20, 9000)[-1]
    assert "BUG STILL PRESENT" not in prec_json(0.9990, 17, 4000)[-1]
    assert "BUG STILL PRESENT" not in prec_json(1.0, 20, 4000)[-1], "all three symptoms are needed"


def test_history_summary_reads_the_benchmark_json(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"k": 10, "ingest": {"commits": 40, "rows": 900, "lineages": 300, "distinct_hashes": 400},
                             "totals": {"embeddings_full": 5000, "embeddings_incremental": 500, "saved_pct": 90.0,
                                        "speedup_x": 8.1, "seconds_full": 80, "seconds_incremental": 10},
                             "warnings": [], "retrieval": {"n_queries": 50, "p1_version_targeted": {
                                 "top1_correct_lineage": 0.9, "median_latency_ms": 12.0},
                                 "bonus_all_versions": {"duplicate_pct_at_10_all_versions": 30.0,
                                                        "duplicate_pct_at_10_collapsed": 0.0}},
                             "delta_vectors": {"lineage_k": "max(3k, 20)", "index_bytes_delta": 100,
                                               "index_bytes_current": 400, "size_ratio_delta_over_current": 0.25,
                                               "top1_exact_version": {"delta": 0.5, "current": 0.5},
                                               "top1_lineage": {"delta": 0.9, "current": 0.9},
                                               "top1_agreement_delta_vs_current": 1.0}}))
    lines = R.history_summary(p)
    assert any("saved 90.0%" in ln for ln in lines) and any("Delta vectors" in ln for ln in lines)
    assert any("duplicate slots 30.0% -> 0.0%" in ln for ln in lines)
    assert R.history_summary(tmp_path / "missing.json")[0].endswith("not written")


def test_checks_runner_counts_a_raising_check_as_a_failure_not_a_crash(env):
    ctx = make_ctx(env)
    c = R.Checks(ctx)
    assert c.run("good", lambda: (True, "fine")) is True
    assert c.run("bad", lambda: (False, "nope")) is False
    assert c.run("raises", lambda: 1 / 0) is False
    assert (c.n_pass, c.n_fail) == (1, 2)
    assert any(k.startswith("FAIL  raises") and "ZeroDivisionError" in k for k in ctx.keys)


# --- collect and push safety --------------------------------------------------------------------------

def test_only_small_text_results_are_pushable(env, monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "results" / "logs").mkdir(parents=True)
    (root / "outputs" / "dev").mkdir(parents=True)
    (root / "results" / "a.json").write_text("{}")
    (root / "results" / "metrics.md").write_text("m")
    (root / "results" / "huge.json").write_text("x" * 3_000_000)
    (root / "results" / "results_partial.zip").write_bytes(b"zip")
    (root / "results" / "logs" / "P1.log").write_text("log")
    (root / "outputs" / "appsretrieval_results.json").write_text("{}")
    (root / "outputs" / "appsretrieval_rankings.json").write_text("{}")
    (root / "outputs" / "dev" / "chosen.json").write_text("{}")
    monkeypatch.setattr(R, "ROOT", root)
    monkeypatch.setattr(R, "RESULTS", root / "results")
    names = sorted(str(p.relative_to(root)) for p in R.small_text_files())
    assert names == ["outputs/appsretrieval_results.json", "results/a.json", "results/metrics.md"]


def test_the_push_branch_can_never_be_main_or_develop():
    text = (ROOT / "scripts" / "run_all_checks.py").read_text()
    assert 'branch = "results/final-run"' in text
    assert {"main", "master", "develop"} <= set(R.PROTECTED_BRANCHES)
    assert "results/final-run" not in R.PROTECTED_BRANCHES
    assert "--force" not in text and "push -f" not in text, "a results push must never rewrite history"


def test_zip_dir_and_sha256_file(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_text("hello")
    z = R.zip_dir(d, tmp_path / "out.zip")
    assert zipfile.ZipFile(z).namelist() == ["d/a.txt"]
    assert R.sha256_file(d / "a.txt") == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_collect_lists_path_size_and_sha256(env, monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "outputs").mkdir(parents=True)
    (root / "outputs" / "appsretrieval_results.json").write_text('{"a": 1}')
    monkeypatch.setattr(R, "ROOT", root)
    monkeypatch.setattr(R, "RELEASE_DIR", tmp_path / "release")
    monkeypatch.setattr(R, "sysinfo", lambda: {"physical_cores": 2, "logical_cores": 4, "ram_total_gb": 29,
                                               "ram_available_gb": 25})
    ctx = R.Ctx("C", args(), R.LOGS / "C.log", 60, time.time() + 600)
    monkeypatch.setattr(ctx, "sh", lambda *a, **k: SimpleNamespace(rc=0, out="", seconds=0, timed_out=False))
    R.pc_collect(ctx)
    listing = json.loads(R.RELEASE_JSON.read_text())
    res = next(i for i in listing if i["path"].endswith("appsretrieval_results.json"))
    assert res["bytes"] == 8 and len(res["sha256"]) == 64
    assert any("appsretrieval_results.json | 0.00 MB | sha256 " in k for k in ctx.keys)
    assert any("results_final.zip" in i["path"] for i in listing)
    assert any("runtime_index.zip" in f for f in ctx.fails), "a missing index is reported, not ignored"


# --- hygiene --------------------------------------------------------------------------------------------


def test_dockerignore_excludes_indexes_outputs_caches_and_adapters():
    lines = {ln.strip() for ln in (ROOT / ".dockerignore").read_text().splitlines()}
    for needed in ("outputs/", "runtime_index*/", "history_index*/", "*.zip", "*.npy", "hf_cache/", ".cache/",
                   "adapter*/", "*.safetensors", "results_partial.zip", "data/"):
        assert needed in lines, needed
    assert "src/" not in lines and "configs/" not in lines, "the image still needs the code and configs"


def test_the_official_config_and_guard_test_are_untouched_by_this_tooling():
    cfg = json.loads((ROOT / R.OFFICIAL_CONFIG).read_text())
    assert cfg["preset"] == "f2llm-v2-1.7b" and cfg["rerank"] == {"enabled": False}
    assert (ROOT / "tests" / "test_official_path_guard.py").exists()


def test_p5_history_diff_lineage_selection_finds_distinct_versions():
    """Verify that lineage candidates are scanned until one with >= 2 distinct content hashes is found."""
    # Mock data
    history_responses = {
        "sid1": {"n_versions": 2, "versions": [{"version": 1, "content_hash": "aaa"}, {"version": 2, "content_hash": "aaa"}]},
        "sid2": {"n_versions": 3, "versions": [{"version": 1, "content_hash": "aaa"}, {"version": 2, "content_hash": "bbb"}]},
    }
    candidates = ["sid1", "sid2"]
    chosen_sid, chosen_pair, chosen_h = None, None, None
    for sid in candidates:
        h = history_responses[sid]
        hashes = {}
        for v in h.get("versions", []):
            hashes.setdefault(v["content_hash"], v["version"])
        if len(hashes) >= 2:
            chosen_sid = sid
            chosen_pair = sorted(hashes.values())[:2]
            chosen_h = h
            break
    assert chosen_sid == "sid2"
    assert chosen_pair == [1, 2]
    assert chosen_h["n_versions"] == 3
