"""Version-delta vectors (experiment, off by default): reconstruction, two-stage search, sizes, and the
benchmark flag. Synthetic vectors and the hashing encoder only; no result about real data is implied."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.versioning.delta_index import DeltaVectors

ROOT = Path(__file__).resolve().parents[1]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_history(n_lineages=30, n_versions=6, dim=96, noise=0.03, seed=0, distinct_every=2):
    """Lineages with random centres; each version = centre + small noise. Every `distinct_every`-th
    version repeats the previous one's content (same hash), like an unchanged function."""
    rng = np.random.RandomState(seed)
    centres = unit(rng.randn(n_lineages, dim))
    rows, lineage, hashes, version = [], [], [], []
    for li in range(n_lineages):
        prev_vec, prev_hash = None, None
        for v in range(1, n_versions + 1):
            if prev_vec is not None and v % distinct_every == 0:
                vec, h = prev_vec, prev_hash                     # unchanged: identical content
            else:
                vec = unit(centres[li] + noise * rng.randn(dim))
                h = f"h{li}-{v}"
            rows.append(vec)
            lineage.append(f"L{li:02d}")
            hashes.append(h)
            version.append(v)
            prev_vec, prev_hash = vec, h
    return np.array(rows, dtype=np.float32), lineage, hashes, version


@pytest.fixture(scope="module")
def hist():
    return make_history()


def test_identical_content_shares_one_residual_entry(hist):
    E, lineage, hashes, _ = hist
    dv = DeltaVectors.build(E, lineage, hashes)
    assert dv.n_rows == len(E) == 180
    assert dv.n_entries == len(set(zip(lineage, hashes))) == 90        # half the versions are repeats
    assert len(dv.lineages) == 30


def test_reconstruction_is_close_to_the_original_vectors(hist):
    E, lineage, hashes, _ = hist
    dv = DeltaVectors.build(E, lineage, hashes)
    full = dv.full_vectors()
    cos = (unit(full) * E).sum(axis=1)
    assert cos.min() > 0.99999
    assert np.abs(full - E).max() < 2e-3                              # int8 residual + fp16 base


def test_the_stored_base_is_what_the_residual_is_measured_against():
    """If the residual were taken against the float32 base but the fp16 base is stored, the fp16
    rounding error would be added on top of the quantisation error."""
    E, lineage, hashes, _ = make_history(n_lineages=4, dim=64, noise=0.001)
    dv = DeltaVectors.build(E, lineage, hashes)
    assert np.abs(dv.full_vectors() - E).max() < 1e-4


def test_zero_residual_does_not_divide_by_zero():
    E = unit(np.random.RandomState(1).randn(1, 16)).astype(np.float32)
    dv = DeltaVectors.build(E, ["a"], ["h"])                          # one version: residual is ~0
    assert np.isfinite(dv.scale).all() and np.abs(dv.full_vectors() - E).max() < 1e-3


def test_sizes_are_accounted_and_smaller_than_per_row_fp16(hist):
    E, lineage, hashes, _ = hist
    dv = DeltaVectors.build(E, lineage, hashes)
    assert dv.baseline_bytes() == 180 * 96 * 2
    expected = (dv.base.nbytes + dv.entry_lineage.nbytes + dv.residual.nbytes + dv.scale.nbytes
                + dv.row_entry.nbytes)
    assert dv.size_bytes() == expected
    assert dv.size_bytes() < dv.baseline_bytes()
    assert dv.residual.dtype == np.int8 and dv.base.dtype == np.float16


def test_no_saving_is_claimed_when_every_version_is_distinct():
    E, lineage, hashes, _ = make_history(n_lineages=10, n_versions=2, distinct_every=99)
    dv = DeltaVectors.build(E, lineage, hashes)
    assert dv.n_entries == dv.n_rows
    # 2 versions per lineage: the base table alone costs as much as the rows it replaces
    assert dv.size_bytes() >= dv.baseline_bytes() * 0.9


def test_search_agrees_with_an_exact_scan_when_every_lineage_is_kept(hist):
    E, lineage, hashes, _ = hist
    dv = DeltaVectors.build(E, lineage, hashes)
    rng = np.random.RandomState(5)
    full = dv.full_vectors()
    for _ in range(25):
        q = unit(E[rng.randint(len(E))] + 0.02 * rng.randn(E.shape[1]))
        got = dv.search(q, k=5, lineage_k=len(dv.lineages))
        exact = np.argsort(-(full @ q), kind="stable")[:5]
        # compare by score, not row id: repeated versions share a vector, so the best row can tie
        assert got[0][1] == pytest.approx(float(full[exact[0]] @ q), abs=1e-5)
        assert [s for _, s in got] == sorted([s for _, s in got], reverse=True)


def test_top1_matches_the_exact_search_on_clustered_versions(hist):
    E, lineage, hashes, _ = hist
    dv = DeltaVectors.build(E, lineage, hashes)
    agree = 0
    for i in range(0, len(E), 3):
        q = E[i]
        exact_top = int(np.argmax(E @ q))
        agree += int(dv.search(q, k=1)[0][0] in np.nonzero(E @ q > (E @ q)[exact_top] - 1e-6)[0])
    assert agree / len(range(0, len(E), 3)) >= 0.95


def test_the_first_stage_really_restricts_the_candidates(hist):
    E, lineage, hashes, _ = hist
    dv = DeltaVectors.build(E, lineage, hashes)
    q = E[100]
    top_lineage = int(np.argmax(dv._base_unit @ unit(q)))
    got = dv.search(q, k=50, lineage_k=1)
    assert got and all(dv.row_lineage[r] == top_lineage for r, _ in got)
    assert len(got) == 6                                              # that lineage has 6 versions


def test_search_never_returns_more_than_k_or_a_repeated_row(hist):
    E, lineage, hashes, _ = hist
    dv = DeltaVectors.build(E, lineage, hashes)
    got = dv.search(E[7], k=4)
    assert len(got) == 4 and len({r for r, _ in got}) == 4


def test_save_and_load_round_trip(tmp_path, hist):
    E, lineage, hashes, _ = hist
    dv = DeltaVectors.build(E, lineage, hashes)
    dv.save(tmp_path)
    back = DeltaVectors.load(tmp_path)
    assert back.lineages == dv.lineages and back.size_bytes() == dv.size_bytes()
    assert np.array_equal(back.full_vectors(), dv.full_vectors())
    assert back.search(E[3], k=3) == dv.search(E[3], k=3)


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError, match="one value per embedding row"):
        DeltaVectors.build(np.zeros((3, 4), dtype=np.float32), ["a"], ["h"])


# --- the benchmark flag ------------------------------------------------------------------------------------

BODY_1 = '''def add(values, offset):
    """Add an offset to every value."""
    result = []
    for value in values:
        result.append(value + offset)
    return result
'''
BODY_2 = BODY_1.replace("result.append(value + offset)", "shifted = value + offset\n        result.append(shifted)")
BODY_3 = BODY_2.replace("return result", "return list(result)")


def git(repo, *args):
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture(scope="module")
def bench_setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("delta")
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "T")
    other = 'def scale(v, f):\n    """Scale."""\n    out = []\n    for x in v:\n        out.append(x * f)\n    return out\n'
    for body in (BODY_1, BODY_2, BODY_3, BODY_3 + "\n# tail comment\n"):
        (repo / "m.py").write_text(body + "\n\n" + other, encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "c")
    idx = tmp / "idx"
    proc = subprocess.run([sys.executable, "src/build_history_index.py", "--repo", str(repo),
                           "--mock-encoder", "--out", str(idx), "--queries-out", str(tmp / "q.json"),
                           "--stats-out", str(tmp / "s.json")], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return repo, idx, tmp


def run_bench(repo, idx, out, *extra):
    proc = subprocess.run([sys.executable, "src/bench_real_history.py", "--repo", str(repo),
                           "--mock-encoder", "--index", str(idx), "--out", str(out), *extra],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout, json.loads(Path(out).read_text(encoding="utf-8"))


def test_the_benchmark_is_unchanged_without_the_flag(bench_setup):
    repo, idx, tmp = bench_setup
    out, data = run_bench(repo, idx, tmp / "plain.json")
    assert "delta_vectors" not in data and "Delta vectors" not in out


def test_the_flag_reports_size_and_top1_accuracy_for_both_approaches(bench_setup):
    repo, idx, tmp = bench_setup
    out, data = run_bench(repo, idx, tmp / "delta.json", "--delta-vectors")
    d = data["delta_vectors"]
    assert d["n_queries"] >= 1 and d["rows"] == 8 and d["lineages"] == 2
    assert d["distinct_entries"] <= d["rows"]
    assert d["index_bytes_current"] == 8 * 64 * 2 and d["index_bytes_delta"] > 0
    assert d["size_ratio_delta_over_current"] == pytest.approx(d["index_bytes_delta"] / d["index_bytes_current"], abs=1e-3)
    for key in ("top1_lineage", "top1_exact_version", "top1_token_consistent_version"):
        assert set(d[key]) == {"current", "delta"} and all(0 <= v <= 1 for v in d[key].values())
    assert 0 <= d["top1_agreement_delta_vs_current"] <= 1
    assert "Delta vectors (experiment)" in out
    assert data["retrieval"] is not None and data["warnings"] == []


def test_delta_lineage_k_is_passed_through(bench_setup):
    repo, idx, tmp = bench_setup
    _, data = run_bench(repo, idx, tmp / "delta_k.json", "--delta-vectors", "--delta-lineage-k", "1")
    assert data["delta_vectors"]["lineage_k"] == 1
