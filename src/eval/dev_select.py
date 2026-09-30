"""Turn dev-table results into the chosen pipeline config (outputs/dev/chosen.json), one stage at a time.

    python src/eval/dev_select.py --group variants     # after dev_variants.py
    python src/eval/dev_select.py --group hybrid       # after dev_hybrid.py
    python src/eval/dev_select.py --group rerank       # after dev_rerank.py
    python src/eval/dev_select.py --group rejudge      # after dev_rejudge.py
    python src/eval/dev_select.py --show

Only rows flagged ADOPT (dNDCG@10 >= +0.005, paired-bootstrap 95% CI > 0, worsened <= 0.5 x improved, all
on the tune set) are eligible; the eligible row with the largest NDCG@10 wins. If none qualifies the
stage is left OFF and the config is unchanged. Selection is deterministic and made on the tune set
only (holdout untouched).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.dev_lib import CHOSEN, RESULTS_JSONL, TEST_NDCG_REFERENCE, row_preset  # noqa: E402
from src.retrieval.pipeline import merge_config  # noqa: E402


def load_rows(group, preset="f2llm-v2-0.6b", split="tune"):
    """Rows for one first-stage preset only, so e.g. a 1.7B rerank sweep is never compared against or
    adopted on top of 0.6B rows. Rows written before multi-preset support carry no 'preset' field and
    are treated as 'f2llm-v2-0.6b' (row_preset's fallback), matching the default here."""
    rows = [json.loads(l) for l in RESULTS_JSONL.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["group"] == group and r["split"] == split and row_preset(r) == preset]


def choose(rows):
    """Best adopted row by NDCG@10 (ties: fewer added ms, then experiment name). None if none adopted."""
    ok = [r for r in rows if r.get("adopt") and not str(r.get("notes", "")).startswith("reference row")]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (-r["ndcg@10"], r.get("added_ms_per_query") or 0.0, r["exp"]))[0]


def apply_patch(cfg, patch):
    cfg = merge_config(cfg)
    for k, v in patch.items():
        if isinstance(cfg[k], dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return merge_config(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["variants", "hybrid", "rerank", "rejudge"])
    ap.add_argument("--preset", default="f2llm-v2-0.6b",
                    help="which first-stage preset's rows to select from (each preset's rows and chosen "
                         "config are kept separate so e.g. a 1.7B sweep can't overwrite 0.6B's)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    chosen_path = CHOSEN if a.preset == "f2llm-v2-0.6b" else CHOSEN.with_name(f"chosen_{a.preset}.json")
    cfg = json.loads(chosen_path.read_text()) if chosen_path.exists() else {}
    if a.group:
        best = choose(load_rows(a.group, a.preset))
        if best is None:
            print(f"[select] no '{a.group}' row for preset {a.preset!r} passed the adoption rule; "
                  "stage stays OFF / unchanged.")
        else:
            cfg = apply_patch(cfg, best["config_patch"])
            print(f"[select] adopted '{best['exp']}' (preset {a.preset}): NDCG@10 {best['ndcg@10']:.4f}, "
                  f"dNDCG {best['delta_ndcg@10']:+.4f} [{best['ci_low']:+.4f},{best['ci_high']:+.4f}], "
                  f"improved/worsened {best['improved']}/{best['worsened']}")
            chosen_path.parent.mkdir(parents=True, exist_ok=True)
            chosen_path.write_text(json.dumps(cfg, indent=2))
    print(f"[select] chosen config ({chosen_path}):\n" + json.dumps(merge_config(cfg), indent=2))
    print(f"(published F2LLM test NDCG@10 for reference: {TEST_NDCG_REFERENCE.get(a.preset, 'unknown')})")


if __name__ == "__main__":
    main()
