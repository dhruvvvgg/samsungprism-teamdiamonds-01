"""Query tool: one question against a built runtime index, on CPU, with per-stage timing.

    python src/cli.py "count the number of primes below n"
    python src/cli.py "binary search over a sorted array" -k 5 --json

Interactive mode keeps the model warm -- load it once, then ask as many questions as you like. This is
the mode to demo: a cold `src/cli.py` run pays the model load (~16 s for the 1.7B on CPU) every time,
and that cost is per *process*, not per query.

    python src/cli.py --interactive
    > count the primes below n
    > :k 5
    > :quit

Indexes are named: `--index full` (the submitted 1.7B system), `--index lite` (the 0.6B
speed/quality tradeoff), `--index versions` (P1/Bonus), or any path.

Versioned index (P1/Bonus):

    python src/cli.py "sum the scores" --index versions --version 2   # only version 2
    python src/cli.py "sum the scores" --index versions --all-versions
    python src/cli.py --history snip0007 --index versions

CPU by default and physical cores only (hyperthreads make single-query latency worse, not better).
It never encodes the corpus: the index already holds the document embeddings, so a query costs one
forward pass plus one mat-vec.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime_index import DEFAULT_SERVING_QUERY_TOKENS  # noqa: E402

HELP = """commands:
  <text>          search for <text>
  :k N            change the number of results
  :version N      search only version N (versioned index)
  :allversions    let every version compete
  :collapse       collapse lineages again (the default)
  :history ID     show one snippet's versions, oldest first
  :info           index / model / timing info
  :help           this list
  :quit           exit
"""


def add_args(ap):
    ap.add_argument("-k", "--top-k", type=int, default=10)
    ap.add_argument("--index", dest="index_dir", default="full",
                    help="index name (full / lite / versions) or a path")
    # older spelling, kept working; SUPPRESS so it cannot overwrite --index's default with None
    ap.add_argument("--index-dir", dest="index_dir", default=argparse.SUPPRESS,
                    help=argparse.SUPPRESS)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads (default: physical cores)")
    ap.add_argument("--cpu-dtype", default="fp32", choices=["fp32", "bf16"],
                    help="CPU weight precision. bf16 halves weight memory; whether it is faster "
                         "depends on the CPU having native bf16 (measure it)")
    ap.add_argument("--int8", action="store_true",
                    help="dynamic int8 quantisation of Linear layers. OFF by default: it can reorder "
                         "the top-10 for Qwen-based embedders -- run src/check_cpu_precision.py first")
    ap.add_argument("--max-query-tokens", type=int, default=None,
                    help=f"truncate the query at this many tokens "
                         f"(default: {DEFAULT_SERVING_QUERY_TOKENS}, the serving cap measured on dev; "
                         f"0 = uncapped). Documents are never truncated by this")
    ap.add_argument("--preview-chars", type=int, default=240)
    ap.add_argument("--mock-encoder", action="store_true",
                    help="hashing encoder: no model download; for smoke tests only")
    ap.add_argument("--verify-index", action="store_true", help="re-hash the index files on load")
    ap.add_argument("--category", default=None,
                    help="only results tagged with this algorithm family or cluster label")
    ap.add_argument("--structural-root", default=None,
                    help="folder to parse for structural queries and agent reads "
                         "(default: the index manifest's source_root)")
    ap.add_argument("--no-query-cache", action="store_true",
                    help="interactive mode keeps an exact-query embedding cache (repeat queries skip the "
                         "model); this turns it off. Also off with QUERY_CACHE=0. One-shot runs and "
                         "benchmarks never use it")
    ap.add_argument("--no-router", action="store_true",
                    help="skip query classification and always run a plain dense search")
    ap.add_argument("--llm-router", action="store_true",
                    help="allow the LLM fallback when the rules are unsure (mock provider by default)")
    ap.add_argument("--suggestions", action="store_true",
                    help="extra: rule-based performance smells in the results, with file:line")
    return ap


def print_hits(res, info, show_tokens=None):
    if res["version_filter"] is not None:
        print(f"filter  : version {res['version_filter']} only")
    elif res["collapsed_lineages"]:
        print("lineage : collapsed to the best-scoring version per snippet (--all-versions to disable)")
    print()
    for h in res["hits"]:
        ver = f"  v{h['version']}" if "version" in h else ""
        if "location" in h:                       # code-folder index: the location IS the answer
            what = f"{h['kind']} {h['qualname']}" if h.get("qualname") else h["doc_id"]
            print(f"  #{h['rank']:<3} {h['location']}  score {h['score']:.4f}")
            print(f"       ({what})")
        else:
            print(f"  #{h['rank']:<3} {h['doc_id']:<20}{ver}  score {h['score']:.4f}")
        print("       " + h["preview"].replace("\n", "\n       ")
              + (" ..." if h["truncated"] else ""))
        print()
    t = res["timings_ms"]
    tok = "" if show_tokens is None else f" | query {show_tokens} tokens"
    print(f"timing  : query encode {t['encode_query_ms']:.1f} ms (cached: {str(t.get('cached', False)).lower()})"
          f" | search {t['search_ms']:.1f} ms{tok}")


def interactive(svc, args):
    """Read queries until EOF or :quit, with the model already loaded."""
    info = svc.describe()
    print(f"\nindex   : {info['index_dir']}  ({info['n_docs']} docs, dim {info['dim']}, {info['kind']})")
    print(f"model   : {info['model']} on {info['device']}"
          + (f", {info['threads']} threads" if info["threads"] else "")
          + (f", {info['cpu_dtype']}" if info["cpu_dtype"] else "")
          + (" +int8" if info["int8"] else ""))
    print(f"loaded  : model {info['model_load_seconds']:.1f}s + index {info['load_index_seconds']:.1f}s "
          f"-- the model stays warm, so queries below pay only the forward pass")
    print(f"\n{HELP}")
    k, version, all_versions = args.top_k, args.version, args.all_versions
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in (":quit", ":q", ":exit"):
            return 0
        if line in (":help", ":h", "?"):
            print(HELP)
            continue
        if line == ":info":
            print(json.dumps(svc.describe(), indent=2, default=str))
            continue
        if line.startswith(":k "):
            try:
                k = max(1, int(line.split(None, 1)[1]))
                print(f"k = {k}")
            except ValueError:
                print("usage: :k N")
            continue
        if line.startswith(":version "):
            try:
                version, all_versions = int(line.split(None, 1)[1]), False
                print(f"version filter = {version}")
            except ValueError:
                print("usage: :version N")
            continue
        if line == ":allversions":
            all_versions, version = True, None
            print("every version competes")
            continue
        if line == ":collapse":
            all_versions, version = False, None
            print("lineages collapsed")
            continue
        if line.startswith(":history "):
            try:
                out = svc.history(line.split(None, 1)[1], preview_chars=args.preview_chars)
                print_history(out)
            except (KeyError, ValueError) as exc:
                print(f"error: {exc}")
            continue
        if line.startswith(":"):
            print(f"unknown command {line!r}; :help for the list")
            continue
        t0 = time.time()
        try:
            res = svc.search(line, k=k, version=version, all_versions=all_versions,
                             preview_chars=args.preview_chars)
        except ValueError as exc:
            print(f"error: {exc}")
            continue
        print_hits(res, svc.describe(), show_tokens=svc.token_count(line))
        print(f"total   : {1000 * (time.time() - t0):.0f} ms (model already warm)\n")


def structural_root_for(svc, args):
    """Where to parse Python from for a structural query: the flag, else the index's own source_root."""
    root = args.structural_root or svc.describe().get("source_root")
    return Path(root) if root and Path(root).is_dir() else None


def load_structural(svc, args):
    from src.indexing.structural import build_from_folder
    root = structural_root_for(svc, args)
    if root is None:
        return None
    return build_from_folder(root)


def run_structural(svc, args, route):
    """Answer a structural question exactly, or return None so the caller falls back to dense."""
    index = load_structural(svc, args)
    if index is None:
        return None
    intent, subject = route.get("intent"), route.get("subject")
    if intent == "call_order":
        if not (route.get("first") and route.get("second")):
            return None
        return {"intent": intent, "subject": f"{route['first']} before {route['second']}",
                "results": index.files_calling_in_order(route["first"], route["second"])}
    if not subject:
        return None
    fn = {"who_calls": index.who_calls, "what_calls": index.what_calls,
          "where_imported": index.where_imported, "where_used": index.where_used}.get(intent)
    if fn is None:
        return None
    return {"intent": intent, "subject": subject, "results": fn(subject),
            "stats": index.stats()}


def print_structural(route, out):
    print(f"\nroute   : structural ({out['intent']}) -- {route['reason']}")
    print(f"subject : {out['subject']!r}")
    results = out["results"]
    print(f"answer  : {len(results)} exact match(es)\n")
    for r in results[:40]:
        loc = r.get("location") or f"{r.get('file')}:{r.get('line', r.get('first_line', ''))}"
        detail = (r.get("caller") or r.get("callee_text") or r.get("callee")
                  or r.get("value") or r.get("module") or "")
        flag = "  (ambiguous)" if r.get("ambiguous") else ""
        print(f"  {loc:<44} {str(detail)[:60]}{flag}")
    if not results:
        print("  (nothing found -- the name may be spelled differently, or defined outside this tree)")


def collect_suggestions(hits):
    from src.indexing.optimizations import analyze_hit
    out = []
    for h in hits:
        out.extend(analyze_hit(h))
    return out


def print_suggestions(suggestions):
    if not suggestions:
        return
    print(f"\nextras  : {len(suggestions)} performance note(s) in the results above "
          f"(rule-based, not part of retrieval)")
    for s in suggestions[:12]:
        print(f"  [{s['severity']}] {s['location']}  {s['check']}")
        print(f"        {s['detail']}")
        print(f"        -> {s['suggestion']}")


def run_agent(svc, args):
    from src.agent.code_agent import CodeAgent
    agent = CodeAgent(svc, structural=load_structural(svc, args), max_steps=args.max_steps,
                      k=args.top_k, use_llm=args.llm_router)
    return agent.run(args.query)


def print_agent(out):
    print(f"\nquestion: {out['question']!r}")
    print(f"route   : {out['route']['kind']} -> {out['route']['route']}  ({out['route']['reason']})")
    print(f"planner : {'LLM-assisted' if out['used_llm'] else 'deterministic (no LLM)'}")
    print(f"\ntrace   : {out['steps_run']} of at most {out['max_steps']} steps")
    for step in out["trace"]:
        head = f"  {step['step']}. {step['tool']}({step['argument']})"
        print(f"{head:<52} {step['n_results']:>3} result(s)  {step['seconds'] * 1000:6.0f} ms")
        print(f"        why: {step['reason']}")
        if step["note"]:
            print(f"        note: {step['note']}")
        for r in step["results"][:3]:
            print(f"          - {r.get('location', '')}  {str(r.get('qualname') or r.get('callee') or r.get('value') or '')[:50]}")
    print(f"\nstopped : {out['stop_reason']}")
    print(f"evidence: {len(out['answers'])} distinct locations, {out['seconds']:.2f}s total")
    for item in out["answers"][:15]:
        print(f"  {item['kind']:<10} {item.get('location', '')}")


def print_history(out):
    print(f"\nsnippet {out['snippet_id']}: {out['n_versions']} versions, oldest first\n")
    for v in out["versions"]:
        print(f"  v{v['version']}  {v['doc_id']:<18} hash {v['content_hash']}  change: {v['mutation']}")
        print("      " + v["preview"].replace("\n", "\n      "))
        print()


def main():
    ap = argparse.ArgumentParser(description="Search a built runtime index.")
    ap.add_argument("query", nargs="?", default=None, help="the query text (omit with --history)")
    add_args(ap)
    ap.add_argument("--version", type=int, default=None, help="versioned index: search only version N")
    ap.add_argument("--all-versions", action="store_true",
                    help="versioned index: do not collapse a lineage to its best version")
    ap.add_argument("--history", default=None, metavar="SNIPPET_ID",
                    help="versioned index: list one snippet's versions oldest to newest")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="keep the model warm and read queries in a loop")
    ap.add_argument("--agent", action="store_true",
                    help="plan -> search -> read -> refine, printing every step as a trace")
    ap.add_argument("--max-steps", type=int, default=6, help="agent step cap")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()
    if not a.query and not a.history and not a.interactive:
        ap.error("give a query, or --history SNIPPET_ID, or --interactive")

    from src.query_cache import resolve_cache_size
    from src.runtime_index import resolve_query_cap
    from src.search_service import SearchService
    # the cache only pays off in a process that stays alive to see a repeat, so a one-shot run has none
    cache_size = resolve_cache_size() if (a.interactive and not a.no_query_cache) else 0
    svc = SearchService(a.index_dir, device=a.device, mock=a.mock_encoder, threads=a.threads,
                        verify=a.verify_index, cpu_dtype=a.cpu_dtype, int8=a.int8,
                        max_query_tokens=resolve_query_cap(a.max_query_tokens),
                        query_cache_size=cache_size)
    info = svc.describe()
    if info["verify_problems"]:
        print("[cli] WARNING: the index does not match its manifest:", file=sys.stderr)
        for p in info["verify_problems"]:
            print(f"[cli]   {p}", file=sys.stderr)

    if a.interactive:
        return interactive(svc, a)

    if a.history:
        out = svc.history(a.history, preview_chars=a.preview_chars)
        if a.json:
            print(json.dumps({"index": info, "history": out}, indent=2, default=str))
            return 0
        print_history(out)
        return 0

    if a.agent:
        out = run_agent(svc, a)
        if a.json:
            print(json.dumps({"index": info, "agent": out}, indent=2, default=str))
            return 0
        print_agent(out)
        return 0

    route = None if a.no_router else svc.route(a.query, allow_llm=a.llm_router)
    if route and route["route"] == "structural":
        out = run_structural(svc, a, route)
        if out is not None:
            if a.json:
                print(json.dumps({"index": info, "route": route, "structural": out},
                                 indent=2, default=str))
                return 0
            print_structural(route, out)
            return 0

    res = svc.search(a.query, k=a.top_k, version=a.version, all_versions=a.all_versions,
                     preview_chars=a.preview_chars, category=a.category)
    if route:
        res["route"] = route
    if a.suggestions:
        res["suggestions"] = collect_suggestions(res["hits"])
    if a.json:
        print(json.dumps({"index": info, "result": res}, indent=2, default=str))
        return 0

    print(f"\nindex   : {info['index_dir']}  ({info['n_docs']} docs, dim {info['dim']}, "
          f"{info['kind']})")
    threads = "" if info["threads"] is None else f", {info['threads']} threads"
    precision = (f", {info['cpu_dtype']}" if info["cpu_dtype"] else "") + (" +int8" if info["int8"] else "")
    print(f"model   : {info['model']} @ {str(info['revision'])[:12]} on {info['device']}"
          f"{threads}{precision}")
    print(f"query   : {res['query']!r}")
    if route:
        print(f"route   : {route['kind']} -> {route['route']}  ({route['reason']})")
    if res.get("category_filter"):
        print(f"filter  : category {res['category_filter']!r}")
    print_hits(res, info, show_tokens=svc.token_count(a.query))
    print_suggestions(res.get("suggestions") or [])
    print(f"load    : model {info['model_load_seconds'] * 1000:.0f} ms | index "
          f"{info['load_index_seconds'] * 1000:.0f} ms   "
          f"(paid once per process -- use --interactive to amortise it)")
    if info["peak_rss_mb"]:
        print(f"memory  : peak RSS {info['peak_rss_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
