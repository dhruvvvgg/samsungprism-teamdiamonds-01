"""Minimal FastAPI service over the same SearchService the CLI uses.

    uvicorn src.api:app --host 0.0.0.0 --port 8000
    curl localhost:8000/health
    curl -s localhost:8000/search -H 'Content-Type: application/json' \
         -d '{"query": "count the primes below n", "k": 5}'

Configuration is environment-driven so the Docker image needs no arguments:

    INDEX_DIR         index name (full / lite / versions) or path; default: full
    SEARCH_DEVICE     cpu (default) or cuda
    SEARCH_THREADS    CPU threads (default: physical cores)
    CPU_DTYPE         fp32 (default) or bf16
    INT8              rejected and disabled (fails closed due to severe quality degradation)
    MAX_QUERY_TOKENS  cap the query length (default 1024, the measured serving cap;
                      0 = uncapped). Documents are never truncated
    QUERY_CACHE       0 to turn off the exact-query embedding cache (default on, 128 entries;
                      QUERY_CACHE_SIZE changes the size). Serving only: benchmarks never cache
    MOCK_ENCODER      1 to use the hashing encoder (smoke tests / CI only)

The model is loaded ONCE, at startup (see `lifespan`), so no user request ever pays for the load. A
load failure is recorded rather than raised, so the process still starts and `/health` can say what
went wrong instead of the container dying with a traceback nobody reads.
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - the CLI path does not need fastapi
    raise SystemExit("The API needs fastapi and uvicorn:  pip install fastapi uvicorn") from exc

_service = None
_error = None


@asynccontextmanager
async def lifespan(app):
    """Load the model ONCE, at startup, before the first request.

    Lazily loading on the first request would make that request pay the model load (~16 s for the 1.7B
    on CPU) -- which, in a demo, is the request someone is watching. Loading here means the server is
    slow to *start* and fast to *answer*, which is the right way round.

    A failure is recorded rather than raised: the process still starts and /health explains what is
    wrong, instead of a container that exits with a traceback nobody sees."""
    try:
        get_service()
    except Exception as exc:  # noqa: BLE001  reported through /health
        globals()["_error"] = globals()["_error"] or f"{type(exc).__name__}: {exc}"
        if os.environ.get("FAIL_CLOSED_STARTUP") == "1":
            raise RuntimeError(f"Service startup failed: {_error}. Use --mock only for smoke tests.") from exc
    yield


app = FastAPI(title="Agentic Code Intelligence - APPS retrieval",
              description="Dense code retrieval over a prebuilt runtime index.", version="1.0",
              lifespan=lifespan)


_service = None
_error = None
_current_index = None
_services = {}          # index name/path -> SearchService


def default_index():
    return os.environ.get("INDEX_DIR", "lite")


def _release_current_service():
    global _service, _current_index
    if _service is not None:
        try:
            if hasattr(_service, "encoder"):
                if hasattr(_service.encoder, "model"):
                    del _service.encoder.model
                del _service.encoder
            if hasattr(_service, "index"):
                del _service.index
            if hasattr(_service, "query_cache"):
                del _service.query_cache
        except Exception:
            pass
        _service = None
    _services.clear()
    _current_index = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _create_service(index_name_or_path):
    from src.query_cache import resolve_cache_size
    from src.runtime_index import resolve_query_cap
    from src.search_service import SearchService
    return SearchService(
        index_name_or_path, device=os.environ.get("SEARCH_DEVICE", "cpu"),
        mock=os.environ.get("MOCK_ENCODER") == "1",
        threads=(int(os.environ["SEARCH_THREADS"]) if os.environ.get("SEARCH_THREADS")
                 else None),
        cpu_dtype=os.environ.get("CPU_DTYPE", "fp32"),
        int8=False,
        max_query_tokens=resolve_query_cap(
            int(os.environ["MAX_QUERY_TOKENS"]) if os.environ.get("MAX_QUERY_TOKENS")
            else None),
        query_cache_size=resolve_cache_size())


def service_for(index):
    """A SearchService for `index`.

    Memory guard: Only one model is kept resident in memory. When the user switches index,
    the previously loaded model is released before loading the new one. If loading fails
    (out of memory or any error), it falls back to the lite index so the server does not die."""
    global _service, _current_index, _error
    index = index or default_index()
    allowed = [x.strip() for x in os.environ.get("ALLOWED_INDEXES", "").split(",") if x.strip()]
    if allowed and index not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"index {index!r} is not in ALLOWED_INDEXES ({allowed})")

    from src.runtime_index import resolve_index_dir
    target_dir = resolve_index_dir(index).resolve()
    current_dir = (resolve_index_dir(_current_index).resolve()
                   if (_current_index and _service is not None) else None)

    if current_dir == target_dir and _service is not None:
        return _service

    if not (target_dir / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail=f"No runtime index at {target_dir}")

    # Release previous model before loading new one
    _release_current_service()

    try:
        new_svc = _create_service(index)
        _service = new_svc
        _current_index = index
        _services[index] = new_svc
        _error = None
        return _service
    except Exception as exc:  # noqa: BLE001
        err_msg = f"{type(exc).__name__}: {exc}"
        fallback_index = "lite" if (resolve_index_dir("lite") / "manifest.json").is_file() else default_index()
        if fallback_index == index:
            fallback_index = default_index()
        _release_current_service()
        try:
            fallback_svc = _create_service(fallback_index)
            _service = fallback_svc
            _current_index = fallback_index
            _services[fallback_index] = fallback_svc
            _error = None
        except Exception as fb_exc:
            _error = f"Fallback to {fallback_index} failed: {fb_exc}"
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load index '{index}': {err_msg}. Falling back to lite index.") from exc


def manifest_facts(path):
    """What the manifest says about an index, without loading it: enough for the UI to decide which
    panels apply (version comparison, file paths) before anything is searched."""
    import json

    from src.runtime_index import MANIFEST
    try:
        m = json.loads((Path(path) / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": False}
    files = m.get("files") or {}
    return {"available": True, "manifest_kind": m.get("kind"), "n_docs": m.get("n_docs"),
            "dim": m.get("dim"), "model": m.get("model"),
            "versioned": "versions.json" in files, "chunked": "chunks.json" in files}


def available_indexes():
    """Which named indexes actually exist on disk, so the UI only offers real choices."""
    from src.runtime_index import INDEX_NAMES, MANIFEST
    from src.reindex import reindexable_info
    from src.runtime_index import resolve_index_dir
    out = []
    order = ["lite", "full", "versions", "history"]
    sorted_names = [k for k in order if k in INDEX_NAMES] + [k for k in INDEX_NAMES if k not in order]
    for name in sorted_names:
        path = INDEX_NAMES[name]
        p = Path(path)
        if p.is_dir() and (p / MANIFEST).is_file():
            facts = manifest_facts(p)
            if facts.get("available"):
                target_dir = p.resolve()
                current_dir = (resolve_index_dir(_current_index).resolve()
                               if (_current_index and _service is not None) else None)
                loaded = (current_dir == target_dir) or (name in _services)
                out.append({"name": name, "path": str(p), "loaded": loaded,
                            **reindexable_info(p), **facts})
    extra = default_index()
    if extra not in INDEX_NAMES and extra not in [o["name"] for o in out]:
        extra_dir = resolve_index_dir(extra)
        facts = manifest_facts(extra_dir)
        target_dir = extra_dir.resolve()
        current_dir = (resolve_index_dir(_current_index).resolve()
                       if (_current_index and _service is not None) else None)
        loaded = (current_dir == target_dir) or (extra in _services)
        out.append({"name": extra, "path": str(extra_dir), "loaded": loaded,
                    **reindexable_info(extra_dir),
                    **facts})
    return out


def get_service():
    """The process-wide SearchService, built on first use. A failed load is remembered and reported."""
    global _service, _current_index, _error
    if _service is None and _error is None:
        idx = default_index()
        try:
            _service = _create_service(idx)
            _current_index = idx
            _services[idx] = _service
        except Exception as exc:  # noqa: BLE001  reported through /health and /search
            _error = f"{type(exc).__name__}: {exc}"
    if _error:
        raise HTTPException(status_code=503, detail=f"index/model unavailable -- {_error}")
    return _service


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="the query text")
    k: int = Field(10, ge=1, le=100, description="how many results")
    index: Optional[str] = Field(None, description="index name (full / lite / versions) or path")
    version: Optional[int] = Field(None, description="versioned index: search only this version")
    all_versions: bool = Field(False, description="versioned index: do not collapse lineages")
    preview_chars: int = Field(240, ge=0, le=4000)
    category: Optional[str] = Field(None, description="only results tagged with this family/cluster")
    route: bool = Field(True, description="classify the query and report the route")
    llm_router: bool = Field(False, description="allow the LLM fallback when rules are unsure")
    suggestions: bool = Field(False, description="extra: rule-based performance notes on the results")
    explain: bool = Field(False, description="add a `why` block to each hit (route, score gap, matched "
                                             "terms); lexical only, no extra model call")


def _loaded_health(svc):
    mock = getattr(svc.encoder, "model_name", "") == "mock/hashing-encoder"
    return {"status": "degraded" if mock else "ok", "loaded": True,
            **({"reason": "mock encoder in use"} if mock else {}), "index": svc.describe()}


@app.get("/health")
def health(index: Optional[str] = None):
    """Liveness plus what is loaded. Never raises: reports `loaded: false` and why instead.

    Without `index` this describes the server's default index. With one, it describes THAT index -- but
    only reports on it, never loads it (a status probe must not cost an 11 GB model load), so an index
    that has not been searched yet comes back `loaded: false` with what its manifest says."""
    if index:
        from src.reindex import allowed_index_names
        from src.runtime_index import resolve_index_dir
        if index not in allowed_index_names(default_index(), os.environ):
            raise HTTPException(status_code=400, detail=f"index {index!r} is not one this server serves")
        target_dir = resolve_index_dir(index).resolve()
        current_dir = (resolve_index_dir(_current_index).resolve()
                       if (_current_index and _service is not None) else None)
        if current_dir == target_dir and _service is not None:
            return dict(_loaded_health(_service), index_name=index)
        svc = _services.get(index)
        if svc is not None:
            return dict(_loaded_health(svc), index_name=index)
        facts = manifest_facts(resolve_index_dir(index))
        if not facts.get("available"):
            return {"status": "ok", "loaded": False, "index_name": index, "available": False,
                    "reason": "no built index at that location"}
        return {"status": "ok", "loaded": False, "index_name": index, "available": True,
                "facts": facts, "note": "not loaded yet; it loads on its first search"}
    if _service is None and _error is None:
        return {"status": "ok", "loaded": False,
                "note": "the model loads at startup; if this persists, startup has not finished yet"}
    if _error:
        return {"status": "degraded", "loaded": False, "error": _error}
    return _loaded_health(_service)


@app.post("/search")
def search(req: SearchRequest):
    svc = service_for(req.index) if req.index else get_service()
    t0 = time.time()
    try:
        res = svc.search(req.query, k=req.k, version=req.version, all_versions=req.all_versions,
                         preview_chars=req.preview_chars, category=req.category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    res["total_ms"] = round(1000 * (time.time() - t0), 2)
    res["performance"] = svc.performance_info(req.query)
    if svc.versioned:
        from src.versioning.lineage import group_by_lineage
        res["groups"] = group_by_lineage(res["hits"])       # a view over the ranking, not a re-rank
    if req.route:
        res["route"] = svc.route(req.query, allow_llm=req.llm_router)
    if req.explain:
        svc.explain_hits(req.query, res["hits"], route=res.get("route") or svc.route(req.query),
                         groups=res.get("groups"))
    if req.suggestions:
        from src.indexing.optimizations import analyze_hit
        notes = []
        for h in res["hits"]:
            notes.extend(analyze_hit(h))
        res["suggestions"] = notes
    return res


# :path, because a real-history lineage id is `file/path.py::qualname` and contains slashes --
# the default converter stops at the first one and the route would never match.
@app.get("/history/{snippet_id:path}")
def snippet_history(snippet_id: str, index: Optional[str] = None):
    """Versioned index only: one snippet's versions oldest to newest."""
    svc = service_for(index) if index else get_service()
    try:
        return svc.history(snippet_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/doc")
def get_doc(id: str = Query(..., min_length=1), index: Optional[str] = None):
    """One document in full (the search hits only carry a preview). A query parameter rather than a
    path segment, because ids such as `pkg/mod.py:12-40` contain slashes and colons."""
    svc = service_for(index) if index else get_service()
    try:
        return svc.get_doc(id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@app.get("/versions")
def versions(index: Optional[str] = None):
    """Version numbers a versioned index holds (empty list for one that is not versioned)."""
    svc = service_for(index) if index else get_service()
    return {"versioned": svc.versioned, "versions": svc.version_numbers}


@app.get("/compare")
def compare(q: str = Query(..., min_length=1), a: int = Query(...), b: int = Query(...),
            k: int = Query(10, ge=1, le=100), preview_chars: int = Query(240, ge=0, le=4000),
            index: Optional[str] = None):
    """The same query against versions `a` and `b`: what appeared, disappeared or moved rank."""
    svc = service_for(index) if index else get_service()
    try:
        return svc.compare(q, a, b, k=k, preview_chars=preview_chars)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/diff")
def diff(snippet_id: str = Query(..., min_length=1), a: int = Query(...), b: int = Query(...),
         context: int = Query(3, ge=0, le=50), index: Optional[str] = None):
    """Unified diff of one lineage between versions `a` and `b`."""
    svc = service_for(index) if index else get_service()
    try:
        return svc.diff(snippet_id, a, b, context=context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


class ReindexRequest(BaseModel):
    index: Optional[str] = Field(None, description="a custom-folder or history index; default: the "
                                 "server's default index")
    dry_run: bool = Field(False, description="report what would change without writing anything")


@app.post("/reindex")
def reindex(req: ReindexRequest):
    """Incremental update (P1): re-scan the index's own source folder / repository and re-encode only
    the chunks whose text is new. The source is read from the index manifest, never from the request;
    the official APPS indexes are refused."""
    from src.reindex import ReindexError, check_index_name, check_reindexable
    try:
        name = check_index_name(req.index or default_index(), default_index(), os.environ)
        check_reindexable(name)                   # manifest only: refuses before any model is loaded
        svc = service_for(name)
        report = svc.reindex(dry_run=req.dry_run)
    except ReindexError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    report["index_name"] = name
    return report


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)
    max_steps: int = Field(6, ge=1, le=12)
    index: Optional[str] = None
    llm_planner: bool = Field(False, description="off by default; the planner is deterministic")
    structural_root: Optional[str] = None


def _structural_for(svc, root=None):
    """Parse the folder this index came from, for structural queries and agent reads."""
    from pathlib import Path

    from src.indexing.structural import build_from_folder
    root = root or svc.describe().get("source_root")
    if not root or not Path(root).is_dir():
        return None
    return build_from_folder(root)


@app.post("/agent")
def agent(req: AgentRequest):
    """Plan -> search -> read -> refine, returning the full step trace."""
    from src.agent.code_agent import CodeAgent
    svc = service_for(req.index) if req.index else get_service()
    loop = CodeAgent(svc, structural=_structural_for(svc, req.structural_root),
                     max_steps=req.max_steps, k=req.k, use_llm=req.llm_planner)
    return loop.run(req.question)


@app.get("/route")
def route(q: str, llm: bool = False):
    """How a query would be classified, without running a search."""
    return get_service().route(q, allow_llm=llm)


@app.get("/categories")
def categories(index: Optional[str] = None):
    """Tags available for filtering, with row counts."""
    svc = service_for(index) if index else get_service()
    return svc.categories()


@app.get("/structural")
def structural(intent: str, subject: str, second: Optional[str] = None,
               index: Optional[str] = None, structural_root: Optional[str] = None):
    """Exact structural answers: who_calls, what_calls, where_imported, where_used, call_order."""
    svc = service_for(index) if index else get_service()
    idx = _structural_for(svc, structural_root)
    if idx is None:
        raise HTTPException(status_code=400,
                            detail="this index has no source_root to parse; pass structural_root")
    if intent == "call_order":
        if not second:
            raise HTTPException(status_code=400, detail="call_order needs `second`")
        return {"intent": intent, "results": idx.files_calling_in_order(subject, second)}
    fn = {"who_calls": idx.who_calls, "what_calls": idx.what_calls,
          "where_imported": idx.where_imported, "where_used": idx.where_used}.get(intent)
    if fn is None:
        raise HTTPException(status_code=400, detail=f"unknown intent {intent!r}")
    return {"intent": intent, "subject": subject, "results": fn(subject), "stats": idx.stats()}


@app.get("/indexes")
def indexes():
    """Which indexes this server can serve, and which are already loaded."""
    return {"default": default_index(), "indexes": available_indexes()}


@app.get("/", response_class=HTMLResponse)
def home():
    """The single-page UI. Static HTML with no build step and no network dependency."""
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail=f"UI not installed at {page}")
    return HTMLResponse(page.read_text(encoding="utf-8"))
