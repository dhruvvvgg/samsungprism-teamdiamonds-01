"""Plan -> search -> read -> refine over an index larger than any context window.

A single dense query answers "find me code like this". It cannot answer "how does X get called, and
what does that path do", because the answer is spread across files and is only reachable by following
references. This agent does that: it plans sub-questions, runs the retriever that fits each one, reads
the code it finds by line range, and follows the calls and imports it discovers -- then stops.

It runs with **no LLM at all**. The planner is deterministic: it reads the question with the same router
that routes a plain search, and expands it using what the structural index actually contains. An LLM
planner is available through `llm_client` (mock by default) and is off unless asked for. That ordering
is deliberate -- the deterministic path is the one that has to work, and the LLM is an optimisation on
top of it, so a missing API key can never be the reason a demo fails.

Three stopping conditions, because an agent that does not stop is a bug:

    hard cap          6 steps by default, counted and reported
    loop detection    a repeated (tool, argument) pair is refused and recorded, not re-run
    evidence check    stop as soon as the collected evidence answers the question

Every step is recorded in a trace: what it did, why it chose that, what it found and how long it took.
The trace is the product as much as the answer is -- an agent whose reasoning you cannot inspect is not
something you can trust a result from.
"""
import time

MAX_STEPS = 6
ENOUGH_EVIDENCE = 5


class Step:
    """One action, with the reason it was taken and what it produced."""

    def __init__(self, n, tool, argument, reason):
        self.n, self.tool, self.argument, self.reason = n, tool, argument, reason
        self.results, self.seconds, self.note = [], 0.0, ""

    def to_dict(self):
        return {"step": self.n, "tool": self.tool, "argument": self.argument, "reason": self.reason,
                "n_results": len(self.results), "results": self.results[:10],
                "seconds": round(self.seconds, 3), "note": self.note}


class CodeAgent:
    """Deterministic agent over a SearchService plus an optional StructuralIndex."""

    def __init__(self, service, structural=None, max_steps=MAX_STEPS, k=5, use_llm=False,
                 call=None, read_lines=40):
        self.svc, self.structural = service, structural
        self.max_steps, self.k, self.use_llm, self.call = max_steps, k, use_llm, call
        self.read_lines = read_lines

    # --- planning -----------------------------------------------------------------------------------
    def plan(self, question):
        """Ordered (tool, argument, reason) tuples. Deterministic unless an LLM planner is enabled."""
        from src.retrieval.query_router import classify
        route = classify(question, allow_llm=self.use_llm, call=self.call)
        steps = []
        subject = route.get("subject")
        if route["kind"] == "structural" and self.structural is not None:
            intent = route.get("intent")
            if intent == "call_order" and route.get("first") and route.get("second"):
                steps.append(("call_order", (route["first"], route["second"]),
                              "the question asks about call ordering across files"))
            elif intent and subject:
                steps.append((intent, subject, f"the question asks {intent.replace('_', ' ')}"))
            if subject:
                steps.append(("search", subject,
                              "also retrieve the definition itself, for context around the hits"))
        else:
            steps.append(("search", question, f"router: {route['reason']}"))
        if self.use_llm:
            extra = self._llm_subqueries(question)
            steps.extend(("search", q, "LLM planner sub-query") for q in extra)
        return route, steps

    def _llm_subqueries(self, question):
        """Optional. Any failure returns nothing and the deterministic plan stands."""
        try:
            if self.call is None:
                from src.agent.llm_client import call_llm as call
            else:
                call = self.call
            out = call("Break this code-search question into at most 2 short sub-queries, "
                       f"one per line, no numbering:\n{question}")
            lines = [ln.strip("-* ").strip() for ln in (out or "").split("\n") if ln.strip()]
            return [ln for ln in lines if 3 < len(ln) < 120][:2]
        except Exception:  # noqa: BLE001  the deterministic plan must still run
            return []

    # --- tools --------------------------------------------------------------------------------------
    def _run_tool(self, tool, argument):
        if tool == "search":
            hits = self.svc.search(str(argument), k=self.k)["hits"]
            return [{"kind": "snippet", "location": h.get("location", h["doc_id"]),
                     "qualname": h.get("qualname"), "score": h["score"],
                     "preview": h.get("preview", "")[:400]} for h in hits]
        if self.structural is None:
            return []
        if tool == "who_calls":
            return [{"kind": "caller", "location": f"{e['file']}:{e['line']}",
                     "caller": e.get("caller"), "callee": e["callee_text"],
                     "ambiguous": e.get("ambiguous", False)}
                    for e in self.structural.who_calls(str(argument))]
        if tool == "what_calls":
            return [{"kind": "callee", "location": f"{e['file']}:{e['line']}",
                     "callee": e["callee_text"], "resolved": "callee_key" in e}
                    for e in self.structural.what_calls(str(argument))]
        if tool == "where_imported":
            return [{"kind": "import", "location": f"{i['file']}:{i['line']}",
                     "module": i["module"], "name": i["name"], "alias": i["alias"]}
                    for i in self.structural.where_imported(str(argument))]
        if tool == "where_used":
            return [{"kind": u["kind"], "location": f"{u['file']}:{u['line']}",
                     "value": u["value"], "scope": u.get("scope")}
                    for u in self.structural.where_used(str(argument))]
        if tool == "call_order":
            first, second = argument
            return [{"kind": "call_order", "location": r["file"],
                     "first_line": r["first_line"], "second_line": r["second_line"]}
                    for r in self.structural.files_calling_in_order(first, second)]
        if tool == "read":
            return self._read(argument)
        return []

    def _read(self, location):
        """Open a file slice by line range: `path.py:12-40`, or `path.py:12` for a window around it."""
        if self.structural is None or not self.structural.root:
            return []
        from pathlib import Path
        text, _, span = str(location).partition(":")
        path = Path(self.structural.root) / text
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").split("\n")
        except (OSError, UnicodeDecodeError):
            return []
        if "-" in span:
            start, end = (int(x) for x in span.split("-", 1))
        elif span.isdigit():
            start = max(1, int(span) - self.read_lines // 2)
            end = start + self.read_lines
        else:
            start, end = 1, min(len(lines), self.read_lines)
        start, end = max(1, start), min(len(lines), max(start, end))
        return [{"kind": "source", "location": f"{text}:{start}-{end}",
                 "preview": "\n".join(lines[start - 1:end])[:1500]}]

    # --- refinement ---------------------------------------------------------------------------------
    def refine(self, question, evidence, done):
        """What to do next, given what has been found. Returns (tool, argument, reason) or None.

        Following what was actually discovered is what makes this an agent rather than a fan-out: the
        next step is chosen from the results of the previous one."""
        for item in evidence:
            if item["kind"] in ("caller", "callee") and item.get("location"):
                key = ("read", item["location"])
                if key not in done:
                    return ("read", item["location"],
                            f"read the {item['kind']} site found at {item['location']}")
        for item in evidence:
            if item["kind"] == "snippet" and item.get("qualname") and self.structural is not None:
                key = ("who_calls", item["qualname"])
                if key not in done:
                    return ("who_calls", item["qualname"],
                            f"the top result defines {item['qualname']}; find out who calls it")
        return None

    @staticmethod
    def enough(question, evidence):
        """Stop when the evidence answers the question rather than burning the remaining budget."""
        located = [e for e in evidence if e.get("location")]
        if len(located) >= ENOUGH_EVIDENCE and any(e["kind"] != "snippet" for e in evidence):
            return True, f"{len(located)} located results including structural facts"
        if len(located) >= ENOUGH_EVIDENCE * 2:
            return True, f"{len(located)} located results"
        return False, ""

    # --- the loop -------------------------------------------------------------------------------------
    def run(self, question):
        t0 = time.time()
        route, planned = self.plan(question)
        trace, evidence, done, stop_reason = [], [], set(), "plan exhausted"
        queue = list(planned)
        n = 0
        while n < self.max_steps:
            if queue:
                tool, argument, reason = queue.pop(0)
            else:
                nxt = self.refine(question, evidence, done)
                if nxt is None:
                    stop_reason = "nothing further to follow"
                    break
                tool, argument, reason = nxt
            key = (tool, str(argument))
            if key in done:
                step = Step(n + 1, tool, argument, reason)
                step.note = "skipped: this exact step already ran (loop detection)"
                trace.append(step.to_dict())
                continue                       # a refused repeat costs no budget
            done.add(key)
            n += 1
            step = Step(n, tool, argument, reason)
            t1 = time.time()
            try:
                step.results = self._run_tool(tool, argument)
            except Exception as exc:  # noqa: BLE001  one failing tool must not kill the run
                step.note = f"tool failed: {type(exc).__name__}: {exc}"
            step.seconds = time.time() - t1
            evidence.extend(step.results)
            trace.append(step.to_dict())
            ok, why = self.enough(question, evidence)
            if ok:
                stop_reason = f"evidence sufficient: {why}"
                break
        else:
            stop_reason = f"hit the {self.max_steps}-step cap"

        located = []
        seen = set()
        for item in evidence:
            loc = item.get("location")
            if loc and loc not in seen:
                seen.add(loc)
                located.append(item)
        return {"question": question, "route": route, "steps_run": n, "max_steps": self.max_steps,
                "stop_reason": stop_reason, "trace": trace, "answers": located[:20],
                "n_evidence": len(evidence), "seconds": round(time.time() - t0, 3),
                "used_llm": bool(self.use_llm)}
