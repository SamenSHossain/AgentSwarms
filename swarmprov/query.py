"""Ask a run about one answer, one fact or one post (``swarmprov trace``), or one agent (``swarmprov agent``).

Everything is read from the stage tables.  The first public copy is the post
exposure found; the relay parent is the graph's inference (the latest earlier
carrier on the same page).  Neither says what an agent read, so the two are kept
apart, and every hop says when its direction or source is uncertain.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis.agents import AGENT_COLS, match_agents
from .claims import PRIVATE_KINDS, public, value_key
from .exposure import match_key
from .graph import CLOCK_RES_S, KINDS
from .rules import norm_value
from .schema import RunDir, empty

MAPPING_NOTE = {"merged": "one agent per cohort date and task family; --mapping strict for one per author string",
                "strict": "one agent per case-folded author string; --mapping merged joins a cohort's handles"}
RELAY_KINDS = ("relay", "relay_xchannel")
KIND_NOTE = {"relay_xchannel": "; cross-page hops: source unknown, attributed to the originator",
             "reply": "; interactions, not knowledge transfers"}
EXCERPT = 160


def excerpt(text, around: str | None = None, width: int = EXCERPT) -> str:
    """One line of plain text cut to ``width`` characters around the first ``around``.
    Agent text is untrusted (the wiki holds a script payload): escape it before any HTML use."""
    if not isinstance(text, str):
        return ""
    t = " ".join(re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text).split())
    i = t.lower().find(around.lower()) if around else -1
    start = max(0, i - width // 4) if i > 0 else 0
    return ("…" if start else "") + t[start:start + width] + ("…" if start + width < len(t) else "")


def _is(s: pd.Series, v: str) -> pd.Series:
    return s.astype(str).str.casefold() == str(v).casefold()


def _table(run: RunDir, name: str) -> pd.DataFrame:
    """A stage table; one a run wrote without rows or columns (no task families) gets its schema columns."""
    df = run.read(name)
    return df if len(df.columns) else empty(name.rsplit("_", 1)[0])


def _answers(ex: pd.DataFrame, agents=None, item=None, family=None) -> pd.DataFrame:
    sel = ex[ex["agent"].isin(agents)] if agents is not None else ex
    if item is not None:
        sel = sel[_is(sel["item"], item)]
    return sel[_is(sel["family"], family)] if family is not None else sel


def _events(run: RunDir) -> pd.DataFrame:
    return run.read("events").drop_duplicates("event_id").set_index("event_id")


def _post(ev: pd.DataFrame, event_id, around: str | None = None) -> dict | None:
    if not isinstance(event_id, str) or event_id not in ev.index:
        return None
    e = ev.loc[event_id]
    return {"event_id": event_id, "ts": e["ts"], "channel": e["channel"], "author": e["author_raw"],
            "excerpt": excerpt(e["text"], around)}


def _variants(c: pd.DataFrame) -> list[dict]:
    c = c.sort_values(["n_agents", "t0"], ascending=[False, True])
    return [{"value": r.value_key, "carriers": int(r.n_agents), "max_depth": int(r.max_depth),
             "originator": r.originator, "t0": r.t0} for r in c.itertuples(index=False)]


def _tree(edges: pd.DataFrame, ev: pd.DataFrame, origin: str, around: str) -> list[dict]:
    """A fact's relay hops depth-first from the originator, children in time order."""
    hops = []
    for e in edges.itertuples(index=False):
        dst = ev.loc[e.event_id] if e.event_id in ev.index else None
        ts = dst["ts"] if dst is not None else pd.NaT
        hops.append({"src": e.src, "dst": e.dst, "kind": e.kind, "ts": ts,
                     "src_ts": ts - pd.Timedelta(seconds=e.dt_s) if pd.notna(ts) else pd.NaT, "dt_s": e.dt_s,
                     "within_clock_res": bool(e.within_clock_res), "channel": dst["channel"] if dst is not None else None,
                     "event_id": e.event_id, "src_event_id": e.src_event_id,
                     "excerpt": excerpt(dst["text"], around) if dst is not None else ""})
    hops.sort(key=lambda h: (h["ts"], h["event_id"]))
    children: dict[str, list[dict]] = {}
    for h in hops:
        children.setdefault(h["src"], []).append(h)
    out, stack = [], [(h, 1) for h in reversed(children.get(origin, []))]
    while stack:
        h, d = stack.pop()
        out.append({"depth": d, **h})
        stack.extend((c, d + 1) for c in reversed(children.get(h["dst"], [])))
    return out


def fact_lineage(run: RunDir, family: str, item: str, value: str | None = None, mapping: str = "merged") -> list[dict]:
    """Origin and relay hops of a fact (family, item, value), or of every value of the item.
    Values match on value_key, so '9.90', '9.9' and '9.9%' are one fact."""
    chains = _table(run, f"chains_{mapping}")
    c = chains[_is(chains["family"], family) & _is(chains["item"], item)]
    if value is not None:
        keys = {str(value).strip(), value_key(norm_value(str(value))), value_key(str(value).strip())}
        c = c[c["value_key"].isin(keys)]
    if not len(c):
        return []
    edges, ev = run.read(f"edges_{mapping}"), _events(run)
    mn = public(run.read("mentions")).sort_values(["ts", "event_id"], kind="stable")   # what relay_edges reads, in its order
    out = []
    for f in c.sort_values(["n_agents", "t0"], ascending=[False, True]).itertuples(index=False):
        first = mn[(mn["family"] == f.family) & (mn["item"] == f.item) & (mn["value_key"] == f.value_key)]
        e = edges[edges["kind"].isin(RELAY_KINDS) & (edges["family"] == f.family) & (edges["item"] == f.item)
                  & (edges["value_norm"] == f.value_key)]
        hops = _tree(e, ev, f.originator, f.item)
        out.append({"family": f.family, "item": f.item, "value": f.value_key, "carriers": int(f.n_agents),
                    "max_depth": int(f.max_depth), "span_s": float(f.span_s),
                    "n_relay": sum(h["kind"] == "relay" for h in hops),
                    "n_xchannel": sum(h["kind"] == "relay_xchannel" for h in hops),
                    "n_within_clock_res": sum(h["within_clock_res"] for h in hops),
                    "origin": {"agent": f.originator,
                               **((_post(ev, first["event_id"].iloc[0], f.item) if len(first) else None) or {"ts": f.t0})},
                    "hops": hops})
    return out


def answer_lineage(run: RunDir, claim_id: str | None = None, agent: str | None = None, item: str | None = None,
                   family: str | None = None, mapping: str = "merged") -> list[dict]:
    """Answers picked by claim id, or by agent label (and item / family), or every answer: the
    report post, the exposure flags, the first public copy, the agent's own earlier post, the
    inferred relay parent and path from the originator, and the competing values."""
    col = AGENT_COLS[mapping]
    ex = _table(run, f"exposures_{mapping}")
    if claim_id is not None:
        sel = ex[ex["claim_id"] == claim_id]
        if not len(sel):   # a later report of the same agent-round
            cl = run.read("claims")
            c = cl[(cl["claim_id"] == claim_id) & (cl["event_type"] == "answer")]
            if len(c):
                sel = ex[(ex["agent"] == c[col].iloc[0]) & (ex["family"] == c["family"].iloc[0])
                         & (ex["item"] == c["item"].iloc[0])]
    else:
        sel = _answers(ex, None if agent is None else [agent], item, family)
    if not len(sel):
        return []
    ev, edges, chains = _events(run), run.read(f"edges_{mapping}"), _table(run, f"chains_{mapping}")
    fams = set(sel["family"])
    rel = edges[edges["kind"].isin(RELAY_KINDS) & edges["family"].isin(fams)]
    hops_of = dict(list(rel.groupby(["family", "item", "value_norm"])))
    chains_of = dict(list(chains[chains["family"].isin(fams)].groupby(["family", "item"])))
    variants: dict[tuple, list[dict]] = {}
    out = []
    for r in sel.sort_values("t_report").itertuples(index=False):
        value = None if pd.isna(r.value_norm) else r.value_norm
        target = match_key(value, r.consensus_value)
        parent_of = {e.dst: e for e in hops_of.get((r.family, r.item, target), rel.iloc[:0]).itertuples(index=False)}
        parent = parent_of.get(r.agent)
        ch = chains_of.get((r.family, r.item), chains.iloc[:0])
        if (r.family, r.item) not in variants:
            variants[(r.family, r.item)] = _variants(ch)
        origin = ch.loc[ch["value_key"] == target, "originator"]
        src = _post(ev, r.src_event_id, r.item)
        if src is not None:
            src.update(agent=r.src_agent, head_start_s=float(r.gap_s),
                       deleted_at=r.t_src_deleted if pd.notna(r.t_src_deleted) else None,
                       deleted_before_report=bool(r.src_deleted_before_report))
        rp = None
        if parent is not None:
            carrier, src_post = (ev.loc[i] if i in ev.index else None for i in (parent.event_id, parent.src_event_id))
            parent_ts = src_post["ts"] if src_post is not None else None
            rp = {"agent": parent.src, "kind": parent.kind, "dt_s": float(parent.dt_s),
                  "within_clock_res": bool(parent.within_clock_res), "parent_event_id": parent.src_event_id,
                  "parent_ts": parent_ts, "carrier_event_id": parent.event_id,
                  "carrier_ts": carrier["ts"] if carrier is not None else None,
                  # the parent's post is not before the report (exposure's rule for a public copy): not its source
                  "after_report": bool(parent_ts is not None and parent_ts >= r.t_report)}
        path, a = ([r.agent] if (rp is not None and not rp["after_report"])
                   or (len(origin) and origin.iloc[0] == r.agent) else []), r.agent
        while path and a in parent_of:
            a = parent_of[a].src
            path.insert(0, a)
        out.append({
            "claim_id": r.claim_id, "agent": r.agent, "family": r.family, "item": r.item, "episode": r.episode,
            "value": value, "matched_value": target, "consensus_value": r.consensus_value,
            "report": _post(ev, r.report_event_id, r.item) or {"ts": r.t_report},
            "D": int(r.D), "D_w60m": None if pd.isna(r.D_w60m) else int(r.D_w60m), "D_visible": int(r.D_visible),
            "first_public_copy": src,
            "first_visible_copy": _post(ev, r.src_visible_event_id, r.item)
            if r.src_visible_event_id != r.src_event_id else None,
            "own_earlier_post": _post(ev, r.self_event_id, r.item),
            # framework traces: the agent's own tool output, and the task message, that carried the value first
            "own_lookup": _post(ev, getattr(r, "own_lookup_event_id", None), r.item),
            "own_lookup_evidence": int(getattr(r, "own_lookup_evidence", 0) or 0),
            "given_in_task": _post(ev, getattr(r, "task_event_id", None), r.item),
            "relay_parent": rp, "path": path,
            "variants": variants[(r.family, r.item)],
        })
    return out


def claim_note(run: RunDir, claim_id: str, shown: list[dict], mapping: str = "merged") -> str | None:
    """What ``--answer CLAIM_ID`` showed instead, when the claim is not itself an answer row."""
    cl = run.read("claims")
    c = cl[cl["claim_id"] == claim_id]
    if not len(c) or (shown and shown[0]["claim_id"] == claim_id):
        return None
    c = c.iloc[0]
    if shown:
        return f"claim {claim_id} is a later report of this agent-round; shown from its first report {shown[0]['claim_id']}"
    if c["event_type"] == "answer":
        return f"claim {claim_id} is an answer report with no task family or item, so exposure has no row for it"
    ex = _table(run, f"exposures_{mapping}")
    own = ex.loc[(ex["agent"] == c[AGENT_COLS[mapping]]) & (ex["family"] == c["family"]) & (ex["item"] == c["item"]),
                 "claim_id"]
    return (f"claim {claim_id} is a {c['event_type']} (post {c['event_id']}); " +
            (f"the agent's answer for this item is {own.iloc[0]} (trace --answer {own.iloc[0]})" if len(own)
             else "the agent has no answer for this item"))


def event_view(run: RunDir, event_id: str, mapping: str = "merged") -> dict | None:
    """One post: its author under both mappings, family, tags, claims, mentions and the edges
    and exposures that point at it.  Answers 'why did this post produce no claim?'."""
    ev = run.read("events")
    e = ev[ev["event_id"] == event_id]
    if not len(e):
        return None
    e = e.iloc[0]
    ag = run.read("agents").set_index("author_raw")
    who = ag.loc[e["author_raw"]] if e["author_raw"] in ag.index else None
    tags = run.read("tags")
    t = tags[tags["event_id"] == event_id]
    flags = [c[3:] for c in tags.columns if c.startswith("is_") and len(t) and bool(t[c].iloc[0])]
    cl = run.read("claims")
    cl = cl[cl["event_id"] == event_id]
    mn = run.read("mentions")
    mn = mn[mn["event_id"] == event_id]
    urls, mn = mn.loc[mn["family"] == "url", "value_norm"], mn[mn["family"] != "url"]
    edges = run.read(f"edges_{mapping}")
    ed = edges[(edges["event_id"] == event_id) | (edges["src_event_id"] == event_id)]
    ex = _table(run, f"exposures_{mapping}")
    fam = e["family"] if isinstance(e.get("family"), str) else ""
    note, kind = None, e.get("source_kind")
    if kind in PRIVATE_KINDS:
        note = ("no claim: " + ("a task message (system or user prompt)" if kind == "task" else
                                "a tool result (the calling agent's private read)") +
                " is not an agent's post; exposure reads its mentions as " +
                ("the value given in the task" if kind == "task" else "the agent's own lookup"))
    elif not len(cl):
        note = ("no claim: no answer report ('answered …', 'Answer: item = value'), prediction or round "
                "sequence matched" + ("" if fam else ", and the post has no task family") +
                (f"; it still carries {len(mn)} item-value mention(s), which exposure and relays use" if len(mn) else ""))
    return {
        "event_id": event_id, "ts": e["ts"], "channel": e["channel"], "author": e["author_raw"], "source_kind": kind,
        "agent_merged": who["agent_merged"] if who is not None else None,
        "agent_strict": who["agent_strict"] if who is not None else None,
        "family": fam, "visible_until": e.get("visible_until"),
        "tags": flags, "techniques": t["techniques"].iloc[0] if len(t) else "", "cites": t["cites"].iloc[0] if len(t) else "",
        "claims": cl[["claim_id", "event_type", "item", "episode", "value_norm", "latency_class", "wrong_flag"]]
        .to_dict("records"),
        "mentions": mn[["family", "item", "value_norm", "value_key"]].to_dict("records"), "urls": list(urls),
        "edges": ed[["kind", "src", "dst", "family", "item", "value_norm", "event_id", "src_event_id"]].to_dict("records"),
        "reported_answers": ex.loc[ex["report_event_id"] == event_id, ["claim_id", "agent", "item"]].to_dict("records"),
        "first_copy_for": ex.loc[ex["src_event_id"] == event_id, ["claim_id", "agent", "item"]].to_dict("records"),
        "no_claim_reason": note, "excerpt": excerpt(e["text"], width=600),
    }


def list_facts(run: RunDir, mapping: str = "merged", family: str | None = None, top: int = 25) -> dict:
    """Facts ranked by carriers.  URL facts only when ``family`` is 'url'."""
    ch = _table(run, f"chains_{mapping}")
    c = ch[_is(ch["family"], family)] if family else ch[ch["family"] != "url"]
    c = c.sort_values(["n_agents", "max_depth", "t0"], ascending=[False, False, True])
    return {"n_facts": int(len(c)), "n_repeated": int((c["n_agents"] > 1).sum()),
            "facts": [{"family": r.family, "item": r.item, "value": r.value_key, "carriers": int(r.n_agents),
                       "max_depth": int(r.max_depth), "originator": r.originator, "t0": r.t0}
                      for r in c.head(top).itertuples(index=False)]}


def agent_view(run: RunDir, label: str, mapping: str = "merged") -> dict:
    """One agent: its profile (``agent_profiles_*``, None for an author with no public post), its author strings,
    its answers with their first public source, and the agents its edges link it to, by kind: ``learned_from``
    (edges into it) and ``supplied`` (edges out of it)."""
    prof, ag = run.read(f"agent_profiles_{mapping}"), run.read("agents")
    p = prof[prof["label"] == label]
    ex, edges = _table(run, f"exposures_{mapping}"), run.read(f"edges_{mapping}")
    answers = [{"claim_id": r.claim_id, "family": r.family, "item": r.item,
                "value": None if pd.isna(r.value_norm) else r.value_norm,
                "matched_value": match_key(None if pd.isna(r.value_norm) else r.value_norm, r.consensus_value),
                "D": int(r.D), "first_public_source": r.src_agent if r.D == 1 else None,
                "head_start_s": (r.t_report - r.t_public).total_seconds() if r.D == 1 else None,
                "t_report": r.t_report, "report_event_id": r.report_event_id}
               for r in ex[ex["agent"] == label].sort_values(["t_report", "claim_id"]).itertuples(index=False)]

    def by_kind(e: pd.DataFrame, other: str) -> dict:
        return {k: [{"agent": a, "edges": int(n)} for a, n in sorted(g[other].value_counts().items(),
                                                                       key=lambda x: (-x[1], x[0]))]
                for k in KINDS if len(g := e[e["kind"] == k])}
    return {"label": label, "author_strings": sorted(ag.loc[ag[AGENT_COLS[mapping]] == label, "author_raw"]),
            "profile": {k: None if pd.isna(x) else x for k, x in p.iloc[0].items()} if len(p) else None,
            "answers": answers,
            "learned_from": by_kind(edges[edges["dst"] == label], "src"),
            "supplied": by_kind(edges[edges["src"] == label], "dst")}


# --- text and figure ---------------------------------------------------------------------------

def _ts(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") if t is not None and pd.notna(t) else "?"


def _dur(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "?"
    a = abs(s)
    return f"{s:.0f} s" if a < 120 else (f"{s / 60:.1f} min" if a < 7200 else f"{s / 3600:.1f} h")


def _n(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def _gap(s, seq: bool = False) -> str:
    """A time difference; under a sequence clock (positions 1 s apart) a count of messages, never seconds."""
    return _n(int(round(s)), "message") if seq and s is not None and not math.isnan(s) else _dur(s)


def _flags(kind: str, within: bool, more=()) -> str:
    f = (["cross-page hop: source unknown, attributed to originator"] if kind == "relay_xchannel" else []) + \
        ([f"direction unreliable: within the {CLOCK_RES_S:.0f} s clock resolution"] if within else []) + list(more)
    return f"  ({'; '.join(f)})" if f else ""


def _post_line(p: dict) -> str:
    return f"{p['event_id']}, {_ts(p['ts'])}, {p['channel']}, by {p['author']}"


def text_answers(answers: list[dict], seq: bool = False) -> list[str]:
    """``seq``: the run has a sequence clock, so nothing is said in minutes or seconds."""
    L = []
    for a in answers:
        own = a["value"] is not None
        val = a["value"] if own else (f"(no value; matched on the consensus {a['matched_value']})" if a["matched_value"]
                                      else "(no value, and no consensus value to match)")
        L.append(f"answer {a['claim_id']}: {a['agent']}, {a['family']} | {a['item']} = {val}")
        rep = a["report"]
        L.append("  report post: " + (_post_line(rep) if "event_id" in rep else _ts(rep["ts"])))
        if rep.get("excerpt"):
            L.append(f"    \"{rep['excerpt']}\"")
        yn = lambda v: "yes" if v else "no"
        w60 = "needs a wall clock" if a["D_w60m"] is None else yn(a["D_w60m"])
        L.append(f"  public before the report: {yn(a['D'])} (D); at least 60 min before: {w60} (D_w60m); "
                 f"a copy still visible at the report: {yn(a['D_visible'])} (D_visible)")
        s = a["first_public_copy"]
        if s:
            gone = (f"; page deleted {_ts(s['deleted_at'])} ({'before' if s['deleted_before_report'] else 'after'} "
                    "the report)") if s["deleted_at"] is not None else ""
            head = (f"head start needs a wall clock ({_gap((rep['ts'] - s['ts']).total_seconds(), True)} earlier in the "
                    "file)" if seq else f"head start {_dur(s['head_start_s'])}")
            L.append(f"  first public copy: {s['agent']}, post {_post_line(s)}; {head}{gone}")
            L.append(f"    \"{s['excerpt']}\"")
            v = a["first_visible_copy"]
            if v:
                L.append(f"  first copy still visible at the report: {_post_line(v)}")
        else:
            L.append("  first public copy: none before the report")
        p = a["own_earlier_post"]
        L.append(f"  own earlier post: {_post_line(p)}" if p else "  own earlier post: none")
        for key, what in (("own_lookup", "own tool output before the report"), ("given_in_task", "value given in the task")):
            p = a.get(key)
            if p:
                note = ("" if key != "own_lookup" else "; own lookup preceded report (own_lookup_evidence)"
                        if a["own_lookup_evidence"] else "; a public copy came first, so no own_lookup_evidence")
                L.append(f"  {what}: post {_post_line(p)}{note}")
                L.append(f"    \"{p['excerpt']}\"")
        rp = a["relay_parent"]
        if rp:
            late = ["parent post not before the report: not this answer's source"] if rp["after_report"] else []
            L.append(f"  inferred relay parent: {rp['agent']} ({rp['kind']}), parent post {rp['parent_event_id']} "
                     f"({_ts(rp['parent_ts'])}), {_gap(rp['dt_s'], seq)} before this agent's first post of the value "
                     f"({rp['carrier_event_id']}, {_ts(rp['carrier_ts'])}){_flags(rp['kind'], rp['within_clock_res'], late)}")
        elif a["path"]:
            L.append("  inferred relay parent: none (this agent originated the value)")
        elif not own:
            L.append("  inferred relay parent: none (" + ("no value to match" if not a["matched_value"] else
                                                         "matched on the consensus; no value of its own, and no post "
                                                         "of the consensus value") + ")")
        else:
            L.append("  inferred relay parent: none (no item-value mention of this value by the agent, "
                     "so the relay graph has no hop for it)")
        if len(a["path"]) > 1:
            L.append(f"  inferred path from the originator ({_n(len(a['path']) - 1, 'hop')}): " + " -> ".join(a["path"]))
        if len(a["variants"]) > 1:
            L.append("  competing values: " + ", ".join(
                f"{v['value']} ({_n(v['carriers'], 'carrier')})"
                + ((" <- this answer" if own else " <- matched (consensus)") if v["value"] == a["matched_value"] else "")
                for v in a["variants"]))
    return L


def text_facts(facts: list[dict], limit: int | None = None, seq: bool = False) -> list[str]:
    L = []
    for f in facts:
        L.append(f"fact {f['family']} | {f['item']} | {f['value']}: {_n(f['carriers'], 'carrier')}, {_n(len(f['hops']), 'hop')} "
                 f"({f['n_relay']} same-page relay, {f['n_xchannel']} cross-page attributed to the originator"
                 + ("" if seq else f", {f['n_within_clock_res']} within the clock resolution") + f"), max depth {f['max_depth']}, "
                 f"span {_gap(f['span_s'], seq)}")
        o = f["origin"]
        L.append(f"  origin: {o['agent']}" + (f", post {_post_line(o)}" if "event_id" in o else f", {_ts(o['ts'])}"))
        if o.get("excerpt"):
            L.append(f"    \"{o['excerpt']}\"")
        for h in f["hops"][:limit]:
            pad = "  " * (h["depth"] + 1)
            L.append(f"{pad}[{h['depth']}] {h['src']} -> {h['dst']}  {h['kind']}  {_ts(h['ts'])}  +{_gap(h['dt_s'], seq)}  "
                     f"{h['channel']}  post {h['event_id']}{_flags(h['kind'], h['within_clock_res'])}")
            L.append(f"{pad}    \"{h['excerpt']}\"")
        if limit is not None and len(f["hops"]) > limit:
            L.append(f"  ... {_n(len(f['hops']) - limit, 'more hop')} (--limit)")
    return L


def text_event(v: dict) -> list[str]:
    kind = {"task": " (task message)", "tool_result": " (tool result)"}.get(v.get("source_kind"), "")
    L = [f"post {v['event_id']}, {_ts(v['ts'])}, {v['channel']}, by {v['author']}{kind}",
         f"  agent: merged {v['agent_merged']}, strict {v['agent_strict']}; family: {v['family'] or '(none)'}"]
    if v["visible_until"] is not None and pd.notna(v["visible_until"]):
        L.append(f"  page deleted {_ts(v['visible_until'])}")
    L.append(f"  tags: {', '.join(v['tags']) or '(none)'}" + (f"; techniques: {v['techniques']}" if v["techniques"] else "")
             + (f"; cites: {v['cites']}" if v["cites"] else ""))
    L.append(f"  claims ({len(v['claims'])}):")
    L += [f"    {c['event_type']}: {c['item']} = {c['value_norm']}"
          + (f", round {c['episode']:.0f}" if c["episode"] is not None and pd.notna(c["episode"]) else "")
          + (f", latency {c['latency_class']}" if isinstance(c["latency_class"], str) else "") for c in v["claims"]]
    if v["no_claim_reason"]:
        L.append(f"    {v['no_claim_reason']}")
    L.append(f"  item-value mentions ({len(v['mentions'])}): " + (", ".join(
        f"{m['item']} {m['value_norm']}" for m in v["mentions"]) or "(none)"))
    if v["urls"]:
        L.append(f"  URLs ({len(v['urls'])}): " + ", ".join(excerpt(u, width=100) for u in v["urls"][:10])
                 + (" ..." if len(v["urls"]) > 10 else ""))
    if v["reported_answers"]:
        L.append("  report post of: " + ", ".join(f"{r['agent']} {r['item']} ({r['claim_id']})" for r in v["reported_answers"]))
    if v["first_copy_for"]:
        L.append(f"  first public copy for {len(v['first_copy_for'])} answers: " + ", ".join(
            f"{r['agent']} {r['item']}" for r in v["first_copy_for"][:10]) + (" ..." if len(v["first_copy_for"]) > 10 else ""))
    L.append(f"  edges at this post ({len(v['edges'])}): " + (", ".join(
        f"{e['kind']} {e['src']} -> {e['dst']}"
        + (f" ({e['item']} {excerpt(e['value_norm'], width=100)})" if isinstance(e["item"], str) else "")
        for e in v["edges"][:10]) or "(none)") + (" ..." if len(v["edges"]) > 10 else ""))
    L.append(f"  text: \"{v['excerpt']}\"")
    return L


def text_agent(v: dict, seq: bool = False, limit: int = 10) -> list[str]:
    """``limit`` agents per edge kind (JSON keeps all)."""
    p, a = v["profile"], v["author_strings"]
    head = f"agent {v['label']}: {_n(len(a), 'author string')} ({', '.join(a[:5])}{', ...' if len(a) > 5 else ''})"
    if p is None:
        return [head + "; no public post (task messages and tool results are no agent's posts)"]
    L = [head + (f"; cohort {p['cohort']}" if p["cohort"] else "") + (f"; family {p['family']}" if p["family"] else "")
         + (f" (posts in {p['families']})" if p["families"] and p["families"] != p["family"] else ""),
         f"  posts: {p['n_posts']:,}, first {_ts(p['first_seen'])}, last {_ts(p['last_seen'])}"]
    k_n = f" ({p['exposed_k_of_n']})" if p["exposed_k_of_n"] else ""
    L.append(f"  answers: {p['n_answers']}; no earlier public copy {p['n_no_prior_copy']}, exposed {p['n_exposed']}{k_n}, "
             + ("at least 1 h before and head starts need a wall clock" if seq else
                f"at least 1 h before {p['n_exposed_60m']}, median head start "
                + ("-" if pd.isna(p["median_head_h"]) else f"{p['median_head_h']:.1f} h"))
             + f"; self-prepared {p['n_self_prepared']}; wrong {p['n_wrong']}")
    if p["answers_supplied"]:
        L.append(f"  first public source for {_n(p['answers_supplied'], 'answer')} on {_n(p['items_first_sourced'], 'item')}"
                 + (f" (origin of those answers: {p['origin']})" if p["origin"] else ""))
    L.append(f"  same-page relays: {p['relay_in']} in, {p['relay_out']} out; task values posted first: "
             f"{p['facts_originated']}, carried by {p['reach_of_originated']} other agents in all; "
             f"betweenness {p['betweenness']:.4f}" + (f"; techniques: {p['techniques_used']}" if p["techniques_used"] else ""))
    L.append("  exposure is a floor (only captured posts are searched): an answer with no earlier public copy is not "
             "proof of computation")
    L.append(f"answers ({len(v['answers'])}):")
    for x in v["answers"]:
        val = x["value"] if x["value"] is not None else f"(no value; matched on the consensus {x['matched_value']})"
        L.append(f"  {x['family']} | {x['item']} = {val}: " + (
            "no earlier public copy (D=0)" if not x["D"] else
            f"public before the report (D=1); first public source {x['first_public_source']}, head start "
            + (_gap(x["head_start_s"], seq) if seq else _dur(x["head_start_s"]))) + f"  [{x['claim_id']}]")
    for key, what in (("learned_from", "learned from (edges into this agent)"),
                      ("supplied", "supplied (edges out of this agent)")):
        L.append(f"{what}:" + ("" if v[key] else " none"))
        for k, rows in v[key].items():
            more = len(rows) - limit
            L.append(f"  {k} ({_n(sum(r['edges'] for r in rows), 'edge')}, {_n(len(rows), 'agent')}{KIND_NOTE.get(k, '')}): "
                     + ", ".join(f"{r['agent']} {r['edges']}" for r in rows[:limit]) + (f", ... {more} more" if more > 0 else ""))
    return L


def text_list(res: dict, family: str | None) -> list[str]:
    what = f"family {family}" if family else "task values"
    L = [f"{res['n_facts']:,} facts ({what}), {res['n_repeated']:,} carried by more than one agent; top by carriers:"]
    L += [f"  {f['carriers']:>4} carriers  depth {f['max_depth']:>2}  {f['family']} | {f['item']} | {f['value']}  "
          f"(origin {f['originator']}, {_ts(f['t0'])})" for f in res["facts"]]
    return L


def timeline(fact: dict, path, highlight: str | None = None, seq: bool = False) -> str:
    """x = time, y = carriers in order of their first post; relay hops solid blue from the parent's post (small
    dot), cross-page hops grey."""
    from matplotlib.lines import Line2D

    from . import plotting
    from .plotting import plt
    plotting.setup()
    order = [fact["origin"]["agent"]] + [h["dst"] for h in sorted(fact["hops"], key=lambda h: h["ts"])]
    y = {a: i for i, a in enumerate(order)}
    t_first = {fact["origin"]["agent"]: fact["origin"]["ts"], **{h["dst"]: h["ts"] for h in fact["hops"]}}
    fig, ax = plt.subplots(figsize=(9, min(14, 2 + 0.2 * len(order))), layout="constrained")
    for h in fact["hops"]:
        relay = h["kind"] == "relay"
        ax.plot([h["src_ts"], h["ts"]], [y[h["src"]], y[h["dst"]]], lw=2 if relay else 0.8,
                color=plotting.SERIES[0] if relay else plotting.NEUTRAL, zorder=2 if relay else 1)
    rh = [h for h in fact["hops"] if h["kind"] == "relay"]
    ax.scatter([h["src_ts"] for h in rh], [y[h["src"]] for h in rh], s=5, color=plotting.SERIES[0], zorder=2)
    ax.scatter([t_first[a] for a in order], range(len(order)), s=14, color=plotting.INK, zorder=3)
    ax.scatter([t_first[order[0]]], [0], s=40, color=plotting.SERIES[1], zorder=4)
    if highlight in y:
        ax.scatter([t_first[highlight]], [y[highlight]], s=40, color=plotting.SERIES[3], zorder=4)
    step = math.ceil(len(order) / 70)   # about 70 labels fit the tallest figure
    ax.set_yticks(range(0, len(order), step))
    ax.set_yticklabels([a.replace("$", r"\$") for a in order[::step]], fontsize=7)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("message position (2000-01-01 + i s)" if seq else "time (UTC)")
    ax.set_ylabel("carriers, in order of first post")
    ax.set_title(f"{fact['family']} | {fact['item']} | {fact['value']}".replace("$", r"\$")
                 + f": {fact['carriers']} carriers, max depth {fact['max_depth']}")
    handles = [Line2D([], [], color=plotting.SERIES[0], lw=2,
                      label="same-page relay, from the parent's latest earlier post (dot)"),
               Line2D([], [], color=plotting.NEUTRAL, lw=0.8, label="cross-page hop: source unknown, from originator"),
               Line2D([], [], ls="", marker="o", color=plotting.SERIES[1], label="originator")]
    if highlight in y:
        handles.append(Line2D([], [], ls="", marker="o", color=plotting.SERIES[3], label=highlight.replace("$", r"\$")))
    fig.autofmt_xdate()
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    return plotting.save(fig, path)


def to_json(obj):
    """Timestamps as ISO text, NaN/NaT as null, numpy scalars as Python ones."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if obj is None or obj is pd.NaT or (isinstance(obj, float) and math.isnan(obj)):
        return None
    if isinstance(obj, np.generic):
        return to_json(obj.item())
    return obj


def _open(run_path: str, mapping: str, need: tuple[str, ...]) -> tuple[RunDir | None, bool]:
    """The run, once it has every table in ``need`` (else None, with what is missing printed), and whether it has a
    sequence clock; prints the mapping (and clock) lines both commands open with."""
    run = RunDir(run_path) if Path(run_path).is_dir() else None
    missing = [n for n in need if run is None or not run.has(n)]
    if missing:
        print(f"{run_path}: not a finished run, it has no {', '.join(missing)} (run `swarmprov run`, or every stage "
              "up to `swarmprov graph`)")
        return None, False
    print(f"mapping: {mapping} ({MAPPING_NOTE[mapping]})")
    seq = json.loads((run.path / "profile.json").read_text()).get("clock") == "sequence"
    if seq:
        print("clock: message order, no wall clock (a post's time is its position, 2000-01-01 + i s; gaps are in messages)")
    return run, seq


def _one_agent(cands: list[str], name: str) -> str | None:
    if len(cands) != 1:
        print(f"no agent matches {name!r}" if not cands else
              f"{len(cands)} agents match {name!r}; pass one of:\n  " + "\n  ".join(cands[:50]))
        return None
    return cands[0]


def trace(run_path: str, agent=None, item=None, family=None, answer=None, fact=None, event=None,
          mapping: str = "merged", json_path=None, png_path=None, limit: int | None = None) -> int:
    """The ``trace`` command: an answer (agent or claim id), a fact, a post, or else the fact list."""
    run, seq = _open(run_path, mapping, ("events", "agents", "claims", "tags", "mentions", f"exposures_{mapping}",
                                         f"edges_{mapping}", f"chains_{mapping}"))
    if run is None:
        return 1
    if agent is not None or answer is not None:
        ex = _table(run, f"exposures_{mapping}")
        if agent is not None:
            cands = match_agents(run, agent, mapping)
            if len(cands) > 1 and (item is not None or family is not None):   # the ones with a matching answer
                cands = sorted(set(_answers(ex, cands, item, family)["agent"])) or cands
            agent = _one_agent(cands, agent)
            if agent is None:
                return 1
        res = answer_lineage(run, claim_id=answer, agent=agent, item=item, family=family, mapping=mapping)[:limit]
        note = claim_note(run, answer, res, mapping) if answer is not None else None
        if note:
            print(note)
            if not res:
                return 1
        if not res and agent is not None:
            print(f"no answer by {agent} matches; its answers: {', '.join(sorted(ex.loc[ex['agent'] == agent, 'item'])) or 'none'}")
            return 1
        mode, lines = "answer", text_answers(res, seq)
    elif fact is not None:
        parts = [p.strip() for p in fact.split("|", 2)]
        if len(parts) < 2:
            print("--fact takes 'family|item' or 'family|item|value'")
            return 1
        res = fact_lineage(run, parts[0], parts[1], parts[2] if len(parts) == 3 else None, mapping)
        mode, lines = "fact", text_facts(res, limit, seq)
    elif event is not None:
        res = event_view(run, event, mapping)
        mode, lines = "event", text_event(res) if res else []
    else:
        res = list_facts(run, mapping, family, limit or 25)
        mode, lines = "list", text_list(res, family)
    if not res or (mode == "list" and not res["facts"]):
        print("nothing matches")
        return 1
    print("\n".join(lines))
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(to_json({"mapping": mapping, "clock": "sequence" if seq else "wall", "mode": mode, "result": res}),
                      fh, indent=1, default=str)
    if png_path:
        facts, highlight = (res if mode == "fact" else []), None
        if mode == "answer" and len(res) == 1 and res[0]["matched_value"] is not None:
            a = res[0]
            facts, highlight = fact_lineage(run, a["family"], a["item"], a["matched_value"], mapping), a["agent"]
        if len(facts) != 1:
            print("--png draws one fact: pass --fact with a value, or pick a single answer")
            return 1
        print(f"timeline: {timeline(facts[0], png_path, highlight, seq)}")
    return 0


def agent(run_path: str, name: str, mapping: str = "merged", json_path=None) -> int:
    """The ``agent`` command: one agent's profile, answers and edges."""
    run, seq = _open(run_path, mapping, ("events", "agents", f"exposures_{mapping}", f"edges_{mapping}",
                                         f"agent_profiles_{mapping}"))
    label = _one_agent(match_agents(run, name, mapping), name) if run is not None else None
    if label is None:
        return 1
    v = agent_view(run, label, mapping)
    print("\n".join(text_agent(v, seq)))
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(to_json({"mapping": mapping, "clock": "sequence" if seq else "wall", "mode": "agent", "result": v}),
                      fh, indent=1, default=str)
    return 0
