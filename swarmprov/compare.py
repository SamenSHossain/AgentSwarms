"""``swarmprov compare RUN_A RUN_B``: what changed between two runs, two configs or two identity mappings.

Three layers, each keyed by meaning rather than position:

* the setup: adapter, source, config file and every setting that differs, as each run's stages recorded them in
  profile.json (a run recorded before that is read from its config as it reads now, and the report says so),
  capabilities, posts, roster, mapping;
* ``summary.json`` (``summary_strict.json`` under the strict mapping), flattened to one value per path, with the
  report's list rows keyed by what they describe (A3 by technique, A4 by slot, A5 attribution by rule, top
  sources by agent, ...).  Numbers match within a relative tolerance and missing (null) equals missing; the
  largest difference the tolerance hides is reported, so a small genuine shift stays visible;
* the rows behind them, on the stable ids: posts by ``event_id``, claims by ``claim_id``, agent-rounds by
  ``claim_id`` (flag flips, source changes), author strings by ``author_raw``, facts by (family, item, value_key).

Chat event ids hash the file path, so a moved transcript gets new ones.  When fewer than half the posts share an
id, posts are aligned on (time, author, text hash) instead; a key two posts share on either side is a collision
and left unpaired, and the counts say how many were aligned that way.  Comparing the two mappings of one run,
agent labels on the strict side are translated to merged labels (merged is a function of strict) before sources
and originators are compared.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

from .analysis.agents import AGENT_COLS
from .fmt import dumps
from .schema import RunDir, suffix

TOL = 1e-4   # fixed-effects estimates move by up to 6e-5 relative between processes on identical tables
LIMIT = 10
TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")
LOW_OVERLAP = 0.5   # below this share of shared event ids, align posts on (ts, author_raw, text hash)
SETUP = ("adapter", "source", "config_path", "roster", "n_events", "clock")
# summary lists whose rows are keyed by what they describe, not by position
KEYS = {"A2": ("outcome", "treatment"), "A2_dose.rows": ("outcome", "term"), "A2_dose.other_mapping": ("outcome", "term"),
        "A3": ("technique",), "A4.rows": ("slot",), "A5.attribution": ("rule", "answers"),
        "A5.origins.no answer round observed, top sources": ("source",), "agents.top_suppliers": ("label",),
        "agents.top_no_prior_copy": ("label",), "A1.threshold_curve.curves": ("answers",),
        "A6.window_sensitivity": ("W_s",), "A6.url domains.top": ("domain",), "integrity.failing": ("check", "table")}
AGENT_ROWS = {"agents.top_suppliers", "agents.top_no_prior_copy", "A5.origins.no answer round observed, top sources"}
CLAIM_FIELDS = ("item", "value_norm", "episode", "family", "event_type")
FLAGS = ("D", "D_w60m", "D_visible", "D_cons", "y_instant", "y_consensus")
# summary sections left out: the mapping is a setup field, and the key findings restate values compared elsewhere
SKIP = {"mapping", "headline"}


def _flatten(obj, labels: dict | None = None, path: str = "", pattern: str = "", row: str | None = None):
    """(path, value, keyed row the value belongs to) for every leaf; a keyed list row reads ``A3[technique]`` and
    leaves its key fields out, and ``labels`` renames the agents that key a top-sources row (strict labels to merged)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, labels, f"{path}.{k}" if path else str(k), f"{pattern}.{k}" if pattern else str(k), row)
    elif isinstance(obj, list) and pattern in KEYS and obj and all(isinstance(r, dict) for r in obj):
        seen: dict[str, int] = {}
        for r in obj:
            key = " | ".join(str(r.get(k)) for k in KEYS[pattern])
            key = (labels or {}).get(key, key) if pattern in AGENT_ROWS else key
            seen[key] = seen.get(key, 0) + 1
            p = f"{path}[{key}]" + (f"#{seen[key]}" if seen[key] > 1 else "")
            yield from _flatten({k: v for k, v in r.items() if k not in KEYS[pattern]}, labels, p, pattern, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, labels, f"{path}[{i}]", pattern, row)
    else:   # NaN, as summaries written before strict JSON hold it, is missing like null
        yield path, None if isinstance(obj, float) and math.isnan(obj) else obj, row


def _delta(a, b) -> float | None:
    """The relative difference of two numbers (0 when equal), or None when either is not a number."""
    num = (int, float)
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, num) or not isinstance(b, num):
        return None
    if a == b:
        return 0.0
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b) / max(abs(a), abs(b))


def _ts(v):
    """A timestamp string in one spelling (``2026-06-20 05:10:21+00:00`` and ``2026-06-20T05:10:21+00:00`` alike)."""
    if isinstance(v, str) and TS_RE.match(v):
        try:
            return pd.Timestamp(v).isoformat()
        except ValueError:
            pass
    return v


def summary_diff(sa: dict, sb: dict, tol: float = TOL, labels_a: dict | None = None, labels_b: dict | None = None) -> dict:
    """Sections only in one summary, keyed rows only in one, and every shared value that differs (numbers beyond
    ``tol`` relative); ``largest_within_tol`` is the biggest difference the tolerance hid.  ``labels_*`` rename
    one side's agents in the top-sources rows (see ``_flatten``)."""
    out = {"sections_only_a": sorted(set(sa) - set(sb) - SKIP), "sections_only_b": sorted(set(sb) - set(sa) - SKIP),
           "rows_only_a": [], "rows_only_b": [], "changed": [], "n_compared": 0, "largest_within_tol": None}
    shared = sorted(set(sa) & set(sb) - SKIP)
    fa = {p: (v, r) for p, v, r in _flatten({k: sa[k] for k in shared}, labels_a) if not p.endswith(".mapping")}
    fb = {p: (v, r) for p, v, r in _flatten({k: sb[k] for k in shared}, labels_b) if not p.endswith(".mapping")}
    rows_a, rows_b = {r for _, r in fa.values() if r}, {r for _, r in fb.values() if r}
    out["rows_only_a"], out["rows_only_b"] = sorted(rows_a - rows_b), sorted(rows_b - rows_a)
    hidden = (0.0, None)
    for p in list(fa) + [p for p in fb if p not in fa]:
        (va, ra), (vb, rb) = fa.get(p, (None, None)), fb.get(p, (None, None))
        if (ra or rb) and ((ra or rb) not in rows_a or (ra or rb) not in rows_b):
            continue   # the whole row is only on one side
        out["n_compared"] += 1
        d = _delta(va, vb)
        if d is None and _ts(va) == _ts(vb) or d is not None and d <= tol:
            if d and d > hidden[0]:
                hidden = (d, p)
            continue
        out["changed"].append({"path": p, "a": va, "b": vb, "rel_delta": d})
    if hidden[1]:
        out["largest_within_tol"] = {"path": hidden[1], "rel_delta": hidden[0]}
    return out


def _settings(run: RunDir, prof: dict):
    """The settings the run's stages last ran with, as profile.json records them (True); for a run recorded before
    that, the config file as it reads now (False), or why it cannot be read."""
    if "config_settings" in prof:
        return prof["config_settings"], True
    from .pipeline import load_config
    try:
        return load_config(run, with_mined=False).settings(), False
    except (OSError, ValueError) as e:
        return f"unreadable ({e})", False


def _stages(prof: dict) -> str | None:
    """The stages that ran with different settings, grouped (``ingest, extract ≠ report``), or None when all agree."""
    groups: dict[str, list[str]] = {}
    for stage, digest in (prof.get("config_stages") or {}).items():
        groups.setdefault(digest, []).append(stage)
    return " ≠ ".join(map(", ".join, groups.values())) if len(groups) > 1 else None


def setup_diff(run_a: RunDir, run_b: RunDir, mapping_a: str, mapping_b: str) -> tuple[list[dict], dict]:
    """The setup fields, capabilities and config settings that differ; and per side whether its settings were
    recorded by its stages or read from its file now, and which stages ran with different ones."""
    pa, pb = run_a.profile(), run_b.profile()
    rows = [{"field": k, "a": pa.get(k), "b": pb.get(k)} for k in SETUP if pa.get(k) != pb.get(k)]
    if mapping_a != mapping_b:
        rows.append({"field": "mapping", "a": mapping_a, "b": mapping_b})
    ca, cb = pa.get("capabilities") or {}, pb.get("capabilities") or {}
    rows += [{"field": f"capabilities.{k}", "a": ca.get(k), "b": cb.get(k)} for k in sorted(set(ca) | set(cb))
             if ca.get(k) != cb.get(k)]
    (fa, rec_a), (fb, rec_b) = _settings(run_a, pa), _settings(run_b, pb)
    info = {"a": {"recorded": rec_a, "stages_differ": _stages(pa)}, "b": {"recorded": rec_b, "stages_differ": _stages(pb)}}
    if isinstance(fa, str) or isinstance(fb, str):
        if fa != fb:
            rows.append({"field": "config", "a": fa if isinstance(fa, str) else "read", "b": fb if isinstance(fb, str) else "read"})
        return rows, info
    rows += [{"field": f"config.{k}", "a": fa.get(k), "b": fb.get(k)} for k in list(fa) + [k for k in fb if k not in fa]
             if json.dumps(fa.get(k), sort_keys=True) != json.dumps(fb.get(k), sort_keys=True)]
    return rows, info


def _text_key(ev: pd.DataFrame) -> pd.Series:
    text = ev["text"].fillna("").map(lambda t: hashlib.sha1(t.encode()).hexdigest()[:16])
    return ev["ts"].astype(str) + "\x1f" + ev["author_raw"].astype(str) + "\x1f" + text


def align_events(ev_a: pd.DataFrame, ev_b: pd.DataFrame) -> tuple[dict, int]:
    """B's event ids paired with A's on (ts, author_raw, sha1(text)), and how many posts were left unpaired because
    their key is shared by another post on either side (a duplicate is not guessed at)."""
    ka, kb = _text_key(ev_a), _text_key(ev_b)
    dup = set(ka[ka.duplicated(keep=False)]) | set(kb[kb.duplicated(keep=False)])
    a = dict(zip(ka, ev_a["event_id"]))
    pairs = {e: a[k] for k, e in zip(kb, ev_b["event_id"]) if k in a and k not in dup}
    n_coll = int(ka.isin(dup).sum() + kb.isin(dup).sum())
    return pairs, n_coll


def _claim_ids(ca: pd.DataFrame, cb: pd.DataFrame, events: dict) -> dict:
    """B's claim ids paired with A's inside aligned posts, by type, item, value, round and order in the post."""
    cols = ["event_type", "item", "value_raw", "episode"]
    cb = cb[cb["event_id"].isin(events)].assign(event_id=lambda d: d["event_id"].map(events))
    ka, kb = (c.assign(n=c.groupby(["event_id", *cols], dropna=False).cumcount()) for c in (ca, cb))
    m = kb.merge(ka, on=["event_id", *cols, "n"], how="inner", suffixes=("_b", "_a"))
    return dict(zip(m["claim_id_b"], m["claim_id_a"]))


def _same(a, b) -> bool:
    """Equal, with missing equal to missing."""
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        return a == b
    return bool(a == b) or bool(pd.isna(a) and pd.isna(b))


def _keyed(a: pd.DataFrame, b: pd.DataFrame, key, fields, limit) -> dict:
    """Rows added and dropped on ``key``, and per field how many shared rows changed (with examples)."""
    ia, ib = a.set_index(key), b.set_index(key)
    added, dropped = ib.index.difference(ia.index), ia.index.difference(ib.index)
    shared = ia.index.intersection(ib.index)
    out = {"a": len(a), "b": len(b), "added": len(added), "dropped": len(dropped),
           "examples_added": [_id(k) for k in added[:limit]], "examples_dropped": [_id(k) for k in dropped[:limit]],
           "changed": {}}
    sa, sb = ia.loc[shared], ib.loc[shared]
    for f in fields:
        if f not in sa or f not in sb:
            continue
        diff = [k for k, x, y in zip(shared, sa[f], sb[f]) if not _same(x, y)]
        if diff:
            ch = pd.Series([f"{_cell(x)}→{_cell(y)}" for x, y in zip(sa.loc[diff, f], sb.loc[diff, f])])
            out["changed"][f] = {"n": len(diff), "by_change": ch.value_counts().head(limit).to_dict(),
                                 "examples": [_id(k) for k in diff[:limit]]}
    return out


def _id(k):
    return " | ".join(map(str, k)) if isinstance(k, tuple) else str(k)


def _cell(v) -> str:
    if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
        return "—"
    return f"{v:g}" if isinstance(v, float) else str(v)


def _read(run: RunDir, name: str) -> pd.DataFrame | None:
    """A stage table, or None when the run lacks it (a committed results/ directory holds only the summary)."""
    return run.read(name) if run.has(name) else None


def row_diff(run_a: RunDir, run_b: RunDir, mapping_a: str, mapping_b: str, limit: int = LIMIT,
             labels_a: dict | None = None, labels_b: dict | None = None) -> tuple[dict, dict]:
    """Per table both runs hold, the rows added, dropped and changed on its stable id; and how posts were aligned.
    ``labels_*`` rename one side's agents (strict labels to merged) before agents, sources and originators compare."""
    t = {k: (_read(run_a, n.format(mapping_a)), _read(run_b, n.format(mapping_b))) for k, n in
         (("events", "events"), ("claims", "claims"), ("exposures", "exposures_{}"), ("agents", "agents"),
          ("chains", "chains_{}"))}
    t = {k: v for k, v in t.items() if v[0] is not None and v[1] is not None and len(v[0].columns) and len(v[1].columns)}
    rows, align, ev_map, cl_map = {}, {"fallback": False}, {}, {}
    if "events" in t:
        ea, eb = t["events"]
        n = min(len(ea), len(eb))
        align["event_id_overlap"] = len(set(ea["event_id"]) & set(eb["event_id"])) / n if n else 1.0
        if align["event_id_overlap"] < LOW_OVERLAP:
            ev_map, align["collisions"] = align_events(ea, eb)
            align |= {"fallback": True, "aligned": len(ev_map)}
            if "claims" in t:
                cl_map = _claim_ids(*t["claims"], ev_map)
        rows["events"] = _keyed(ea, eb.assign(event_id=eb["event_id"].map(lambda e: ev_map.get(e, e))), "event_id", (),
                                limit)
    if "claims" in t:
        ca, cb = t["claims"]
        rows["claims"] = _keyed(ca, cb.assign(claim_id=cb["claim_id"].map(lambda c: cl_map.get(c, c))), "claim_id",
                                CLAIM_FIELDS, limit)
    ta, tb = ((lambda s, d=d or {}: s.map(lambda v: d.get(v, v))) for d in (labels_a, labels_b))
    if "exposures" in t:
        xa, xb = t["exposures"]
        xa = xa.assign(agent=ta(xa["agent"]), src_agent=ta(xa["src_agent"]))
        xb = xb.assign(agent=tb(xb["agent"]), src_agent=tb(xb["src_agent"]), claim_id=xb["claim_id"].map(lambda c: cl_map.get(c, c)))
        rows["exposures"] = _keyed(xa, xb, "claim_id", (*FLAGS, "src_agent", "agent"), limit)
    if "agents" in t:
        rows["agents"] = _agents(*t["agents"], mapping_a, mapping_b, limit)
    if "chains" in t:
        ka, kb = t["chains"]
        ka, kb = ka.assign(originator=ta(ka["originator"])), kb.assign(originator=tb(kb["originator"]))
        rows["chains"] = _keyed(ka, kb, ["family", "item", "value_key"], ("n_agents", "originator", "max_depth"), limit)
    return rows, align


def _merged_labels(agents: pd.DataFrame) -> dict:
    return dict(zip(agents["agent_strict"], agents["agent_merged"]))


def _agents(aa: pd.DataFrame, ab: pd.DataFrame, mapping_a: str, mapping_b: str, limit: int) -> dict:
    """Author strings added and dropped; label changes (one mapping only); author strings regrouped, i.e. whose
    agent joins a different set of author strings (any mappings)."""
    la, lb = aa.set_index("author_raw")[AGENT_COLS[mapping_a]], ab.set_index("author_raw")[AGENT_COLS[mapping_b]]
    out = _keyed(la.rename("label").reset_index(), lb.rename("label").reset_index(), "author_raw",
                 ("label",) if mapping_a == mapping_b else (), limit)
    shared = la.index.intersection(lb.index)
    ga, gb = (s.loc[shared].groupby(s.loc[shared]).groups for s in (la, lb))
    regrouped = [a for a in shared if set(ga[la[a]]) != set(gb[lb[a]])]
    out |= {"agents_a": int(la.nunique()), "agents_b": int(lb.nunique()), "regrouped": len(regrouped),
            "examples_regrouped": regrouped[:limit]}
    return out


def _summary(run: RunDir, mapping: str):
    p = run.path / f"summary{suffix(mapping)}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def compare(run_a, run_b, mapping_a: str = "merged", mapping_b: str = "merged", tol: float = TOL,
            limit: int = LIMIT) -> dict:
    """Everything ``compare.json`` holds; ``n_changes`` is 0 when nothing differs."""
    ra, rb = RunDir.open(run_a), RunDir.open(run_b)
    res = {"a": {"run": str(ra.path), "mapping": mapping_a}, "b": {"run": str(rb.path), "mapping": mapping_b},
           "tolerance": tol}
    res["setup"], res["config"] = setup_diff(ra, rb, mapping_a, mapping_b)
    # across mappings, labels compare in one namespace: the strict side's read as merged (merged is a function of strict)
    side = ra if mapping_a == "strict" else rb
    to_merged = _merged_labels(side.read("agents")) if mapping_a != mapping_b and side.has("agents") else {}
    lab_a, lab_b = (to_merged if m == "strict" else None for m in (mapping_a, mapping_b))
    sa, sb = _summary(ra, mapping_a), _summary(rb, mapping_b)
    if sa is None or sb is None:
        miss = [f"{r.path}/summary{suffix(m)}.json" for r, m, s in ((ra, mapping_a, sa), (rb, mapping_b, sb)) if s is None]
        res["summary"] = {"missing": miss}
    else:
        res["summary"] = summary_diff(sa, sb, tol, lab_a, lab_b)
    res["rows"], res["alignment"] = row_diff(ra, rb, mapping_a, mapping_b, limit, lab_a, lab_b)
    res["tables_not_compared"] = [n for n in TABLE_NAMES if n not in res["rows"]]
    s = res["summary"]
    res["n_changes"] = (len(res["setup"]) + sum(len(s.get(k, [])) for k in
                                                ("sections_only_a", "sections_only_b", "rows_only_a", "rows_only_b",
                                                 "changed"))
                        + sum(t["added"] + t["dropped"] + sum(c["n"] for c in t["changed"].values())
                              + t.get("regrouped", 0) for t in res["rows"].values()))
    return res


TABLE_NAMES = {"events": ("posts", "event_id"), "claims": ("claims", "claim_id"),
               "exposures": ("agent-rounds", "claim_id"), "agents": ("author strings", "author_raw"),
               "chains": ("facts", "(family, item, value_key)")}


def _esc(text: str) -> str:
    return str(text).replace("|", "\\|")   # a pipe would split a table cell


def _code(x) -> str:
    return "`" + str(x).replace("`", "'") + "`"


def _changes(t: dict) -> str:
    parts = [f"{f} {c['n']:,} (" + ", ".join(f"{k} {n:,}" for k, n in list(c["by_change"].items())[:3])
             + (", ..." if len(c["by_change"]) > 3 else "") + ")" for f, c in t["changed"].items()]
    if t.get("regrouped"):
        parts.append(f"regrouped {t['regrouped']:,}")
    return _esc("; ".join(parts) or "0")


def _show(v) -> str:
    text = json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (list, dict)) else _cell(v)
    return _esc((text[:117] + "...") if len(text) > 120 else text)


def _ids(xs) -> str:
    return ", ".join(map(_code, xs))


def _config_notes(info: dict) -> list[str]:
    """Where each side's settings come from, when not from what its stages recorded, and stages that disagree."""
    out = []
    for side in ("a", "b"):
        i = info[side]
        if not i["recorded"]:
            out.append(f"{side.upper()}'s settings are its config as it reads now (the adapter's built-in one and any "
                       "config file): the run predates the settings its stages record")
        if i["stages_differ"]:
            out.append(f"{side.upper()}'s stages ran with different settings ({i['stages_differ']}); the config rows "
                       "give the latest stage's")
    return out


def to_markdown(res: dict, limit: int = LIMIT) -> str:
    a, b, s, al = res["a"], res["b"], res["summary"], res["alignment"]
    hid = s.get("largest_within_tol")
    L = ["# What changed between two runs\n",
         f"A: `{a['run']}` ({a['mapping']} mapping); B: `{b['run']}` ({b['mapping']} mapping). "
         f"**{res['n_changes']:,} change{'s' * (res['n_changes'] != 1)}** in all: setup fields, summary values and "
         "rows, each counted once.\n",
         "## Setup\n"]
    if res["setup"]:
        L += ["| field | A | B |\n|---|---|---|", *(f"| {r['field']} | {_show(r['a'])} | {_show(r['b'])} |"
                                                   for r in res["setup"]), ""]
    else:
        L.append("Same adapter, source, config, capabilities, posts, roster, clock and mapping.\n")
    L += [f"{note}.\n" for note in _config_notes(res["config"])]
    L.append("## Summary\n")
    if "missing" in s:
        L.append("Not compared: no " + " and no ".join(map(_code, s["missing"]))
                 + " (`swarmprov report RUN [--mapping strict]` writes it).\n")
    else:
        L.append(f"{s['n_compared']:,} values compared in the sections both summaries hold; {len(s['changed']):,} differ. "
                 f"Numbers count as equal within a relative tolerance of {res['tolerance']:g}, and a missing value equals "
                 "a missing one; " + (f"the largest difference the tolerance hid is {hid['rel_delta']:.1e} "
                                      f"({_code(hid['path'])})." if hid else "it hid no difference.")
                 + (" Top-source rows are keyed by merged labels on both sides." if a["mapping"] != b["mapping"] else "")
                 + "\n")
        for side, k in (("A", "sections_only_a"), ("B", "sections_only_b"), ("A", "rows_only_a"), ("B", "rows_only_b")):
            if s[k]:
                L.append(f"{'Sections' if k.startswith('sections') else 'Rows'} only in {side} ({len(s[k]):,}): "
                         + _ids(s[k][:limit]) + (", ..." if len(s[k]) > limit else "") + "\n")
        if s["changed"]:
            L.append("| value | A | B | relative difference |\n|---|---|---|---|")
            L += [f"| {_esc(_code(c['path']))} | {_show(c['a'])} | {_show(c['b'])} | "
                  + ("" if c["rel_delta"] is None else f"{c['rel_delta']:.2g}") + " |" for c in s["changed"][:limit]]
            L.append(f"\n{len(s['changed']) - limit:,} more in `compare.json`.\n" if len(s["changed"]) > limit else "")
    L.append("## Rows\n")
    if "event_id_overlap" in al:
        L.append(f"Posts share {al['event_id_overlap']:.0%} of their event ids"
                 + (f", so they were aligned on (time, author, text hash): {al['aligned']:,} posts of B paired with A's, "
                    f"{al['collisions']:,} left unpaired because another post on the same side shares their key (claims "
                    "inside paired posts are paired by type, item, value, round and order)." if al["fallback"] else
                    ", so rows are aligned on their ids.")
                 + (" Agent labels on the strict side are translated to merged labels before sources and originators "
                    "are compared, so a change there is more than the mapping's." if a["mapping"] != b["mapping"] else "")
                 + "\n")
    if res["tables_not_compared"]:
        L.append("Not in both runs, so not compared: " + ", ".join(TABLE_NAMES[t][0] for t in res["tables_not_compared"])
                 + ".\n")
    if res["rows"]:
        L.append("| table | key | A | B | added | dropped | changed |\n|---|---|---|---|---|---|---|")
        L += [f"| {TABLE_NAMES[t][0]} | {TABLE_NAMES[t][1]} | {r['a']:,} | {r['b']:,} | {r['added']:,} | {r['dropped']:,} "
              f"| {_changes(r)} |" for t, r in res["rows"].items()]
    if "agents" in res["rows"]:
        r = res["rows"]["agents"]
        L.append(f"\nAgents: {r['agents_a']:,} in A, {r['agents_b']:,} in B; an author string is *regrouped* when its agent "
                 "joins a different set of author strings" + ("" if a["mapping"] == b["mapping"] else
                                                              " (labels are not compared across mappings)") + ".")
    for t, r in res["rows"].items():
        lines = [f"- added: {_ids(r['examples_added'])}" if r["examples_added"] else "",
                 f"- dropped: {_ids(r['examples_dropped'])}" if r["examples_dropped"] else "",
                 *(f"- {f} changed: {_ids(c['examples'])}" for f, c in r["changed"].items()),
                 f"- regrouped: {_ids(r['examples_regrouped'])}" if r.get("examples_regrouped") else ""]
        if any(lines):
            L += [f"\n**{TABLE_NAMES[t][0]}** (at most {limit} examples each):\n", *(x for x in lines if x)]
    return "\n".join(L) + "\n"


def headline(res: dict) -> list[str]:
    """The terminal summary."""
    s, al = res["summary"], res["alignment"]
    out = [f"compare {res['a']['run']} ({res['a']['mapping']}) vs {res['b']['run']} ({res['b']['mapping']}): "
           f"{res['n_changes']:,} change{'s' * (res['n_changes'] != 1)}",
           "  setup: " + (", ".join(r["field"] for r in res["setup"]) or "same"),
           *(f"  note: {n}" for n in _config_notes(res["config"]))]
    if "missing" in s:
        out.append("  summary: not compared (no " + ", ".join(s["missing"]) + ")")
    else:
        hid = s["largest_within_tol"]
        out.append(f"  summary: {s['n_compared']:,} values compared, {len(s['changed']):,} differ beyond a relative "
                   f"{res['tolerance']:g}" + (f" (largest within it {hid['rel_delta']:.1e} at {hid['path']})" if hid else "")
                   + (f"; {len(s['rows_only_a'])} rows only in A, {len(s['rows_only_b'])} only in B"
                      if s["rows_only_a"] or s["rows_only_b"] else ""))
    out.append("  rows: " + ("; ".join(f"{TABLE_NAMES[t][0]} +{r['added']:,} -{r['dropped']:,}"
                                       + (f", {sum(c['n'] for c in r['changed'].values()):,} changed" if r["changed"] else "")
                                       + (f", {r['regrouped']:,} regrouped" if r.get("regrouped") else "")
                                       for t, r in res["rows"].items()) or "no table in both runs"))
    flips = res["rows"].get("exposures", {}).get("changed", {})
    if flips:
        out.append("  agent-rounds changed: " + ", ".join(f"{f} {c['n']:,}" for f, c in flips.items()))
    if al["fallback"]:
        out.append(f"  posts aligned on (time, author, text hash): {al['aligned']:,} paired, {al['collisions']:,} "
                   "unpaired collisions")
    return out


def main(run_a: str, run_b: str, mapping_a: str = "merged", mapping_b: str = "merged", out: str = "runs/compare",
         limit: int | None = None, tol: float = TOL) -> int:
    """``swarmprov compare``: write ``compare.md`` and ``compare.json`` to ``out`` and print the summary."""
    limit = limit or LIMIT
    res = compare(run_a, run_b, mapping_a, mapping_b, tol, limit)
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "compare.json").write_text(dumps(res, indent=2), encoding="utf-8")
    (d / "compare.md").write_text(to_markdown(res, limit), encoding="utf-8")
    print("\n".join(headline(res)))
    print(f"wrote {d / 'compare.md'} and {d / 'compare.json'}")
    return 0
