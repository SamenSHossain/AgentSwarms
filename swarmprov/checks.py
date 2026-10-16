"""``swarmprov check RUN``: do a run's tables agree with each other?

``check_run`` re-derives what it can without the code that built it and compares, per identity mapping:

* exposure against the mentions: a plain loop over each fact's public mentions finds the first other agent's post
  before every report, and ``D``, ``t_public``, ``src_agent``, ``src_event_id``, ``gap_s`` and the 10 min / 1 h flags
  must follow from it (no helper is shared with ``exposure.build``, or the check would prove nothing);
* the flags nest: ``D_w60m <= D_w10m <= D``, ``D_visible <= D`` and ``D_cons_w60m <= D_cons``;
* the graph against the chains and the exposures: every multi-carrier fact has ``n_agents - 1`` relay hops, a chain
  counts the fact's distinct timed carriers, and each exposed round has exactly one exposure edge, from its source;
* the posts behind the edges and exposures exist: every edge's ``event_id`` and ``src_event_id`` is a post, every
  report post is one, and every ``src_event_id`` is a mention of the round's value by its ``src_agent``;
* the strict mapping refines the merged one (identity resolves per strict handle).

Those are errors.  Warnings: exposed rounds the agent had itself posted before the first public copy
(``t_self_prepared < t_public``: it was holding the answer), and ``summary.json``'s A1 and A5 numbers against a
recomputation from the tables (within ``REL_TOL``; a stale report after a rerun stage shows here).  The result is one
row per check and table, also written to ``checks.json`` (``checks_strict.json`` against the strict report).
"""

from __future__ import annotations

import json
import math

import pandas as pd

from .analysis.agents import AGENT_COLS
from .claims import PRIVATE_KINDS, value_key
from .schema import RunDir, suffix

COLS = ["check", "table", "n_violations", "severity", "example_ids"]
SEVERITY = {"exposure_matches_mentions": "error", "flag_nesting": "error", "relay_hops_per_chain": "error",
            "chain_carriers": "error", "exposure_edges": "error", "edge_posts_resolve": "error",
            "exposure_posts_resolve": "error", "strict_refines_merged": "error",
            "self_prepared_before_public": "warning", "summary_matches_tables": "warning"}
# what a check's violations are, for the report and ``swarmprov check``
MEANING = {"exposure_matches_mentions": "rounds whose exposure flags, first public copy or head start do not follow "
                                        "from the mentions",
           "flag_nesting": "rounds with a stricter exposure flag set and a looser one not",
           "relay_hops_per_chain": "facts without one relay hop per carrier after the first",
           "chain_carriers": "facts whose chain does not count their distinct timed carriers",
           "exposure_edges": "rounds without exactly one exposure edge from their first public source",
           "edge_posts_resolve": "edges naming a post the run does not hold",
           "exposure_posts_resolve": "rounds whose report post or first public copy is not among the run's posts and "
                                     "mentions",
           "strict_refines_merged": "author handles split between merged agents",
           "self_prepared_before_public": "exposed rounds whose agent had posted the value itself before the first "
                                          "public copy",
           "summary_matches_tables": "summary.json numbers that differ from the tables (a stage rerun after the "
                                     "report?)"}
N_EXAMPLES = 5
REL_TOL = 1e-6
RELAY = ("relay", "relay_xchannel")


def _given(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and v is not pd.NaT


def _key(r):
    """The value a round is matched on: its own, else the consensus (the rule exposure documents)."""
    return (value_key(r.value_norm) if _given(r.value_norm) else None) or \
        (r.consensus_value if isinstance(r.consensus_value, str) else None)


def _facts(mentions: pd.DataFrame, agent_col: str) -> dict:
    """(family, item, value_key) -> its timed public mentions as (ts, event_id, agent, channel), earliest first."""
    out: dict = {}
    for r in mentions.itertuples(index=False):
        if r.family != "url" and r.source_kind not in PRIVATE_KINDS and _given(r.ts):
            out.setdefault((r.family, r.item, r.value_key), []).append(
                (r.ts, r.event_id, getattr(r, agent_col), getattr(r, "channel", None)))
    return {k: sorted(v, key=lambda m: (m[0], m[1])) for k, v in out.items()}


def _exposure(ex: pd.DataFrame, facts: dict, audience: dict, wall: bool) -> list:
    bad = []
    for r in ex.itertuples(index=False):
        first = next((m for m in facts.get((r.family, r.item, _key(r)), []) if m[0] < r.t_report and m[2] != r.agent
                      and (m[3] not in audience or r.agent in audience[m[3]])), None)
        ok = r.D == int(first is not None)
        if ok and first is not None:
            gap = (r.t_report - first[0]).total_seconds()
            ok = (r.t_public == first[0] and r.src_agent == first[2] and r.src_event_id == first[1]
                  and (not wall or (abs(r.gap_s - gap) < 1e-6 and r.D_w10m == int(gap >= 600)
                                    and r.D_w60m == int(gap >= 3600))))
        if not ok:
            bad.append(r.claim_id)
    return bad


def _nesting(ex: pd.DataFrame) -> list:
    pairs = [("D_w60m", "D_w10m"), ("D_w10m", "D"), ("D_visible", "D"), ("D_cons_w60m", "D_cons")]
    bad = pd.Series(False, index=ex.index)
    for lo, hi in pairs:
        bad |= ex[lo].notna() & (ex[lo] > ex[hi])   # NaN under a sequence clock: no flag to nest
    return ex.loc[bad, "claim_id"].tolist()


def _relay(edges: pd.DataFrame, chains: pd.DataFrame) -> list:
    hops = edges[edges["kind"].isin(RELAY)].groupby(["family", "item", "value_norm"]).size()
    c = chains.set_index(["family", "item", "value_key"])["n_agents"] - 1
    both = pd.concat([c.rename("want"), hops.rename("got").rename_axis(c.index.names)], axis=1).fillna(0)
    return ["|".join(map(str, k)) for k in both.index[both["want"] != both["got"]]]


def _carriers(mentions: pd.DataFrame, chains: pd.DataFrame, agent_col: str) -> list:
    m = mentions[~mentions["source_kind"].isin(PRIVATE_KINDS) & mentions["ts"].notna()]
    n = m.groupby(["family", "item", "value_key"])[agent_col].nunique()
    c = chains.set_index(["family", "item", "value_key"])["n_agents"]
    both = pd.concat([n.rename("want"), c.rename("got")], axis=1)
    return ["|".join(map(str, k)) for k in both.index[both["want"] != both["got"]]]


def _exposure_edges(ex: pd.DataFrame, edges: pd.DataFrame) -> list:
    e = edges[edges["kind"] == "exposure"]
    per = e.groupby("claim_id").agg(n=("src", "size"), src=("src", "first"), dst=("dst", "first"))
    x = ex.set_index("claim_id")
    bad = [c for c in per.index if c not in x.index]
    for c, r in x.iterrows():
        n = per.loc[c, "n"] if c in per.index else 0
        if n != r["D"] or (n and (per.loc[c, "src"] != r["src_agent"] or per.loc[c, "dst"] != r["agent"])):
            bad.append(c)
    return bad


def _edge_posts(edges: pd.DataFrame, posts: set) -> list:
    bad = edges["event_id"].isna() | ~edges["event_id"].isin(posts) \
        | (edges["src_event_id"].notna() & ~edges["src_event_id"].isin(posts))
    return edges.loc[bad, "event_id"].fillna("(none)").tolist()


def _exposure_posts(ex: pd.DataFrame, mentions: pd.DataFrame, posts: set, agent_col: str) -> list:
    carried = set(zip(mentions["event_id"], mentions[agent_col], mentions["family"], mentions["item"],
                      mentions["value_key"]))
    bad = []
    for r in ex.itertuples(index=False):
        if r.report_event_id not in posts or (_given(r.src_event_id) and
                                              (r.src_event_id, r.src_agent, r.family, r.item, _key(r)) not in carried):
            bad.append(r.claim_id)
    return bad


def _summary(summary: dict, run: RunDir) -> list:
    """The A1 and A5 numbers ``summary.json`` states that differ from the tables by more than ``REL_TOL``."""
    mapping = summary.get("mapping", "merged")
    ex = run.read(f"exposures_{mapping}")
    edges = run.read(f"edges_{mapping}") if run.has(f"edges_{mapping}") else None
    want = {}
    if "A1" in summary and len(ex):
        want |= {"A1.n_answers": len(ex), "A1.n_independent": int((ex["D"] == 0).sum()),
                 "A1.n_agents": ex["agent"].nunique(), "A1.share_exposed": ex["D"].mean(),
                 "A1.share_self_prepared": ex["self_prepared"].mean()}
        if summary["A1"].get("share_exposed_60m") is not None:
            want |= {"A1.share_exposed_10m": ex["D_w10m"].mean(), "A1.share_exposed_60m": ex["D_w60m"].mean(),
                     "A1.median_gap_h": ex["gap_s"].median() / 3600}
    if "A5" in summary and edges is not None:
        kinds = edges["kind"].value_counts()
        want |= {f"A5.n_{k}_edges": int(kinds.get(kind, 0))
                 for k, kind in (("exposure", "exposure"), ("relay", "relay"), ("xchannel", "relay_xchannel"),
                                 ("citation", "citation"))}
        src = ex.loc[ex["D"] == 1, "src_agent"].value_counts()
        if src.sum():
            want["A5.top10_share_exposure"] = src.head(10).sum() / src.sum()
    bad = []
    for k, v in want.items():
        sec, name = k.split(".")
        got = summary[sec].get(name)
        if got is None or not math.isclose(float(got), float(v), rel_tol=REL_TOL, abs_tol=1e-9):
            bad.append(k)
    return bad


def _row(check: str, table: str, bad: list) -> dict:
    return {"check": check, "table": table, "n_violations": len(bad), "severity": SEVERITY[check],
            "example_ids": [str(b) for b in bad[:N_EXAMPLES]]}


def check_run(run: RunDir, summary: dict | None = None, mapping: str | None = None) -> pd.DataFrame:
    """One row per check and table (``COLS``); written to ``checks.json`` (``checks_strict.json`` for a strict
    summary).  ``summary`` is the report's (the report passes the one it is writing); by default the ``mapping``
    report's ``summary.json`` (merged) or ``summary_strict.json`` when the run has one, else that check is left out."""
    from . import village
    from .pipeline import _lists
    prof = json.loads((run.path / "profile.json").read_text())
    wall = prof.get("clock") != "sequence"
    mapping = mapping or (summary or {}).get("mapping", "merged")
    sfx = suffix(mapping)
    if summary is None and (run.path / f"summary{sfx}.json").exists():
        summary = json.loads((run.path / f"summary{sfx}.json").read_text())
    rows = []
    if run.has("agents"):
        ag = run.read("agents")
        n = ag.groupby("agent_strict")["agent_merged"].nunique()
        rows.append(_row("strict_refines_merged", "agents", n.index[n > 1].tolist()))
    if run.has("mentions"):
        mn, events = run.read("mentions"), run.read("events")
        posts = set(events["event_id"])
        for name, col in AGENT_COLS.items():
            ex = run.read(f"exposures_{name}") if run.has(f"exposures_{name}") else pd.DataFrame()
            if len(ex):
                aud = village.audience(_lists(run.read("channels")), run.read("agents"), col,
                                       run.read("directory") if run.has("directory") else None) \
                    if run.has("channels") else {}
                t = f"exposures_{name}"
                rows += [_row("exposure_matches_mentions", t, _exposure(ex, _facts(mn, col), aud or {}, wall)),
                         _row("flag_nesting", t, _nesting(ex)),
                         _row("exposure_posts_resolve", t, _exposure_posts(ex, mn, posts, col)),
                         _row("self_prepared_before_public", t,
                              ex.loc[(ex["D"] == 1) & (ex["t_self_prepared"] < ex["t_public"]), "claim_id"].tolist())]
            if run.has(f"edges_{name}"):
                edges, chains = run.read(f"edges_{name}"), run.read(f"chains_{name}")
                t = f"edges_{name}"
                rows += [_row("relay_hops_per_chain", t, _relay(edges, chains) if len(chains) else []),
                         _row("chain_carriers", f"chains_{name}", _carriers(mn, chains, col) if len(mn) else []),
                         _row("edge_posts_resolve", t, _edge_posts(edges, posts))]
                if len(ex):
                    rows.append(_row("exposure_edges", t, _exposure_edges(ex, edges)))
    if summary and run.has("mentions"):
        rows.append(_row("summary_matches_tables", f"summary{sfx}.json", _summary(summary, run)))
    out = pd.DataFrame(rows, columns=COLS)
    fail = out[out["n_violations"] > 0]
    (run.path / f"checks{sfx}.json").write_text(json.dumps({
        "n_checks": len(out), "n_pass": int((out["n_violations"] == 0).sum()),
        "n_errors": int((fail["severity"] == "error").sum()), "n_warnings": int((fail["severity"] == "warning").sum()),
        "checks": out.to_dict("records")}, indent=2))
    return out


def main(run: RunDir, mapping: str = "merged") -> int:
    """``swarmprov check``: print each check, write ``checks.json`` (``checks_strict.json`` against the strict
    report's summary); exit status 1 when an error-level check fails."""
    if not (run.path / "profile.json").exists():
        print(f"{run.path}: not a run directory (no profile.json); run `swarmprov run` or `ingest` first")
        return 2
    out = check_run(run, mapping=mapping)
    print(f"checks on {run.path}:")
    for r in out.itertuples(index=False):
        status = "ok" if not r.n_violations else ("ERROR" if r.severity == "error" else "warn")
        print(f"  {status:<5} {r.check:<28} {r.table}" + (
            f": {r.n_violations:,} {MEANING[r.check]} (e.g. {', '.join(r.example_ids)})" if r.n_violations else ""))
    fail = out[out["n_violations"] > 0]
    n_err = int((fail["severity"] == "error").sum())
    print(f"{len(out) - len(fail)} of {len(out)} checks pass: {n_err} error{'s' * (n_err != 1)}, "
          f"{len(fail) - n_err} warning{'s' * (len(fail) - n_err != 1)}; written to {run.path / f'checks{suffix(mapping)}.json'}")
    return 1 if n_err else 0
