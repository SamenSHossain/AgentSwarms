"""``swarmprov check RUN``: do a run's tables agree with each other?

``check_run`` re-derives what it can without the code that built it and compares, per identity mapping:

* exposure against the mentions: a plain loop over each fact's public mentions the agent could read before every
  report finds the first and the latest other agent's post, the first still undeleted at the report, the agent's own
  earliest one and the first other agent's post of the consensus value, and every exposure column built from them must
  follow (``D``, ``D_visible``, ``self_prepared``, ``self_originated``, ``D_cons``, their times, posts and sources,
  the head starts and the 10 min / 1 h flags; no helper is shared with ``exposure.build``, or the check would prove
  nothing);
* the flags nest: ``D_w60m <= D_w10m <= D``, ``D_visible <= D`` and ``D_cons_w60m <= D_cons``;
* the graph against the chains and the exposures: every multi-carrier fact has ``n_agents - 1`` relay hops, a chain
  counts the fact's distinct timed carriers, each exposed round has exactly one exposure edge, from its source, and
  a self-originated one (the agent had posted the value before the first public copy, ``t_self_prepared <
  t_public``) has none;
* the posts behind the edges and exposures exist: every edge's ``event_id`` and ``src_event_id`` is a post, every
  report post is one, and every ``src_event_id`` is a mention of the round's value by its ``src_agent``;
* the strict mapping refines the merged one (identity resolves per strict handle), except where a handle's case
  variants carry conflicting cohorts or roster ids, which ``identity.resolve`` keeps apart by design.

Those are errors.  Warnings: those designed splits (``strict_handles_kept_apart``; ``agent_aliases.csv`` lists them),
and ``summary.json``'s A1 and A5 numbers against a recomputation from the tables (within
``REL_TOL``; a stale report after a rerun stage shows here; left out when the summary states none of them).  The result
is one row per check and table, also written to ``checks.json`` (``checks_strict.json`` against the strict report).
"""

from __future__ import annotations

import json
import math

import pandas as pd

from .analysis.agents import AGENT_COLS
from .claims import PRIVATE_KINDS, value_key
from .fmt import dumps
from .graph import RELAY_KINDS
from .schema import RunDir, suffix

COLS = ["check", "table", "n_violations", "severity", "example_ids"]
# name -> (severity, what its violations are, what it compares), in the order the report's integrity line names what
# they compare (a shared phrase is named once; the report names its own summary file)
CHECKS = {
    "exposure_matches_mentions": ("error", "rounds whose exposure columns (named in brackets) do not follow from the "
                                           "mentions", "exposure re-derived from the mentions"),
    "flag_nesting": ("error", "rounds with a stricter exposure flag set and a looser one not", "flag nesting"),
    "exposure_posts_resolve": ("error", "rounds whose report post or first public copy is not among the run's posts "
                                        "and mentions", "the posts behind edges and exposures"),
    "edge_posts_resolve": ("error", "edges naming a post the run does not hold", "the posts behind edges and exposures"),
    "relay_hops_per_chain": ("error", "facts without one relay hop per carrier after the first",
                             "relay hops and chain carriers against the mentions"),
    "chain_carriers": ("error", "facts whose chain does not count their distinct timed carriers",
                       "relay hops and chain carriers against the mentions"),
    "exposure_edges": ("error", "rounds without exactly one exposure edge from their first public source",
                       "exposure edges against the rounds"),
    "self_originated_no_edge": ("error", "self-originated rounds (the agent had posted the value before the first "
                                         "public copy) that carry an exposure edge", "exposure edges against the rounds"),
    "strict_refines_merged": ("error", "author handles split between merged agents with no conflicting cohort or "
                                       "roster id behind it", "the strict mapping inside the merged one"),
    "strict_handles_kept_apart": ("warning", "author handles whose case variants carry conflicting cohorts or roster "
                                             "ids, kept apart in the merged mapping as designed; agent_aliases.csv "
                                             "lists them", "the strict mapping inside the merged one"),
    "summary_matches_tables": ("warning", "summary.json numbers that differ from the tables (a stage rerun after the "
                                          "report?)", "summary.json against the tables"),
}
N_EXAMPLES = 5
REL_TOL = 1e-6


def _given(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and v is not pd.NaT


def _key(r):
    """The value a round is matched on: its own, else the consensus (the rule exposure documents)."""
    return (value_key(r.value_norm) if _given(r.value_norm) else None) or \
        (r.consensus_value if isinstance(r.consensus_value, str) else None)


def _facts(mentions: pd.DataFrame, agent_col: str) -> dict:
    """(family, item, value_key) -> its timed public mentions as (ts, event_id, agent, channel, visible_until),
    earliest first."""
    out: dict = {}
    for r in mentions.itertuples(index=False):
        if r.family != "url" and r.source_kind not in PRIVATE_KINDS and _given(r.ts):
            out.setdefault((r.family, r.item, r.value_key), []).append(
                (r.ts, r.event_id, getattr(r, agent_col), getattr(r, "channel", None),
                 getattr(r, "visible_until", None)))
    return {k: sorted(v, key=lambda m: (m[0], m[1])) for k, v in out.items()}


def _same(got, want) -> bool:
    if not _given(got) or not _given(want):
        return not _given(got) and not _given(want)
    return math.isclose(got, want, abs_tol=1e-6) if isinstance(want, float) else got == want


def _expected(r, facts: dict, audience: dict, wall: bool) -> dict:
    """A round's exposure columns as the mentions give them, from the public posts of its value (``D_cons``: of the
    consensus value) the agent could read before its report."""
    def seen(key):
        return [m for m in facts.get((r.family, r.item, key), []) if m[0] < r.t_report
                and (m[3] not in audience or r.agent in audience[m[3]])]
    none = (None,) * 5
    before = seen(_key(r))
    others = [m for m in before if m[2] != r.agent]
    still = [m for m in others if not _given(m[4]) or m[4] > r.t_report]
    own = [m for m in before if m[2] == r.agent]
    cons = [m for m in seen(r.consensus_value) if m[2] != r.agent] if isinstance(r.consensus_value, str) else []
    first, last = (others[0], others[-1]) if others else (none, none)
    vis, mine, c = (still or [none])[0], (own or [none])[0], (cons or [none])[0]
    out = {"D": int(bool(others)), "t_public": first[0], "src_event_id": first[1], "src_agent": first[2],
           "src_channel": first[3], "t_latest": last[0], "latest_event_id": last[1], "src_latest": last[2],
           "D_visible": int(bool(still)), "t_public_visible": vis[0], "src_visible_event_id": vis[1],
           "src_deleted_before_report": int(_given(first[4]) and first[4] <= r.t_report), "t_src_deleted": first[4],
           "self_prepared": int(bool(own)), "t_self_prepared": mine[0], "self_event_id": mine[1],
           "self_originated": int(bool(others) and bool(own) and mine[0] < first[0]),
           "D_cons": int(bool(cons)), "t_cons": c[0]}
    if not wall:   # message positions are not seconds: no head start, no windowed flag
        return out | dict.fromkeys(("gap_s", "D_w10m", "D_w60m", "gap_cons_s", "D_cons_w60m"))
    gap = (r.t_report - first[0]).total_seconds() if others else None
    gap_c = (r.t_report - c[0]).total_seconds() if cons else None
    return out | {"gap_s": gap, "D_w10m": int(bool(others) and gap >= 600), "D_w60m": int(bool(others) and gap >= 3600),
                  "gap_cons_s": gap_c, "D_cons_w60m": int(bool(cons) and gap_c >= 3600)}


def _exposure(ex: pd.DataFrame, facts: dict, audience: dict, wall: bool) -> list:
    """Each round whose columns differ from ``_expected``, as ``claim_id (column, ...)`` (a column the table lacks,
    from a run built before it existed, is not compared)."""
    bad = []
    for r in ex.itertuples(index=False):
        off = [k for k, v in _expected(r, facts, audience, wall).items() if k in ex and not _same(getattr(r, k), v)]
        if off:
            bad.append(f"{r.claim_id} ({', '.join(off)})")
    return bad


def _nesting(ex: pd.DataFrame) -> list:
    pairs = [("D_w60m", "D_w10m"), ("D_w10m", "D"), ("D_visible", "D"), ("D_cons_w60m", "D_cons")]
    bad = pd.Series(False, index=ex.index)
    for lo, hi in pairs:
        bad |= ex[lo].notna() & (ex[lo] > ex[hi])   # NaN under a sequence clock: no flag to nest
    return ex.loc[bad, "claim_id"].tolist()


def _relay(edges: pd.DataFrame, chains: pd.DataFrame) -> list:
    hops = edges[edges["kind"].isin(RELAY_KINDS)].groupby(["family", "item", "value_norm"]).size()
    c = chains.set_index(["family", "item", "value_key"])["n_agents"] - 1
    both = pd.concat([c.rename("want"), hops.rename("got").rename_axis(c.index.names)], axis=1).fillna(0)
    return ["|".join(map(str, k)) for k in both.index[both["want"] != both["got"]]]


def _carriers(mentions: pd.DataFrame, chains: pd.DataFrame, agent_col: str) -> list:
    m = mentions[~mentions["source_kind"].isin(PRIVATE_KINDS) & mentions["ts"].notna()]
    n = m.groupby(["family", "item", "value_key"])[agent_col].nunique()
    c = chains.set_index(["family", "item", "value_key"])["n_agents"]
    both = pd.concat([n.rename("want"), c.rename("got")], axis=1)
    return ["|".join(map(str, k)) for k in both.index[both["want"] != both["got"]]]


def _own_first(ex: pd.DataFrame) -> pd.Series:
    """Exposed rounds whose agent had posted the value before the first public copy: no source is credited."""
    return (ex["D"] == 1) & (ex["t_self_prepared"] < ex["t_public"])


def _exposure_edges(ex: pd.DataFrame, edges: pd.DataFrame) -> tuple[list, list]:
    """The rounds without exactly one exposure edge from their first public source (none when D = 0), and the
    self-originated rounds that carry one."""
    e = edges[edges["kind"] == "exposure"]
    per = e.groupby("claim_id").agg(n=("src", "size"), src=("src", "first"), dst=("dst", "first"))
    x = ex.set_index("claim_id")
    own = set(x.index[_own_first(x).to_numpy()])
    bad = [c for c in per.index if c not in x.index]
    for c, r in x[~x.index.isin(own)].iterrows():
        n = per.loc[c, "n"] if c in per.index else 0
        if n != r["D"] or (n and (per.loc[c, "src"] != r["src_agent"] or per.loc[c, "dst"] != r["agent"])):
            bad.append(c)
    return bad, sorted(own & set(per.index))


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


def _split(agents: pd.DataFrame) -> tuple[list, list]:
    """Strict handles mapped to several merged agents: those whose variants carry one cohort and one roster id (a
    bug), and those ``identity.resolve`` and the roster keep apart on purpose (conflicting cohorts or roster ids)."""
    bug, kept = [], []
    for strict, g in agents.groupby("agent_strict"):
        if g["agent_merged"].nunique() > 1:
            apart = g["cohort"].dropna().nunique() > 1 or (
                "roster_agent" in g and g["roster_agent"].nunique(dropna=False) > 1)
            (kept if apart else bug).append(strict)
    return bug, kept


def _summary(summary: dict, run: RunDir) -> list | None:
    """The A1 and A5 numbers ``summary.json`` states that differ from the tables by more than ``REL_TOL``; None when
    it states none of them (nothing compared)."""
    mapping = summary.get("mapping", "merged")
    ex = run.read(f"exposures_{mapping}")
    edges = run.read(f"edges_{mapping}") if run.has(f"edges_{mapping}") else None
    want = {}
    if "A1" in summary and len(ex):
        want |= {"A1.n_answers": len(ex), "A1.n_independent": int((ex["D"] == 0).sum()),
                 "A1.n_agents": ex["agent"].nunique(), "A1.share_exposed": ex["D"].mean(),
                 "A1.share_self_prepared": ex["self_prepared"].mean()}
        if "n_self_originated" in summary["A1"]:
            want["A1.n_self_originated"] = int(_own_first(ex).sum())
        if summary["A1"].get("share_exposed_60m") is not None:
            want |= {"A1.share_exposed_10m": ex["D_w10m"].mean(), "A1.share_exposed_60m": ex["D_w60m"].mean(),
                     "A1.median_gap_h": ex["gap_s"].median() / 3600}
    if "A5" in summary and edges is not None:
        kinds = edges["kind"].value_counts()
        want |= {f"A5.n_{k}_edges": int(kinds.get(kind, 0))
                 for k, kind in (("exposure", "exposure"), ("relay", "relay"), ("xchannel", "relay_xchannel"),
                                 ("citation", "citation"))}
        src = ex.loc[(ex["D"] == 1) & ~_own_first(ex), "src_agent"].value_counts()
        if src.sum():
            want["A5.top10_share_exposure"] = src.head(10).sum() / src.sum()
    if not want:
        return None
    bad = []
    for k, v in want.items():
        sec, name = k.split(".")
        got = summary[sec].get(name)
        if got is None or not math.isclose(float(got), float(v), rel_tol=REL_TOL, abs_tol=1e-9):
            bad.append(k)
    return bad


def _row(check: str, table: str, bad: list) -> dict:
    return {"check": check, "table": table, "n_violations": len(bad), "severity": CHECKS[check][0],
            "example_ids": [str(b) for b in bad[:N_EXAMPLES]]}


def check_run(run: RunDir, summary: dict | None = None, mapping: str | None = None) -> pd.DataFrame:
    """One row per check and table (``COLS``); written to ``checks.json`` (``checks_strict.json`` for a strict
    summary).  ``summary`` is the report's (the report passes the one it is writing); by default the ``mapping``
    report's ``summary.json`` (merged) or ``summary_strict.json`` when the run has one, else that check is left out."""
    from . import village
    prof = run.profile()
    wall = prof.get("clock") != "sequence"
    mapping = mapping or (summary or {}).get("mapping", "merged")
    sfx = suffix(mapping)
    if summary is None and (run.path / f"summary{sfx}.json").exists():
        summary = json.loads((run.path / f"summary{sfx}.json").read_text())
    rows = []
    if run.has("agents"):
        bug, kept = _split(run.read("agents"))
        rows += [_row("strict_refines_merged", "agents", bug), _row("strict_handles_kept_apart", "agents", kept)]
    if run.has("mentions"):
        mn, events = run.read("mentions"), run.read("events")
        posts = set(events["event_id"])
        for name, col in AGENT_COLS.items():
            ex = run.read(f"exposures_{name}") if run.has(f"exposures_{name}") else pd.DataFrame()
            if len(ex):
                aud = village.audience(village.channel_lists(run.read("channels")), run.read("agents"), col,
                                       run.read("directory") if run.has("directory") else None) \
                    if run.has("channels") else {}
                t = f"exposures_{name}"
                rows += [_row("exposure_matches_mentions", t, _exposure(ex, _facts(mn, col), aud or {}, wall)),
                         _row("flag_nesting", t, _nesting(ex)),
                         _row("exposure_posts_resolve", t, _exposure_posts(ex, mn, posts, col))]
            if run.has(f"edges_{name}"):
                edges, chains = run.read(f"edges_{name}"), run.read(f"chains_{name}")
                t = f"edges_{name}"
                rows += [_row("relay_hops_per_chain", t, _relay(edges, chains) if len(chains) else []),
                         _row("chain_carriers", f"chains_{name}", _carriers(mn, chains, col) if len(mn) else []),
                         _row("edge_posts_resolve", t, _edge_posts(edges, posts))]
                if len(ex):
                    bad, own = _exposure_edges(ex, edges)
                    rows += [_row("exposure_edges", t, bad), _row("self_originated_no_edge", t, own)]
    if summary and run.has("mentions") and (bad := _summary(summary, run)) is not None:
        rows.append(_row("summary_matches_tables", f"summary{sfx}.json", bad))
    out = pd.DataFrame(rows, columns=COLS)
    fail = out[out["n_violations"] > 0]
    (run.path / f"checks{sfx}.json").write_text(dumps({
        "n_checks": len(out), "n_pass": int((out["n_violations"] == 0).sum()),
        "n_errors": int((fail["severity"] == "error").sum()), "n_warnings": int((fail["severity"] == "warning").sum()),
        "checks": out.to_dict("records")}, indent=2))
    return out


def main(run: RunDir, mapping: str = "merged") -> int:
    """``swarmprov check``: print each check, write ``checks.json`` (``checks_strict.json`` against the strict
    report's summary); exit status 1 when an error-level check fails, 2 when the run holds no exposure or edge table
    (nothing to check)."""
    if not any(run.has(f"{t}_{m}") for t in ("exposures", "edges") for m in AGENT_COLS):
        print(f"{run.path}: no exposure or edge tables to check; run `swarmprov expose` and `swarmprov graph` on it "
              "(or `swarmprov run`) first")
        return 2
    out = check_run(run, mapping=mapping)
    print(f"checks on {run.path}:")
    for r in out.itertuples(index=False):
        status = "ok" if not r.n_violations else ("ERROR" if r.severity == "error" else "warn")
        print(f"  {status:<5} {r.check:<28} {r.table}" + (
            f": {r.n_violations:,} {CHECKS[r.check][1]} (e.g. {', '.join(r.example_ids)})" if r.n_violations else ""))
    fail = out[out["n_violations"] > 0]
    n_err = int((fail["severity"] == "error").sum())
    print(f"{len(out) - len(fail)} of {len(out)} checks pass: {n_err} error{'s' * (n_err != 1)}, "
          f"{len(fail) - n_err} warning{'s' * (len(fail) - n_err != 1)}; written to {run.path / f'checks{suffix(mapping)}.json'}")
    return 1 if n_err else 0
