"""Exposure table: was the answer already public before the agent answered?

Unit = agent-round (agent, family, item).  Wall clock only: the agents'
task clocks are dilated, so a reported task-clock time is never compared
with a wiki timestamp.

* ``t_report``  first post in which the agent reports answering (an upper
  bound on when the question arrived).
* ``t_public``  first post by a *different* agent carrying the same item
  with the agent's value (or the consensus value when the agent did not
  restate it).
* ``D``         t_public < t_report.  Because t_report is an upper bound on
  arrival, D=0 is a conservative "independent" label; ``D_w10m`` /
  ``D_w60m`` require the value to have been public 10 / 60 minutes before the
  report, which is robust to the unknown report lag.
* ``t_self_prepared``  the agent itself posted the item+value before its
  report (it was holding the answer in advance).
* ``D_cons``    the *consensus* value was public before the report.  This is the
  causal treatment: unlike D it does not depend on the agent's own answer, so
  "wrong answer => nothing matching was public" cannot manufacture an effect.
  ``t_cons`` is that first public consensus copy and ``gap_cons_s`` its head start
  over the report (NaN when D_cons = 0), the dose A2's exploratory block reads.
* ``y_consensus``  the answer matches the consensus value (the value most distinct
  agents carried); ``y_winner`` scores it the same way against A4's winner (the
  later majority), only on slots A4 calls overturned (``winners``), NaN elsewhere.
* ``D_visible`` like D, but a copy whose channel was deleted before the report
  (``visible_until <= t_report``) does not count: the earliest copy *still
  readable* at report time.  Deletion ends public visibility, not knowledge an
  agent already took, so D_visible under-counts exposure and D stays the headline;
  ``src_deleted_before_report`` marks the D=1 rows whose first source was gone.

* ``src_latest`` / ``t_latest``  the *latest* other carrier before the report, a
  second attribution rule beside the first public copy (A5 reports both).

Framework traces add two private sources (``claims.PRIVATE_KINDS``), which never
count as a public copy, for the agent or for anyone else:

* ``t_own_lookup``  the agent's own tool output carried the item and value
  before the report; ``own_lookup_evidence`` = D=0 and such a lookup, i.e. *own
  lookup preceded report*.  Not "computed": a tool can echo a value the agent
  passed in.
* ``t_task``  a task message (system or user prompt) carried them before the
  report: the value was given in the task.

Each time is kept with the post behind it: ``report_event_id`` (the report),
``src_event_id`` / ``src_channel`` (the first public copy), ``latest_event_id`` (the
latest), ``src_visible_event_id`` (the first copy still visible) and ``self_event_id``
(the agent's own earlier post), ``own_lookup_event_id`` and ``task_event_id``.  Copies
posted in the same second are taken in event_id order, so the chosen post does not depend on row order.

Under a sequence clock (``wall_clock=False``: times are message positions) ``gap_s``,
``D_w10m``, ``D_w60m``, ``D_cons_w60m`` and ``gap_cons_s`` are NA, since positions are not seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .claims import PRIVATE_KINDS, value_key


def match_key(value_norm, consensus) -> str | None:
    """The value an agent-round is matched on: its own value's ``value_key``, or the consensus
    value when the agent did not restate one.  Exposure, edges, trace and A5 all join on it."""
    return value_key(value_norm) or (consensus if isinstance(consensus, str) else None)


def _score(r, target) -> float:
    """1 if the round's answer is ``target``: its report's own correct / wrong flag first, else its stated value
    (NaN when it states none, or there is no target)."""
    if r.correct_flag:
        return 1.0
    if r.wrong_flag:
        return 0.0
    return float(value_key(r.value_norm) == target) if r.value_norm is not None and target is not None else np.nan


def build(rounds: pd.DataFrame, mentions: pd.DataFrame, cons: dict, agent_col: str,
          audience: dict[str, set[str]] | None = None, wall_clock: bool = True,
          winners: dict | None = None) -> pd.DataFrame:
    """``winners`` maps an overturned A4 slot (family, item) to its winning value (``y_winner``).

    ``audience`` maps a restricted channel to the agent labels that could read it;
    a mention there counts as public only for those agents (channels absent from
    the map are public to all).  A round whose report has no timestamp is dropped:
    nothing can be ordered against it (the report counts these)."""
    m = mentions[mentions["family"] != "url"]
    m = m.assign(_kind=m["source_kind"] if "source_kind" in m else None)
    groups = {k: g.sort_values(["ts", "event_id"], kind="stable") for k, g in m.groupby(["family", "item", "value_key"])}
    audience = audience or {}

    def readable(g: pd.DataFrame, agent) -> pd.DataFrame:
        if not audience or "channel" not in g or not len(g):
            return g
        ok = np.array([c not in audience or agent in audience[c] for c in g["channel"]], dtype=bool)
        return g.loc[ok]
    rows = []
    rounds = rounds[rounds["t_report"].notna()]
    rounds = rounds.astype(object).where(rounds.notna(), None)
    for r in rounds.itertuples(index=False):
        cv, win = cons.get((r.family, r.item)), (winners or {}).get((r.family, r.item))
        target = match_key(r.value_norm, cv)
        t_pub = src = t_self = src_ev = src_ch = vis_ev = self_ev = t_last = src_last = last_ev = None
        g = groups.get((r.family, r.item, target)) if target else None
        t_vis = t_src_del = t_look = look_ev = t_task = task_ev = None
        src_gone = False
        if g is not None:
            before = readable(g[g["ts"] < r.t_report], r.agent)
            look = before[(before["_kind"] == "tool_result") & (before[agent_col] == r.agent)]
            if len(look):
                t_look, look_ev = look["ts"].iloc[0], look["event_id"].iloc[0]
            task = before[before["_kind"] == "task"]
            if len(task):
                t_task, task_ev = task["ts"].iloc[0], task["event_id"].iloc[0]
            before = before[~before["_kind"].isin(PRIVATE_KINDS)]
            others = before[before[agent_col] != r.agent]
            if len(others):
                first, last = others.iloc[0], others.iloc[-1]
                t_pub, src, src_ev = first["ts"], first[agent_col], first["event_id"]
                t_last, src_last, last_ev = last["ts"], last[agent_col], last["event_id"]
                src_ch = first["channel"] if "channel" in others else None
                if "visible_until" in others:
                    vu = pd.to_datetime(others["visible_until"], utc=True, errors="coerce")
                    still = others[vu.isna() | (vu > r.t_report)]
                    if len(still):
                        t_vis, vis_ev = still["ts"].iloc[0], still["event_id"].iloc[0]
                    src_gone = bool(pd.notna(vu.iloc[0]) and vu.iloc[0] <= r.t_report)
                    t_src_del = vu.iloc[0] if pd.notna(vu.iloc[0]) else None
                else:
                    t_vis, vis_ev = t_pub, src_ev
            own = before[before[agent_col] == r.agent]
            if len(own):
                t_self, self_ev = own["ts"].iloc[0], own["event_id"].iloc[0]
        gap = (r.t_report - t_pub).total_seconds() if t_pub is not None else np.nan
        # Treatment for the causal model: was the *consensus* value public?  Independent
        # of what this agent answered, so a wrong answer cannot mechanically force D=0.
        t_cons = None
        gc = groups.get((r.family, r.item, cv)) if cv else None
        if gc is not None:
            oc = readable(gc[(gc["ts"] < r.t_report) & (gc[agent_col] != r.agent) & ~gc["_kind"].isin(PRIVATE_KINDS)],
                          r.agent)
            if len(oc):
                t_cons = oc["ts"].iloc[0]
        gap_c = (r.t_report - t_cons).total_seconds() if t_cons is not None else np.nan
        rows.append({
            "claim_id": r.claim_id, "report_event_id": getattr(r, "report_event_id", None),
            "agent": r.agent, "family": r.family, "episode": r.episode,
            "item": r.item, "value_norm": r.value_norm, "t_report": r.t_report,
            "t_public": t_pub, "src_agent": src, "t_self_prepared": t_self, "gap_s": gap,
            "D": int(t_pub is not None),
            "D_w10m": int(t_pub is not None and gap >= 600),
            "D_w60m": int(t_pub is not None and gap >= 3600),
            "self_prepared": int(t_self is not None),
            "D_cons": int(t_cons is not None),
            "D_cons_w60m": int(t_cons is not None and gap_c >= 3600),
            "t_cons": t_cons, "gap_cons_s": gap_c,
            "y_instant": int(r.latency_class == "instant"),
            "latency_known": int(r.latency_class is not None),
            "y_consensus": _score(r, cv), "y_winner": _score(r, win) if win is not None else np.nan,
            "consensus_value": cv,
            "latency_class": r.latency_class, "wrong_flag": r.wrong_flag,
            "t_public_visible": t_vis, "D_visible": int(t_vis is not None),
            "src_deleted_before_report": int(src_gone), "t_src_deleted": t_src_del,
            "src_event_id": src_ev, "src_channel": src_ch, "src_visible_event_id": vis_ev, "self_event_id": self_ev,
            "src_latest": src_last, "t_latest": t_last, "latest_event_id": last_ev,
            "t_own_lookup": t_look, "own_lookup_event_id": look_ev,
            "own_lookup_evidence": int(t_pub is None and t_look is not None),
            "t_task": t_task, "task_event_id": task_ev,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["item_fe"] = out["family"] + "|" + out["item"]
        if not wall_clock:
            out[["gap_s", "D_w10m", "D_w60m", "D_cons_w60m", "gap_cons_s"]] = np.nan
    return out
