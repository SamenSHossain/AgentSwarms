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
* ``self_originated``  it did so before any other agent's copy (``t_self_prepared <
  t_public``): D stays 1, as the value was public before the report, but no source is
  credited with the answer (``credited``), so it gives no exposure edge and no A5 credit.
* ``D_cons``    the *consensus* value was public before the report.  This is the
  causal treatment: unlike D it does not depend on the agent's own answer, so
  "wrong answer => nothing matching was public" cannot manufacture an effect.
  ``t_cons`` is that first public consensus copy and ``gap_cons_s`` its head start
  over the report (NaN when D_cons = 0), the dose A2's exploratory block reads.
* ``y_consensus``  the answer matches the consensus value (the value most distinct
  agents carried); ``y_winner`` whether the stated value is A4's winner (the later
  majority), only on slots A4 calls overturned (``winners``), NaN elsewhere and for
  a round that states no value: a report's own correct / wrong flag grades it against
  the task, not against the winner.
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


def credited(ex: pd.DataFrame) -> pd.Series:
    """The exposed rounds a source is credited with: D == 1 and not self-originated (the agent posted the value
    before any other agent did).  Exposure edges, A5's attribution and origins, and the agent profiles read it."""
    return (ex["D"] == 1) & (ex["self_originated"] == 0)


def _score(r, target) -> float:
    """1 if the round's answer is ``target``: its report's own correct / wrong flag first, else its stated value
    (NaN when it states none, or there is no target)."""
    if r.correct_flag:
        return 1.0
    if r.wrong_flag:
        return 0.0
    return float(value_key(r.value_norm) == target) if r.value_norm is not None and target is not None else np.nan


def _facts(m: pd.DataFrame, agent_col: str, audience: dict) -> dict:
    """(family, item, value_key) -> its mentions as arrays in (ts, event_id) order: ``t`` (datetime64, searched),
    ``ts`` (the times themselves), ``agent``, ``event_id``, ``channel``, the ``private`` / ``tool`` / ``task`` kind
    flags, ``restricted`` (a channel in ``audience``) and ``vu`` / ``vu_ts`` (visible_until), when there is one."""
    if not len(m):
        return {}
    m = m.sort_values(["ts", "event_id"], kind="stable")
    kind = m["source_kind"] if "source_kind" in m else pd.Series(None, index=m.index, dtype=object)
    cols = {"t": pd.to_datetime(m["ts"], utc=True).dt.tz_convert(None).to_numpy("datetime64[ns]"),
            "ts": m["ts"].to_numpy(object), "agent": m[agent_col].to_numpy(object),
            "event_id": m["event_id"].to_numpy(object), "private": kind.isin(PRIVATE_KINDS).to_numpy(bool),
            "tool": (kind == "tool_result").to_numpy(bool), "task": (kind == "task").to_numpy(bool)}
    if "channel" in m:
        cols["channel"] = m["channel"].to_numpy(object)
        cols["restricted"] = np.array([c in audience for c in cols["channel"]], dtype=bool)
    if "visible_until" in m:
        vu = pd.to_datetime(m["visible_until"], utc=True, errors="coerce")
        cols |= {"vu": vu.dt.tz_convert(None).to_numpy("datetime64[ns]"), "vu_ts": vu.to_numpy(object)}
    return {k: {c: v[i] for c, v in cols.items()}
            for k, i in m.groupby(["family", "item", "value_key"], sort=False).indices.items()}


def build(rounds: pd.DataFrame, mentions: pd.DataFrame, cons: dict, agent_col: str,
          audience: dict[str, set[str]] | None = None, wall_clock: bool = True,
          winners: dict | None = None) -> pd.DataFrame:
    """``winners`` maps an overturned A4 slot (family, item) to its winning value (``y_winner``).

    ``audience`` maps a restricted channel to the agent labels that could read it;
    a mention there counts as public only for those agents (channels absent from
    the map are public to all).  A round whose report has no timestamp is dropped:
    nothing can be ordered against it (the report counts these)."""
    audience = audience or {}
    m = mentions[mentions["family"] != "url"]
    facts = _facts(m, agent_col, audience)

    def before(f: dict, agent, t) -> tuple[int, np.ndarray]:
        """How many of the fact's mentions precede ``t``, and which of those ``agent`` could read."""
        n = int(np.searchsorted(f["t"], t, side="left"))
        ok = np.ones(n, dtype=bool)
        if audience and "channel" in f:
            for i in np.flatnonzero(f["restricted"][:n]):
                ok[i] = agent in audience[f["channel"][i]]
        return n, ok
    rows = []
    rounds = rounds[rounds["t_report"].notna()]
    rounds = rounds.astype(object).where(rounds.notna(), None)
    for r in rounds.itertuples(index=False):
        cv, win = cons.get((r.family, r.item)), (winners or {}).get((r.family, r.item))
        target = match_key(r.value_norm, cv)
        t = np.datetime64(pd.Timestamp(r.t_report).value, "ns")
        t_pub = src = t_self = src_ev = src_ch = vis_ev = self_ev = t_last = src_last = last_ev = None
        f = facts.get((r.family, r.item, target)) if target else None
        t_vis = t_src_del = t_look = look_ev = t_task = task_ev = None
        src_gone = False
        if f is not None:
            n, ok = before(f, r.agent, t)
            mine = f["agent"][:n] == r.agent
            if len(look := np.flatnonzero(ok & f["tool"][:n] & mine)):
                t_look, look_ev = f["ts"][look[0]], f["event_id"][look[0]]
            if len(task := np.flatnonzero(ok & f["task"][:n])):
                t_task, task_ev = f["ts"][task[0]], f["event_id"][task[0]]
            pub = ok & ~f["private"][:n]
            others = np.flatnonzero(pub & ~mine)
            if len(others):
                i, j = others[0], others[-1]
                t_pub, src, src_ev = f["ts"][i], f["agent"][i], f["event_id"][i]
                t_last, src_last, last_ev = f["ts"][j], f["agent"][j], f["event_id"][j]
                src_ch = f["channel"][i] if "channel" in f else None
                if "vu" in f:
                    vu = f["vu"][others]
                    still = others[np.isnat(vu) | (vu > t)]
                    if len(still):
                        t_vis, vis_ev = f["ts"][still[0]], f["event_id"][still[0]]
                    src_gone = bool(not np.isnat(vu[0]) and vu[0] <= t)
                    t_src_del = None if np.isnat(vu[0]) else f["vu_ts"][i]
                else:
                    t_vis, vis_ev = t_pub, src_ev
            if len(own := np.flatnonzero(pub & mine)):
                t_self, self_ev = f["ts"][own[0]], f["event_id"][own[0]]
        gap = (r.t_report - t_pub).total_seconds() if t_pub is not None else np.nan
        # Treatment for the causal model: was the *consensus* value public?  Independent
        # of what this agent answered, so a wrong answer cannot mechanically force D=0.
        t_cons = None
        c = facts.get((r.family, r.item, cv)) if cv else None
        if c is not None:
            n, ok = before(c, r.agent, t)
            if len(oc := np.flatnonzero(ok & (c["agent"][:n] != r.agent) & ~c["private"][:n])):
                t_cons = c["ts"][oc[0]]
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
            "self_originated": int(t_self is not None and t_pub is not None and t_self < t_pub),
            "D_cons": int(t_cons is not None),
            "D_cons_w60m": int(t_cons is not None and gap_c >= 3600),
            "t_cons": t_cons, "gap_cons_s": gap_c,
            "y_instant": int(r.latency_class == "instant"),
            "latency_known": int(r.latency_class is not None),
            "y_consensus": _score(r, cv),
            "y_winner": float(value_key(r.value_norm) == win) if win is not None and r.value_norm is not None else np.nan,
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
