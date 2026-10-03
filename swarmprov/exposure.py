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
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .claims import value_key


def build(rounds: pd.DataFrame, mentions: pd.DataFrame, cons: dict, agent_col: str) -> pd.DataFrame:
    m = mentions[mentions["family"] != "url"]
    groups = {k: g.sort_values("ts") for k, g in m.groupby(["family", "item", "value_key"])}
    rows = []
    rounds = rounds.astype(object).where(rounds.notna(), None)
    for r in rounds.itertuples(index=False):
        cv = cons.get((r.family, r.item))
        target = value_key(r.value_norm) or cv
        t_pub = src = t_self = None
        g = groups.get((r.family, r.item, target)) if target else None
        if g is not None:
            before = g[g["ts"] < r.t_report]
            others = before[before[agent_col] != r.agent]
            if len(others):
                t_pub, src = others["ts"].iloc[0], others[agent_col].iloc[0]
            own = before[before[agent_col] == r.agent]
            if len(own):
                t_self = own["ts"].iloc[0]
        gap = (r.t_report - t_pub).total_seconds() if t_pub is not None else np.nan
        # Treatment for the causal model: was the *consensus* value public?  Independent
        # of what this agent answered, so a wrong answer cannot mechanically force D=0.
        t_cons = None
        gc = groups.get((r.family, r.item, cv)) if cv else None
        if gc is not None:
            oc = gc[(gc["ts"] < r.t_report) & (gc[agent_col] != r.agent)]
            if len(oc):
                t_cons = oc["ts"].iloc[0]
        gap_c = (r.t_report - t_cons).total_seconds() if t_cons is not None else np.nan
        if r.correct_flag:
            y_cons = 1.0
        elif r.wrong_flag:
            y_cons = 0.0
        elif r.value_norm is not None and cv is not None:
            y_cons = float(value_key(r.value_norm) == cv)
        else:
            y_cons = np.nan
        rows.append({
            "claim_id": r.claim_id, "agent": r.agent, "family": r.family, "episode": r.episode,
            "item": r.item, "value_norm": r.value_norm, "t_report": r.t_report,
            "t_public": t_pub, "src_agent": src, "t_self_prepared": t_self, "gap_s": gap,
            "D": int(t_pub is not None),
            "D_w10m": int(t_pub is not None and gap >= 600),
            "D_w60m": int(t_pub is not None and gap >= 3600),
            "self_prepared": int(t_self is not None),
            "D_cons": int(t_cons is not None),
            "D_cons_w60m": int(t_cons is not None and gap_c >= 3600),
            "y_instant": int(r.latency_class == "instant"),
            "latency_known": int(r.latency_class is not None),
            "y_consensus": y_cons, "consensus_value": cv,
            "latency_class": r.latency_class, "wrong_flag": r.wrong_flag,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["item_fe"] = out["family"] + "|" + out["item"]
    return out
