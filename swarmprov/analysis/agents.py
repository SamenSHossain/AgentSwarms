"""Agent profiles: one row per agent under one identity mapping, from the stage tables.

The report's Agents section, ``swarmprov agent`` and the GraphML node attributes all read
``profile``, so each count has one definition.  Exposure is a floor (only captured posts are
searched), so ``n_no_prior_copy`` (answers with no earlier public copy, D == 0) is a ceiling on an
agent's answers with no earlier copy anywhere.  It is no bound on computed answers: an exposed answer
may still have been computed, and one with no earlier public copy may copy a post the run lacks.  One
or two answers say little about an agent, so no per-agent share is given below ``MIN_ROUNDS`` answers.
"""

from __future__ import annotations

import pandas as pd

from ..claims import PRIVATE_KINDS
from ..graph import to_networkx
from . import origins, structure

AGENT_COLS = {"merged": "agent_merged", "strict": "agent_strict"}
MIN_ROUNDS = 3
COLUMNS = ["label", "n_author_strings", "cohort", "family", "families", "n_posts", "first_seen", "last_seen",
           "n_answers", "n_no_prior_copy", "n_exposed", "exposed_k_of_n", "n_exposed_60m", "median_head_h",
           "n_self_prepared", "n_wrong", "answers_supplied", "items_first_sourced", "origin", "relay_in", "relay_out",
           "facts_originated", "reach_of_originated", "agents_reached", "betweenness", "techniques_used"]


def match_agents(run, name: str, mapping: str = "merged") -> list[str]:
    """Agent labels whose label or author string equals ``name`` ignoring case, else contains it."""
    ag, col, q = run.read("agents"), AGENT_COLS[mapping], name.casefold()
    lab, raw = ag[col].astype(str).str.casefold(), ag["author_raw"].astype(str).str.casefold()
    for hit in (lab.eq(q) | raw.eq(q), lab.str.contains(q, regex=False) | raw.str.contains(q, regex=False)):
        if hit.any():
            return sorted(set(ag.loc[hit, col]))
    return []


def _ranked(s: pd.Series) -> str:
    """Values by frequency (ties alphabetical), comma-joined."""
    return ", ".join(s.value_counts().sort_index().sort_values(ascending=False, kind="stable").index)


def profile(agents: pd.DataFrame, events: pd.DataFrame, tags: pd.DataFrame, ex: pd.DataFrame, edges: pd.DataFrame,
            chains: pd.DataFrame, mapping: str = "merged", wall_clock: bool = True,
            btw: pd.Series | None = None) -> pd.DataFrame:
    """One row per agent with a public post (task messages and tool results are no agent's posts).

    * identity: ``n_author_strings`` (as in ``agent_aliases.csv``), ``cohort``, ``family`` (the label's family
      part when the label was built as ``cohort|family``, else the agent's most frequent post family) and
      ``families`` (its post families by frequency);
    * answers (the exposure table): ``n_no_prior_copy`` (D == 0), ``n_exposed`` (D == 1), ``exposed_k_of_n`` (from
      ``MIN_ROUNDS`` answers on), ``n_exposed_60m`` and ``median_head_h`` (NA without a wall clock);
    * supply and graph: ``answers_supplied`` (exposure out-degree, as A5), ``items_first_sourced``, ``origin``
      (the origins of the answers it supplied, ``origins.by_source``), same-page ``relay_in`` / ``relay_out``,
      ``facts_originated`` (task values it posted first), ``reach_of_originated`` (their other carriers summed
      over those values: an agent carrying two of them counts twice) and ``agents_reached`` (the distinct other
      agents that carried any of them), ``betweenness`` (A5's, passed as ``btw`` or computed the same way).
    """
    col = AGENT_COLS[mapping]
    label = agents.drop_duplicates("author_raw").set_index("author_raw")[col]
    pub = events[~events["source_kind"].isin(PRIVATE_KINDS)] if "source_kind" in events else events
    posts = pub.assign(label=pub["author_raw"].map(label)).dropna(subset=["label"])
    out = posts.groupby("label")["ts"].agg(n_posts="size", first_seen="min", last_seen="max")
    out["n_author_strings"] = agents.groupby(col)["author_raw"].nunique()
    out["cohort"] = agents.dropna(subset=["cohort"]).groupby(col)["cohort"].agg(lambda s: s.mode().iloc[0])
    fam = tags[tags["family"].fillna("") != ""].groupby(col)["family"]
    out["families"] = fam.agg(_ranked)
    built = {lab: f for lab, c, f in zip(agents[col], agents["cohort"], agents["family"]) if lab == f"{c}|{f}" and c and f}
    out["family"] = [built.get(i) or (f.split(", ")[0] if isinstance(f, str) else "")
                     for i, f in zip(out.index, out["families"])]
    tech = tags[tags["techniques"].fillna("") != ""]
    out["techniques_used"] = (tech.assign(t=tech["techniques"].str.split(",")).explode("t")
                              .groupby(col)["t"].agg(lambda s: ", ".join(sorted(set(s.str.strip()) - {""}))))
    x = ex if len(ex) else pd.DataFrame(columns=["agent", "D", "self_prepared", "wrong_flag", "D_w60m", "gap_s"])
    g = x.groupby("agent")
    out = out.join(pd.DataFrame({"n_answers": g.size(), "n_no_prior_copy": g["D"].agg(lambda d: int((d == 0).sum())),
                                 "n_exposed": g["D"].sum(), "n_exposed_60m": g["D_w60m"].sum(),
                                 "n_self_prepared": g["self_prepared"].sum(),
                                 "n_wrong": g["wrong_flag"].agg(lambda w: int(w.fillna(False).astype(bool).sum())),
                                 "median_head_h": x[x["D"] == 1].groupby("agent")["gap_s"].median() / 3600}))
    tallies = ["n_answers", "n_no_prior_copy", "n_exposed", "n_exposed_60m", "n_self_prepared", "n_wrong"]
    out[tallies] = out[tallies].fillna(0).astype(int)
    out["n_exposed_60m"] = out["n_exposed_60m"].astype("Int64") if wall_clock else pd.array([pd.NA] * len(out), "Int64")
    out["median_head_h"] = out["median_head_h"].astype(float) if wall_clock else float("nan")
    out["exposed_k_of_n"] = [f"{k}/{n}" if n >= MIN_ROUNDS else None for k, n in zip(out["n_exposed"], out["n_answers"])]
    kind = edges["kind"]
    xe, rel = edges[kind == "exposure"], edges[kind == "relay"]
    task = chains[chains["family"] != "url"] if len(chains) else chains
    # every other carrier of a multi-carrier fact has one relay or cross-page hop into it for that fact
    hops = edges[kind.isin(["relay", "relay_xchannel"])].merge(
        task[["family", "item", "value_key", "originator"]].rename(columns={"value_key": "value_norm"}),
        on=["family", "item", "value_norm"]) if len(task) else pd.DataFrame(columns=["originator", "dst"])
    counts = {"answers_supplied": xe.groupby("src").size(),
              "items_first_sourced": xe.drop_duplicates(["src", "family", "item"]).groupby("src").size(),
              "relay_in": rel.groupby("dst").size(), "relay_out": rel.groupby("src").size(),
              "facts_originated": task.groupby("originator").size() if len(task) else pd.Series(dtype=int),
              "reach_of_originated": (task["n_agents"] - 1).groupby(task["originator"]).sum() if len(task)
              else pd.Series(dtype=int),
              "agents_reached": hops.groupby("originator")["dst"].nunique()}
    for c, v in counts.items():
        out[c] = v.reindex(out.index).fillna(0).astype(int)
    if len(ex) and (ex["D"] == 1).any():
        out["origin"] = origins.by_source(origins.origin_status(ex), ex)["origin"]
    if btw is None:
        btw = structure.betweenness(to_networkx(edges, kinds=structure.BROKER_KINDS))[0]
    out["betweenness"] = btw.reindex(out.index).fillna(0.0).astype(float)
    out = out.reindex(columns=COLUMNS[1:])
    for c in ("cohort", "families", "exposed_k_of_n", "origin", "techniques_used"):
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out.reset_index()[COLUMNS]
