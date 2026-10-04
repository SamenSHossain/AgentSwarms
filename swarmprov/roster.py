"""Roster: who the agents are and what each was asked to do, per time window.

The AI Village publishes ``agent_goals`` (one row per goal assignment: agent id,
short role name, full goal text, start/end).  A roster is not a transcript, so
it produces no posts; it does three things for the pipeline:

* **families** - an agent's current goal is its task family, so two agents
  given the same goal are compared with each other and not with the rest;
* **identity** - the goal assignment batch (goals created together) is the
  cohort, and ``agent_merged`` becomes ``cohort|role`` from the roster
  instead of from parsed signatures;
* **lifecycle** - goal windows say when an agent was active on what, so posts
  outside every window and goals with no posts are flagged in the report.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .schema import conform

# column aliases, first match wins
FIELDS = {
    "goal_id": ["id", "goal_id"],
    "agent_id": ["agent_id", "agent", "member_id", "agent_uuid"],
    "role": ["short_name", "role", "short", "label"],
    "goal": ["name", "goal", "objective", "task", "title"],
    "detail": ["description", "detail", "notes", "instructions"],
    "start": ["start_time", "start", "started_at", "since", "from"],
    "end": ["end_time", "end", "ended_at", "until", "to"],
    "created": ["created_at", "created"],
    "updated": ["updated_at", "updated"],
}
# goals created within this many seconds of each other were assigned together
BATCH_GAP_S = 600
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def looks_like_roster(rows: list[dict]) -> bool:
    """Dict rows with an agent id, a role or goal, a start, and no message text."""
    if not rows or not all(isinstance(r, dict) for r in rows):
        return False
    keys = set().union(*(r.keys() for r in rows))
    has = lambda f: any(k in keys for k in FIELDS[f])
    texty = {"content", "text", "message", "body", "msg"} & keys
    return has("agent_id") and (has("role") or has("goal")) and has("start") and not texty


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def _pick(r: dict, field: str):
    for k in FIELDS[field]:
        v = r.get(k)
        if v is not None and v != "" and not (isinstance(v, float) and v != v):  # NaN from CSV
            return v
    return None


def _ts(v) -> pd.Timestamp:
    return pd.NaT if v is None else pd.to_datetime(v, utc=True, errors="coerce")


def normalize(rows: list[dict]) -> pd.DataFrame:
    """Rows from any roster-like export -> the canonical ``roster`` table."""
    out = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or _pick(r, "agent_id") is None:
            continue
        role = _pick(r, "role") or _pick(r, "goal") or "unknown"
        out.append({
            "goal_id": str(_pick(r, "goal_id") or f"goal{i}"),
            "agent_id": str(_pick(r, "agent_id")),
            "role": slug(role),
            "goal": str(_pick(r, "goal") or role),
            "detail": str(_pick(r, "detail") or ""),
            "start": _ts(_pick(r, "start")),
            "end": _ts(_pick(r, "end")),
            "created": _ts(_pick(r, "created")),
            "updated": _ts(_pick(r, "updated")),
        })
    if not out:
        raise ValueError("no roster rows with an agent id")
    df = pd.DataFrame(out).sort_values(["start", "agent_id"], kind="stable").reset_index(drop=True)
    df["batch"] = batches(df)
    return conform(df, "roster")


def batches(df: pd.DataFrame) -> pd.Series:
    """Cohort token (``Jul03``) for the assignment batch each goal belongs to.

    Goals are batched by creation time (falling back to start) with gaps under
    ``BATCH_GAP_S``; a batch is named after the day its first goal was created."""
    t = df["created"].where(df["created"].notna(), df["start"])
    order = t.sort_values(kind="stable").index
    tokens, cur, prev = {}, None, None
    for i in order:
        ti = t[i]
        if pd.isna(ti):
            tokens[i] = ""
            continue
        if cur is None or (ti - prev).total_seconds() > BATCH_GAP_S:
            cur = f"{MONTHS[ti.month - 1]}{ti.day:02d}"
            n = sum(1 for v in set(tokens.values()) if v.startswith(cur))
            cur = cur if n == 0 else f"{cur}{chr(ord('a') + n)}"  # second batch the same day -> Jul03b
        tokens[i] = cur
        prev = ti
    return pd.Series([tokens[i] for i in df.index], index=df.index)


def eras(roster: pd.DataFrame) -> pd.DataFrame:
    """Per agent, its goals in order with what changed between consecutive ones."""
    rows = []
    for aid, g in roster.sort_values("start").groupby("agent_id", sort=False):
        prev = None
        for _, r in g.iterrows():
            change = "first"
            if prev is not None:
                change = "reworded" if prev["role"] == r["role"] else "reassigned"
                if pd.notna(prev["end"]) and pd.notna(r["start"]) and r["start"] < prev["end"]:
                    change += "+overlap"
            rows.append({"agent_id": aid, "role": r["role"], "start": r["start"], "end": r["end"],
                         "open": pd.isna(r["end"]), "change": change, "batch": r["batch"], "goal": r["goal"]})
            prev = r
    return pd.DataFrame(rows)


def annotate(events: pd.DataFrame, roster: pd.DataFrame, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """For each event: the roster agent and the goal it was working at the time.

    An author is matched to a roster agent by ``author_alt`` or ``author_raw``
    equal to the agent id, by a config alias, or by its name equalling a role
    that only one agent holds (an ambiguous role name stays unmatched and is
    listed in the report, so the analyst can add an alias).
    The goal is the window containing ``ts``; a post before an agent's first
    goal or after its last closed one gets the nearest goal and ``in_window``
    False, so coverage problems are visible instead of silently absorbed."""
    aliases = {str(k).lower(): v for k, v in (aliases or {}).items()}
    ids = set(roster["agent_id"])
    holders = roster.groupby("role")["agent_id"].unique()
    by_role = {role: ids[0] for role, ids in holders.items() if len(ids) == 1}
    windows = {aid: g.sort_values("start") for aid, g in roster.groupby("agent_id")}

    def agent_of(raw, alt) -> str | None:
        for cand in (str(alt), str(raw)):
            if cand in ids:
                return cand
            if cand.lower() in aliases and aliases[cand.lower()] in ids:
                return aliases[cand.lower()]
        return by_role.get(slug(raw))

    out = {"roster_agent": [], "role": [], "goal_id": [], "batch": [], "in_window": []}
    for raw, alt, ts in zip(events["author_raw"], events["author_alt"].fillna(""), events["ts"]):
        aid = agent_of(raw, alt)
        if aid is None:
            for k in out:
                out[k].append(None)
            continue
        w = windows[aid]
        inside = w[(w["start"].isna() | (w["start"] <= ts)) & (w["end"].isna() | (ts < w["end"]))]
        if len(inside):
            r, ok = inside.iloc[-1], True
        else:  # nearest window edge
            dist = pd.concat([(w["start"] - ts).abs(), (w["end"] - ts).abs()], axis=1).min(axis=1)
            r, ok = w.loc[dist.idxmin()] if dist.notna().any() else w.iloc[-1], False
        out["roster_agent"].append(aid)
        out["role"].append(r["role"])
        out["goal_id"].append(r["goal_id"])
        out["batch"].append(r["batch"])
        out["in_window"].append(ok)
    return pd.DataFrame({k: pd.Series(v, index=events.index, dtype="object") for k, v in out.items()})


def merged_id(roster: pd.DataFrame) -> dict[str, str]:
    """Roster agent id -> ``cohort|role`` label, unique per agent.

    When one batch gave the same role to several agents (two Twitterati on
    2026-07-03), the label carries the start of the agent id so the two are
    never merged: the roster is ground truth for *who* is distinct."""
    first = roster.sort_values("start").drop_duplicates("agent_id")
    first = first.assign(cohort=first["batch"].where(first["batch"].astype(bool), None))
    multiplicity = first.groupby(["cohort", "role"], dropna=False)["agent_id"].transform("nunique")
    out = {}
    for aid, cohort, role, n in zip(first["agent_id"], first["cohort"], first["role"], multiplicity):
        label = f"{cohort}|{role}" if cohort else role
        out[aid] = label if n == 1 else f"{label}|{aid[:8]}"
    return out


def merge_agents(agents: pd.DataFrame, events: pd.DataFrame, ann: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Rewrite ``agent_merged``/``cohort`` for authors the roster identifies.

    The roster is ground truth for the merge: cohort = the batch of the agent's
    first goal, family = its role, and distinct roster agents stay distinct."""
    agents = agents.copy()
    agents["roster_agent"] = None
    hit = ann["roster_agent"].notna()
    if not hit.any():
        return agents
    labels = merged_id(roster)
    first_batch = roster.sort_values("start").drop_duplicates("agent_id").set_index("agent_id")["batch"]
    for author, g in ann[hit].assign(_a=events.loc[hit, "author_raw"]).groupby("_a"):
        aid = Counter(g["roster_agent"]).most_common(1)[0][0]
        sel = agents["author_raw"] == author
        agents.loc[sel, "cohort"] = first_batch[aid] or None
        agents.loc[sel, "agent_merged"] = labels[aid]
        agents.loc[sel, "roster_agent"] = aid
    return agents


def summary(roster: pd.DataFrame, events: pd.DataFrame | None = None, ann: pd.DataFrame | None = None) -> dict:
    er = eras(roster)
    roles = (roster.groupby("role")
             .agg(agents=("agent_id", "nunique"), goals=("goal_id", "count"),
                  first_start=("start", "min"), last_end=("end", "max"), open=("end", lambda s: int(s.isna().sum())))
             .sort_values(["agents", "first_start"], ascending=[False, True]))
    bt = (roster.groupby("batch").agg(goals=("goal_id", "count"), agents=("agent_id", "nunique"),
                                      created=("created", "min"), starts=("start", "min"),
                                      roles=("role", lambda s: ", ".join(sorted(set(s)))))
          .sort_values("created"))
    with_detail = roster[roster["detail"].astype(bool)][["role", "detail"]]
    out = {
        "n_goals": int(len(roster)), "n_agents": int(roster["agent_id"].nunique()),
        "n_roles": int(roster["role"].nunique()), "n_open": int(roster["end"].isna().sum()),
        "first_start": roster["start"].min(), "last_start": roster["start"].max(), "last_end": roster["end"].max(),
        "changes": er["change"].value_counts().to_dict(),
        "shared_roles": roles[roles["agents"] > 1].index.tolist(),
        "roles": roles, "batches": bt, "eras": er[er["change"] != "first"], "detail": with_detail,
    }
    if events is not None and ann is not None and len(events):
        hit = ann["roster_agent"].notna()
        out["coverage"] = {
            "posts": int(len(events)), "posts_matched": int(hit.sum()),
            "authors": int(events["author_raw"].nunique()),
            "authors_matched": int(events.loc[hit, "author_raw"].nunique()),
            "posts_outside_window": int((hit & ~ann["in_window"].fillna(False).astype(bool)).sum()),
            "goals_with_posts": int(ann.loc[hit, "goal_id"].nunique()),
            "unmatched_authors": events.loc[~hit, "author_raw"].value_counts().head(10).to_dict(),
        }
    return out


def section(s: dict, md_table) -> list[str]:
    """Markdown lines for the report's roster block."""
    L = ["## Roster\n"]
    ch = s["changes"]
    L.append(f"{s['n_goals']} goal assignments to {s['n_agents']} agents in {s['n_roles']} roles, "
             f"starting between {s['first_start']:%Y-%m-%d} and {s['last_start']:%Y-%m-%d}; "
             f"{s['n_open']} still open at export. Agents given a second goal: {ch.get('reworded', 0)} reworded, "
             f"{ch.get('reassigned', 0)} reassigned. Roles held by more than one agent (comparable tasks): "
             f"{', '.join(s['shared_roles']) or 'none'}.\n")
    L.append("Assignment batches (goals created together; the batch is the agent's cohort):\n")
    L.append(md_table(s["batches"]))
    L.append("\nRoles:\n")
    L.append(md_table(s["roles"]))
    if len(s["eras"]):
        L.append("\nGoal changes:\n")
        L.append(md_table(s["eras"][["agent_id", "role", "change", "start", "end", "goal"]], index=False))
    if len(s["detail"]):
        L.append("\nGoals with extra instructions:\n")
        L.append(md_table(s["detail"], index=False))
    if "coverage" in s:
        c = s["coverage"]
        L.append(f"\nCoverage against the transcript: {c['posts_matched']:,}/{c['posts']:,} posts by "
                 f"{c['authors_matched']}/{c['authors']} authors matched a roster agent; "
                 f"{c['posts_outside_window']:,} matched posts fall outside every goal window of their agent; "
                 f"{c['goals_with_posts']}/{s['n_goals']} goals have posts.\n")
        if c["unmatched_authors"]:
            L.append("Unmatched authors (add `roster_aliases` to the config to map them): "
                     + ", ".join(f"`{a}` ({n})" for a, n in c["unmatched_authors"].items()) + "\n")
    return L
