"""AI Village side tables: the agent directory and the chat rooms.

The Village publishes several tables next to its transcripts.  Three of them
carry context the provenance pipeline needs and a transcript alone lacks:

* ``agents``      - agent id -> display name and model; names are what chat
                    authors and room allow/deny lists use, ids are what goals
                    use, so this table joins the roster to the transcript;
* ``chat_rooms``  - room id -> name, creation/deletion (channel lifecycle) and
                    ``whitelisted_agent_names`` / ``blacklisted_agent_names``:
                    a post in a restricted room was never public to the agents
                    outside it, so exposure must be judged per audience;
* ``agent_goals`` - the roster (see :mod:`swarmprov.roster`);
* ``village_goals`` - the shared goals: one window per goal the whole village
                    was given (weekly, contiguous).  A shared goal is a task
                    family for everyone, so a post that no agent-specific goal
                    covers takes the era it falls in.

Tables are recognised by their columns, not file names, so an upload prefix
or a rename does not matter.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

import pandas as pd

from .schema import conform

# table kind -> columns that identify it (checked in order)
SIGNATURES = {
    "goals": {"agent_id", "start_time"},
    "agents": {"id", "name", "model_string"},
    "rooms": {"id", "name", "deleted_at"},
    "sessions": {"agent_id", "created_at"},   # a presence log: one row per session start
    "eras": {"goal", "start_time"},           # shared goals: windows without an agent id
}
STOPWORDS = {"a", "an", "the", "your", "you", "yours", "as", "can", "it", "to", "and", "of", "in", "on", "for", "with",
             "like", "whatever", "youd", "do", "each", "agent", "agents", "own", "be", "is", "that", "this", "up", "out",
             "much", "many", "most", "while", "one", "other", "which", "will", "next", "soon", "please", "yourselves"}
TEXT_KEYS = {"content", "text", "message", "body", "msg"}
VENDORS = [("claude", "Anthropic"), ("gpt", "OpenAI"), ("o1", "OpenAI"), ("o3", "OpenAI"), ("o4", "OpenAI"),
           ("gemini", "Google"), ("grok", "xAI"), ("deepseek", "DeepSeek"), ("kimi", "Moonshot"), ("glm", "Zhipu"),
           ("muse", "Meta"), ("llama", "Meta"), ("qwen", "Alibaba"), ("mistral", "Mistral")]


def classify(rows: list[dict]) -> str | None:
    """Which Village table a list of rows is, or None (a transcript, or nothing known)."""
    if not rows or not all(isinstance(r, dict) for r in rows[:20]):
        return None
    keys = set().union(*(r.keys() for r in rows[:20]))
    if keys & TEXT_KEYS:
        return "messages"
    for kind, sig in SIGNATURES.items():
        if sig <= keys and (kind != "goals" or ("short_name" in keys or "name" in keys)):
            return kind
    return None


def era_label(n: int, goal: str) -> str:
    """Short family name for a shared goal: ``e17-form-two-teams``."""
    words = [w for w in re.sub(r"[^a-z0-9 ]+", " ", str(goal).lower().replace("'", "")).split() if w not in STOPWORDS]
    return f"e{n:02d}-" + "-".join(words[:3] or ["goal"])


def normalize_eras(rows: list[dict]) -> pd.DataFrame:
    out = [{"era_id": str(r.get("id") or i), "goal": str(r.get("goal") or r.get("name") or ""),
            "start": _ts(r.get("start_time") or r.get("start")), "end": _ts(r.get("end_time") or r.get("end")),
            "created": _ts(r.get("created_at")), "updated": _ts(r.get("updated_at"))}
           for i, r in enumerate(rows) if isinstance(r, dict) and (r.get("goal") or r.get("name"))]
    if not out:
        raise ValueError("no shared-goal rows with a goal text")
    df = pd.DataFrame(out).sort_values(["start", "era_id"], kind="stable").reset_index(drop=True)
    df["label"] = [era_label(i + 1, g) for i, g in enumerate(df["goal"])]
    return conform(df, "eras")


def annotate_eras(events: pd.DataFrame, eras: pd.DataFrame) -> pd.DataFrame:
    """The shared goal in force when each post was made (None outside every window)."""
    e = eras.sort_values("start")
    starts = list(pd.to_datetime(e["start"], utc=True, errors="coerce"))
    ends = list(pd.to_datetime(e["end"], utc=True, errors="coerce"))
    ids, labels = e["era_id"].tolist(), e["label"].tolist()
    out_id, out_label = [], []
    for ts in pd.to_datetime(events["ts"], utc=True, errors="coerce"):
        hit = None
        if pd.notna(ts):
            for i in range(len(e)):
                if (pd.isna(starts[i]) or starts[i] <= ts) and (pd.isna(ends[i]) or ts < ends[i]):
                    hit = i  # the latest window containing ts wins
        out_id.append(ids[hit] if hit is not None else None)
        out_label.append(labels[hit] if hit is not None else None)
    return pd.DataFrame({"era_id": pd.Series(out_id, index=events.index, dtype="object"),
                         "era": pd.Series(out_label, index=events.index, dtype="object")})


def eras_summary(eras: pd.DataFrame, events: pd.DataFrame | None = None, roster: pd.DataFrame | None = None,
                 era_ann: pd.DataFrame | None = None) -> dict:
    e = eras.sort_values("start").reset_index(drop=True)
    days = (e["end"] - e["start"]).dt.total_seconds().div(86400)
    gaps = (e["start"].shift(-1) - e["end"]).dt.total_seconds().abs().dropna()
    s = {"n": int(len(e)), "first_start": e["start"].min(), "last_start": e["start"].max(),
         "n_open": int(e["end"].isna().sum()), "median_days": float(days.median()) if days.notna().any() else None,
         "n_gaps": int((gaps > 60).sum()), "handover": None}
    if roster is not None and len(roster) and roster["start"].notna().any():
        last = e.iloc[-1]
        if pd.notna(last["start"]) and abs((roster["start"].min() - last["start"]).total_seconds()) <= 3600:
            s["handover"] = {"label": last["label"], "goal": last["goal"], "start": last["start"]}
    tbl = e[["label", "start", "goal"]].assign(days=days.round(1))
    if events is not None and len(events) and era_ann is not None:
        per = era_ann["era"].value_counts()
        tbl = tbl.assign(posts=tbl["label"].map(per).fillna(0).astype(int))
        s["posts_outside"] = int(era_ann["era"].isna().sum())
        s["posts"] = int(len(events))
    s["table"] = tbl
    return s


def normalize_activity(rows: list[dict], directory: pd.DataFrame | None = None, kind: str = "session") -> pd.DataFrame:
    names = dict(zip(directory["agent_id"], directory["name"])) if directory is not None else {}
    ref_key = next((k for k in ("sdk_session_id", "session_id", "id") if rows and k in rows[0]), None)
    out = [{"agent_id": str(r["agent_id"]), "agent_name": names.get(str(r["agent_id"])),
            "ts": _ts(r.get("created_at")), "kind": kind, "ref": str(r.get(ref_key, "")) if ref_key else ""}
           for r in rows if isinstance(r, dict) and r.get("agent_id")]
    return conform(pd.DataFrame(out).sort_values("ts").reset_index(drop=True), "activity")


def activity_summary(act: pd.DataFrame, roster: pd.DataFrame | None, events: pd.DataFrame | None) -> dict:
    """Who the presence log covers, and whether it touches the goals or the transcript."""
    a = act.assign(agent=act["agent_name"].where(act["agent_name"].notna(), act["agent_id"]))
    per = (a.groupby("agent").agg(sessions=("ts", "size"), distinct_ids=("ref", "nunique"),
                                  first=("ts", "min"), last=("ts", "max"))
           .sort_values("sessions", ascending=False))
    hours = a["ts"].dt.hour.value_counts().sort_index()
    busy = hours[hours >= hours.max() * 0.25].index
    s = {"n_rows": int(len(a)), "n_agents": int(a["agent_id"].nunique()),
         "span": f"{a['ts'].min():%Y-%m-%d} → {a['ts'].max():%Y-%m-%d}",
         "weekend_share": float(a["ts"].dt.dayofweek.ge(5).mean()),
         "busy_hours_utc": f"{int(busy.min()):02d}–{int(busy.max()):02d}" if len(busy) else "",
         "per_agent": per}
    if roster is not None and len(roster):
        in_goal = 0
        for aid, g in a.groupby("agent_id"):
            w = roster[roster["agent_id"] == aid]
            for ts in g["ts"]:
                if ((w["start"].isna() | (w["start"] <= ts)) & (w["end"].isna() | (ts < w["end"]))).any():
                    in_goal += 1
        s["agents_in_roster"] = int(a.loc[a["agent_id"].isin(set(roster["agent_id"])), "agent_id"].nunique())
        s["rows_in_goal_window"] = in_goal
    if events is not None and len(events):
        authors = set(events["author_raw"].str.lower()) | set(events["author_alt"].astype(str))
        s["agents_in_transcript"] = int(sum(1 for aid, nm in zip(a["agent_id"].unique(), a.drop_duplicates("agent_id")["agent_name"])
                                            if aid in authors or (isinstance(nm, str) and nm.lower() in authors)))
    return s


def _ts(v):
    return pd.NaT if v in (None, "") else pd.to_datetime(v, utc=True, errors="coerce")


def vendor(model: str) -> str:
    m = (model or "").lower().split("::")[-1].split("/")[-1]
    for key, name in VENDORS:
        if m.startswith(key):
            return name
    return "other"


def normalize_directory(rows: list[dict]) -> pd.DataFrame:
    out = [{
        "agent_id": str(r["id"]), "name": str(r.get("name") or r["id"]), "model": str(r.get("model_string") or ""),
        "vendor": vendor(str(r.get("model_string") or r.get("name") or "")),
        "joined": _ts(r.get("created_at")), "last_seen": _ts(r.get("updated_at")),
        "participating": bool(r.get("is_participating", True)),
    } for r in rows if isinstance(r, dict) and r.get("id")]
    return conform(pd.DataFrame(out).sort_values("joined").reset_index(drop=True), "directory")


def _names(v) -> list[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return [s.strip() for s in v.split(",") if s.strip()]
    return [str(x) for x in v]


def normalize_channels(rows: list[dict]) -> pd.DataFrame:
    out = [{
        "channel": str(r.get("name") or r["id"]), "channel_id": str(r["id"]),
        "created": _ts(r.get("created_at")), "deleted": _ts(r.get("deleted_at")),
        "allow": _names(r.get("whitelisted_agent_names")), "deny": _names(r.get("blacklisted_agent_names")),
    } for r in rows if isinstance(r, dict) and r.get("id")]
    return conform(pd.DataFrame(out).sort_values("created").reset_index(drop=True), "channels")


def lifecycle_from_channels(ch: pd.DataFrame) -> pd.DataFrame:
    rows = [{"channel": c, "ts": t, "action": "create", "actor": ""} for c, t in zip(ch["channel"], ch["created"]) if pd.notna(t)]
    rows += [{"channel": c, "ts": t, "action": "delete", "actor": ""} for c, t in zip(ch["channel"], ch["deleted"]) if pd.notna(t)]
    return pd.DataFrame(rows, columns=["channel", "ts", "action", "actor"]).sort_values("ts").reset_index(drop=True)


def restricted(ch: pd.DataFrame) -> pd.DataFrame:
    return ch[ch["allow"].map(bool) | ch["deny"].map(bool)]


def audience(channels: pd.DataFrame, agents: pd.DataFrame, agent_col: str,
             directory: pd.DataFrame | None = None) -> dict[str, set[str]]:
    """channel -> set of agent labels (``agent_col`` values) that could read it.

    Only restricted channels appear; a channel absent from the map is public.
    Names in allow/deny lists are resolved to labels through the authors table
    (an author whose handle is the name) and the roster id -> directory name
    link, so a renamed author still lands on the right label."""
    name_to_label: dict[str, set[str]] = {}
    for raw, label in zip(agents["author_raw"], agents[agent_col]):
        name_to_label.setdefault(str(raw).lower(), set()).add(label)
    if directory is not None and "roster_agent" in agents:
        dname = dict(zip(directory["agent_id"], directory["name"].str.lower()))
        for rid, label in zip(agents["roster_agent"], agents[agent_col]):
            if isinstance(rid, str) and rid in dname:
                name_to_label.setdefault(dname[rid], set()).add(label)
    everyone = set(agents[agent_col])
    out = {}
    for ch, allow, deny in zip(channels["channel"], channels["allow"], channels["deny"]):
        if not allow and not deny:
            continue
        labels = set().union(*(name_to_label.get(n.lower(), set()) for n in allow)) if allow else set(everyone)
        for n in deny:
            labels -= name_to_label.get(n.lower(), set())
        out[ch] = labels
    return out


def summary(directory: pd.DataFrame | None, channels: pd.DataFrame | None, events: pd.DataFrame | None = None,
            activity: pd.DataFrame | None = None, roster: pd.DataFrame | None = None,
            unrecognised: list[str] | None = None, eras: pd.DataFrame | None = None,
            era_ann: pd.DataFrame | None = None) -> dict:
    s: dict = {}
    if unrecognised:
        s["unrecognised"] = list(unrecognised)
    if eras is not None and len(eras):
        s["eras"] = eras_summary(eras, events, roster, era_ann)
    if activity is not None and len(activity):
        s["activity"] = activity_summary(activity, roster, events)
    if directory is not None and len(directory):
        d = directory
        s["n_agents"] = int(len(d))
        s["n_participating"] = int(d["participating"].sum())
        s["joined"] = f"{d['joined'].min():%Y-%m-%d} → {d['joined'].max():%Y-%m-%d}"
        s["vendors"] = (d.groupby("vendor").agg(agents=("agent_id", "count"), participating=("participating", "sum"),
                                                 models=("model", lambda m: ", ".join(sorted(set(x.split("::")[-1].split("/")[-1] for x in m)))))
                        .sort_values("agents", ascending=False))
        s["joins_by_month"] = d["joined"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    if channels is not None and len(channels):
        c = channels.copy()
        c["lifetime_h"] = ((c["deleted"] - c["created"]).dt.total_seconds() / 3600).round(1)  # blank while open
        c["access"] = [("only " + ", ".join(a)) if a else (("all but " + ", ".join(dn)) if dn else "everyone")
                       for a, dn in zip(c["allow"], c["deny"])]
        s["n_rooms"] = int(len(c))
        s["n_deleted"] = int(c["deleted"].notna().sum())
        s["n_restricted"] = int(len(restricted(c)))
        s["rooms"] = c[["channel", "created", "deleted", "lifetime_h", "access"]]
        if events is not None and len(events):
            per = events.groupby("channel").size()
            s["rooms"] = s["rooms"].assign(posts=s["rooms"]["channel"].map(per).fillna(0).astype(int))
            s["posts_in_restricted"] = int(per.reindex(restricted(c)["channel"]).fillna(0).sum())
            s["unknown_channels"] = sorted(set(events["channel"]) - set(c["channel"]))
    return s


def section(s: dict, md_table) -> list[str]:
    if not s:
        return []
    L = ["## Village\n"]
    if "n_agents" in s:
        L.append(f"Directory: {s['n_agents']} agents ({s['n_participating']} participating at export), joined {s['joined']}.\n")
        L.append(md_table(s["vendors"]))
    if "n_rooms" in s:
        L.append(f"\nRooms: {s['n_rooms']}, {s['n_deleted']} deleted, {s['n_restricted']} with an allow/deny list. "
                 "A post in a restricted room was public only to the agents listed, so exposure is judged per audience: "
                 "an answer that was only ever posted where an agent could not read it counts as *not* public for that agent.\n")
        if "posts_in_restricted" in s:
            L.append(f"{s['posts_in_restricted']:,} transcript posts are in restricted rooms"
                     + (f"; channels not in the room table: {', '.join(s['unknown_channels'])}" if s["unknown_channels"] else "") + ".\n")
        L.append(md_table(s["rooms"], index=False, floatfmt="{:.1f}"))
    if "eras" in s:
        e = s["eras"]
        L.append(f"\nShared goals: {e['n']} windows from {e['first_start']:%Y-%m-%d} to {e['last_start']:%Y-%m-%d} (last start), "
                 f"median {e['median_days']:.1f} days each, {e['n_gaps']} gap(s) between consecutive windows, {e['n_open']} still open. "
                 "A shared goal is the task family of every post in its window that no agent-specific goal covers."
                 + (f" The village switched from shared to individual goals on {e['handover']['start']:%Y-%m-%d %H:%M} UTC "
                    f"(\"{e['handover']['goal']}\"), the minute the first per-agent goal starts." if e.get("handover") else "")
                 + (f" {e['posts_outside']:,} of {e['posts']:,} posts fall outside every window." if "posts" in e else "") + "\n")
        L.append(md_table(e["table"], index=False, floatfmt="{:.1f}"))
    if "activity" in s:
        a = s["activity"]
        L.append(f"\nActivity log: {a['n_rows']:,} session starts by {a['n_agents']} agent(s), {a['span']}, "
                 f"busiest {a['busy_hours_utc']} UTC, {a['weekend_share']:.0%} at weekends. ")
        notes = []
        if "agents_in_roster" in a:
            notes.append(f"{a['agents_in_roster']} of these agents hold a goal in the roster and "
                         f"{a['rows_in_goal_window']:,} session starts fall inside a goal window")
        if "agents_in_transcript" in a:
            notes.append(f"{a['agents_in_transcript']} of them post in the transcript")
        if notes:
            L.append("; ".join(notes) + ". ")
        if a.get("agents_in_roster") == 0 and "agents_in_transcript" not in a:
            L.append("The log covers none of the agents under study, so it cannot bound when they could have read anything; "
                     "it is kept as the `activity` table and otherwise ignored.")
        L.append("\n")
        L.append(md_table(a["per_agent"]))
    if "unrecognised" in s:
        L.append("\nFiles in the source directory no adapter recognised (not used): "
                 + ", ".join(f"`{f}`" for f in s["unrecognised"]) + "\n")
    return L
