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
                    was given (contiguous, typically a week: 1 to 38 days).  A
                    shared goal is a task family for everyone, so a post that
                    neither an agent-specific goal nor a configured text or
                    channel family covers takes the era it falls in;
* ``summaries``   - LLM-written digests (daily, per goal, per agent).  The daily
                    ones list timestamped events in Pacific time naming the
                    agents; the latest version of each day is parsed into a
                    *derived* timeline, kept apart from real posts unless the
                    config asks for ``digest_as_posts``;
* ``villages``    - one row of metadata: the export cut (``updated_at``), the
                    operating schedule (daily windows, timezone not stated),
                    whether chat was open, and which agent held the turn.  The
                    village runs one agent at a time; turn boundaries are not
                    exported, so timing stays on the wall clock.

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
    if {"type", "summary_target", "content"} <= keys:   # LLM digests carry text but are not messages
        return "summaries"
    if keys & TEXT_KEYS:
        return "messages"
    if {"id", "name"} <= keys and keys & {"schedule", "village_goal", "is_chat_open"}:
        return "meta"
    for kind, sig in SIGNATURES.items():
        if sig <= keys and (kind != "goals" or ("short_name" in keys or "name" in keys)):
            return kind
    return None


DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DIGEST_TZ = "America/Los_Angeles"   # the digests stamp events "PT"
EVENT_RX = re.compile(r"^\s*\d+\.\s*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)\s*(PT|PST|PDT|UTC)?\]\s*(.+?)\s*$", re.M)
STAMP_RX = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)\s*(PT|PST|PDT|UTC)?\]")
BULLET_RX = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$", re.M)
QUOTE_RX = re.compile(r"<quote[^>]*>(.*?)</quote>", re.S)


def normalize_summaries(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        if not isinstance(r, dict) or not r.get("content"):
            continue
        target = r.get("summary_target")
        day = int(target) if isinstance(target, str) and target.isdigit() else None
        content = str(r["content"])
        out.append({"summary_id": str(r.get("id")), "type": str(r.get("type") or ""), "target": target,
                    "date": _ts(r.get("summary_date")), "day": day, "generated_by": r.get("generated_by"),
                    "created": _ts(r.get("created_at")), "updated": _ts(r.get("updated_at")),
                    "n_events": len(EVENT_RX.findall(content)), "chars": len(content), "content": content})
    if not out:
        raise ValueError("no summary rows with content")
    df = pd.DataFrame(out).sort_values("created").reset_index(drop=True)
    date = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    key = [f"{t}|{g or ''}|{d}" for t, g, d in zip(df["type"], df["target"], date)]   # plain strings: no NA keys
    df["latest"] = ~pd.Series(key, index=df.index).duplicated(keep="last")   # the newest regeneration wins
    return conform(df, "summaries")


def _actor(text: str, names: list[str]) -> str | None:
    t = STAMP_RX.sub("", text).lstrip("*_\"' :-")   # a leading stamp or bold marker is not part of the name
    return next((n for n in names if t.startswith(n) and (len(t) == len(n) or not t[len(n)].isalnum())), None)


def parse_digest(summaries: pd.DataFrame, directory: pd.DataFrame | None = None) -> pd.DataFrame:
    """Timestamped lines of the latest summaries, in UTC, with the named actor.

    Three layouts occur: numbered event lines (``1. [2025-06-30 11:01:40 PT] o3 ...``,
    kind ``event``), narrative bullets with an inline stamp (kind ``note``), and
    ``<quote>`` blocks (``Speaker: words [stamp]``, kind ``quote``: the agent's
    own words as the summariser quoted them).  Only the latest version of each
    summary is read, so regenerated days are not counted twice."""
    names = sorted(directory["name"].astype(str), key=len, reverse=True) if directory is not None else []
    ids = dict(zip(directory["name"].astype(str), directory["agent_id"])) if directory is not None else {}
    rows = []
    latest = summaries[summaries["latest"].astype(bool)]
    for s in latest.itertuples(index=False):
        content, i = str(s.content), 0
        body = QUOTE_RX.sub(" ", content)          # quotes are parsed separately
        if s.type == "daily":
            events = EVENT_RX.findall(body)
            if events:
                for stamp, tz, text in events:
                    rows.append({"event_id": f"{s.summary_id}:{i}", "kind": "event", "stamp": stamp, "tz": tz or "PT",
                                 "actor": _actor(text, names), "text": text, "summary_id": s.summary_id, "day": s.day,
                                 "generated_by": s.generated_by}); i += 1
            else:
                for text in BULLET_RX.findall(body):
                    m = STAMP_RX.search(text)
                    if not m:
                        continue
                    rows.append({"event_id": f"{s.summary_id}:{i}", "kind": "note", "stamp": m.group(1), "tz": m.group(2) or "PT",
                                 "actor": _actor(text, names), "text": text, "summary_id": s.summary_id, "day": s.day,
                                 "generated_by": s.generated_by}); i += 1
        for q in QUOTE_RX.findall(content):
            m = STAMP_RX.search(q)
            if not m:
                continue
            q = q.strip()
            speaker = q.split(":", 1)[0].strip().strip("*_\"' ") if ":" in q else ""
            actor = speaker if speaker in ids else _actor(q, names)
            rows.append({"event_id": f"{s.summary_id}:{i}", "kind": "quote", "stamp": m.group(1), "tz": m.group(2) or "PT",
                         "actor": actor, "text": STAMP_RX.sub("", q).strip(), "summary_id": s.summary_id, "day": s.day,
                         "generated_by": s.generated_by}); i += 1
    if not rows:
        return conform(pd.DataFrame(), "digest")
    d = pd.DataFrame(rows)
    local = pd.to_datetime(d["stamp"], format="mixed", errors="coerce")
    d["ts"] = local.dt.tz_localize(DIGEST_TZ, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    utc = d["tz"].eq("UTC")
    if utc.any():
        d.loc[utc, "ts"] = local[utc].dt.tz_localize("UTC")
    d["actor_id"] = d["actor"].map(ids)
    return conform(d.sort_values(["ts", "event_id"]).reset_index(drop=True), "digest")


def digest_to_events(digest: pd.DataFrame) -> pd.DataFrame:
    """The digest as an events table, so the pipeline can run on it (derived text)."""
    d = digest[digest["ts"].notna()]
    return pd.DataFrame({
        "event_id": "digest:" + d["event_id"].astype(str), "ts": d["ts"], "channel": "digest",
        "author_raw": d["actor"].fillna("narrator"), "author_alt": d["actor_id"].fillna(""), "text": d["text"],
        "parent_id": d["summary_id"], "visible_until": pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]"),
        "channel_family": "", "source_ref": d["event_id"], "site": "summaries", "source_kind": "digest_" + d["kind"].astype(str),
    }).reset_index(drop=True)


def summaries_summary(summaries: pd.DataFrame, digest: pd.DataFrame | None, directory: pd.DataFrame | None = None) -> dict:
    s = summaries
    latest = s[s["latest"].astype(bool)]
    daily = latest[latest["type"] == "daily"]
    out = {"n": int(len(s)), "n_latest": int(len(latest)), "n_superseded": int(len(s) - len(latest)),
           "by_type": latest["type"].value_counts().to_dict(),
           "generators": s["generated_by"].value_counts().to_dict(),
           "created": f"{s['created'].min():%Y-%m-%d} → {s['created'].max():%Y-%m-%d}"}
    if len(daily):
        dates = daily["date"].dropna()
        out["daily"] = {"n": int(len(daily)), "first": dates.min(), "last": dates.max(),
                        "day_first": int(daily["day"].min()) if daily["day"].notna().any() else None,
                        "day_last": int(daily["day"].max()) if daily["day"].notna().any() else None}
    if digest is not None and len(digest):
        named = digest["actor"].notna()
        out["digest"] = {"n_events": int(len(digest)), "n_named": int(named.sum()),
                         "by_kind": digest["kind"].value_counts().to_dict(),
                         "n_actors": int(digest["actor"].nunique()),
                         "top_actors": digest["actor"].value_counts().head(8).to_dict(),
                         "n_bad_ts": int(digest["ts"].isna().sum()),
                         "span": f"{digest['ts'].min():%Y-%m-%d} → {digest['ts'].max():%Y-%m-%d}"}
    return out



def _hhmm(s: str) -> float:
    h, m = str(s).split(":")[:2]
    return int(h) + int(m) / 60


def normalize_meta(rows: list[dict], directory: pd.DataFrame | None = None) -> dict:
    """The village row as JSON-ready facts: export cut, schedule, turn holder."""
    r = next((x for x in rows if isinstance(x, dict) and x.get("id")), None)
    if r is None:
        raise ValueError("no village row with an id")
    names = dict(zip(directory["agent_id"], directory["name"])) if directory is not None else {}
    windows = [{"days": [str(d).lower()[:3] for d in w.get("days", [])], "start": str(w.get("start")), "end": str(w.get("end"))}
               for w in ((r.get("schedule") or {}).get("windows") or []) if isinstance(w, dict)]
    hours = sum(max(0.0, _hhmm(w["end"]) - _hhmm(w["start"])) * len(w["days"]) for w in windows if w["start"] and w["end"])
    created, cut = _ts(r.get("created_at")), _ts(r.get("updated_at"))
    return {
        "village_id": str(r["id"]), "name": str(r.get("name") or r.get("slug") or r["id"]), "slug": r.get("slug"),
        "created": None if pd.isna(created) else created.isoformat(),
        "export_cut": None if pd.isna(cut) else cut.isoformat(),
        "schedule": {"windows": windows, "open_hours_per_week": hours},
        "is_chat_open": bool(r.get("is_chat_open")) if r.get("is_chat_open") is not None else None,
        "active_agent_id": r.get("active_agent_id"),
        "active_agent": names.get(str(r.get("active_agent_id"))),
        "turn_id": r.get("turn_id"),
        "legacy_goal": (r.get("village_goal") or "").strip() or None,
    }


def in_schedule(ts: pd.Series, windows: list[dict], tz: str) -> pd.Series:
    """True for timestamps inside one of the schedule's daily windows, read in ``tz``."""
    local = pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(tz)
    day = local.dt.dayofweek.map(lambda d: DAYS[int(d)] if pd.notna(d) else None)
    hour = local.dt.hour + local.dt.minute / 60
    ok = pd.Series(False, index=ts.index)
    for w in windows:
        if not (w.get("start") and w.get("end")):
            continue
        ok |= day.isin(w["days"]) & (hour >= _hhmm(w["start"])) & (hour < _hhmm(w["end"]))
    return ok


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
                 era_ann: pd.DataFrame | None = None, cut: pd.Timestamp | None = None) -> dict:
    e = eras.sort_values("start").reset_index(drop=True)
    end = e["end"].fillna(cut) if cut is not None else e["end"]   # an open window runs to the export cut
    days = (end - e["start"]).dt.total_seconds().div(86400)
    diff = (e["start"].shift(-1) - e["end"]).dt.total_seconds().dropna()   # + gap, - overlap
    s = {"n": int(len(e)), "first_start": e["start"].min(), "last_start": e["start"].max(),
         "n_open": int(e["end"].isna().sum()), "median_days": float(days.median()) if days.notna().any() else None,
         "min_days": float(days.min()) if days.notna().any() else None, "max_days": float(days.max()) if days.notna().any() else None,
         "n_gaps": int((diff > 60).sum()), "n_overlaps": int((diff < -60).sum()), "handover": None}
    if roster is not None and len(roster) and roster["start"].notna().any():
        last = e.iloc[-1]
        offset = (roster["start"].min() - last["start"]).total_seconds() if pd.notna(last["start"]) else None
        if offset is not None and abs(offset) <= 3600:
            s["handover"] = {"label": last["label"], "goal": last["goal"], "start": last["start"],
                             "roster_start": roster["start"].min(), "offset_s": float(offset)}
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
            era_ann: pd.DataFrame | None = None, meta: dict | None = None, schedule_tz: str | None = None,
            summaries: pd.DataFrame | None = None, digest: pd.DataFrame | None = None,
            family_from_roster: bool = True) -> dict:
    s: dict = {"family_from_roster": family_from_roster}
    cut = pd.Timestamp(meta["export_cut"]) if meta and meta.get("export_cut") else None
    if summaries is not None and len(summaries):
        s["summaries"] = summaries_summary(summaries, digest, directory)
        if schedule_tz is None and digest is not None and len(digest):
            schedule_tz = DIGEST_TZ          # the digests stamp events in PT: evidence for the schedule's zone
            s["schedule_tz_inferred"] = True
    if meta:
        s["meta"] = dict(meta)
        s["meta"]["schedule_tz"] = schedule_tz
        if events is not None and len(events) and schedule_tz and meta["schedule"]["windows"]:
            inside = in_schedule(events["ts"], meta["schedule"]["windows"], schedule_tz)
            s["meta"]["posts_outside_schedule"] = int((~inside).sum())
            s["meta"]["posts"] = int(len(events))
    if unrecognised:
        s["unrecognised"] = list(unrecognised)
    if eras is not None and len(eras):
        s["eras"] = eras_summary(eras, events, roster, era_ann, cut)
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
        end = c["deleted"].fillna(cut) if cut is not None else c["deleted"]   # open rooms: to the export cut
        c["lifetime_h"] = ((end - c["created"]).dt.total_seconds() / 3600).round(1)
        s["lifetime_to_cut"] = cut is not None
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


def _schedule_text(sch: dict) -> str:
    parts = []
    for w in sch.get("windows", []):
        days = w["days"]
        run = f"{days[0]}–{days[-1]}" if len(days) > 2 and days == DAYS[DAYS.index(days[0]):DAYS.index(days[0]) + len(days)] else ", ".join(days)
        parts.append(f"{run} {w['start']}–{w['end']}")
    return "; ".join(parts) or "no windows"


def section(s: dict, md_table) -> list[str]:
    if not s:
        return []
    L = ["## Village\n"]
    if "meta" in s:
        m = s["meta"]
        tz = m.get("schedule_tz")
        L.append(f"Village `{m['name']}`, created {m['created'][:16] if m.get('created') else '?'} UTC, exported "
                 f"{m['export_cut'][:16] if m.get('export_cut') else '?'} UTC (the export cut: open goals, rooms and windows are measured to it). "
                 f"Operating schedule: {_schedule_text(m['schedule'])} ({m['schedule']['open_hours_per_week']:.0f} h/week), "
                 + (f"read in {tz}{' (inferred: the daily digests stamp events in PT)' if s.get('schedule_tz_inferred') else ''}: "
                    f"{m['posts_outside_schedule']:,} of {m['posts']:,} posts fall outside it. " if "posts_outside_schedule" in m
                    else (f"timezone {tz}, inferred from the daily digests' PT stamps. " if s.get("schedule_tz_inferred")
                          else "timezone not stated in the export (set `schedule_tz` in the config to check posts against it). "))
                 + "The village runs one agent at a time"
                 + (f"; at export the turn was held by {m['active_agent']}" if m.get("active_agent") else "")
                 + f" and chat was {'open' if m.get('is_chat_open') else 'closed'}. Turn boundaries are not exported, so timing stays on the wall clock."
                 + (f" The row's `village_goal` field still reads \"{m['legacy_goal']}\", the first shared goal, not the current one." if m.get("legacy_goal") else "")
                 + "\n")
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
        L.append(md_table(s["rooms"].rename(columns={"lifetime_h": "lifetime_h (open: to export)" if s.get("lifetime_to_cut") else "lifetime_h"}),
                          index=False, floatfmt="{:.1f}"))
    if "eras" in s:
        e = s["eras"]
        h = e.get("handover")
        if h and abs(h["offset_s"]) <= 60:
            hand = (f" The village switched from shared to individual goals on {h['start']:%Y-%m-%d %H:%M} UTC "
                    f"(\"{h['goal']}\"), the minute the first per-agent goal starts.")
        elif h:
            hand = (f" The last shared goal (\"{h['goal']}\") opened at {h['start']:%Y-%m-%d %H:%M} UTC, within an hour of the first "
                    f"per-agent goal ({h['roster_start']:%H:%M}).")
        else:
            hand = ""
        L.append(f"\nShared goals: {e['n']} windows from {e['first_start']:%Y-%m-%d} to {e['last_start']:%Y-%m-%d} (last start), "
                 f"median {e['median_days']:.1f} days each ({e['min_days']:.1f} to {e['max_days']:.1f}), {e['n_gaps']} gap(s) and "
                 f"{e['n_overlaps']} overlap(s) between consecutive windows, {e['n_open']} still open. "
                 + ("A shared goal is the task family of every post in its window that neither an agent-specific goal nor a "
                    "configured text or channel family covers." if s.get("family_from_roster", True) else
                    "`family_from_roster` is off, so the eras are recorded per post but not used as task families.")
                 + hand
                 + (f" {e['posts_outside']:,} of {e['posts']:,} posts fall outside every window." if "posts" in e else "") + "\n")
        L.append(md_table(e["table"], index=False, floatfmt="{:.1f}"))
    if "summaries" in s:
        u = s["summaries"]
        types = ", ".join(f"{k} {v}" for k, v in sorted(u["by_type"].items(), key=lambda kv: -kv[1]))
        gens = ", ".join(f"{k} {v}" for k, v in u["generators"].items())
        L.append(f"\nSummaries: {u['n']:,} LLM-written summaries ({u['n_superseded']:,} superseded regenerations; latest versions: {types}), "
                 f"written {u['created']} by {gens}.")
        if "daily" in u:
            d = u["daily"]
            L.append(f" Daily digests cover {d['n']} village days, {d['first']:%Y-%m-%d} → {d['last']:%Y-%m-%d}"
                     + (f" (Day {d['day_first']} → Day {d['day_last']})" if d.get("day_first") is not None else "") + ".")
        if "digest" in u:
            g = u["digest"]
            top = ", ".join(f"{k} {v:,}" for k, v in g["top_actors"].items())
            kinds = ", ".join(f"{v:,} {k}s" for k, v in g["by_kind"].items())
            L.append(f" Their timestamped lines give a derived timeline of {g['n_events']:,} entries ({kinds}; {g['span']}, stamped PT and read as "
                     f"{DIGEST_TZ}; {g['n_bad_ts']} unparseable); {g['n_named']:,} name a directory agent ({g['n_actors']} agents: {top}). "
                     "Events and notes are an LLM's account of what agents did, quotes are their words as the summariser quoted them: "
                     "fit for who-did-what-when and technique mentions, not for the provenance of specific values. "
                     "Set `digest_as_posts` in the config to run the pipeline on it.")
        L.append("\n")
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
