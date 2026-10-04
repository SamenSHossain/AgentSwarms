"""Canonical tables shared by every stage.

Adapters produce ``events`` (and optionally ``lifecycle`` / ``reads``); every
later stage reads and writes the tables below as Parquet files in a run
directory.  Columns are listed here once so stages and tests agree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES: dict[str, list[str]] = {
    # One row per post: a single authored unit of text that became visible to
    # other agents at ``ts`` (a chat message, or the new paragraph a wiki
    # revision added).
    "events": [
        "event_id",       # stable id
        "ts",             # wall clock, UTC
        "channel",        # page / room / thread
        "author_raw",     # author as the source reports it (signature, sender)
        "author_alt",     # secondary handle (account label, ip prefix), may be ""
        "text",
        "parent_id",      # revision / message this post arrived in
        "visible_until",  # UTC time the channel was deleted, NaT if never
        "channel_family", # task family implied by the channel, "" if none
        "source_ref",
        "site",           # surface the post lives on: "dse", "paste.linuxiarz.pl", "rmn.re", ...
        "source_kind",    # adapter-specific provenance of the row: "revision", "paste", "shortener_link", ...
    ],
    # Channel deletions / recreations (wiki admin deletions, archived rooms).
    "lifecycle": ["channel", "ts", "action", "actor"],
    # Observed reads, when the source logs them (page views, context windows).
    "reads": ["agent_raw", "ts", "channel", "event_id", "method"],
    # Roster: who the agents are and what each was asked to do, per time window
    # (AI Village ``agent_goals``).  ``role`` is the short task label, ``goal``
    # the full instruction; ``end`` is NaT while the assignment is still open.
    "roster": ["goal_id", "agent_id", "role", "goal", "detail", "start", "end", "created", "updated", "batch",
               "agent_name", "model"],  # the last two come from the agent directory when one is present
    # Agent directory (AI Village ``agents``): id -> display name, model, join date.
    "directory": ["agent_id", "name", "model", "vendor", "joined", "last_seen", "participating"],
    # Channel table (AI Village ``chat_rooms``): lifetime and audience restrictions (lists of agent names).
    "channels": ["channel", "channel_id", "created", "deleted", "allow", "deny"],
    # Identity resolution result: one row per author_raw.
    # ``roster_agent`` is the roster id when a roster identified the author, else None.
    "agents": ["author_raw", "agent_strict", "agent_merged", "cohort", "n_events", "roster_agent"],
    # Structured facts extracted from posts.
    "claims": [
        "claim_id", "event_id", "ts", "agent_strict", "agent_merged", "family",
        "episode", "item", "value_raw", "value_norm", "event_type",
        "latency_class", "latency_s", "timer_s", "wrong_flag", "correct_flag", "extractor",
    ],
    # Per-post flags (multi-label) and raw citation tokens.
    "tags": [
        "event_id", "ts", "agent_strict", "agent_merged", "family", "is_answer", "is_arrival",
        "is_confirm", "is_prediction", "is_correction", "is_request", "is_independent",
        "techniques", "cites",
    ],
    # Every public occurrence of an (item, value) pair; drives T_public.
    "mentions": ["event_id", "ts", "agent_strict", "agent_merged", "family", "item", "value_norm"],
    # One row per answer claim (agent x episode x item).
    "exposures": [
        "claim_id", "agent", "family", "episode", "item", "value_norm", "t_report",
        "t_public", "src_agent", "t_self_prepared", "gap_s", "D", "D_w10m", "D_w60m", "D_cons", "D_cons_w60m",
        "y_instant", "y_consensus", "consensus_value", "latency_class", "wrong_flag",
    ],
    # Provenance graph edges.
    "edges": ["src", "dst", "family", "item", "value_norm", "kind", "dt_s", "event_id"],
}


def empty(table: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in TABLES[table]})


def conform(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Return ``df`` with exactly the canonical columns (missing ones filled)."""
    cols = TABLES[table]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols]


class RunDir:
    """A directory holding one pipeline run's tables, figures and report."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "figures").mkdir(exist_ok=True)

    def table_path(self, name: str) -> Path:
        return self.path / f"{name}.parquet"

    def has(self, name: str) -> bool:
        return self.table_path(name).exists()

    def write(self, name: str, df: pd.DataFrame) -> Path:
        p = self.table_path(name)
        df = df.copy()
        for c in df.columns:
            # Parquet needs homogeneous columns; lists/dicts are stored as text.
            if df[c].dtype == object and df[c].map(lambda v: isinstance(v, (list, dict))).any():
                df[c] = df[c].map(lambda v: None if v is None else str(v))
        df.to_parquet(p, index=False)
        return p

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.table_path(name))

    def figure(self, name: str) -> Path:
        return self.path / "figures" / name
