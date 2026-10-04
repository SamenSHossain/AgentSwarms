"""AI Village adapter: a directory (or single file) of the Village's exported tables.

Recognised by columns, whatever the file names: ``agents`` (id, name,
model_string), ``chat_rooms`` (id, name, deleted_at, allow/deny lists),
``agent_goals`` (agent_id, short_name, start_time) and a message table
(any text field).  Everything found is assembled into one bundle: posts
with room names as channels and agent names as authors, the roster with
names attached, the agent directory, the channel table and room lifecycle.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import roster as roster_mod, village as village_mod
from ..schema import empty
from .base import Adapter, Bundle, Capabilities
from .chat import ChatAdapter
from .roster import read_rows

SUFFIXES = (".json", ".jsonl", ".gz", ".csv", ".tsv")


def discover(path: Path) -> dict[str, Path]:
    """kind -> file for every Village table under ``path`` (a file is its own directory)."""
    files = sorted(p for p in (path.iterdir() if path.is_dir() else [path]) if p.is_file() and p.name.endswith(SUFFIXES))
    found: dict[str, Path] = {}
    for p in files:
        try:
            kind = village_mod.classify(read_rows(p)[:20])
        except Exception:
            kind = None
        if kind and kind not in found:
            found[kind] = p
    return found


class VillageAdapter(Adapter):
    name = "village"

    def sniff(self, path: Path) -> bool:
        path = Path(path)
        found = discover(path)
        if path.is_dir():
            return bool(found.keys() & {"goals", "agents", "rooms"})
        return bool(found.keys() & {"agents", "rooms"})   # a lone goals file is the roster adapter's

    def load(self, path: Path) -> Bundle:
        path = Path(path)
        found = discover(path)
        directory = village_mod.normalize_directory(read_rows(found["agents"])) if "agents" in found else None
        channels = village_mod.normalize_channels(read_rows(found["rooms"])) if "rooms" in found else None
        lifecycle = village_mod.lifecycle_from_channels(channels) if channels is not None else None
        roster = notes = None
        notes = {"tables": {k: v.name for k, v in found.items()}}
        if "goals" in found:
            rows = read_rows(found["goals"])
            roster = roster_mod.normalize(rows)
            notes["timestamp_issues"] = roster_mod.timestamp_issues(rows)
            if directory is not None:
                d = directory.set_index("agent_id")
                roster["agent_name"] = roster["agent_id"].map(d["name"])
                roster["model"] = roster["agent_id"].map(d["model"])
        events, caps = empty("events"), Capabilities(has_wall_clock=True, has_explicit_author=True,
                                                     has_lifecycle=channels is not None)
        if "messages" in found:
            b = ChatAdapter().load(found["messages"])
            events, caps = b.events, b.capabilities
            caps.has_lifecycle = channels is not None
            if directory is not None:  # ids -> names
                names = dict(zip(directory["agent_id"], directory["name"]))
                alt = events["author_alt"].map(names)
                events["author_raw"] = alt.where(alt.notna(), events["author_raw"])
            if channels is not None:
                cmap = dict(zip(channels["channel_id"], channels["channel"]))
                events["channel"] = events["channel"].map(lambda c: cmap.get(str(c), str(c)))
                gone = dict(zip(channels["channel"], channels["deleted"]))
                events["visible_until"] = events["channel"].map(gone)
            notes.update(b.notes)
        notes.update({"n_agents": 0 if directory is None else int(len(directory)),
                      "n_rooms": 0 if channels is None else int(len(channels)),
                      "n_goals": 0 if roster is None else int(len(roster))})
        return Bundle(events=events, lifecycle=lifecycle, roster=roster, directory=directory,
                      channels=channels, capabilities=caps, notes=notes)
