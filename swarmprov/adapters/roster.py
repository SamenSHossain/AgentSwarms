"""Roster adapter: agent-goal tables (AI Village ``agent_goals``).

A roster has no posts, so the bundle's ``events`` table is empty and the
interesting output is the ``roster`` table.  Attach it to a transcript run
with ``swarmprov run TRANSCRIPT --roster agent_goals.jsonl``; on its own,
``swarmprov run agent_goals.jsonl`` writes a roster report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import roster as roster_mod
from ..schema import empty
from .base import Adapter, Bundle, Capabilities
from .chat import _read_any


TABULAR = (".csv", ".tsv", ".txt")


def _is_tabular(path: Path) -> bool:
    name = path.name[:-3] if path.suffix == ".gz" else path.name
    return name.endswith(TABULAR)


def read_rows(path: Path) -> list[dict]:
    path = Path(path)
    if _is_tabular(path):
        # a spreadsheet export: comma or tab separated, delimiter sniffed; everything read as text
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
        return df.to_dict("records")
    obj = _read_any(path)
    if isinstance(obj, dict):  # {"goals": [...]} / {"data": [...]}
        lists = [v for v in obj.values() if isinstance(v, list)]
        obj = lists[0] if lists else []
    return [r for r in obj if isinstance(r, dict)]


class RosterAdapter(Adapter):
    name = "roster"

    def sniff(self, path: Path) -> bool:
        path = Path(path)
        if not path.is_file() or not (path.name.endswith((".json", ".jsonl", ".gz")) or _is_tabular(path)):
            return False
        return roster_mod.looks_like_roster(read_rows(path)[:50])

    def load(self, path: Path) -> Bundle:
        rows = read_rows(Path(path))
        ros = roster_mod.normalize(rows)
        caps = Capabilities(has_wall_clock=True, has_explicit_author=True)
        notes = {"n_goals": int(len(ros)), "n_agents": int(ros["agent_id"].nunique()),
                 "n_roles": int(ros["role"].nunique()), "n_open": int(ros["end"].isna().sum()),
                 "timestamp_issues": roster_mod.timestamp_issues(rows)}
        return Bundle(events=empty("events"), roster=ros, capabilities=caps, notes=notes)
