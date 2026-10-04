"""Adapter registry.  ``detect(path)`` picks the first adapter whose sniff() matches."""

from __future__ import annotations

from pathlib import Path

from .base import Adapter, AdapterConfig, Bundle, Capabilities, Technique
from .chat import ChatAdapter
from .roster import RosterAdapter
from .wiki import WikiAdapter

# detection order: wiki (revisions.jsonl) before corpus (records.jsonl) before roster
# (agent_goals: ids and windows, no text) before generic chat
ADAPTERS: dict[str, type[Adapter]] = {"wiki": WikiAdapter}
try:  # the cross-site corpus adapter is optional
    from .corpus import CorpusAdapter
    ADAPTERS["corpus"] = CorpusAdapter
except ImportError:  # pragma: no cover
    pass
ADAPTERS["roster"] = RosterAdapter
ADAPTERS["chat"] = ChatAdapter


def get(name: str) -> Adapter:
    return ADAPTERS[name]()


def detect(path: str | Path) -> Adapter:
    for cls in ADAPTERS.values():
        a = cls()
        try:
            if a.sniff(Path(path)):
                return a
        except Exception:
            continue
    raise ValueError(f"no adapter recognises {path}; pass --adapter explicitly")


__all__ = ["Adapter", "AdapterConfig", "Bundle", "Capabilities", "Technique", "get", "detect"]
