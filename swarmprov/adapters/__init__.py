"""Adapter registry.  ``detect(path)`` picks the first adapter whose sniff() matches."""

from __future__ import annotations

from pathlib import Path

from .base import Adapter, AdapterConfig, Bundle, Capabilities, Technique, did_you_mean
from .chat import ChatAdapter
from .roster import RosterAdapter
from .swarmtraces import SwarmTracesAdapter
from .village import VillageAdapter
from .wiki import WikiAdapter

# detection order: wiki (revisions.jsonl) before corpus (records.jsonl) before
# swarmtraces (redacted.jsonl: id/cite/kind/parent_id/time_utc/tags/text) before
# village (a directory of AI Village tables, or its agents/chat_rooms files)
# before roster (agent_goals: ids and windows, no text) before generic chat,
# which takes any message file or directory of them, so it stays last
ADAPTERS: dict[str, type[Adapter]] = {"wiki": WikiAdapter}
try:  # the cross-site corpus adapter is optional
    from .corpus import CorpusAdapter
    ADAPTERS["corpus"] = CorpusAdapter
except ImportError:  # pragma: no cover
    pass
ADAPTERS["swarmtraces"] = SwarmTracesAdapter
ADAPTERS["village"] = VillageAdapter
ADAPTERS["roster"] = RosterAdapter
ADAPTERS["chat"] = ChatAdapter


def get(name: str, fields: dict | None = None) -> Adapter:
    """The adapter called ``name``; ``fields`` (canonical field -> source key) reaches the chat adapter."""
    if name not in ADAPTERS:
        raise ValueError(f"unknown adapter `{name}`{did_you_mean(name, ADAPTERS)}; the adapters are auto, {', '.join(ADAPTERS)}")
    return ChatAdapter(fields) if name == "chat" else ADAPTERS[name]()


def sniffs(path: str | Path, fields: dict | None = None):
    """(adapter, True / False / the exception its sniff raised) for each adapter, in detection order."""
    for name in ADAPTERS:
        a = get(name, fields)
        try:
            yield a, bool(a.sniff(Path(path)))
        except Exception as e:   # a sniff that cannot read the input says no; the reason is kept for the error
            yield a, e


def detect(path: str | Path, fields: dict | None = None) -> Adapter:
    errors = []
    for a, r in sniffs(path, fields):
        if r is True:
            return a
        if isinstance(r, Exception):
            errors.append(f"\n  {a.name}: {type(r).__name__}: {r}")
    raise ValueError(f"no adapter recognises {path}; pass --adapter ({', '.join(ADAPTERS)}) or map its fields with "
                     f"--field (`swarmprov inspect {path}` shows what the chat reader finds)"
                     + (f". Sniff errors:{''.join(errors)}" if errors else ""))


__all__ = ["Adapter", "AdapterConfig", "Bundle", "Capabilities", "Technique", "get", "sniffs", "detect"]
