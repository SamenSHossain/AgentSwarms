"""Adapter registry.  ``detect(path)`` picks the first adapter whose sniff() matches."""

from __future__ import annotations

from pathlib import Path

from .base import Adapter, AdapterConfig, Bundle, Capabilities, Technique
from .chat import ChatAdapter
from .wiki import WikiAdapter

ADAPTERS: dict[str, type[Adapter]] = {"wiki": WikiAdapter, "chat": ChatAdapter}


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
