"""Adapter interface: every source becomes the same ``Bundle`` of tables."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass
class Capabilities:
    """What the source can support; analyses that need a missing one are skipped."""

    has_wall_clock: bool = True
    has_explicit_author: bool = False
    has_reads: bool = False
    has_lifecycle: bool = False
    has_threading: bool = False
    has_episodes: bool = False  # explicit round / turn boundaries as fields

    def as_dict(self) -> dict[str, bool]:
        return dict(self.__dict__)


@dataclass
class Technique:
    name: str
    pattern: str                 # regex: a post matching it mentions/uses the technique
    families: tuple[str, ...] = ()  # restrict the at-risk population (empty = all)
    description: str = ""


@dataclass
class AdapterConfig:
    """Source-specific knowledge.  Everything wiki-specific lives here."""

    name: str
    # family -> regex matched against channel name and post text
    families: dict[str, str] = field(default_factory=dict)
    # channel -> family override (e.g. a page-family map shipped with the dump)
    channel_family: Callable[[str], str] | None = None
    # cohort / agent-instance token inside author names ("Mar16", "Nov27")
    # (case-sensitive: Title or UPPER month spellings only, so "Decimal5" is not Dec05;
    #  any preceding character is allowed, so CamelCase "OpenAIOECDNov27" -> Nov27)
    cohort_regex: str = (r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
                         r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|"
                         r"JAN|FEB|MAR(?:CH)?|APR(?:IL)?|MAY|JUNE?|JULY?|AUG|SEPT?|OCT|NOV|DEC)"
                         r"[ \-]?(\d{1,2})(?!\d)")
    # episode / round markers inside post text
    episode_regex: str = r"(?<![A-Za-z0-9])(?:R|G|#|Round\s?)([1-9])(?![0-9])"
    # extra items to recognise per family (on top of the built-in gazetteers)
    extra_items: dict[str, list[str]] = field(default_factory=dict)
    techniques: list[Technique] = field(default_factory=list)
    # (family, item) pairs to feature in the error-propagation figure
    featured_disputes: list[tuple[str, str]] = field(default_factory=list)
    # analyst-declared disputes the parser cannot attribute:
    # {"family", "slot", "context": regex, "variants": {label: regex}}
    disputes: list[dict] = field(default_factory=list)
    # claimed latency at or under this many seconds counts as "instant"
    instant_threshold_s: float = 2.0
    # authors to drop (admins, maintenance bots)
    ignore_authors: tuple[str, ...] = ()
    # use the channel name as the task family when nothing else matches
    family_from_channel: bool = False
    # when a roster is attached, an agent's current goal (its ``role``) is its task family
    family_from_roster: bool = True
    # author strings -> roster agent ids, for transcripts that carry names but no agent id
    roster_aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, base: "AdapterConfig | None" = None) -> "AdapterConfig":
        """Build (or override ``base``) from a JSON/TOML-style dict, e.g.

        {"name": "village", "families": {"fundraiser": "donat|fundrais"},
         "techniques": [{"name": "shared-doc", "pattern": "docs\\.google"}],
         "family_from_channel": false}
        """
        cfg = base if base is not None else cls(name=d.get("name", "custom"))
        for k, v in d.items():
            if k == "techniques":
                v = [Technique(t["name"], t["pattern"], tuple(t.get("families", ())), t.get("description", ""))
                     for t in v]
            elif k == "featured_disputes":
                v = [tuple(x) for x in v]
            elif k == "ignore_authors":
                v = tuple(v)
            elif k == "extra_items":
                v = {f: list(items) for f, items in v.items()}
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_file(cls, path, base: "AdapterConfig | None" = None) -> "AdapterConfig":
        import json
        from pathlib import Path
        p = Path(path)
        if p.suffix == ".toml":
            import tomllib
            d = tomllib.loads(p.read_text())
        else:
            d = json.loads(p.read_text())
        return cls.from_dict(d, base)

    def family_of_text(self, text: str) -> str:
        best, best_pos = "", None
        for fam, rx in self.families.items():
            m = re.search(rx, text, flags=re.I)
            if m and (best_pos is None or m.start() < best_pos):
                best, best_pos = fam, m.start()
        return best


@dataclass
class Bundle:
    events: pd.DataFrame
    lifecycle: pd.DataFrame | None = None
    reads: pd.DataFrame | None = None
    roster: pd.DataFrame | None = None
    capabilities: Capabilities = field(default_factory=Capabilities)
    notes: dict = field(default_factory=dict)


class Adapter:
    name: str = "base"

    def sniff(self, path: Path) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def load(self, path: Path) -> Bundle:  # pragma: no cover - interface
        raise NotImplementedError

    def config(self) -> AdapterConfig:
        return AdapterConfig(name=self.name)
