"""Adapter interface: every source becomes the same ``Bundle`` of tables."""

from __future__ import annotations

import dataclasses
import difflib
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

# the chat adapter's canonical fields, each mappable to a source key with ``fields`` / ``--field``
FIELD_NAMES = ("ts", "author", "author_alt", "text", "channel", "id", "parent", "visible_until")


def did_you_mean(name: str, options) -> str:
    m = difflib.get_close_matches(name, list(options), n=1)
    return f" (did you mean `{m[0]}`?)" if m else ""


def check_fields(mapping: dict) -> dict[str, str]:
    """A ``fields`` mapping (canonical field -> source key; a dotted key reaches into nested objects), validated."""
    bad = [f"`{k}`{did_you_mean(k, FIELD_NAMES)}" for k in mapping if k not in FIELD_NAMES]
    if bad:
        raise ValueError(f"unknown field {', '.join(bad)}; the fields are {', '.join(FIELD_NAMES)}")
    empty = [k for k, v in mapping.items() if not isinstance(v, str) or not v.strip()]
    if empty:
        raise ValueError(f"field {', '.join(empty)} maps to no source key")
    return {k: v.strip() for k, v in mapping.items()}


@dataclass
class Capabilities:
    """What the source can support; analyses that need a missing one are skipped."""

    has_wall_clock: bool = True
    has_order: bool = True      # posts can be ordered: by the wall clock, or by message position when there is none
    has_explicit_author: bool = False
    has_reads: bool = False
    has_lifecycle: bool = False
    has_threading: bool = False
    has_episodes: bool = False  # explicit round / turn boundaries as fields
    has_activity: bool = False  # presence log: when each agent was running
    has_request_log: bool = False  # request-level rows exist (here: script-injection probes only; no page views)
    derived_text: bool = False  # posts are an LLM digest of what agents did, not the agents' own words

    def as_dict(self) -> dict[str, bool]:
        return {k: bool(v) for k, v in self.__dict__.items()}   # a numpy bool would reach profile.json as "False"


def read_capabilities(profile: dict) -> dict[str, bool]:
    """Capabilities from profile.json; runs written before ``as_dict`` cast to bool may hold the string "False"."""
    return {k: v is True or str(v).lower() == "true" for k, v in profile["capabilities"].items()}


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
    # also take the cohort from the date a handle's posts open with ("Mar16 R3 ..."); None = only when the
    # source has no explicit author field (a chat whose messages start "May 5 standup" is not a cohort)
    cohort_from_text: bool | None = None
    # episode / round markers inside post text
    episode_regex: str = r"(?<![A-Za-z0-9])(?:R|G|#|Round\s?)([1-9])(?![0-9])"
    # extra items to recognise per family (on top of the built-in gazetteers)
    extra_items: dict[str, list[str]] = field(default_factory=dict)
    techniques: list[Technique] = field(default_factory=list)
    # with no techniques configured, mine URL ones at extract (``suggest.mine_url_techniques``: domains whose URLs
    # wrap another URL, such as fetch proxies) and label them "(mined)" in A3; off unless a config turns it on
    auto_techniques: bool = False
    # (family, item) pairs to feature in the error-propagation figure
    featured_disputes: list[tuple[str, str]] = field(default_factory=list)
    # analyst-declared disputes the parser cannot attribute:
    # {"family", "slot", "context": regex, "variants": {label: regex}}
    disputes: list[dict] = field(default_factory=list)
    # analyst-supplied correct values {"family|item": value}: A4 labels each disputed slot's outcome
    # against them (reported as analyst input; they never change the slot's status)
    reference_values: dict[str, float | str] = field(default_factory=dict)
    # claimed latency at or under this many seconds counts as "instant"
    instant_threshold_s: float = 2.0
    # resamples behind the report's bootstrap intervals (A1 by item, A6 by fact; seeded); 0 turns them off
    bootstrap_reps: int = 1000
    # authors to drop (admins, maintenance bots)
    ignore_authors: tuple[str, ...] = ()
    # use the channel name as the task family when nothing else matches
    family_from_channel: bool = False
    # catch-all family for every post when no families are configured and no roster or channel names one
    # (the chat adapter's "all"), so a transcript without a config still reaches A1
    default_family: str | None = None
    # when a roster is attached, an agent's current goal (its ``role``) is its task family
    family_from_roster: bool = True
    # author strings -> roster agent ids, for transcripts that carry names but no agent id
    roster_aliases: dict[str, str] = field(default_factory=dict)
    # IANA zone of the village's operating schedule (the export does not state it); when set, the
    # report counts posts outside the schedule's windows
    schedule_tz: str | None = None
    # the Limitations bullet on what the exposure search could not see (None: only this transcript is searched)
    coverage_note: str | None = None
    # with no primary message table, run the pipeline on the digest parsed from LLM-written daily
    # summaries (derived text: timing and names are second-hand, specific values unreliable)
    digest_as_posts: bool = False
    # chat adapter: canonical field -> source key, e.g. {"author": "source", "parent": "reference.messageId"};
    # a field left out is found through the built-in aliases
    fields: dict[str, str] = field(default_factory=dict)
    # a config key that is not a setting raises instead of being ignored with a warning
    strict_config: bool = False

    @classmethod
    def from_dict(cls, d: dict, base: "AdapterConfig | None" = None) -> "AdapterConfig":
        """Build (or override ``base``) from a JSON/TOML-style dict, e.g.

        {"name": "village", "families": {"fundraiser": "donat|fundrais"},
         "techniques": [{"name": "shared-doc", "pattern": "docs\\.google"}],
         "family_from_channel": false}

        Only dataclass fields are settings: any other key warns with the nearest setting, or raises under
        ``strict_config``.
        """
        cfg = base if base is not None else cls(name=d.get("name", "custom"))
        known = [f.name for f in dataclasses.fields(cls)]
        unknown = [k for k in d if k not in known]
        if unknown and d.get("strict_config", cfg.strict_config):
            raise ValueError("; ".join(f"config key `{k}` is not a setting{did_you_mean(k, known)}" for k in unknown))
        for k in unknown:
            warnings.warn(f"config key `{k}` is not a setting{did_you_mean(k, known)} and is ignored "
                          "(`\"strict_config\": true` makes this an error)", stacklevel=2)
        for k, v in d.items():
            if k not in known:
                continue
            if k == "techniques":
                v = [Technique(t["name"], t["pattern"], tuple(t.get("families", ())), t.get("description", ""))
                     for t in v]
            elif k == "featured_disputes":
                v = [tuple(x) for x in v]
            elif k == "ignore_authors":
                v = tuple(v)
            elif k == "extra_items":
                v = {f: list(items) for f, items in v.items()}
            elif k == "fields":
                v = check_fields(v)
            setattr(cfg, k, v)
        return cfg

    @staticmethod
    def read_file(path) -> dict:
        p = Path(path)
        if p.suffix == ".toml":
            import tomllib
            return tomllib.loads(p.read_text())
        return json.loads(p.read_text())

    @classmethod
    def from_file(cls, path, base: "AdapterConfig | None" = None) -> "AdapterConfig":
        return cls.from_dict(cls.read_file(path), base)

    def settings(self) -> dict:
        """Every setting as plain JSON (a callable by its qualified name), as profile.json records a run's config."""
        return json.loads(json.dumps({f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)},
                                     default=str))

    def family_of_text(self, text: str) -> str:
        best, best_pos = "", None
        for fam, rx in self.families.items():
            m = re.search(rx, text, flags=re.I)
            if m and (best_pos is None or m.start() < best_pos):
                best, best_pos = fam, m.start()
        return best


def _plain(v):
    if callable(v):
        return getattr(v, "__qualname__", str(v))
    if dataclasses.is_dataclass(v):
        return _plain(dataclasses.asdict(v))
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def config_snippet(**settings) -> str:
    """A ready-to-paste ``--config`` JSON holding the given settings (``inspect`` writes ``fields``)."""
    known = {f.name for f in dataclasses.fields(AdapterConfig)}
    if set(settings) - known:
        raise ValueError(f"not config settings: {sorted(set(settings) - known)}")
    return json.dumps(settings, indent=2, ensure_ascii=False)


@dataclass
class Bundle:
    events: pd.DataFrame
    lifecycle: pd.DataFrame | None = None
    reads: pd.DataFrame | None = None
    roster: pd.DataFrame | None = None
    directory: pd.DataFrame | None = None
    channels: pd.DataFrame | None = None
    activity: pd.DataFrame | None = None
    probes: pd.DataFrame | None = None
    eras: pd.DataFrame | None = None
    summaries: pd.DataFrame | None = None
    digest: pd.DataFrame | None = None
    capabilities: Capabilities = field(default_factory=Capabilities)
    notes: dict = field(default_factory=dict)


class Adapter:
    name: str = "base"
    reads_fields: bool = False   # its messages go through the chat reader, so a ``fields`` mapping applies

    def sniff(self, path: Path) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def load(self, path: Path) -> Bundle:  # pragma: no cover - interface
        raise NotImplementedError

    def config(self) -> AdapterConfig:
        return AdapterConfig(name=self.name)
