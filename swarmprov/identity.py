"""Identity resolution: author strings -> agent ids.

Two mappings are produced and every headline number is reported under both:

* ``agent_strict`` - the author handle itself (case-folded).  Over-splits an
  agent that signs under several names.
* ``agent_merged`` - ``cohort|family`` when the handle carries a cohort token
  such as ``Mar16`` (or, on a source without an explicit author field, the
  handle's posts open with one; ``cohort_from_text`` overrides); the cohort is
  the date the agent instance was assigned, so the same cohort working the
  same task family is treated as one agent.  Under-splits if two agents share both.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .adapters.base import AdapterConfig

MONTHS = {m.lower(): m for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}


def cohort_token(s: str, rx: re.Pattern) -> str | None:
    if not s:
        return None
    for m in rx.finditer(s):
        mon, day = m.group(1), int(m.group(2))
        if 1 <= day <= 31:
            return f"{MONTHS.get(mon[:3].lower(), mon[:3].title())}{day:02d}"
    return None


def strict_handle(author) -> str:
    return re.sub(r"\W+$", "", str(author)).lower()


def resolve(events: pd.DataFrame, post_family: pd.Series, cfg: AdapterConfig, explicit_author: bool = False) -> pd.DataFrame:
    """One row per author string.  Cohort and family are resolved per strict handle (case variants pooled) and
    fanned out, so ``agent_merged`` is a function of ``agent_strict``, unless the variants carry conflicting
    cohorts: those stay split and show up in ``agent_aliases.csv``."""
    rx = re.compile(cfg.cohort_regex)
    lead_rx = re.compile(r"^\W*(?:'''|\*\*)?(?:[A-Za-z]+[- ])?" + cfg.cohort_regex)
    from_text = not explicit_author if cfg.cohort_from_text is None else cfg.cohort_from_text
    ev = events.assign(_fam=post_family, _strict=events["author_raw"].map(strict_handle))
    rows = []
    for strict, g in ev.groupby("_strict"):
        fams = [f for f in g["_fam"] if f]
        fam = Counter(fams).most_common(1)[0][0] if fams else ""
        own = {}
        for author, ga in g.groupby("author_raw"):
            cohort = cohort_token(str(author), rx)
            if cohort is None and from_text:
                # fall back to the cohort the agent announces at the start of its posts
                leads = [cohort_token(m.group(0), rx) for t in ga["text"] if (m := lead_rx.search(t[:40]))]
                leads = [c for c in leads if c]
                if leads:
                    cohort = Counter(leads).most_common(1)[0][0]
            own[author] = (cohort, len(ga))
        found = {c for c, _ in own.values() if c}
        for author, (cohort, n) in own.items():
            cohort = next(iter(found)) if len(found) == 1 else cohort
            merged = f"{cohort}|{fam}" if cohort and fam else strict
            rows.append({"author_raw": author, "agent_strict": strict, "agent_merged": merged,
                         "cohort": cohort, "family": fam, "n_events": n})
    cols = ["author_raw", "agent_strict", "agent_merged", "cohort", "family", "n_events"]
    return pd.DataFrame(rows, columns=cols).sort_values("author_raw").reset_index(drop=True)


def alias_table(agents: pd.DataFrame) -> pd.DataFrame:
    """The audit of both mappings: one row per author string with the size of its strict and merged groups;
    ``n_merged_for_strict`` > 1 marks a handle whose variants were kept apart (conflicting cohorts, or a roster)."""
    a = agents.assign(n_strict_aliases=agents.groupby("agent_strict")["author_raw"].transform("nunique"),
                      n_merged_aliases=agents.groupby("agent_merged")["author_raw"].transform("nunique"),
                      n_merged_for_strict=agents.groupby("agent_strict")["agent_merged"].transform("nunique"))
    cols = ["agent_merged", "agent_strict", "author_raw", "cohort", "family", "n_events", "roster_agent",
            "n_strict_aliases", "n_merged_aliases", "n_merged_for_strict"]
    return (a.sort_values(["n_merged_aliases", "agent_merged", "n_events", "author_raw"], ascending=[False, True, False, True])
            [[c for c in cols if c in a]])


def catch_all_family(cfg: AdapterConfig, roster: bool = False) -> str | None:
    """The config's ``default_family``, if it applies: only with no families configured and no roster (a
    configured swarm's unmatched posts stay family-less rather than pooled into one task)."""
    return cfg.default_family if not cfg.families and not roster else None


def post_families(events: pd.DataFrame, cfg: AdapterConfig, roster: bool = False) -> pd.Series:
    """Channel family when the channel belongs to one task, else classify the text, else the catch-all family."""
    default = catch_all_family(cfg, roster) or ""
    out = []
    for ch_fam, ch, text in zip(events["channel_family"].fillna(""), events["channel"], events["text"]):
        if ch_fam:
            out.append(ch_fam)
        else:
            f = cfg.family_of_text(text[:600]) or cfg.family_of_text(str(ch))
            out.append(f or (str(ch) if cfg.family_from_channel else default))
    return pd.Series(out, index=events.index)
