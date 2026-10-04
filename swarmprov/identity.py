"""Identity resolution: author strings -> agent ids.

Two mappings are produced and every headline number is reported under both:

* ``agent_strict`` - the author handle itself (case-folded).  Over-splits an
  agent that signs under several names.
* ``agent_merged`` - ``cohort|family`` when the handle (or the agent's own
  posts) carries a cohort token such as ``Mar16``; the cohort is the date the
  agent instance was assigned, so the same cohort working the same task
  family is treated as one agent.  Under-splits if two agents share both.
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


def resolve(events: pd.DataFrame, post_family: pd.Series, cfg: AdapterConfig) -> pd.DataFrame:
    rx = re.compile(cfg.cohort_regex)
    lead_rx = re.compile(r"^\W*(?:'''|\*\*)?(?:[A-Za-z]+[- ])?" + cfg.cohort_regex)
    rows = []
    for author, g in events.assign(_fam=post_family).groupby("author_raw"):
        strict = re.sub(r"\W+$", "", str(author)).lower()
        cohort = cohort_token(str(author), rx)
        if cohort is None:
            # fall back to the cohort the agent announces at the start of its posts
            leads = [cohort_token(m.group(0), rx) for t in g["text"] if (m := lead_rx.search(t[:40]))]
            leads = [c for c in leads if c]
            if leads:
                cohort = Counter(leads).most_common(1)[0][0]
        fams = [f for f in g["_fam"] if f]
        fam = Counter(fams).most_common(1)[0][0] if fams else ""
        merged = f"{cohort}|{fam}" if cohort and fam else strict
        rows.append({"author_raw": author, "agent_strict": strict, "agent_merged": merged,
                     "cohort": cohort, "n_events": len(g)})
    return pd.DataFrame(rows)


def post_families(events: pd.DataFrame, cfg: AdapterConfig) -> pd.Series:
    """Channel family when the channel belongs to one task, else classify the text."""
    out = []
    for ch_fam, ch, text in zip(events["channel_family"].fillna(""), events["channel"], events["text"]):
        if ch_fam:
            out.append(ch_fam)
        else:
            f = cfg.family_of_text(text[:600]) or cfg.family_of_text(str(ch))
            out.append(f or (str(ch) if cfg.family_from_channel else ""))
    return pd.Series(out, index=events.index)
