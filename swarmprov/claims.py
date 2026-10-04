"""Canonicalise claims and index every public (item, value) mention.

* fills missing episodes from the (family, item) -> episode mapping other
  agents reported (task sequences are shared across cohorts);
* collapses repeated reports into one row per agent-round;
* computes the consensus value per (family, item);
* scans every post for "item ... value" pairs so exposure can ask when a
  value first became public.  URLs are indexed too, so the reach analysis
  also works on free-form chat.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .rules import ItemMatcher, first_value, norm_value

URL_RE = re.compile(r"https?://[^\s\]\)\|'\"<>]+")


def value_key(v: str | None) -> str | None:
    """Comparison key: first component of a multi-number answer."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v).split("/")[0]


def fill_episodes(claims: pd.DataFrame, min_share: float = 0.6, min_support: int = 3,
                  sequences: dict | None = None) -> pd.DataFrame:
    """Fill missing episodes from (family, item) -> round, learned from other
    agents' answer reports, else from sequence chains ('MA -> CT -> MI')."""
    c = claims.copy()
    known = c[c["event_type"] == "answer"].dropna(subset=["episode", "item"])
    ep_map = {}
    for (fam, item), g in known.groupby(["family", "item"]):
        ep, n = Counter(g["episode"]).most_common(1)[0]
        if n >= min_support and n / len(g) >= min_share:
            ep_map[(fam, item)] = ep
    for k, pos in (sequences or {}).items():
        ep_map.setdefault(k, float(pos))
    miss = c["episode"].isna() & c["item"].notna()
    filled = [ep_map.get((f, i), float("nan")) for f, i in zip(c.loc[miss, "family"], c.loc[miss, "item"])]
    c["episode"] = c["episode"].astype(float)
    c.loc[miss, "episode"] = pd.Series(filled, index=c.index[miss], dtype=float)
    return c


def agent_rounds(claims: pd.DataFrame, agent_col: str) -> pd.DataFrame:
    """One row per (agent, family, item): the agent's earliest answer report."""
    a = claims[(claims["event_type"] == "answer") & claims["item"].notna() & (claims["family"] != "")]
    a = a.sort_values("ts")
    rows = []
    for (agent, fam, item), g in a.groupby([agent_col, "family", "item"], sort=False):
        first = g.iloc[0]
        vals = g["value_norm"].dropna()
        lat = g["latency_class"].dropna()
        rows.append({
            "claim_id": first["claim_id"], "agent": agent, "family": fam, "item": item,
            "episode": g["episode"].dropna().iloc[0] if g["episode"].notna().any() else None,
            "value_norm": vals.iloc[0] if len(vals) else None,
            "t_report": first["ts"],
            "latency_class": lat.iloc[0] if len(lat) else None,
            "timer_s": g["timer_s"].dropna().iloc[0] if g["timer_s"].notna().any() else None,
            "wrong_flag": bool(g["wrong_flag"].any()),
            "correct_flag": bool(g["correct_flag"].fillna(False).any()),
            "n_reports": len(g),
        })
    return pd.DataFrame(rows)


def consensus(claims: pd.DataFrame, mentions: pd.DataFrame | None = None,
              agent_col: str = "agent_merged", min_agents: int = 2) -> dict[tuple[str, str], str]:
    """Value backed by the most distinct agents per (family, item), pooling the
    values agents answered with and the values they posted next to the item.
    Ties go to the variant that appeared first."""
    a = claims[(claims["event_type"] == "answer") & claims["value_norm"].notna() & ~claims["wrong_flag"].astype(bool)]
    support = pd.DataFrame({"ts": a["ts"], "agent": a[agent_col], "family": a["family"], "item": a["item"],
                            "value_key": a["value_norm"].map(value_key)})
    if mentions is not None and len(mentions):
        m = mentions[mentions["family"] != "url"]
        support = pd.concat([support, pd.DataFrame({"ts": m["ts"], "agent": m[agent_col], "family": m["family"],
                                                    "item": m["item"], "value_key": m["value_key"]})])
    support = support.dropna(subset=["value_key", "item"]).sort_values("ts")
    out = {}
    for (fam, item), g in support.groupby(["family", "item"]):
        cnt = g.drop_duplicates(["agent", "value_key"])["value_key"].value_counts()
        if cnt.iloc[0] < min_agents and len(cnt) > 1:
            continue
        top = cnt.iloc[0]
        out[(fam, item)] = next(k for k in g["value_key"] if cnt.get(k, 0) == top)
    return out


def index_mentions(events: pd.DataFrame, families: pd.Series, agents: pd.DataFrame,
                   items: ItemMatcher, targets: set[tuple[str, str]] | None = None,
                   window: int = 60, with_urls: bool = True) -> pd.DataFrame:
    """Every 'item <= window chars => number' pair (and URL) in every post."""
    amap = agents.set_index("author_raw")[["agent_strict", "agent_merged"]].to_dict("index")
    rows = []
    for ev, fam in zip(events.itertuples(index=False), families):
        ids = amap.get(ev.author_raw, {"agent_strict": ev.author_raw, "agent_merged": ev.author_raw})
        base = {"event_id": ev.event_id, "ts": ev.ts, **ids}
        text = ev.text
        if fam:
            found = items.find(fam, text)
            for k, (pos, item) in enumerate(found):
                if targets is not None and (fam, item) not in targets:
                    continue
                end = found[k + 1][0] if k + 1 < len(found) else len(text)
                seg = text[pos: min(end, pos + window)]
                seg = re.sub(r"(?<![A-Za-z0-9])[RG#][1-9]\b", " ", seg)
                v = first_value(seg)
                if v:
                    rows.append({**base, "channel": ev.channel, "family": fam, "item": item,
                                 "value_norm": norm_value(v)})
        if with_urls:
            for u in sorted(set(URL_RE.findall(text))):   # deterministic order: set order follows the hash seed
                u = u.rstrip(".,;")
                dom = re.sub(r"^https?://", "", u).split("/")[0]
                rows.append({**base, "channel": ev.channel, "family": "url", "item": dom, "value_norm": u})
    m = pd.DataFrame(rows)
    if len(m):
        m["value_key"] = [v if f == "url" else value_key(v) for f, v in zip(m["family"], m["value_norm"])]
    return m
