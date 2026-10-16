"""``swarmprov suggest RUN``: a starter config mined from a run's own posts, written for review and never applied.

* ``techniques``: URL domains that wrap other URLs, i.e. fetch proxies and readers such as ``r.jina.ai/https://...``
  (``mine_url_techniques``).  A URL wraps another when, after its scheme and host, the path or query holds
  ``http(s):`` or ``%3A%2F%2F``, or the path begins with another hostname (``markdown.new/example.com/page``); a bare
  ``.tld/`` somewhere in the path is not enough, or every URL would wrap.  A domain becomes the technique
  ``proxy:<domain>`` when at least ``MIN_AGENTS`` agents posted it and at least ``MIN_WRAP`` of its distinct URLs wrap
  one.  Domains from ``NEAR`` up to ``MIN_WRAP`` (shorteners, redirectors, archives) are listed as near the threshold
  and left out.  Only URL-wrapping techniques are found: seed cracking or a clock trick needs a hand-written pattern.
* ``families``: the channels, when at least two each carry ``MIN_POSTS`` posts by two or more authors
  (``suggest_families``); otherwise none, so the chat config's catch-all family applies.
* ``extra_items``: capitalised phrases followed within ``ITEM_WINDOW`` characters by a number and posted by two or
  more authors, per family, leaving out items the run already knows (``suggest_items``).

A setting the run's config already has is kept: its families, and its techniques and items before the new ones.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from .adapters.base import Technique, config_snippet
from .claims import public, url_domain, urls
from .gazetteer import COUNTRIES, US_STATES
from .rules import FUNCTION_WORDS, STOP_LOWER
from .schema import RunDir

MIN_AGENTS, MIN_WRAP, NEAR = 3, 0.8, 0.1
MIN_POSTS = 10
ITEM_WINDOW, MAX_ITEMS = 60, 10
SOURCES_SHOWN = 10
MINED = "mined from URL mentions"   # the description prefix of a mined technique
HOST_RE = re.compile(r"(?:[a-z0-9-]+\.)+[a-z]{2,24}(?::\d+)?(?=[/?#]|$)", re.I)
# a first path segment ending in one of these is a file (/wiki.cgi, /index.html), not a wrapped host
FILE_EXT = {"htm", "html", "xhtml", "php", "cgi", "asp", "aspx", "jsp", "json", "xml", "js", "css", "txt", "csv", "png",
            "jpg", "jpeg", "gif", "svg", "pdf", "py", "zip", "gz"}
PHRASE_RE = re.compile(r"(?<![\w-])([A-Z][a-z][A-Za-z'&.-]*(?:\s+(?:of\s+|and\s+|&\s+)?[A-Z][a-z][A-Za-z'&.-]*){0,3})(?!\w)")
MONTHS = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "january",
          "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"}


def wraps(url: str) -> bool:
    """Does ``url`` carry another URL after its own host?  After the scheme and host, decoded (an inner URL is often
    percent-encoded, sometimes twice): the path or query holds a scheme (``http:``, ``https:``, ``://``), or the
    path begins with another hostname, or a query value does and continues with a path (``?url=www.sec.gov/x``).
    A host-like first path segment ending in a file extension (``/wiki.cgi``) is a file."""
    m = re.match(r"https?://([^/?#]*)(.*)", url, re.I | re.S)
    if not m:
        return False
    host, rest = m.groups()
    for _ in range(3):
        rest = unquote(rest)
    if re.search(r"https?:|://", rest, re.I):
        return True
    path, _, query = rest.partition("?")
    starts = [(path.lstrip("/"), False), *((v, True) for v in re.findall(r"(?:^|&)[^=&]*=([^&]*)", query))]
    for text, needs_path in starts:
        h = HOST_RE.match(text)
        if (h and h.group(0).rsplit(".", 1)[-1].split(":")[0].lower() not in FILE_EXT
                and url_domain(h.group(0)) != url_domain(host) and (not needs_path or text[h.end():].startswith("/"))):
            return True
    return False


def url_mentions(events: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    """Every URL in the public posts: the post, its time, both agent labels, the domain and the URL."""
    amap = agents.set_index("author_raw")[["agent_strict", "agent_merged"]].to_dict("index")
    rows = [{"event_id": ev.event_id, "ts": ev.ts, **amap.get(ev.author_raw, {"agent_strict": ev.author_raw,
                                                                              "agent_merged": ev.author_raw}),
             "domain": url_domain(host), "url": u}
            for ev in public(events).itertuples(index=False) for host, u in urls(ev.text)]
    return pd.DataFrame(rows, columns=["event_id", "ts", "agent_strict", "agent_merged", "domain", "url"])


def domain_stats(m: pd.DataFrame, agent_col: str = "agent_merged") -> pd.DataFrame:
    """Per domain: distinct agents, posts and URLs, the share of its distinct URLs that wrap another, and its first
    post (time and agent); most agents first."""
    if not len(m):
        return pd.DataFrame(columns=["agents", "posts", "urls", "wrap_share", "first_seen", "first_agent"])
    m = m.sort_values(["ts", "event_id"], kind="stable")
    u = m.drop_duplicates("url")
    g = m.groupby("domain")
    out = pd.DataFrame({"agents": g[agent_col].nunique(), "posts": g["event_id"].nunique(),
                        "urls": u.groupby("domain")["url"].size(),
                        "wrap_share": u.assign(w=u["url"].map(wraps)).groupby("domain")["w"].mean(),
                        "first_seen": g["ts"].min(), "first_agent": g[agent_col].first()})
    return out.sort_values(["agents", "posts"], ascending=False, kind="stable")


def pattern(domain: str) -> str:
    """The technique regex for a domain: the domain itself, not inside a longer host."""
    return rf"(?<![\w.-]){re.escape(domain)}(?![\w-])"


def mine_url_techniques(m: pd.DataFrame, agent_col: str = "agent_merged", min_agents: int = MIN_AGENTS,
                        min_wrap: float = MIN_WRAP) -> list[Technique]:
    """``proxy:<domain>`` for each domain that ``min_agents`` agents posted and whose distinct URLs wrap another in
    at least ``min_wrap`` of cases, most agents first."""
    st = domain_stats(m, agent_col)
    st = st[(st["agents"] >= min_agents) & (st["wrap_share"] >= min_wrap)]
    return [Technique(f"proxy:{d}", pattern(d), (), f"{MINED}: {r.agents} agents, {r.wrap_share:.0%} of its URLs wrap another")
            for d, r in st.iterrows()]


def spread_domains(m: pd.DataFrame, agent_col: str = "agent_merged", min_agents: int = 2) -> pd.DataFrame:
    """Data sources: domains under ``NEAR`` wrap share that two or more agents posted, most agents first."""
    st = domain_stats(m, agent_col)
    return st[(st["agents"] >= min_agents) & (st["wrap_share"] < NEAR)]


def suggest_families(events: pd.DataFrame, min_posts: int = MIN_POSTS) -> dict[str, str]:
    """One family per channel, matching the channel name exactly, when two or more channels each carry ``min_posts``
    public posts by two or more authors; else none (one catch-all family)."""
    g = public(events).groupby("channel").agg(n=("event_id", "size"), authors=("author_raw", "nunique"))
    keep = g[(g["n"] >= min_posts) & (g["authors"] >= 2)].sort_values("n", ascending=False, kind="stable")
    return {str(c): f"^{re.escape(str(c))}$" for c in keep.index} if len(keep) >= 2 else {}


def suggest_items(events: pd.DataFrame, known: dict[str, set[str]] | None = None, cohort_regex: str | None = None,
                  min_authors: int = 2, window: int = ITEM_WINDOW, top: int = MAX_ITEMS) -> dict[str, list[str]]:
    """Per family, capitalised phrases with a number within ``window`` characters after them that ``min_authors``
    authors posted, most authors first (at most ``top``); round words, months, cohort tokens and items the run
    already knows (``known`` per family, the built-in states and countries) are left out."""
    seen = {n.lower() for n in [*US_STATES.values(), *COUNTRIES]}
    cohort = re.compile(cohort_regex) if cohort_regex else None
    posts = public(events)
    hits: dict[tuple[str, str], set] = {}
    for text, fam, author in zip(posts["text"], posts["family"].fillna(""), posts["author_raw"]):
        if not fam:
            continue
        for mt in PHRASE_RE.finditer(text):
            if not re.search(r"\d", text[mt.end():mt.end() + window]):
                continue
            words = mt.group(1).split()
            while words and words[-1].lower() in STOP_LOWER | MONTHS:
                words.pop()
            while words and words[0].lower() in STOP_LOWER | MONTHS:
                words.pop(0)
            phrase = " ".join(words)
            if (len(phrase) < 3 or any(w.lower() in FUNCTION_WORDS for w in words) or phrase.lower() in seen
                    or (cohort and cohort.fullmatch(phrase))):
                continue
            hits.setdefault((fam, phrase), set()).add(author)
    known = {f: {i.lower() for i in v} for f, v in (known or {}).items()}
    out: dict[str, list[tuple[int, str]]] = {}
    for (fam, phrase), who in hits.items():
        if len(who) >= min_authors and phrase.lower() not in known.get(fam, set()):
            out.setdefault(fam, []).append((len(who), phrase))
    return {f: [p for _, p in sorted(v, key=lambda x: (-x[0], x[1]))[:top]] for f, v in sorted(out.items())}


def _tech(t: Technique) -> dict:
    return {"name": t.name, "pattern": t.pattern} | ({"families": list(t.families)} if t.families else {}) \
        | ({"description": t.description} if t.description else {})


def suggest(run: RunDir) -> tuple[dict, dict]:
    """The suggested settings and the evidence behind them."""
    from .pipeline import load_config
    cfg = load_config(run, with_mined=False)   # techniques auto_techniques mined are mined again here, as mined
    events, agents = run.read("events"), run.read("agents")
    m = url_mentions(events, agents)
    st = domain_stats(m)
    found = {t.name.split(":", 1)[1]: t for t in mine_url_techniques(m)}
    covered = {d: next((o.name for o in cfg.techniques if re.search(o.pattern, d, re.I)), None) for d in found}
    mined = [t for d, t in found.items() if covered[d] is None]
    learned = json.loads((run.path / "items.json").read_text()) if (run.path / "items.json").exists() else {}
    known = {f: set(learned.get(f, [])) | set(cfg.extra_items.get(f, [])) for f in set(learned) | set(cfg.extra_items)}
    families = dict(cfg.families) or suggest_families(events)
    items = suggest_items(events, known, cfg.cohort_regex)
    extra = {f: list(cfg.extra_items.get(f, [])) + items.get(f, []) for f in sorted(set(cfg.extra_items) | set(items))}
    settings = {"families": families, "techniques": [*map(_tech, cfg.techniques), *map(_tech, mined)],
                "extra_items": extra}
    evidence = {"domains": st, "mined": [d for d in found if covered[d] is None],
                "covered": {d: o for d, o in covered.items() if o},
                "near": st[(st["agents"] >= MIN_AGENTS) & (st["wrap_share"] >= NEAR) & (st["wrap_share"] < MIN_WRAP)],
                "sources": spread_domains(m), "kept_families": bool(cfg.families), "kept_techniques": len(cfg.techniques),
                "items": items, "channels": events["channel"].nunique()}
    return settings, evidence


def _table(st: pd.DataFrame, note) -> list[str]:
    out = []
    for d, r in st.iterrows():
        seen = "" if pd.isna(r.first_seen) else pd.Timestamp(r.first_seen).strftime("%Y-%m-%d %H:%M")
        out.append(f"  {d:<34} {r.agents:>6,} {r.posts:>6,} {r.urls:>6,} {r.wrap_share:>6.0%}  {seen:<16}  "
                   f"{str(r.first_agent)[:28]:<28}  {note(d)}")
    return out


def main(run_path: str, out: str = "suggested_config.json") -> int:
    """``swarmprov suggest``: print the evidence and write the suggested config."""
    if not (Path(run_path) / "profile.json").exists():
        print(f"{run_path}: not a run directory (no profile.json); `swarmprov run` makes one")
        return 2
    run = RunDir(run_path)
    if not run.has("agents"):
        print(f"{run.path}: no agents table; run `swarmprov extract {run.path}` first")
        return 2
    settings, ev = suggest(run)
    head = (f"  {'domain':<34} {'agents':>6} {'posts':>6} {'URLs':>6} {'wrap':>6}  {'first seen':<16}  "
            f"{'first agent':<28}  suggestion")
    tech, near, src = ev["domains"].loc[ev["mined"]], ev["near"], ev["sources"].head(SOURCES_SHOWN)
    print(f"URL domains in {run.path} (agents and posts under the merged mapping; wrap = share of the domain's "
          "distinct URLs that carry another URL after its host):")
    print(head)
    print("\n".join(_table(tech, lambda d: f"technique proxy:{d}")
                    + _table(ev["domains"].loc[list(ev["covered"])], lambda d: f"already configured ({ev['covered'][d]})")
                    + _table(near, lambda d: "near the threshold: a shortener, redirector or archive? left out")
                    + _table(src, lambda d: "data source (not a technique)")) or "  (no URLs)")
    if len(ev["sources"]) > SOURCES_SHOWN:
        print(f"  ... {len(ev['sources']) - SOURCES_SHOWN:,} more data sources")
    fams = settings["families"]
    print(f"\nfamilies: " + (f"the config's {len(fams)} kept" if ev["kept_families"] else
                            f"{len(fams)} channels with {MIN_POSTS}+ posts by 2+ authors: {', '.join(fams)}" if fams else
                            f"none: fewer than two of the {ev['channels']:,} channel{'s' * (ev['channels'] != 1)} carry "
                            f"{MIN_POSTS}+ posts by 2+ authors, so one catch-all family"))
    n_items = sum(map(len, ev["items"].values()))
    print(f"extra_items: {n_items:,} new" + (": " + "; ".join(f"{f}: {', '.join(v[:8])}" + (", ..." if len(v) > 8 else "")
                                                         for f, v in ev["items"].items()) if n_items else ""))
    print(f"techniques: {len(ev['mined']):,} mined" + (f", after the config's {ev['kept_techniques']}"
                                                       if ev["kept_techniques"] else ""))
    Path(out).write_text(config_snippet(**settings) + "\n")
    print(f"\nwrote {out}: a starting point to review, not applied. Pass it to `swarmprov run INPUT --config {out}`; "
          "only URL-wrapping techniques are found, so add hand-written patterns for the rest.")
    return 0
