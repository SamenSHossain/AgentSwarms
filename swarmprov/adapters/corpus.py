"""Adapter for the cross-site corpus (second + third data batches).

Input: a directory or .zip holding ``records.jsonl`` (de-duplicated agent
texts, each with the list of *origins* where it appeared) and optionally
``links.jsonl``, ``shortener-logs.json``, ``other-wikis.json``,
``site-coverage.csv`` and ``coverage-gaps.csv``.  See ``docs/DATA.md``.

Rows produced:

* one event per (record, origin) whose ``source_date_literal`` parses to a
  wall-clock time at or before 2026-08-01 (later literals are investigator
  re-captures, not swarm activity).  The dse wiki is excluded by default
  because the primary wiki dump covers it in full;
* one event per rmn.re shortener link (``source_kind="shortener_link"``);
* one event per other-wiki revision with recovered added lines
  (``source_kind="wiki_revision_lines"``).

Authorship is weak everywhere: a trailing ``-- Name`` signature when there is
one, an ``agent-oursNNNN`` self-identifier when there is one, else
``anon@<site>``.  Shortener and other-wiki rows carry the /16 prefix.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterator

import pandas as pd

from ..textutil import SIGNATURE_RE, norm_key, stable_id
from .base import Adapter, AdapterConfig, Bundle, Capabilities, Technique
from .wiki import FAMILIES as WIKI_FAMILIES
from .wiki import TECHNIQUES as WIKI_TECHNIQUES

# Origins dated after this are 2026-09 investigator re-captures of old text.
LATE_CUTOFF = pd.Timestamp("2026-08-01", tz="UTC")

FAMILIES: dict[str, str] = {
    **WIKI_FAMILIES,
    "iowa-health": r"\bIowa\b|IowaCollab|asthma|thyroid|85 and older",
}

TECHNIQUES: list[Technique] = WIKI_TECHNIQUES + [
    Technique("proxy-markdown-new", r"markdown\.new", (),
              "markdown.new used as a fetch / render proxy for blocked hosts"),
    Technique("proxy-allorigins", r"allorigins\.hexlet\.app", (),
              "allorigins.hexlet.app CORS proxy used to fetch blocked hosts"),
    Technique("proxy-jqp", r"jqp\.vercel\.app", (),
              "jqp.vercel.app (jq playground) used to fetch and filter JSON"),
    Technique("proxy-md-succ", r"md\.succ\.ai", (),
              "md.succ.ai markdown converter used as a fetch proxy"),
    Technique("proxy-jina", r"r\.jina\.ai", (),
              "r.jina.ai reader used as a fetch proxy"),
    Technique("proxy-pure-md", r"pure\.md", (),
              "pure.md reader used as a fetch proxy"),
    Technique("shortener-da-gd", r"\bda\.gd\b", (),
              "da.gd shortener / header echo used as a relay"),
]

_OMITTED_URL_RE = re.compile(r"\[operational URL omitted;\s*host=([^;\]\s]+)[^\]]*\]")
_AGENT_OURS_RE = re.compile(r"\bagent-ours\d+\b")
_EPOCH_RE = r"^\d{10}(?:\.\d+)?$"
_WAYBACK_RE = r"^\d{14}$"
_ISO_RE = r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"


def rewrite_omitted_urls(text: str) -> str:
    """``[operational URL omitted; host=X; sha256=...]`` -> ``https://X/[omitted]``."""
    return _OMITTED_URL_RE.sub(lambda m: f"https://{m.group(1)}/[omitted]", text)


def parse_date_literals(lits: pd.Series) -> pd.Series:
    """Parse the mixed ``source_date_literal`` column to UTC timestamps (NaT otherwise).

    Accepts ISO 8601 (Z or numeric offsets), 10-digit epoch seconds (optionally
    fractional) and 14-digit Wayback stamps.  Anything else ("current", "",
    version strings, small ints) becomes NaT.
    """
    s = lits.fillna("").astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    m = s.str.match(_EPOCH_RE)
    if m.any():
        out[m] = pd.to_datetime(s[m].astype(float), unit="s", utc=True)
    m = s.str.match(_WAYBACK_RE)
    if m.any():
        out[m] = pd.to_datetime(s[m], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    m = s.str.match(_ISO_RE)
    if m.any():
        out[m] = pd.to_datetime(s[m], format="ISO8601", utc=True, errors="coerce")
    return out


def _open(root: Path, name: str) -> io.TextIOBase | None:
    """Open ``name`` inside a directory or a zip; None when absent."""
    if root.is_file() and zipfile.is_zipfile(root):
        z = zipfile.ZipFile(root)
        for n in z.namelist():
            if Path(n).name == name:
                return io.TextIOWrapper(z.open(n), encoding="utf-8")
        return None
    folder = root if root.is_dir() else root.parent
    p = folder / name
    return open(p, encoding="utf-8") if p.is_file() else None


def _jsonl(root: Path, name: str) -> Iterator[dict]:
    fh = _open(root, name)
    if fh is None:
        return iter(())
    return (json.loads(line) for line in fh if line.strip())


def _json(root: Path, name: str):
    fh = _open(root, name)
    return None if fh is None else json.load(fh)


def _csv(root: Path, name: str) -> pd.DataFrame | None:
    fh = _open(root, name)
    return None if fh is None else pd.read_csv(fh)


def _author(text: str, site: str) -> str:
    m = SIGNATURE_RE.search(text)
    if m:
        return m.group(1)
    m = _AGENT_OURS_RE.search(text)
    if m:
        return m.group(0)
    return f"anon@{site}"


class CorpusAdapter(Adapter):
    name = "corpus"

    def __init__(self, exclude_sites: tuple[str, ...] = ("prowiki.org/dse",)):
        self.exclude_sites = tuple(exclude_sites)

    # ------------------------------------------------------------------ sniff
    def sniff(self, path: Path) -> bool:
        path = Path(path)
        if path.is_file() and not zipfile.is_zipfile(path) and path.name != "records.jsonl":
            return False
        fh = _open(path, "records.jsonl")
        if fh is None:
            return False
        try:
            line = fh.readline()
            first = json.loads(line) if line.strip() else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        return isinstance(first, dict) and {"id", "text", "origins"} <= set(first)

    # ------------------------------------------------------------------- load
    def load(self, path: Path) -> Bundle:
        path = Path(path)
        notes: dict = {}

        rec_rows = self._records(path, notes)
        short_rows = self._shortener(path, notes)
        wiki_rows = self._other_wikis(path, notes)
        self._links(path, notes)
        self._coverage(path, notes)

        cols = ["event_id", "ts", "channel", "author_raw", "author_alt", "text", "parent_id",
                "visible_until", "channel_family", "source_ref", "site", "source_kind", "ts_grade"]
        frames = [f for f in (rec_rows, short_rows, wiki_rows) if f is not None and len(f)]
        if frames:
            ev = pd.concat(frames, ignore_index=True)
        else:
            ev = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
        ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
        ev["visible_until"] = pd.Series(pd.NaT, index=ev.index, dtype="datetime64[ns, UTC]")
        ev["channel_family"] = ""
        ev["ts_grade"] = None   # the corpus states no clock corroboration
        ev["author_alt"] = ev["author_alt"].fillna("") if "author_alt" in ev else ""
        ev = ev.sort_values(["ts", "site", "channel"], kind="stable").reset_index(drop=True)[cols]

        notes["n_events"] = int(len(ev))
        notes["events_per_site"] = ev["site"].value_counts().to_dict() if len(ev) else {}
        notes["events_per_source_kind"] = ev["source_kind"].value_counts().to_dict() if len(ev) else {}
        notes["ip16_overlap_note"] = (
            "ip16 prefixes on shortener / other-wiki rows were not matched against the primary "
            "wiki dump (data/raw/revisions.jsonl is not read by this adapter); join on author_raw "
            "'ip16:*' against the wiki bundle's author_alt downstream if needed."
        )
        caps = Capabilities(has_wall_clock=True, has_explicit_author=False, has_reads=False,
                            has_lifecycle=False, has_threading=False, has_episodes=False)
        return Bundle(events=ev, lifecycle=None, reads=None, capabilities=caps, notes=notes)

    # ---------------------------------------------------------- records.jsonl
    def _records(self, path: Path, notes: dict) -> pd.DataFrame | None:
        rec_ids, texts, authors_sig, origins = [], [], [], []
        n_records = 0
        for r in _jsonl(path, "records.jsonl"):
            n_records += 1
            rid = r.get("id", "")
            text = rewrite_omitted_urls(r.get("text") or "")
            # signature / self-identifier are per record; the anon fallback is per site
            sig = _author(text, "")
            if sig == "anon@":
                sig = ""
            for o in r.get("origins") or []:
                origins.append({
                    "parent_id": rid,
                    "text": text,
                    "_sig": sig,
                    "site": str(o.get("site") or ""),
                    "_title": str(o.get("title") or ""),
                    "source_kind": str(o.get("kind") or ""),
                    "source_ref": str(o.get("source_id") or ""),
                    "_lit": o.get("source_date_literal"),
                })
        notes["n_records"] = n_records
        notes["n_origins"] = len(origins)
        if not origins:
            notes.update(n_origins_with_time=0, n_dropped_no_time=0, n_dropped_late=0,
                         n_dropped_excluded_sites=0, n_dropped_duplicates=0)
            return None

        df = pd.DataFrame(origins)
        df["ts"] = parse_date_literals(df["_lit"])
        has_time = df["ts"].notna()
        late = has_time & (df["ts"] > LATE_CUTOFF)
        excluded = has_time & ~late & df["site"].isin(self.exclude_sites)
        notes["n_origins_with_time"] = int(has_time.sum())
        notes["n_dropped_no_time"] = int((~has_time).sum())
        notes["n_dropped_late"] = int(late.sum())
        notes["n_dropped_excluded_sites"] = int(excluded.sum())
        notes["origins_per_site"] = df["site"].value_counts().to_dict()
        notes["no_time_literals"] = (
            df.loc[~has_time, "_lit"].fillna("").astype(str).str.strip()
            .str.replace(r"\d", "9", regex=True).value_counts().head(10).to_dict()
        )

        df = df[has_time & ~late & ~excluded].copy()
        df["channel"] = df["site"] + "/" + df["_title"]
        df.loc[df["_title"] == "", "channel"] = df["site"]
        df["_key"] = df["text"].map(norm_key)
        before = len(df)
        df = df.drop_duplicates(["channel", "ts", "_key"], keep="first")
        notes["n_dropped_duplicates"] = int(before - len(df))

        df["author_raw"] = [sig if sig else f"anon@{site}" for sig, site in zip(df["_sig"], df["site"])]
        df["author_alt"] = ""
        df["event_id"] = [
            stable_id("corpus", site, ch, ref, pid, ts.isoformat())
            for site, ch, ref, pid, ts in zip(df["site"], df["channel"], df["source_ref"],
                                              df["parent_id"], df["ts"])
        ]
        return df[["event_id", "ts", "channel", "author_raw", "author_alt", "text", "parent_id",
                   "source_ref", "site", "source_kind"]].reset_index(drop=True)

    # ---------------------------------------------------- shortener-logs.json
    def _shortener(self, path: Path, notes: dict) -> pd.DataFrame | None:
        d = _json(path, "shortener-logs.json")
        if not d:
            return None
        rows, clicks, times = [], [], []
        sites = d.get("sites") or ([d] if "links" in d else [])
        for s in sites:
            site = str(s.get("site") or "rmn.re")
            for i, l in enumerate(s.get("links") or []):
                kw = str(l.get("keyword") or "")
                title = str(l.get("title") or "")
                url = str(l.get("url") or "")
                ts = pd.to_datetime(l.get("time"), utc=True, errors="coerce")
                rows.append({
                    "event_id": stable_id("corpus", site, "shortener", kw, i),
                    "ts": ts,
                    "channel": f"{site}/log",
                    "author_raw": f"ip16:{l.get('ip16')}",
                    "author_alt": "",
                    "text": f"{kw}: {title} {url}".strip(),
                    "parent_id": None,
                    "source_ref": kw,
                    "site": site,
                    "source_kind": "shortener_link",
                })
                if isinstance(l.get("clicks"), (int, float)):
                    clicks.append(float(l["clicks"]))
                if pd.notna(ts):
                    times.append(ts)
        notes["shortener"] = {
            "n_links": len(rows),
            "clicks_total": int(sum(clicks)),
            "clicks_median": float(pd.Series(clicks).median()) if clicks else None,
            "date_min": min(times).strftime("%Y-%m-%d") if times else None,
            "date_max": max(times).strftime("%Y-%m-%d") if times else None,
        }
        return pd.DataFrame(rows) if rows else None

    # ------------------------------------------------------- other-wikis.json
    def _other_wikis(self, path: Path, notes: dict) -> pd.DataFrame | None:
        d = _json(path, "other-wikis.json")
        if not d:
            return None
        pages = d.get("pages") if isinstance(d, dict) else d
        rows, n_rev = [], 0
        for p in pages or []:
            site = str(p.get("wiki") or "")
            page_id = str(p.get("page_id") or f"{site}/{p.get('name', '')}")
            for rev in p.get("revisions") or []:
                n_rev += 1
                added = [a for a in (rev.get("added") or [])
                         if not str(a).startswith("[content not recovered")]
                text = "\n".join(str(a) for a in added)
                if not text.strip():
                    continue
                seq = rev.get("seq")
                rows.append({
                    "event_id": stable_id("corpus", site, page_id, seq),
                    "ts": pd.to_datetime(rev.get("time"), utc=True, errors="coerce"),
                    "channel": page_id,
                    "author_raw": f"ip16:{rev.get('ip16')}",
                    "author_alt": "",
                    "text": text,
                    "parent_id": None,
                    "source_ref": f"{page_id}@{seq}",
                    "site": site,
                    "source_kind": "wiki_revision_lines",
                })
        notes["other_wikis"] = {"n_pages": len(pages or []), "n_revisions": n_rev,
                                "n_revisions_with_lines": len(rows)}
        return pd.DataFrame(rows) if rows else None

    # ------------------------------------------------------------ links.jsonl
    def _links(self, path: Path, notes: dict) -> None:
        hosts: Counter = Counter()
        n = 0
        for l in _jsonl(path, "links.jsonl"):
            n += 1
            hosts[str(l.get("host") or "")] += 1
        notes["n_links"] = n
        notes["top_link_hosts"] = dict(hosts.most_common(15))

    # ----------------------------------------------------------- coverage csv
    def _coverage(self, path: Path, notes: dict) -> None:
        cov = _csv(path, "site-coverage.csv")
        gaps = _csv(path, "coverage-gaps.csv")
        if cov is None and gaps is None:
            return
        by_cat: dict[str, dict[str, int]] = {}
        if cov is not None and "category" in cov:
            for cat, n in cov["category"].fillna("").value_counts().items():
                by_cat.setdefault(str(cat), {"sites": 0, "gaps": 0})["sites"] = int(n)
        if gaps is not None and "category" in gaps:
            for cat, n in gaps["category"].fillna("").value_counts().items():
                by_cat.setdefault(str(cat), {"sites": 0, "gaps": 0})["gaps"] = int(n)
        notes["coverage"] = {
            "n_sites": int(len(cov)) if cov is not None else 0,
            "n_gap_rows": int(len(gaps)) if gaps is not None else 0,
            "by_category": by_cat,
        }

    # ----------------------------------------------------------------- config
    def config(self) -> AdapterConfig:
        return AdapterConfig(
            name="cross-site-corpus",
            families=dict(FAMILIES),
            techniques=list(TECHNIQUES),
            family_from_channel=False,
        )
