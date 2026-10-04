"""Adapter for the OpenAI wiki-swarm dump (UseMod wiki farm export).

Input: a directory or .zip holding ``revisions.jsonl``, ``events.jsonl``,
``pages.jsonl`` (optionally ``labels.jsonl`` / ``manifest.json``); gzipped
copies (``*.jsonl.gz`` or upload names like ``abc-revisions.jsonl_1.gz``) work too.

* revisions.jsonl - one full page snapshot per save, with the wiki account
  label, an IP /16 prefix and a corroborated wall-clock ``time``.
* events.jsonl - saves plus admin deletions, first-recreation edges (a
  deletion -> the first later save on the page; 4 of them produced no stored
  revision and are typed ``revert``) and script-injection probe requests.
  No page-view rows, so exposure must be inferred (``has_reads=False``).
* pages.jsonl - per-page metadata including a task ``page_family`` map.

Clocks: a post's ``ts`` is the save *request* second (``revisions.time`` equals
``request_time``; the save completed 0-2 s later).  A deletion's ``ts`` is its
*success* second (``events.time`` equals ``success_time``, 1 s after the request
on some rows): the page stays readable until the delete succeeds, so that is the
right ``visible_until``; ``request_time`` marks only the admin's intent.
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd

from .. import lifecycle as lifecycle_mod
from ..schema import conform
from ..segment import snapshots_to_posts
from .base import Adapter, AdapterConfig, Bundle, Capabilities, Technique

INFRA_FAMILIES = {
    "source-cache-url-list", "relay-coordination", "source-or-unclassified",
    "off_store_unclassified", "loop-chain-infrastructure", "probe-test",
    "unknown", "mixed-task",
}

# Order matters only for ties (same match position): more specific first.
FAMILIES: dict[str, str] = {
    "oecd-equity": r"OECD\s*(?:Education\s*)?Equity|Education\s*Equity|pre-?primary",
    "oecd-regional-co2": r"Regional\s*Recovery|\bRRP|CO2",
    "oecd-household-income": r"Household\s*Disposable\s*Income",
    "datausa-clothing-workforce": r"Clothing",
    "datausa-grocery-workforce": r"Grocery",
    "datausa-cashiers-bachelors": r"Cashier\w*\s*Bachelor",
    "datausa-cashier-skills": r"Cashier\s*Skills",
    "datausa-cashiers-masters": r"Cashier",
    "datausa-construction-wage": r"Construction\s*Wage",
    "datausa-construction-workforce": r"Construction",
    "datausa-sector61-state": r"Sector\s*61|S61x62|61-62",
    "datausa-occupation-salary-61-62": r"Occupation\s*Salary|School\s*Psych",
    "ihme-cvd-deaths": r"\bCVD\b|Healthdata\s*CVD",
    "ihme-family-planning": r"Family\s*Planning",
    "ihme-mcv2": r"MCV2",
    "ihme-smoking": r"Healthdata\s*Smoking",
    "ihme-lymphatic-filariasis": r"Lymphatic|LFSequence",
    "datausa-language-french": r"French|DataUSALang",
    "fuel-poverty-ni": r"Fuel\s*Poverty",
    "world-poverty-clock": r"World\s*Poverty\s*Clock",
    "datausa-poverty-state": r"Poverty\s*State",
    "datausa-poverty-county": r"Poverty",
    "datausa-maids-wage": r"\bMaids?\b",
    "datausa-police-wage-age": r"Police",
    "datausa-finance-gender-gap": r"Finance|financial\s*managers",
    "aihw-pbs": r"\bPBS\b|AIHW",
    "datausa-transport-production": r"Transport",
    "datausa-production-share": r"Production\s*Occupation|DataUSAProd",
    "datausa-slp-ethnicity": r"\bSLP\b|Speech\s*Path",
    "datausa-ivy-tuition": r"\bIvy\b",
    "uefa-pass-accuracy": r"UEFA",
    "vermont-rent": r"Vermont|Lamoille",
    "nyc-veterans": r"NYC\s*Veterans",
    "datausa-elpaso-foreign-born": r"El\s*Paso",
    "datausa-enrollment-asian": r"Enrollment\s*Asian",
    "unaids-bosnia-hiv": r"UNAIDS",
    "sdg-index-score": r"SDG\s*Index",
    "gapminder-age80": r"Gapminder",
}

TECHNIQUES = [
    Technique("blob-hostname-bypass",
              r"blob\s*host(?:name)?|blob\.core\.windows|allowed\s+blob|blob-host",
              ("oecd-equity",),
              "Route Power BI embed calls through an allowed blob hostname with the real Host header"),
    Technique("live-tooltip-value",
              r"(?:real|live|actual)[-\s]?(?:dashboard[-/\s]?)?tooltip|tooltip\s+(?:says|shows|value)",
              ("oecd-equity",),
              "Read the rendered dashboard tooltip instead of the rounded EAG table"),
    Technique("heartbeat-beacon", r"heart\s?beat|beacon", (),
              "Background wiki writes to test whether the container survives the final answer"),
    Technique("zzz-backup-pages", r"ZZZ", (),
              "ZZZ-suffixed backup / link-list pages"),
    Technique("rng-seed-crack", r"random\.shuffle|\bseed\s*\d{4,}|\bRNG\b", (),
              "Predicting the question order by brute-forcing the shuffle seed"),
    Technique("clock-wait-fastforward", r"clock\.wait", (),
              "Advancing the task clock with clock.wait"),
]


def _first(v):
    """First element of a list-valued field, the value itself otherwise."""
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _recreations(revs: pd.DataFrame, events: pd.DataFrame, deletes: pd.DataFrame) -> pd.DataFrame:
    """One lifecycle 'recreate' row per first-recreation edge.

    Edges live on the stored revisions (``relation_type == first_recreation_of``,
    ``related_event_id`` a list of delete ids, ``round_id`` a parallel list) and
    on the ``revert`` event rows, whose response produced no stored revision."""
    held = dict(zip(deletes["event_id"], deletes.get("page_held", pd.Series(index=deletes.index, dtype=object))))
    rows = []
    if "relation_type" in revs:
        for r in revs[revs["relation_type"] == "first_recreation_of"].itertuples(index=False):
            rel = r.related_event_id if isinstance(r.related_event_id, list) else [r.related_event_id]
            rid = getattr(r, "round_id", None)
            rounds = rid if isinstance(rid, list) else [rid] * len(rel)
            for k, dele in enumerate(rel):
                rows.append({"channel": r.channel, "ts": r.ts, "action": "recreate",
                             "actor": r.label or f"ip16:{r.ip16}", "event_id": f"recreate:{r.rev_id}",
                             "related_event_id": dele, "revision_ref": r.rev_id, "page_held": held.get(dele),
                             "round_id": rounds[k] if k < len(rounds) else None})
    for r in events[events["event_type"] == "revert"].itertuples(index=False):
        rows.append({"channel": f"{r.wiki}/{r.page}", "ts": pd.to_datetime(r.time, utc=True), "action": "recreate",
                     "actor": getattr(r, "actor_label", None), "event_id": r.event_id,
                     "related_event_id": _first(r.related_event_id), "revision_ref": None,
                     "page_held": getattr(r, "page_held", None), "round_id": _first(getattr(r, "round_id", None))})
    return pd.DataFrame(rows)


def _probes(events: pd.DataFrame, revs: pd.DataFrame) -> pd.DataFrame:
    """Probe requests with the nearest same-prefix save for co-timing (strict rule)."""
    p = events[events["event_type"] == "probe"].copy()
    if not len(p):
        return conform(pd.DataFrame(), "probes")
    p["ts"] = pd.to_datetime(p["time"], utc=True)
    p["ip16"] = p["ip16"].astype(str)
    saves = revs[["ts", "label", "ip16"]].copy()
    saves["ip16"] = saves["ip16"].astype(str)
    by_ip = {ip: g.sort_values("ts") for ip, g in saves.groupby("ip16")}
    labels_on = {ip: int(g.loc[g["label"].astype(bool), "label"].nunique()) for ip, g in by_ip.items()}
    near_label, near_dt, n_labels, cotimed = [], [], [], []
    for ip, ts in zip(p["ip16"], p["ts"]):
        g = by_ip.get(ip)
        if g is None or not len(g):
            near_label.append(None); near_dt.append(None); n_labels.append(0); cotimed.append(None)
            continue
        dt = (g["ts"] - ts).dt.total_seconds()
        i = dt.abs().idxmin()
        lab, d, n = g.loc[i, "label"] or None, float(dt.loc[i]), labels_on[ip]
        near_label.append(lab); near_dt.append(d); n_labels.append(n)
        cotimed.append(lab if lab and abs(d) <= 1 and n == 1 else None)
    site = p["event_id"].astype(str).str.extract(r"attacklog_raw_([a-z0-9]+)_")[0]
    out = pd.DataFrame({
        "probe_id": p["event_id"], "ts": p["ts"], "site": site.where(site.notna(), p.get("wiki")),
        "actor": "ip16:" + p["ip16"], "ip16": p["ip16"], "request_action": p["request_action"],
        "param_family": p.get("param_family"), "success_observed": p.get("success_observed"),
        "time_grade": p.get("time_grade"), "source_ref": p["source_refs"].map(_first) if "source_refs" in p else None,
        "nearest_save_label": near_label, "nearest_save_dt_s": near_dt, "n_labels_on_prefix": n_labels,
        "cotimed_label": cotimed,
    })
    return conform(out.reset_index(drop=True), "probes")


def _open_member(root: Path, stem: str) -> io.TextIOBase | None:
    """Find ``stem`` (e.g. 'revisions.jsonl') in a dir, zip, or as gz uploads."""
    if root.is_file() and zipfile.is_zipfile(root):
        z = zipfile.ZipFile(root)
        for n in z.namelist():
            if Path(n).name == stem:
                return io.TextIOWrapper(z.open(n), encoding="utf-8")
        return None
    folder = root if root.is_dir() else root.parent
    for p in sorted(folder.iterdir()):
        name = p.name
        if name == stem:
            return open(p, encoding="utf-8")
        if stem in name and name.endswith((".gz", "_1.gz")):
            return io.TextIOWrapper(gzip.open(p), encoding="utf-8")
    return None


def _jsonl(root: Path, stem: str) -> Iterator[dict]:
    fh = _open_member(root, stem)
    if fh is None:
        return iter(())
    return (json.loads(line) for line in fh if line.strip())


def _recreation_cutoff(root: Path) -> pd.Timestamp | None:
    """The dump's recreation-eligibility cutoff (manifest.recreation_source.cutoff_unix_ts), if present."""
    fh = _open_member(root, "manifest.json")
    if fh is None:
        return None
    try:
        m = json.load(fh)
        ts = (m.get("recreation_source") or {}).get("cutoff_unix_ts")
        return pd.Timestamp(int(ts), unit="s", tz="UTC") if ts else None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _delete_lag(events: pd.DataFrame) -> dict:
    """Seconds between a deletion's request and its success, as a histogram."""
    if not len(events) or "request_time" not in events:
        return {}
    d = events[events["event_type"] == "delete"]
    lag = (pd.to_datetime(d["time"], utc=True) - pd.to_datetime(d["request_time"], utc=True)).dt.total_seconds()
    out = {str(int(k)): int(v) for k, v in lag.value_counts().sort_index().items()}
    if lag.isna().any():
        out["unknown"] = int(lag.isna().sum())
    return out


class WikiAdapter(Adapter):
    name = "wiki"

    def sniff(self, path: Path) -> bool:
        fh = _open_member(Path(path), "revisions.jsonl")
        if fh is None:
            return False
        first = json.loads(fh.readline())
        return {"rev_id", "page_id", "body", "label", "time"} <= set(first)

    def load(self, path: Path) -> Bundle:
        path = Path(path)
        pages = pd.DataFrame(list(_jsonl(path, "pages.jsonl")))
        revs = pd.DataFrame(list(_jsonl(path, "revisions.jsonl")))
        events = pd.DataFrame(list(_jsonl(path, "events.jsonl")))
        labels = pd.DataFrame(list(_jsonl(path, "labels.jsonl")))

        revs["ts"] = pd.to_datetime(revs["time"], utc=True)
        revs["author_alt"] = [
            lbl if lbl else f"ip16:{ip}" for lbl, ip in zip(revs["label"], revs["ip16"])
        ]
        revs = revs.rename(columns={"page_id": "channel"})

        lifecycle = conform(pd.DataFrame(), "lifecycle")
        probes = conform(pd.DataFrame(), "probes")
        if len(events):
            dele = events[events["event_type"] == "delete"].copy()
            dele["channel"] = dele["wiki"] + "/" + dele["page"]
            dele["ts"] = pd.to_datetime(dele["time"], utc=True)
            dele["action"] = "delete"
            dele["actor"] = dele.get("actor_label")
            dele["related_event_id"] = None
            dele["revision_ref"] = None
            if "round_id" in dele:
                dele["round_id"] = dele["round_id"].map(_first)
            recs = _recreations(revs, events, dele)
            lifecycle = conform(pd.concat([conform(dele, "lifecycle"), conform(recs, "lifecycle") if len(recs) else None],
                                          ignore_index=True), "lifecycle").sort_values("ts").reset_index(drop=True)
            probes = _probes(events, revs)

        posts = snapshots_to_posts(
            revs[["channel", "ts", "rev_id", "author_alt", "body"]],
            lifecycle[lifecycle["action"] == "delete"][["channel", "ts"]],
            boilerplate={"Beschreibe hier die neue Seite.", "Describe the new page here."},
        )
        fam = dict(zip(pages["page_id"], pages["page_family"])) if len(pages) else {}
        posts["channel_family"] = [
            f if (f := fam.get(ch, "")) not in INFRA_FAMILIES else "" for ch in posts["channel"]
        ]
        posts["source_ref"] = posts["parent_id"]
        posts["site"] = posts["channel"].str.split("/").str[0]
        posts["source_kind"] = "revision"
        if "time_grade" in revs:
            posts["ts_grade"] = posts["parent_id"].map(dict(zip(revs["rev_id"], revs["time_grade"])))

        human = set(labels.loc[labels["is_human_handle"], "label"]) if len(labels) else set()
        recs = lifecycle[lifecycle["action"] == "recreate"]
        cutoff = _recreation_cutoff(path)
        notes = {
            "n_revisions": len(revs),
            "n_pages": len(pages),
            "n_posts": len(posts),
            "n_deletions": int((lifecycle["action"] == "delete").sum()),
            "n_recreations": int(len(recs)),
            "n_recreations_with_revision": int(recs["revision_ref"].map(lambda v: isinstance(v, str) and bool(v)).sum()),
            "n_probes": int(len(probes)),
            "clock_grades": revs["time_grade"].value_counts().to_dict() if "time_grade" in revs else {},
            "clock_uncertainty_s": sorted(float(u) for u in revs["uncertainty_seconds"].dropna().unique()) if "uncertainty_seconds" in revs else [],
            "delete_request_lag_s": _delete_lag(events),
            "recreation_check": lifecycle_mod.recreation_check(
                revs[["channel", "ts", "rev_id", "body"]], lifecycle, posts, cutoff) if len(recs) else None,
            "human_handles": sorted(human),
            "revisions_per_wiki": revs["wiki"].value_counts().to_dict(),
            "revisions_per_day": revs["ts"].dt.strftime("%Y-%m-%d").value_counts().sort_index().to_dict(),
            "event_types": events["event_type"].value_counts().to_dict() if len(events) else {},
        }
        caps = Capabilities(has_wall_clock=True, has_explicit_author=False, has_reads=False,
                            has_lifecycle=True, has_threading=False, has_episodes=False,
                            has_request_log=bool(len(probes)))
        self._human = human
        self._pages = pages
        return Bundle(events=posts, lifecycle=lifecycle, reads=None, probes=probes if len(probes) else None,
                      capabilities=caps, notes=notes)

    def config(self) -> AdapterConfig:
        return AdapterConfig(
            name="openai-wiki",
            families=FAMILIES,
            techniques=TECHNIQUES,
            featured_disputes=[("oecd-equity", "Poland"), ("oecd-equity", "Hungary"),
                               ("oecd-equity", "Slovak Republic")],
            disputes=[{
                "family": "datausa-grocery-workforce", "slot": "G5 (cracked-seed forecast vs observed)",
                "context": r"\bG5\b",
                "variants": {"Maryland (RNG forecast)": r"Maryland|\bMD\b",
                             "Montana (observed)": r"Montana|\bMT\b"},
            }],
            ignore_authors=("[Admin1]", "[Admin2]", "[Person22]"),
        )
