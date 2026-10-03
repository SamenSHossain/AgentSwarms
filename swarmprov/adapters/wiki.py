"""Adapter for the OpenAI wiki-swarm dump (UseMod wiki farm export).

Input: a directory or .zip holding ``revisions.jsonl``, ``events.jsonl``,
``pages.jsonl`` (optionally ``labels.jsonl`` / ``manifest.json``); gzipped
copies (``*.jsonl.gz`` or upload names like ``abc-revisions.jsonl_1.gz``) work too.

* revisions.jsonl - one full page snapshot per save, with the wiki account
  label, an IP /16 prefix and a corroborated wall-clock ``time``.
* events.jsonl - saves plus admin deletions, recreations and attack probes.
  No page-view rows, so exposure must be inferred (``has_reads=False``).
* pages.jsonl - per-page metadata including a task ``page_family`` map.
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd

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

        lifecycle = pd.DataFrame(columns=["channel", "ts", "action", "actor"])
        if len(events):
            ev = events[events["event_type"].isin(["delete", "revert"])].copy()
            ev["channel"] = ev["wiki"] + "/" + ev["page"]
            ev["ts"] = pd.to_datetime(ev["time"], utc=True)
            ev["action"] = ev["event_type"].map({"delete": "delete", "revert": "recreate"})
            ev["actor"] = ev.get("actor_label")
            lifecycle = ev[["channel", "ts", "action", "actor"]].reset_index(drop=True)

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

        human = set(labels.loc[labels["is_human_handle"], "label"]) if len(labels) else set()
        notes = {
            "n_revisions": len(revs),
            "n_pages": len(pages),
            "n_posts": len(posts),
            "n_deletions": int((lifecycle["action"] == "delete").sum()),
            "human_handles": sorted(human),
            "revisions_per_wiki": revs["wiki"].value_counts().to_dict(),
            "revisions_per_day": revs["ts"].dt.strftime("%Y-%m-%d").value_counts().sort_index().to_dict(),
            "event_types": events["event_type"].value_counts().to_dict() if len(events) else {},
        }
        caps = Capabilities(has_wall_clock=True, has_explicit_author=False, has_reads=False,
                            has_lifecycle=True, has_threading=False, has_episodes=False)
        self._human = human
        self._pages = pages
        return Bundle(events=posts, lifecycle=lifecycle, reads=None, capabilities=caps, notes=notes)

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
