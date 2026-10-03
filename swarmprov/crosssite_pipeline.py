"""Cross-site stage: combine the primary wiki run with the cross-site corpus and
ask how techniques moved *between surfaces*, plus the activity timeline and the
collectors' coverage bounds.  Written against ``analysis/crosssite.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import adapters
from .adapters.base import Technique
from .analysis import crosssite
from .schema import RunDir, conform

EXTRA_TECHNIQUES = [
    Technique("proxy-markdown-new", r"markdown\.new", (), "markdown.new fetch proxy"),
    Technique("proxy-allorigins", r"allorigins\.hexlet\.app", (), "allorigins CORS proxy"),
    Technique("proxy-jqp", r"jqp\.vercel\.app", (), "jqp jq-over-URL proxy"),
    Technique("proxy-md-succ", r"md\.succ\.ai", (), "md.succ.ai markdown proxy"),
    Technique("proxy-jina", r"r\.jina\.ai", (), "r.jina.ai reader proxy"),
    Technique("proxy-pure-md", r"pure\.md", (), "pure.md reader proxy"),
]

DEFAULT_MARKERS = [
    ("2026-05-26 10:21", "markdown.new on dse"),
    ("2026-06-16 10:54", "cracked seed (G5 Maryland)"),
    ("2026-06-18 19:51", "rmn.re burst"),
    ("2026-06-20 05:10", "blob bypass / tooltip"),
]


def _wiki_events(run: RunDir) -> pd.DataFrame:
    ev = run.read("events")
    if "site" not in ev.columns or ev["site"].isna().all():
        ev["site"] = ev["channel"].astype(str).str.split("/").str[0]
    if "source_kind" not in ev.columns:
        ev["source_kind"] = "revision"
    return ev


def run_crosssite(wiki_run: str | Path, corpus_path: str | Path, out: str | Path,
                  markers=DEFAULT_MARKERS) -> RunDir:
    run = RunDir(out)
    wiki = RunDir(wiki_run)
    ev_w = _wiki_events(wiki)
    a = adapters.get("corpus")
    bundle = a.load(Path(corpus_path))
    ev_c = bundle.events.copy()
    # the corpus also holds redacted copies of the dumped wikis (probier, fractal, dorfwiki): keep the dump's rows
    ev_c["site"] = ev_c["site"].map(crosssite.canonical_site)
    dumped = set(ev_w["site"].map(crosssite.canonical_site))
    n_dup_sites = int(ev_c["site"].isin(dumped).sum())
    ev_c = ev_c[~ev_c["site"].isin(dumped)]
    ev_c["family"] = ""
    cols = [c for c in conform(ev_w, "events").columns] + ["family"]
    ev = pd.concat([ev_w[cols], ev_c[cols]], ignore_index=True).sort_values("ts")
    run.write("events", ev)

    techs = list(a.config().techniques)
    names = {t.name for t in techs}
    techs += [t for t in EXTRA_TECHNIQUES if t.name not in names]
    spread = crosssite.technique_spread(ev, techs)
    summary = crosssite.technique_site_summary(spread, ev, techs)
    run.write("technique_spread", spread)
    run.write("technique_site_summary", summary)
    fig_spread = crosssite.technique_spread_figure(spread, run.figure("x1_technique_spread.png"))
    fig_time = crosssite.timeline(ev, run.figure("x2_timeline.png"),
                                  [(pd.Timestamp(t, tz="UTC"), lab) for t, lab in markers])
    cov_path = Path(corpus_path) / "site-coverage.csv"
    cov = crosssite.coverage_summary(cov_path) if cov_path.exists() else None

    L = ["# Cross-site view\n",
         f"{len(ev):,} posts across {ev['site'].nunique()} surfaces: {len(ev_w):,} from the primary wiki run "
         f"`{wiki.path}` and {len(ev_c):,} from the corpus `{corpus_path}` "
         f"({bundle.notes.get('n_origins_with_time', '?')} timed origins out of {bundle.notes.get('n_origins', '?')}; "
         f"{n_dup_sites} corpus rows dropped as redacted copies of wikis already in the dump).\n",
         "## Posts per surface\n"]
    per_site = ev.groupby("site").agg(posts=("event_id", "size"), authors=("author_raw", "nunique"),
                                      first=("ts", "min"), last=("ts", "max")).sort_values("posts", ascending=False)
    L.append(_md(per_site.head(20)))
    L.append("\n## Technique spread between surfaces\n")
    L.append("A technique's origin is its first matching post anywhere; a surface adopts it at its first matching post. "
             "At-risk surfaces are those with any activity after the origin. Timings are bounded by what was captured "
             "(see coverage below).\n")
    if fig_spread:
        L.append(f"![spread](figures/{Path(fig_spread).name})\n")
    L.append(_md(summary, index=False))
    L.append("\n## Activity timeline\n")
    L.append(f"![timeline](figures/{Path(fig_time).name})\n")
    if cov:
        L.append("\n## Coverage bounds (from the collectors' site inventory)\n")
        L.append(crosssite.coverage_markdown(cov))
    (run.path / "report.md").write_text("\n".join(L))
    (run.path / "summary.json").write_text(json.dumps({
        "n_events": int(len(ev)), "n_sites": int(ev["site"].nunique()), "corpus_notes": bundle.notes,
        "technique_site_summary": summary.to_dict("records"),
        "coverage": {k: v for k, v in (cov or {}).items() if not isinstance(v, pd.DataFrame)},
    }, indent=2, default=str))
    print(f"[swarmprov] wrote {run.path / 'report.md'}")
    return run


def _md(df: pd.DataFrame, index: bool = True) -> str:
    from .report import md_table
    return md_table(df, index=index, floatfmt="{:.1f}")
