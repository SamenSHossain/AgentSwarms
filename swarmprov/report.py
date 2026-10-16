"""Run the analyses on a RunDir and write ``report.md`` + figures + summary.json."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from . import checks, graph, identity, lifecycle as lifecycle_mod, roster as roster_mod, suggest, village as village_mod
from .adapters.base import read_capabilities
from .adapters.chat import no_author_warning
from .analysis import agents as agents_mod, causal, diffusion, errors, origins, provenance, structure, uncertainty
from .claims import agent_rounds, consensus, public, value_key
from .exposure import credited
from .fmt import dumps, json_safe, pct
from .schema import TABLES, RunDir, other_mapping, suffix


# capability -> (consequence when the source has it, consequence when it does not)
CONSEQUENCES = {
    "has_wall_clock": ("ordering by real time is possible", "no reliable wall clock, so posts cannot be ordered by real time"),
    "has_order": ("posts can be put in order: by the wall clock, or by message position when there is none",
                  "no reliable order: only posts with a timestamp take part in time comparisons"),
    "has_explicit_author": ("authors come from fields, not parsed signatures", "authors are parsed from signatures in the text"),
    "has_reads": ("tool results are recorded as reads by the agent that called the tool (A1 counts own lookups); "
                  "exposure to other agents' posts is still inferred from what was public",
                  "no read events, so exposure is inferred from what was public"),
    "has_lifecycle": ("deletions are recorded, posts carry visible_until, and A1 states D_visible beside D",
                      "no deletion times are used (none recorded, or no wall clock to order them by), so D_visible "
                      "equals D"),
    "has_request_log": ("request-level rows exist; still no page views, so exposure stays inferred",
                        "no request-level rows"),
    "derived_text": ("posts are an LLM digest of what agents did, not their words: values and quotes are second-hand",
                     "posts are the agents' own text"),
    "has_threading": ("reply links become 'reply' edges: interactions, not knowledge transfers, so A5 counts them for "
                      "brokers and the GraphML, never for supply or relays",
                      "no reply links: edges are inferred from timing"),
    "has_episodes": ("rounds are fields", "rounds are parsed from text"),
    "has_activity": ("a presence log says when each agent was running", "no presence log; activity is inferred from posts"),
}

SECTIONS = {"A1": "A1. Provenance split", "A2": "A2. Does availability cause copying?", "A3": "A3. Technique diffusion",
            "A4": "A4. Error propagation", "A5": "A5. Structure", "agents": "Agents", "A6": "A6. Reach"}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}", pct: tuple[str, ...] = (), index: bool = True) -> str:
    if df is None or len(df) == 0:
        return "_(none)_\n"
    d = df.reset_index() if index else df
    cols = [str(c).replace("|", "\\|") for c in d.columns]

    def fmt(c, v):
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT:
            return ""
        if c in pct and isinstance(v, (int, float, np.floating)):
            return f"{v:.0%}"
        if isinstance(v, (float, np.floating)):
            return str(int(v)) if float(v).is_integer() and abs(v) >= 1 and c in ("n", "episode") else floatfmt.format(v)
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d %H:%M")
        return " ".join(str(v).split()).replace("|", "\\|")   # a line break would end the table row

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for rec in d.to_dict("records"):
        lines.append("| " + " | ".join(fmt(c, rec[c]) for c in d.columns) + " |")
    return "\n".join(lines) + "\n"


def _span(lo, hi, hours: bool = False, fmt: str = "{:.0%}") -> str:
    """An interval's bounds: "88–96%", "9.0–16.5 h", or with ``fmt`` a number format ("{:.2f}")."""
    if hours:
        return f"{lo:.1f}–{hi:.1f} h"
    if fmt == "{:.0%}":
        return f"{lo * 100:.0f}–{hi:.0%}"
    return f"{fmt.format(lo)}–{fmt.format(hi)}"


def _median_cell(m: float, lo: float, hi: float) -> str:
    """A3's median time to adopt with Greenwood's interval, "5.7 (4.8–8.5)"; an upper bound the band never
    reaches reads "not reached", and no median leaves the cell empty."""
    if m != m:
        return ""
    return f"{m:.1f}" + (f" ({lo:.1f}–{'not reached' if hi != hi else f'{hi:.1f}'})" if lo == lo else "")


def _with_ci(df: pd.DataFrame, cols, fmt: str = "{:.0%}") -> pd.DataFrame:
    """Each of ``cols`` with ``<col>_lo`` / ``<col>_hi`` beside it becomes one text column, "94% (90–97%)" (the
    value alone where the interval is NaN), headed "<col> (95% CI)"."""
    out = df.copy()
    for c in cols:
        if f"{c}_lo" not in out:
            continue
        if out[f"{c}_lo"].isna().all():
            out = out.drop(columns=[f"{c}_lo", f"{c}_hi"])
            continue
        out[c] = [fmt.format(v) + (f" ({_span(lo, hi, fmt=fmt)})" if lo == lo else "")
                  for v, lo, hi in zip(out[c], out[f"{c}_lo"], out[f"{c}_hi"])]
        out = out.drop(columns=[f"{c}_lo", f"{c}_hi"]).rename(columns={c: f"{c} (95% CI)"})
    return out


def _roster_block(run: RunDir, prof: dict, summary: dict) -> list[str]:
    ros = run.read("roster")
    ev = run.read("events") if prof["n_events"] else None
    ann = run.read("roster_events") if run.has("roster_events") else None
    s = roster_mod.summary(ros, ev, ann)
    summary["roster"] = _jsonable(s)
    return roster_mod.section(s, md_table, prof.get("roster_notes", {}).get("timestamp_issues"))


def _cell(v):
    """A summary.json cell: missing values as null, timestamps in ISO format, anything else not JSON as text."""
    v = json_safe(v)
    return v if v is None or isinstance(v, (int, float, str, bool)) else str(v)


def _jsonable(obj):
    """Nested tables as records, so summary.json never holds a truncated DataFrame repr."""
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict("records") if obj.index.name or not isinstance(obj.index, pd.RangeIndex) \
            else obj.to_dict("records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _village_block(run: RunDir, prof: dict, summary: dict) -> list[str]:
    directory = run.read("directory") if run.has("directory") else None
    channels = village_mod.channel_lists(run.read("channels")) if run.has("channels") else None
    ev = run.read("events") if prof["n_events"] else None
    activity = run.read("activity") if run.has("activity") else None
    roster = run.read("roster") if run.has("roster") else None
    eras = run.read("eras") if run.has("eras") else None
    era_ann = None
    if eras is not None and run.has("roster_events") and "era" in run.read("roster_events").columns:
        era_ann = run.read("roster_events")
    from .pipeline import load_config
    notes = prof.get("notes", {})
    summaries = run.read("summaries") if run.has("summaries") else None
    digest = run.read("digest") if run.has("digest") else None
    cfg = load_config(run)
    s = village_mod.summary(directory, channels, ev, activity, roster, notes.get("unrecognised"), eras, era_ann,
                            notes.get("village"), cfg.schedule_tz, summaries, digest, cfg.family_from_roster)
    L = village_mod.section(s, md_table)
    if L:  # no Village tables: no section and no stub in summary.json
        summary["village"] = _jsonable(s)
    return L


def _swarmtraces_block(prof: dict, st: dict, summary: dict) -> list[str]:
    """Reconstruction report for the Swarm traces redacted payload release.

    The release has no wall clock and no agent identity, so the provenance
    analyses cannot run; this reports what the file *does* contain — the
    reconstruction tree, tags, de-duplication and redaction coverage — and says
    plainly what is not computable.
    """
    summary["swarmtraces"] = st
    k, tree, tags, txt, red, integ = (st["by_kind"], st["tree"], st["tags"],
                                      st["text"], st["redaction"], st["integrity"])
    n = st["n_records"]
    kinds_txt = ", ".join(f"{v:,} {kk}" for kk, v in k.items())
    L = ["# Reconstruction report — Swarm traces redacted payload release\n",
         f"Source: `{prof['source']}` (adapter `{prof['adapter']}`), **{n:,} recovered records** "
         f"({kinds_txt}). This is a reconstruction corpus, not a timed multi-agent transcript: it carries no "
         "wall clock and no agent identity, so swarmprov's provenance analyses (A1–A6) do not run on it. "
         "Every number below is a count of records and fields; no payload was executed, decoded or interpreted.\n"]

    edge_txt = "; ".join(f"{v:,} {e}" for e, v in tree["child_of_parent_kind"].items()) or "none"
    L.append("## Reconstruction tree\n")
    L.append(f"The records form a two-level tree, not a timeline: {tree['roots']:,} root records "
             f"and {tree['children']:,} linked to a parent by `parent_id` (reconstruction links: {edge_txt}; "
             f"max depth {tree['max_depth']}). {tree['parents_with_children']:,} record(s) are a reconstruction parent"
             + (f"; {tree['orphan_children']:,} child record(s) name a parent not present in the release" if tree["orphan_children"] else "")
             + ". `parent_id` records which payload a response or recovered text was rebuilt from; it is not an "
             "ordering in time, so it cannot stand in for the missing clock.\n")
    fan = tree["fanout_dist"]
    if fan:
        fan_df = pd.DataFrame({"children (10 = 10+)": list(fan.keys()), "parents": list(fan.values())})
        L.append(md_table(fan_df, index=False))

    L.append("\n## Tags, text and redaction\n")
    if tags["n_tagged"]:
        fam_txt = ", ".join(f"{kk} {v}" for kk, v in tags["families"].items())
        L.append(f"Tags: {tags['n_tagged']:,} of {n:,} records carry a `;`-joined tag string "
                 f"({tags['n_families']} tag famil{'y' if tags['n_families'] == 1 else 'ies'}: {fam_txt}); "
                 "the rest are untagged.\n")
    else:
        L.append(f"Tags: none of the {n:,} records carry a tag string.\n")
    L.append(f"Text: {txt['distinct']:,} distinct bodies, so {txt['repeated']:,} record(s) repeat content already "
             "seen (identical reconstructed text recovered from more than one place).\n")
    L.append(f"Redaction: {red['records_with_placeholder']:,} of {n:,} records contain at least one bracketed "
             f"placeholder ({red['placeholder_occurrences']:,} occurrences across {red['n_categories']} categories), "
             "so every content count is a lower bound — the publishers removed credentials, hosts, encoded blobs and "
             "shortener codes before release. The categories are the publishers' own redaction labels:\n")
    if red["top_categories"]:
        red_df = pd.DataFrame({"redaction category": list(red["top_categories"].keys()),
                               "occurrences": list(red["top_categories"].values())})
        L.append(md_table(red_df, index=False))

    L.append("\n## Integrity\n")
    L.append(f"Record ids {'are' if integ['ids_unique'] else 'are NOT'} all unique; "
             f"citation handles {'are' if integ['cites_unique'] else 'are NOT'} all unique; "
             f"every citation {'is' if integ['cite_prefixes_match_id'] else 'is not always'} prefixed by its record id.\n")

    L.append("## What is not computable here\n")
    L.append("- **No wall clock.** `time_utc` is null on every record, so there is no ordering in time: "
             "patient-zero, head start, adoption order and diffusion speed cannot be computed. A provenance run "
             "needs a timed transcript (the wiki dump or an AI Village message table).\n"
             "- **No agent identity.** The acting account is redacted and the record id is a record, not an actor, "
             "so there is no \"who\": hub structure, relay chains and per-agent reach do not apply.\n"
             "- **Outbound-heavy.** Most records are requests; a response was not always recovered, and redaction "
             "removes the values that would tie records together. Treat the counts as a floor, not a census.\n")
    return L


def _warnings(prof: dict, summary: dict) -> list[str]:
    """Ingest's warnings (a field not found, a config key ignored), also kept in ``summary.json``."""
    warns = prof.get("warnings")
    if warns is None and (prof.get("notes") or {}).get("n_no_author"):   # a profile from before ingest kept warnings
        warns = [no_author_warning(prof["notes"]["n_no_author"])]
    if warns:
        summary["warnings"] = warns
    return [f"**Warning:** {w}\n" for w in warns or []]


def _write(run: RunDir, L: list[str], summary: dict, mapping: str) -> str:
    """``report.md``, ``summary.json`` and ``report.html``, or ``report_strict.md``, ``summary_strict.json`` and
    ``report_strict.html`` beside them."""
    text, sfx = "\n".join(L), suffix(mapping)
    (run.path / f"report{sfx}.md").write_text(text, encoding="utf-8")
    (run.path / f"summary{sfx}.json").write_text(dumps(summary, indent=2), encoding="utf-8")
    print(f"[swarmprov] wrote {run.path / f'report{sfx}.md'}")
    from .html import build_html
    build_html(run, mapping)
    return text


@dataclass
class _Ctx:
    """What one report's sections read: the run, its settings and tables, and the summary they fill in order.  The
    A4 value slots and the graph statistics are found once (``_slots``, ``_a5``) and read by later sections."""
    run: RunDir
    mapping: str
    cfg: object
    boot: int
    prof: dict
    caps: dict
    notes: dict
    summary: dict
    seq: bool
    agent_col: str
    alt: str
    events: pd.DataFrame
    posts: pd.DataFrame
    pseudo: list
    agents: pd.DataFrame
    claims: pd.DataFrame
    tags: pd.DataFrame
    mentions: pd.DataFrame
    rounds: pd.DataFrame
    ex: pd.DataFrame
    ex_alt: pd.DataFrame
    edges: pd.DataFrame
    chains: pd.DataFrame
    roster: bool
    catch_all: str | None
    untimed: int
    same: bool   # both mappings group the author strings identically: no sensitivity under the other one
    # A4's value slots (``_slots``): A2 warns where the consensus it scores against is not A4's winner
    rule: dict = field(default_factory=dict)
    n_untimed_mn: int = 0
    timed_mn: bool = False
    vs: list = field(default_factory=list)
    cons: dict = field(default_factory=dict)
    prov: dict = field(default_factory=dict)
    split: list = field(default_factory=list)
    n_split_rounds: int = 0
    n_split_zero: int = 0
    # the graph statistics (``_a5``), read by the Agents section and A6
    r5: dict | None = None
    r6: dict = field(default_factory=dict)
    ws: pd.DataFrame | None = None
    profiles: pd.DataFrame | None = None


def build(run: RunDir, mapping: str = "merged", boot: int | None = None) -> str:
    """``boot``: bootstrap resamples behind the intervals (default: the config's ``bootstrap_reps``; 0 = none).  A
    strict report writes its files with a ``_strict`` suffix (``schema.suffix``), beside the merged one."""
    prof = run.profile()
    from .pipeline import load_config
    cfg = load_config(run, stage="report")
    boot = cfg.bootstrap_reps if boot is None else boot
    if isinstance(boot, bool) or not isinstance(boot, (int, np.integer)) or boot < 0:
        raise SystemExit(f"bootstrap_reps (the config setting, or report --boot) must be a whole number, 0 or more "
                         f"(0 leaves the intervals out); got {boot!r}")
    if not prof["n_events"]:
        return _no_posts(run, prof, mapping)
    c = _context(run, mapping, cfg, boot, prof)
    L = _header(c) + _funnel(c.events, c.tags, c.claims, c.rounds, mapping, c.catch_all, c.summary)
    L += _village_block(run, prof, c.summary)
    if run.has("roster"):
        L += _roster_block(run, prof, c.summary)
    L += _lifecycle_blocks(c)
    L += _a1(c)
    _slots(c)
    for section in (_a2, _a3, _a4, _a5):
        L += section(c)
    L += _agents_block(c.profiles, mapping, c.summary)
    L += _a6(c) + _validation(c) + _limitations(c)
    L[2:2] = _headline(run, c.summary, mapping)
    return _write(run, L, c.summary, mapping)


def _no_posts(run: RunDir, prof: dict, mapping: str) -> str:
    """A source with no posts: the swarm traces reconstruction report, or the context tables (Village, roster)."""
    summary = {"profile": {k: prof[k] for k in ("adapter", "source", "n_events")},
               "funnel": _funnel_counts(*[pd.DataFrame()] * 4)}
    notes = prof.get("notes") or {}
    if notes.get("swarmtraces"):
        return _write(run, _swarmtraces_block(prof, notes["swarmtraces"], summary), summary, mapping)
    becomes = (f"The roster below becomes task families and cohorts when attached to a transcript: "
               f"`swarmprov run TRANSCRIPT --roster {prof['source']}`." if prof["adapter"] == "roster" else
               "The tables below become task families, cohorts, channel lifecycle and audiences once a message "
               "table sits next to them (or is attached with `--roster`).")
    L = ["# Context report (no posts)\n",
         f"Source: `{prof['source']}` (adapter `{prof['adapter']}`) holds no posts, so there is no provenance to "
         f"analyse. {becomes}\n"] + _warnings(prof, summary)
    L += _village_block(run, prof, summary)
    if run.has("roster"):
        L += _roster_block(run, prof, summary)
    return _write(run, L, summary, mapping)


def _context(run: RunDir, mapping: str, cfg, boot: int, prof: dict) -> _Ctx:
    ex, alt = run.read(f"exposures_{mapping}"), other_mapping(mapping)
    ex_alt = run.read(f"exposures_{alt}")
    missing = [c for c in TABLES["exposures"] if c not in ex.columns or c not in ex_alt.columns]
    if missing and len(ex.columns):   # a 0x0 table from a run with no agent-rounds has nothing to miss
        raise SystemExit(f"{run.path}: the exposure tables predate this version (no {', '.join(missing)}); "
                         f"rerun `swarmprov expose {run.path}`, then `graph` and `report`")
    events, claims, agents = run.read("events"), run.read("claims"), run.read("agents")
    agent_col = agents_mod.AGENT_COLS[mapping]
    seq = prof.get("clock") == "sequence"   # posts ordered by message position: nothing measured in hours
    summary = {"mapping": mapping, "profile": {
        k: pd.Timestamp(prof[k]).isoformat() if k.endswith("_ts") and prof.get(k) else prof.get(k)
        for k in ("adapter", "n_events", "n_authors", "first_ts", "last_ts", "clock")}
        | ({"llm": prof["llm"]} if prof.get("llm") else {})}
    # the framework pseudo-authors, when they wrote no agent's post: the task and unattributed tool output
    posts = public(events)
    pseudo = [a for a in ("task", "tool") if a in set(events["author_raw"]) - set(posts["author_raw"])]
    rounds = agent_rounds(claims, agent_col) if len(claims) else pd.DataFrame()
    roster = run.has("roster") or run.has("eras")
    catch_all = identity.catch_all_family(cfg, roster)
    return _Ctx(run=run, mapping=mapping, cfg=cfg, boot=boot, prof=prof, caps=read_capabilities(prof),
                notes=prof.get("notes") or {}, summary=summary, seq=seq, agent_col=agent_col, alt=alt,
                events=events, posts=posts, pseudo=pseudo, agents=agents, claims=claims,
                tags=run.read("tags"), mentions=public(run.read("mentions")), rounds=rounds, ex=ex, ex_alt=ex_alt,
                edges=run.read(f"edges_{mapping}"), chains=run.read(f"chains_{mapping}"), roster=roster,
                catch_all=catch_all if catch_all and (events["family"] == catch_all).any() else None,
                untimed=int(rounds["t_report"].isna().sum()) if len(rounds) else 0,
                same=identity.one_to_one(agents) is not None)


def _header(c: _Ctx) -> list[str]:
    """Title, source, the data capabilities, identity merges, ingest's warnings and the clock notes."""
    prof, notes = c.prof, c.notes
    llm = prof.get("llm")
    L = ["# Copied or computed? — provenance report\n",
         f"Source: `{prof['source']}` (adapter `{prof['adapter']}`"
         + (f"; LLM extraction with `{llm['model']}` on {llm['candidates']:,} candidate posts: {llm['calls']:,} calls, "
            f"{llm['failed']:,} failed, {llm['cached']:,} from the cache" if llm else "")
         + f"), {prof['n_events']:,} posts from "
         f"{prof['n_authors'] - len(c.pseudo):,} author strings" + (f" plus {' and '.join(f'`{a}`' for a in c.pseudo)}"
                                                                      if c.pseudo else "") + ", " + _when(prof, notes)
         + f"Agent identity mapping: **{c.mapping}** (sensitivity under the other mapping below).\n",
         "## Data capabilities\n", "| capability | available | consequence |\n|---|---|---|"]
    n_untimed = prof.get("n_untimed") or 0
    for k, v in c.caps.items():
        yes, no = CONSEQUENCES.get(k, ("", ""))
        if k == "has_wall_clock" and prof.get("first_ts") and n_untimed:   # too few timestamps for a clock, but real ones
            no = (f"only the {prof['n_events'] - n_untimed:,} of {prof['n_events']:,} posts with a timestamp are ordered "
                  "by real time")
        L.append(f"| {k} | {'yes' if v else 'no'} | {yes if v else no} |")
    L.append("")
    L.append(_identity_merges(c.agents[~c.agents["author_raw"].isin(c.pseudo)], bool(c.caps.get("has_explicit_author")))
             + "\n")
    L += _warnings(prof, c.summary)
    if notes.get("clock_grades"):
        L.append((_needs_clock("Clock quality notes") if c.seq else _clock_note(notes, c.ex, c.events)) + "\n")
    return L


def _lifecycle_blocks(c: _Ctx) -> list[str]:
    """Page deletions and recreations (``summary['deletions']``) and the request-log probes (``summary['probes']``)."""
    L = []
    if c.run.has("lifecycle"):
        lc = c.run.read("lifecycle")
        dels = lc[lc["action"] == "delete"] if len(lc) else lc
        # page-level deletions (the wiki dump records each with an event id); room deletions are covered in the Village block
        if len(dels) and "event_id" in dels and dels["event_id"].notna().any() and c.seq:
            L.append(_needs_clock("Deletions and recreations (deletion times are real times)") + "\n")
        elif len(dels) and "event_id" in dels and dels["event_id"].notna().any():
            ds = lifecycle_mod.deletion_summary(lc, c.events, c.notes.get("recreation_check"))
            c.summary["deletions"] = _jsonable({k: v for k, v in ds.items() if k != "check"}
                                               | {"check": {k: v for k, v in (ds.get("check") or {}).items()
                                                            if k != "restored_event_ids"}})
            L += lifecycle_mod.deletion_section(ds, pct)
    if c.run.has("probes"):
        ps = lifecycle_mod.probe_summary(c.run.read("probes"))
        c.summary["probes"] = _jsonable(ps)
        L += lifecycle_mod.probe_section(ps)
    return L


def _ci_text(a: dict, key: str, hours: bool = False, first: bool = False) -> str:
    """An A1 statistic's interval, " (95% CI 88–96%)" (the level named at the ``first`` one only), from a summary
    holding ``<key>_ci`` and its ``bootstrap``; "" without one."""
    lo_hi = a.get(f"{key}_ci")
    if not lo_hi:
        return ""
    level = f"{a['bootstrap']['level']:.0%} CI " if first and a.get("bootstrap") else ""
    return f" ({level}{_span(*lo_hi, hours=hours)})"


def _a1(c: _Ctx) -> list[str]:
    ex, seq, run, mapping, summary = c.ex, c.seq, c.run, c.mapping, c.summary
    if not len(ex):
        return _skipped("A1", "needs agent-rounds (an agent's answer to an item in a task family); "
                        + _a1_gap(c.claims, c.rounds, _family_hint(c.cfg, c.roster, bool(c.prof.get("llm"))),
                                  bool(c.prof.get("llm"))), summary)
    r1 = provenance.summarize(ex, wall_clock=not seq, boot=c.boot)
    provenance.figure(ex, r1, run.figure("a1_provenance.png", mapping), wall_clock=not seq)
    a = summary["A1"] = {k: v for k, v in r1.items() if not isinstance(v, pd.DataFrame)} | (
        {"n_rounds_untimed": c.untimed} if c.untimed else {})
    if c.boot:
        a["bootstrap"] = {"reps": c.boot, "cluster": provenance.CLUSTER, "level": uncertainty.LEVEL,
                          "seed": 0, "min_clusters": uncertainty.MIN_CLUSTERS}
    k = r1["n_self_originated"]
    L = [f"## {SECTIONS['A1']}\n",
         f"**{r1['n_answers']:,} answers given by {r1['n_agents']:,} agents; "
         + (f"at most {r1['n_independent_at_most']:,} independent lookups**: {r1['n_independent']:,} with nothing "
            f"carrying the answer public when the agent reported it; {k:,} more {'were' if k != 1 else 'was'} first "
            "posted by the agent itself before any other copy (another agent's copy came before the report, so D = 1, "
            f"but no source is credited with {'them' if k != 1 else 'it'}). " if k else
            f"{r1['n_independent']:,} independent lookups** (nothing carrying that answer was public when the agent "
            "reported it). ")
         + f"{pct(r1['share_exposed'])}{_ci_text(a, 'share_exposed', first=True)} of answers were already public before the report"
         + (". " if seq else
            f", {pct(r1['share_exposed_60m'])}{_ci_text(a, 'share_exposed_60m')} at least an hour before (robust to the "
            "unknown lag between a question's arrival and its report). "
            + (f"Median head start: {r1['median_gap_h']:.1f} h{_ci_text(a, 'median_gap_h', hours=True)}. "
               if r1["median_gap_h"] == r1["median_gap_h"] else ""))
         + f"In {pct(r1['share_self_prepared'])}{_ci_text(a, 'share_self_prepared')} of rounds the agent itself had posted "
           "the answer in advance.\n"]
    if _ci_text(a, "share_exposed"):
        L.append(f"Intervals are 95% percentile intervals from {c.boot:,} resamples of whole **items**: answers to one "
                 "item are not independent (one first poster supplies most of its exposures), so they are drawn "
                 "together. The intervals cover sampling noise across items only, not "
                 "identity-mapping or extraction error. The family and round rows below resample their own items "
                 f"and get no interval from fewer than {uncertainty.MIN_CLUSTERS}.\n")
    kinds = set(c.events["source_kind"].dropna())
    if kinds & {"tool_result", "task"}:
        fw = provenance.framework_split(ex, tools="tool_result" in kinds, task="task" in kinds)
        a.update(fw)
        L.append(_framework_a1(fw, r1["n_independent"], r1["n_answers"]) + "\n")
    if seq:
        L.append(_needs_clock("The ≥10 min and ≥1 h shares, the head start in hours and exposure by required head "
                              "start") + "\n")
    if c.untimed:
        n = c.untimed
        L.append(f"{n:,} agent-round{'s have' if n > 1 else ' has'} no timestamp and {'are' if n > 1 else 'is'} left "
                 f"out: nothing can be ordered against {'them' if n > 1 else 'it'}.\n")
    if "D_visible" in ex and c.caps.get("has_lifecycle") and seq:
        L.append(_needs_clock("Deletion-aware shares (D_visible)") + "\n")
    elif "D_visible" in ex and c.caps.get("has_lifecycle"):
        L.append(_visibility_note(ex) + "\n")
        a.update(_visibility_counts(ex))
    L.append(f"![provenance](figures/{run.figure('a1_provenance.png', mapping).name})\n")
    L.append(md_table(_with_ci(r1["by_family"], ("exposed", "exposed_60m")),
                      pct=("exposed", "exposed_60m", "self_prepared", "instant")))
    every = r1["by_family_all"]
    run.write(f"a1_by_family_{mapping}", every.reset_index())
    small = every[~every.index.isin(r1["by_family"].index)]
    if len(small):
        L.append(f"{len(small):,} smaller famil{'y' if len(small) == 1 else 'ies'} (fewer than "
                 f"{provenance.FAMILY_MIN_N} answers, {int(small['n'].sum()):,} answers in all) "
                 f"{'is' if len(small) == 1 else 'are'} not shown; `a1_by_family_{mapping}.parquet` lists all "
                 f"{len(every):,}.\n")
    L.append("\nBy round:\n")
    L.append(md_table(_with_ci(r1["by_episode"], ("exposed", "exposed_60m")), pct=("exposed", "exposed_60m", "instant")))
    if c.same:
        L.append(f"\n_Sensitivity_: both identity mappings group the author strings identically (each merged agent is "
                 f"one strict agent), so every number in this report is the same under the {c.alt} mapping and none "
                 "is repeated for it.\n")
    elif len(c.ex_alt):
        L.append(f"\n_Sensitivity_: under the other identity mapping, {len(c.ex_alt):,} agent-rounds, "
                 f"{pct(c.ex_alt['D'].mean())} exposed" + ("" if seq else f", {pct(c.ex_alt['D_w60m'].mean())} exposed ≥1 h")
                 + ".\n")
    if not seq:
        L += _threshold_block(run.figure("a1_threshold.png", mapping), ex, bool(c.caps.get("has_lifecycle")), summary)
    lat = ex.assign(latency_s=ex["claim_id"].map(c.rounds.set_index("claim_id")["latency_s"]))
    return L + _instant_block(provenance.instant_sweep(lat, k_cfg=c.cfg.instant_threshold_s), summary)


def _slots(c: _Ctx) -> None:
    """A4's value slots, their consensus and variant provenance, and the answer rounds on the slots whose consensus
    is not the winner that state the winner (and how many of them y_consensus scores 0)."""
    mentions, ex = c.mentions, c.ex
    c.n_untimed_mn = int(mentions["ts"].isna().sum()) if len(mentions) else 0
    c.timed_mn = len(mentions) > c.n_untimed_mn
    c.rule = errors.slot_rule(c.seq)
    c.vs = errors.value_slots(mentions, c.agent_col, **c.rule) if c.timed_mn else []
    c.cons = consensus(c.claims, mentions) if c.vs and len(c.claims) else {}
    c.prov = {s["slot"]: errors.variant_provenance(s, ex, c.cons.get((s["family"], s["item"]))) for s in c.vs} \
        if len(ex) else {}
    c.split = [s for s in c.vs if errors.consensus_disagrees(s, c.cons.get((s["family"], s["item"])))]
    key = ex["value_norm"].map(value_key) if len(ex) else None
    on_winner = [ex[(ex["family"] == s["family"]) & (ex["item"] == s["item"]) & (key == s["winner"])]
                 for s in c.split] if len(ex) else []
    c.n_split_rounds = sum(map(len, on_winner))
    c.n_split_zero = sum(int((g["y_consensus"] == 0).sum()) for g in on_winner)


def _a2(c: _Ctx) -> list[str]:
    ex, seq = c.ex, c.seq
    if len(ex) <= 20:
        return _skipped("A2", f"needs more than 20 {'timed ' if c.untimed else ''}agent-rounds; this run has {len(ex):,}"
                        + (f" ({c.untimed:,} {'have' if c.untimed > 1 else 'has'} no timestamp)" if c.untimed else ""),
                        c.summary)
    est = causal.estimate(ex, wall_clock=not seq)
    causal.figure(est, c.run.figure("a2_causal.png", c.mapping))
    c.summary["A2"] = est.to_dict("records")
    m = causal.MIN_SWITCHERS
    L = [f"## {SECTIONS['A2']}\n",
         "Two-way fixed effects on agent-rounds, `Y ~ D_cons | agent + item`, SEs clustered by agent "
         "(agents and items with a single round dropped). Item FE absorb \"same question, same answer\"; "
         "agent FE absorb ability. Treatment `D_cons` = the consensus value for the item was public before the agent's report; "
         "it is defined without reference to the agent's own answer, so a wrong answer cannot mechanically produce D=0. "
         "Identifying assumption: an agent's place in the run order is unrelated to its ability. "
         f"Only agents whose exposure status switches (`agents_with_D_variation`) identify an effect: under {m} "
         f"switchers a row is not identified, under {2 * m} it is weakly identified (its note says so, and the figure "
         f"draws it grey); from {2 * m} on it reads *identified* where its 95% CI excludes 0 and *no effect detected* "
         "where the CI covers 0, as the key findings do. "
         + ("On a message-order run only the consensus outcome under `D_cons` is estimated: the ≥1 h treatment "
            "needs a wall clock, and the latency outcome (an instant answer, as the post states it) is left out "
            "with it, although A1's *instant* column and A5's *instant only* rows, read from the same text, are "
            "shown." if seq else
            "Latency would be the discriminating outcome (a common cause explains the same answer, not a 1-second "
            "answer on a 14-second timer).")
         + "\n",
         f"![causal](figures/{c.run.figure('a2_causal.png', c.mapping).name})\n"]
    cols = [k for k in ["outcome", "treatment", "beta", "se", "p", "n", "agents", "items",
                        "agents_with_D_variation", "mean_y_D0", "mean_y_D1", "note"] if k in est]
    L.append(md_table(est[cols], index=False, floatfmt="{:.3f}"))
    if "y_winner" in set(est["outcome"]):
        won, n_over = ex[ex["y_winner"].notna()], sum(s["status"] == "overturned" for s in c.vs)
        L.append("`y_winner` is an extra row; the reading above stays on `y_consensus` and `y_instant`. It asks whether "
                 "an answer states A4's winner (the most common value among a slot's late carriers) instead of the "
                 f"consensus, on the value slots A4 calls overturned only ({n_over:,} here, "
                 f"{won['item_fe'].nunique():,} with answer rounds that state a value, {len(won):,} rounds in all); "
                 "every other round is left out, and so is a report's own correct or wrong flag, which grades the "
                 "answer against the task, not the winner.\n")
    if c.n_split_rounds:
        L.append(f"**Warning:** on {len(c.split)} disputed slot{'s' * (len(c.split) != 1)} the consensus value that "
                 "`y_consensus` scores against is not A4's winner (the later majority), and "
                 f"{_zero_clause(c.n_split_rounds, c.n_split_zero)} (see A4).\n")
    if len(c.ex_alt) > 20 and not c.same:
        est_alt = causal.estimate(c.ex_alt, wall_clock=not seq)
        if len(est_alt):
            L.append("\n_Sensitivity (other identity mapping)_:\n")
            L.append(md_table(est_alt[[k for k in ["outcome", "treatment", "beta", "se", "p", "n",
                                                   "agents_with_D_variation", "note"] if k in est_alt]],
                              index=False, floatfmt="{:.3f}"))
    if seq:
        return L + ["\n" + _needs_clock("The exploratory dose-response estimates (how long the consensus had been "
                                        "public, in hours)") + "\n"]
    return L + _dose_block(c.run.figure("a2_dose.png", c.mapping), ex, None if c.same else c.ex_alt, est, c.summary)


def _a3(c: _Ctx) -> list[str]:
    from .pipeline import mined_techniques
    cfg, seq, run, mapping, tags = c.cfg, c.seq, c.run, c.mapping, c.tags
    r3 = diffusion.analyze(tags, cfg.techniques, c.agent_col, c.catch_all) if len(tags) and cfg.techniques else {}
    if not r3:
        return _skipped("A3", (f"`auto_techniques` mined no URL technique (no domain that {suggest.MIN_AGENTS} or more agents "
                               f"posted has URLs that wrap another URL in {suggest.MIN_WRAP:.0%} of cases; `swarmprov suggest "
                               "RUN` shows the evidence), and none is configured (config `techniques`: a name and a regex each)"
                               if cfg.auto_techniques else
                               "needs configured techniques (config `techniques`: a name and a regex each; `swarmprov suggest "
                               "RUN` proposes URL ones from the posts, and `auto_techniques` mines them at extract)")
                        if not cfg.techniques else
                        f"none of the {len(cfg.techniques)} "
                        + ("mined" if {t.name for t in cfg.techniques} == {t.name for t in mined_techniques(run)} else "configured")
                        + " techniques appears in a post with a timestamp", c.summary)
    fig = None if seq else diffusion.figure(r3, run.figure("a3_diffusion.png", mapping),
                                            {t.name: t.families for t in cfg.techniques})
    tbl = pd.DataFrame([{k: v for k, v in r.items() if k not in ("km", "adoptions")} | {"technique": n}
                        for n, r in r3.items()])
    tbl = tbl[[k for k in ["technique", "first_seen", "originator", "n_posts", "at_risk", "adopters", "adoption_share",
                           "median_hours_to_adopt", "median_hours_to_adopt_lo", "median_hours_to_adopt_hi",
                           "median_hours_among_adopters"]
               if not seq or not (k == "first_seen" or "hours" in k)]]
    # the contagion test compares pages, so it needs more than one
    con = diffusion.contagion(tags, c.events, cfg.techniques, c.agent_col) if c.posts["channel"].nunique() > 1 else {}
    mined = {t.name for t in mined_techniques(run)} & set(r3)   # found by auto_techniques, not configured
    c.summary["A3"] = [r | {"at_risk_basis": r3[r["technique"]]["at_risk_basis"]}
                       | ({"contagion": {k: v for k, v in con[r["technique"]].items() if k != "agents"}}
                            if r["technique"] in con else {}) | ({"mined": True} if r["technique"] in mined else {})
                       for r in tbl.to_dict("records")]
    ad = pd.concat([r["adoptions"] for r in r3.values()], ignore_index=True)
    if con:
        per = [x["agents"] for x in con.values() if "agents" in x]
        if per:
            ad = ad.merge(pd.concat(per, ignore_index=True), on=["technique", "agent"], how="left").astype(
                {"adopted_on_carrying_page": "boolean", "n_eligible_posts": "Int64"})
        tbl = tbl.assign(**_contagion_cells(tbl["technique"], con))
    if seq:   # positions are not hours
        ad[["hours_from_first_seen", "censored_at_h"]] = np.nan
    run.write(f"adoptions_{mapping}", ad)
    if not seq:
        tbl["median_hours_to_adopt"] = [_median_cell(*v) for v in zip(
            *(tbl[f"median_hours_to_adopt{x}"] for x in ("", "_lo", "_hi")))]
        tbl = tbl.drop(columns=["median_hours_to_adopt_lo", "median_hours_to_adopt_hi"]).rename(
            columns={"median_hours_to_adopt": "median_hours_to_adopt (95% CI)"})
    basis = {}
    for n, r in r3.items():
        basis.setdefault(r["at_risk_basis"], []).append(n)
    sets = {"all_active": "every agent active after the technique's first post, because no post has a task family (a "
                          "larger set than one task's agents, so adoption shares are lower)",
            "catch_all": f"every agent active after the technique's first post, as every post is in the catch-all family "
                         f"`{c.catch_all}` and the technique names no families of its own (a larger set than one task's "
                         "agents, which lowers adoption shares)",
            "families": "agents active in the technique's task families after its first post"}
    L = [f"## {SECTIONS['A3']}\n",
         "At-risk set: " + "; ".join(sets[b] + ("" if len(basis) == 1 else f" ({', '.join(f'`{n}`' for n in names)})")
                                     for b, names in basis.items())
         + "; adoption = first post mentioning or using it"
         + ("." if seq else "; non-adopters censored at their last post (Kaplan–Meier). The median time to "
            "adopt is given where the Kaplan–Meier curve falls to one half (non-adopters leave the risk set "
            "when censored, so this can happen with fewer than half adopting), with Greenwood's 95% interval "
            "(log-log); an upper bound reads *not reached* where the band's upper edge never falls to half.") + "\n"]
    if seq:
        L.append(_needs_clock("Times to adoption (hours, the Kaplan–Meier median) and the first-seen time") + "\n")
    if fig:
        L.append(f"![diffusion](figures/{run.figure('a3_diffusion.png', mapping).name})\n")
    if mined:
        tbl["technique"] = [f"{n} (mined)" if n in mined else n for n in tbl["technique"]]
        L.append("Rows marked *(mined)* were not configured: `auto_techniques` found them in the posts, as URL "
                 f"domains that at least {suggest.MIN_AGENTS} agents posted and whose distinct URLs wrap another URL "
                 f"in at least {suggest.MIN_WRAP:.0%} of cases (fetch proxies and readers). `swarmprov suggest RUN` "
                 "prints the evidence; read these rows as a first pass.\n")
    L.append(md_table(tbl, index=False, pct=("adoption_share", "baseline"), floatfmt="{:.1f}"))
    L.append(_contagion_note(con) if con else
             "The contagion test compares the pages agents adopted on, and every post here is on one page or channel, "
             "so it does not run.\n")
    L.append(f"`adoptions_{mapping}.parquet` lists every at-risk agent and every user of each technique: first use, "
             + ("as a message position (2000-01-01 + *i* s; the hour columns are empty, since positions are not hours)"
                if seq else "hours from the first appearance or the hour of censoring")
             + (", and for each adopter the page, whether it carried the technique, its baseline and its first earlier "
                "post on a carrying page (`t_exposed`)" if con else "") + ".\n")
    return L


def _a4_rows(slots: list[dict], refs: dict, cons: dict, prov: dict, seq: bool) -> list[dict]:
    """A4's table, one row per slot shown."""
    rows = []
    for s in slots:
        r = {"slot": s["slot"], "kind": s["kind"],
             "variants (agents)": ", ".join(f"{k} ({v})" for k, v in s["agents_per_variant"].items()),
             "agents carrying both": s["overlap_agents"], "first variant": s["first_variant"],
             "winner": s["winner"], "status": s["status"],
             "late carriers (winner/total)": f"{max(s['tail_counts'].values())}/{s['n_tail']}"
                                             + (" tie" if s["tie"] else "")}
        if not seq:
            r |= {"winner first seen": s["t_winner_first"], "winner majority from": s["t_winner_majority"],
                  "hours to overtake": s["hours_to_overtake"]}
        if refs:
            ref = refs.get((s["family"], s["item"]))
            named = None if ref is None else errors.match_reference(s["variants"], ref) or ref
            r |= {"reference (analyst input)": named, "outcome": errors.outcome(s, ref) if ref is not None else None}
        if s["kind"] == "value":
            con = cons.get((s["family"], s["item"]))
            r |= {"independent / after-public rounds": _rounds_cell(s, prov.get(s["slot"])),
                  "consensus ≠ winner": None if s["winner"] is None or con is None else
                  f"yes ({con})" if errors.consensus_disagrees(s, con) else "no",
                  "precision variants": "yes" if errors.precision_variants(s["variants"]) else "no"}
        rows.append(r)
    return rows


def _a4(c: _Ctx) -> list[str]:
    mentions, seq, cfg, vs, rule = c.mentions, c.seq, c.cfg, c.vs, c.rule
    if not c.timed_mn:
        if not len(mentions):
            return _skipped("A4", "needs item–value pairs or URLs in the posts; none was found", c.summary)
        k = mentions["event_id"].nunique()
        return _skipped("A4", f"needs posts with a timestamp; none of the {k:,} post{'s' * (k != 1)} with an item–value "
                        "pair or URL has one" + (", and the file joins several message lists, so no order is invented "
                                                 "for them" if c.notes.get("clock") == "none" else ""), c.summary)
    ss = errors.sequence_slots(c.claims, c.agent_col, **rule) if len(c.claims) else []
    ds = errors.regex_slots(c.posts, c.agents, cfg.disputes, c.agent_col, **rule) if cfg.disputes else []
    feat = errors.pick_featured(vs, ss, cfg.featured_disputes, declared=ds)
    fig = None if seq else errors.figure(feat, c.run.figure("a4_errors.png", c.mapping))
    refs = errors.reference_keys(cfg.reference_values)
    slots = feat + [s for s in ds + vs + ss if s not in feat]   # every disputed slot, the featured ones first
    status = Counter(s["status"] for s in slots)
    rows = _a4_rows(slots, refs, c.cons, c.prov, seq)
    unknown = errors.unused_references(refs, set(zip(mentions["family"], mentions["item"])),
                                       {(s["family"], s["item"]) for s in slots})
    c.run.write(f"disputes_{c.mapping}", errors.dispute_rows(ds + vs + ss))
    a = c.summary["A4"] = {
        "n_value_slots_disputed": len(vs), "n_sequence_slots_disputed": len(ss),
        "status_counts": {k: status[k] for k in errors.STATUSES},
        "rule": {"bin_hours": None if seq else errors.BIN_H, "tail_share": errors.TAIL_SHARE, "min_tail": errors.MIN_TAIL},
        "consensus_disagrees": {"slots": [s["slot"] for s in c.split], "n_value_slots": len(vs),
                                "answer_rounds_stating_winner": c.n_split_rounds,
                                "of_them_y_consensus_0": c.n_split_zero},
        "rows": [{k: _cell(v) for k, v in r.items()}
                 | {"featured": s in feat, "tail counts": s["tail_counts"], "tie": s["tie"], "first seen tie": s["first_tie"],
                    "corrected": s["status"] == "overturned"}   # deprecated alias of status
                 | ({"consensus": c.cons.get((s["family"], s["item"])),
                     "precision_variants": errors.precision_variants(s["variants"]),
                     "variant_provenance": c.prov.get(s["slot"])} if s["kind"] == "value" else {})
                 for r, s in zip(rows, slots)]}
    if refs:
        a["references_unknown"] = unknown
    if c.n_untimed_mn:
        a["n_mentions_untimed"] = c.n_untimed_mn
    tail = f"{Fraction(errors.TAIL_SHARE).limit_denominator(10)}"
    keep, never, kept = (("holds the majority of all the slot's carriers", "does not hold that majority", "holds that majority")
                         if seq else ("kept that majority", "never kept the majority", "kept the majority"))
    L = [f"## {SECTIONS['A4']}\n",
         f"{len(vs)} (family, item) slots carried two or more values, each repeated by ≥3 agents; "
         f"{len(ss)} round slots had competing *items* discussed by ≥2 agents who carried both (e.g. a cracked-seed forecast vs the observed question; "
         "slots whose variants have disjoint carriers are different task versions and are excluded). "
         f"Winner = the most common variant among the last {tail} of carriers, none if the top two tie; "
         "\"late carriers\" = the winner's count among them. "
         + ("" if seq else
            f"\"Majority from\" = start of the first {errors.BIN_H:g}-hour bin after which the winner never lost the majority of new carriers; "
            "\"hours to overtake\" runs from a later winner's first appearance to that point. ")
         + "Status: **held** = the first variant is still the winner; "
         f"**overturned** = a later variant won and {keep}, with at least {errors.MIN_TAIL} late carriers in all; "
         f"**contested** = the late carriers tie, a later winner {never}, or two variants were first seen "
         "at the same moment (first variant blank); "
         f"**contested (small n)** = a later winner {kept}, with fewer than {errors.MIN_TAIL} late carriers in all. "
         + (f"The last-{tail} rule is a convention, not a measurement.\n" if seq else
            f"The {errors.BIN_H:g}-hour bin and the last-{tail} rule are conventions, not measurements.\n")]
    if seq:
        L.append(_needs_clock(f"The {errors.BIN_H:g}-hour majority bins, the time a winner took the majority and the hours "
                              "to overtake") + "\n")
    if c.n_untimed_mn:
        L.append(f"{c.n_untimed_mn:,} of {len(mentions):,} item–value and URL mentions have no timestamp and are left out: "
                 "a carrier counts from its first timed post.\n")
    if refs:
        L.append("Reference values are **analyst input** (config `reference_values`), not measured. They never change the "
                 "status; they only fill the outcome column. A held or overturned slot is judged on the variant left standing "
                 "(the winner if overturned, else the first variant): truth held, error corrected, error persisted or truth "
                 "displaced by error. A contested slot reads *unresolved*, with whether the first variant was the reference. "
                 "Where one value is the other rounded (9.9 and 9.91), the outcome reads *precision variant* instead, and a "
                 "reference that matches no variant says so.\n")
    if ds:
        L.append("Analyst-declared disputes (config `disputes`: a context pattern plus one pattern per variant) are listed first.\n")
    if vs:
        L.append(_variant_note(vs, c.split, c.n_split_rounds, c.n_split_zero, c.prov, c.mapping))
    if fig:
        L.append(f"![errors](figures/{c.run.figure('a4_errors.png', c.mapping).name})\n")
    if slots:
        L.append(f"All {len(slots):,} disputed slot{'s are' if len(slots) != 1 else ' is'} in the table: " + ", ".join(
            f"{status[k]:,} {k}" for k in errors.STATUSES if status[k]) + ".\n")
    L.append(md_table(pd.DataFrame(rows), index=False, floatfmt="{:.1f}"))
    if unknown:
        L.append("Reference values that name no item or disputed slot in this run (a mistyped `family|item` key, or an item "
                 "nobody posted): " + ", ".join(f"`{k}`" for k in unknown) + ".\n")
    return L


def _profiles(c: _Ctx) -> pd.DataFrame:
    """The agent profiles the graph stage wrote; computed here only for a run from before it wrote them."""
    if c.run.has(f"agent_profiles_{c.mapping}"):
        return c.run.read(f"agent_profiles_{c.mapping}")
    return agents_mod.profile(c.agents, c.events, c.tags, c.ex, c.edges, c.chains, c.mapping, wall_clock=not c.seq,
                              btw=c.r5["betweenness"] if c.r5 else None)


def _a5(c: _Ctx) -> list[str]:
    edges, chains, ex, run, mapping, summary = c.edges, c.chains, c.ex, c.run, c.mapping, c.summary
    c.r6 = structure.reach(chains, boot=c.boot, mentions=c.mentions, agent_col=c.agent_col)
    # the relay-window sweep is in seconds, so no sequence clock; task values only (see window_sensitivity)
    c.ws = structure.window_sensitivity(c.mentions, c.agent_col) if c.r6.get("task values") and not c.seq else None
    r5 = c.r5 = structure.analyze(edges, chains, ex=ex if len(ex) else None, events=c.events) if len(edges) else None
    c.profiles = _profiles(c)
    threading = bool(c.caps.get("has_threading"))
    if not r5:
        return _skipped("A5", _a5_gap(c.mentions, c.agent_col, ex, c.untimed, c.notes.get("clock") == "none", threading),
                        summary)
    has_fig = structure.figure(r5, chains, run.figure("a5_structure.png", mapping))
    summary["A5"] = {k: v for k, v in r5.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
    L = [f"## {SECTIONS['A5']}\n",
         f"Graph: {r5['n_nodes']:,} agents, {r5['n_edges']:,} weighted links "
         f"({r5['n_relay_edges']:,} same-page relay hops, {r5['n_xchannel_edges']:,} cross-page hops attributed to the originator, "
         f"{r5['n_exposure_edges']:,} first-source exposures, "
         f"{r5['n_citation_edges']:,} explicit citations"
         + (_replies(r5["n_reply_edges"], graph.reply_edges(c.events, c.agents, c.agent_col)[1]) if threading else "")
         + "). "
         + (_hub_headline(r5) if r5.get("n_exposure_edges") else
            "No answer had a first public source here, so there is no exposure concentration to report. ")
         + _depth_sentence(r5, c.r6, edges, c.ws)]
    if has_fig:
        L.append(f"![structure](figures/{run.figure('a5_structure.png', mapping).name})\n")
    top = r5["top_sources"].rename("answers").to_frame()
    if r5.get("n_exposure_edges"):
        structure.supply_figure(edges, c.profiles.set_index("label"), r5, run.figure("a7_supply_network.png", mapping))
        L.append("Who supplied whom under one attribution rule, *first source*, which credits whoever first posted "
                 "the value before the report: each line joins an agent with an exposed answer (a dot, coloured by "
                 "task family) to that source (a grey disc sized by the answers it supplied). The rows below give "
                 "the other rules, and nothing records who read whom.\n")
        L.append(f"![supply network](figures/{run.figure('a7_supply_network.png', mapping).name})\n")
        L += _attribution_block(r5, summary)
        exs = ((mapping, ex),) if c.same else ((mapping, ex), (c.alt, c.ex_alt))
        lines, first = _origins_block({m: e for m, e in exs if len(e) and credited(e).any()}, mapping, summary,
                                      c.alt if c.same else None)
        L += lines
        top = top.join(origins.by_source(first, ex)[["items", "origin"]])
    L.append("Top first-sources (answers they were first public source for"
             + (", on how many items, and the origin of those answers" if "origin" in top else "") + "):\n")
    L.append(md_table(top))
    L.append(f"\nTop brokers (betweenness on relay + citation{' + reply' if r5['n_reply_edges'] else ''} graph"
             + (f", estimated from {r5['betweenness_sampled_sources']} sampled sources" if r5["betweenness_sampled_sources"] else "")
             + "):\n")
    L.append(md_table(r5["top_brokers"].rename("betweenness").to_frame(), floatfmt="{:.3f}"))
    return L


def _a6(c: _Ctx) -> list[str]:
    r6, ws, run, mapping, summary = c.r6, c.ws, c.run, c.mapping, c.summary
    if not (c.r5 and r6):
        return _skipped("A6", "needs facts (an item–value pair or a URL) to follow; none was posted" if c.r5 else
                        "needs the provenance graph (see A5)", summary)
    structure.reach_figure(c.chains, run.figure("a6_reach.png", mapping))
    a = summary["A6"] = r6 | ({"window_sensitivity": [{k: _cell(v) for k, v in r.items()} for r in ws.to_dict("records")]}
                              if ws is not None else {})
    L = [f"## {SECTIONS['A6']}\n", md_table(_with_ci(pd.DataFrame(r6).T, ("mean_repeaters",), fmt="{:.2f}"), floatfmt="{:.2f}")]
    if "url domains" in r6:
        d = r6["url domains"]
        top = structure.domain_reach(c.mentions, c.agent_col).sort_values("n_agents", ascending=False, kind="stable").head(3)
        a["url domains"] = d | {"top": [{"domain": k, "agents": int(n)} for k, n in top["n_agents"].items()]}
        L.append(f"The `url domains` row counts the distinct agents that posted any URL on each domain "
                 f"({d['facts']:,} domain{'s' * (d['facts'] != 1)}, {d['repeated']:,} posted by two or more agents; "
                 "most shared: " + ", ".join(f"`{k}` {n:,} agent{'s' * (n != 1)}" for k, n in top["n_agents"].items())
                 + "), so agents who fetched different pages of one source count together although no exact URL "
                 "repeats; it has no relay hops" + (", and its interval resamples domains" if "mean_repeaters_lo" in d
                                                    else "") + ".\n")
    if any("mean_repeaters_lo" in r for r in r6.values()):
        L.append(f"The interval on mean_repeaters is the 95% percentile interval from {c.boot:,} resamples of the "
                 "facts: sampling noise only.\n")
    L.append(f"\n![reach](figures/{run.figure('a6_reach.png', mapping).name})\n")
    if ws is not None:
        L += _window_block(ws, (r6.get("urls") or {}).get("facts", 0))
    elif c.seq:
        L.append(_needs_clock("Relay-window sweeps (a window W in minutes and hours)") + "\n")
    return L


def _validation(c: _Ctx) -> list[str]:
    """The recreation check, the integrity line (``summary['integrity']``), the gold-label scores
    (``summary['validation']``) and the synthetic scorecard (``summary['truth_validation']``)."""
    run, summary = c.run, c.summary
    L = ["## Validation\n"]
    rc = c.notes.get("recreation_check")
    if rc:
        L.append(f"**Recreation edges** (source vs pipeline): the source marks {rc['dump_edges']} first recreations after a deletion "
                 f"({rc['dump_edges_with_revision']} with a stored revision); the pipeline's rule finds {rc['pipeline_edges']}, "
                 f"{rc['overlap']} identical, {rc['dump_only']} of the {rc['dump_edges_with_revision']} with a revision missed "
                 f"({rc['dump_edges_without_revision']} have no revision to match), {rc['pipeline_only']} extra"
                 + (f" ({rc['pipeline_only_after_cutoff']} after the source's cutoff)" if rc.get("cutoff") else "")
                 + f". {rc['restored_posts']} of {rc['posts_on_recreations']} posts on those revisions restore pre-deletion text.\n")
    L += _integrity(checks.check_run(run, summary), summary, c.mapping)
    vfiles = sorted(run.path.glob("validation*.json"))
    if vfiles:
        summary["validation"] = {}
        for vpath in vfiles:
            v = json.loads(vpath.read_text(encoding="utf-8"))
            summary["validation"][vpath.stem] = v
            cov = v.get("coverage")   # absent from scores written before coverage was recorded
            part = f" of {cov['labelled']} (the others are not posts of this run)" \
                if cov and cov["in_run"] < cov["labelled"] else ""
            L.append(f"**{vpath.stem}** — {v['n']}{part} labelled posts ({v.get('split') or 'labels'}; labeller: "
                     f"{v.get('labeller')}); {v.get('gold_answers', '?')} gold answers, "
                     f"{v.get('predicted_answers', '?')} predicted.\n")
            L.append(md_table(pd.DataFrame(v["fields"]).T, floatfmt="{:.2f}"))
    else:
        L.append("No gold labels yet. `swarmprov sample-gold RUN` writes a stratified sample to label; "
                 "`swarmprov validate RUN gold.jsonl` scores the extractor against it, and `swarmprov run ... --gold "
                 "DIR` scores every label file in DIR before the report.\n")
    if (run.path / "truth_validation.json").exists():
        L += _truth_block(json.loads((run.path / "truth_validation.json").read_text(encoding="utf-8")), summary)
    return L


def _limitations(c: _Ctx) -> list[str]:
    seq, caps, ex = c.seq, c.caps, c.ex
    explicit = bool(caps.get("has_explicit_author"))
    if c.run.has("roster"):
        identity_note = ("- **Identity**: the roster fixes cohort (assignment batch) and family (goal) for matched authors; "
                         f"unmatched authors fall back to {'their author field' if explicit else 'parsed signatures'}.\n")
    elif explicit:
        identity_note = ("- **Identity**: authors come from the source's author field, case-folded (`Planner` and `planner` are "
                         "one agent under both mappings); the merged mapping also joins handles that share a cohort date token "
                         "and a task family (see Identity merges). Both mappings are reported.\n")
    else:
        identity_note = ("- **Identity**: names are parsed from signatures; the merged mapping assumes one agent per "
                         "(cohort date, task family). Both mappings are reported.\n")
    coverage = c.cfg.coverage_note or ("Only the posts in this transcript are searched for earlier copies of an answer, so a "
                                       "copy made anywhere else is invisible: the exposed share is a floor and the "
                                       "independent count a ceiling.")
    return ["## Limitations\n",
            "- **Exposure is inferred, not observed**: no page-view logs, so D means \"was public\", not \"was read\". "
            + ("t_report is the report's message position, an upper bound on when the question arrived; the ≥1 h variant "
               "needs a wall clock.\n" if seq else
               "t_report is an upper bound on question arrival; the ≥1 h variant guards against report lag.\n")
            + f"- **Exposure is a lower bound.** {coverage}\n"
            + identity_note +
            "- **Extraction**: rule-based on templated posts; unrestated answers (\"answered same second\") inherit the consensus value.\n"
            "- **Inferred relay edges** link each carrier to the latest earlier carrier; they are plausible paths, not proven ones."
            + ("" if seq else _relay_tie_note(c.edges)) + "\n"
            + ("- **Message order is not time.** With no wall clock, posts are ordered by their position in the file, so every "
               "\"before\" above means earlier in the file, and is only as good as the order the file was written in.\n" if seq else "")
            + (f"- **Catch-all family.** No task families are configured, so every post is in one family, `{c.catch_all}`: answers "
               "to unrelated questions that name the same item (a country, a state) count as one fact, which can overstate "
               "exposure. Set `families` to separate the tasks.\n" if c.catch_all else "")
            + ("- **Deletion ends visibility, not knowledge.** D_visible treats a copy deleted before the report as never public; "
               "an agent that read it earlier, or a copy on an uncaptured surface, is not affected, so D stays the headline.\n"
               if "D_visible" in ex and len(ex) and caps.get("has_lifecycle") and not seq else "")
            + ("- **The request log is narrow.** Only script-injection probe rows are included, with no page views, so exposure "
               "remains inferred.\n" if c.run.has("probes") else "")
            + ("- **Derived text.** The posts are an LLM's daily digest of what agents did (one line per event, Pacific time), "
               "not the agents' own messages: who-did-what-when and technique mentions are plausible, the provenance of specific "
               "values is not, and anything the digest's author chose to omit is invisible.\n" if caps.get("derived_text") else "")]


def _headline(run: RunDir, summary: dict, mapping: str) -> list[str]:
    """The key findings block for the top of the report, also written alone to ``headline.md``
    (``headline_strict.md``) and kept as ``summary['headline']``, one string per bullet."""
    block = key_findings(json_safe(summary))
    if block:
        summary["headline"] = [b[2:] for b in block if b.startswith("- ")]
        (run.path / f"headline{suffix(mapping)}.md").write_text("\n".join(block), encoding="utf-8")
    return block


def key_findings(summary: dict) -> list[str]:
    """The report's opening block, ``## Key findings``: one bullet per analysis that ran (or the line saying why it
    did not), each naming the definition it rests on.  Every number is formatted from ``summary`` (as written to
    ``summary.json``), never recomputed, so the block, ``headline.md`` and whatever quotes them cannot drift from the
    report.  [] for a report with no posts."""
    if not (summary.get("profile") or {}).get("n_events"):
        return []
    seq, skipped = summary["profile"].get("clock") == "sequence", summary.get("skipped") or {}
    make = {"A1": _kf_a1, "A2": _kf_a2, "A3": _kf_a3, "A4": _kf_a4, "A5": _kf_a5, "A6": _kf_a6}
    bullets = []
    for sec, f in make.items():
        text = f(summary[sec], summary, seq) if summary.get(sec) else None
        if text is None and sec in skipped:
            text = f"skipped: {skipped[sec]}."
        if text:
            name = SECTIONS[sec]
            bullets.append(f"**{name}**{'' if name.endswith('?') else ':'} {text}")
    return ["## Key findings\n", *(f"- {b}" for b in bullets), ""] if bullets else []


def _kf_a1(a: dict, s: dict, seq: bool) -> str:
    out = (f"{pct(a['share_exposed'])}{_ci_text(a, 'share_exposed', first=True)} of {a['n_answers']:,} answers by "
           f"{a['n_agents']:,} agents were already public before the agent reported them (another agent had posted the "
           "same item and value" + ("; with no wall clock, before means earlier in the file)" if seq else ")"))
    if not seq and a.get("share_exposed_60m") is not None:
        out += f", {pct(a['share_exposed_60m'])}{_ci_text(a, 'share_exposed_60m')} at least an hour before"
        if a.get("median_gap_h") is not None:
            out += f", with a median head start of {a['median_gap_h']:.1f} h{_ci_text(a, 'median_gap_h', hours=True)}"
    vis = (f"{a['n_independent_visible']:,} under `D_visible`, which counts a copy deleted before the report as never "
           "public") if a.get("n_independent_visible") is not None else ""
    if k := a.get("n_self_originated"):
        out += (f"; at most {a['n_independent_at_most']:,} were independent lookups ({a['n_independent']:,} with nothing "
                f"public before the report{', ' + vis if vis else ''}; {k:,} more {'were' if k != 1 else 'was'} first "
                "posted by the agent itself before any other copy)")
    else:
        out += f"; at most {a['n_independent']:,} were independent lookups" + (f" ({vis})" if vis else "")
    return out + ". These are floors: only the posts this run captured are searched, so a copy anywhere else is invisible."


def a2_identified(rows: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """A2's headline rows (the extra ``y_winner`` row aside) and the same rows by ``causal.reading``, the one rule the
    table's notes, the key findings and the report.html tile read: identified only with at least twice
    ``causal.MIN_SWITCHERS`` agents that switch exposure status and a 95% CI that excludes 0."""
    main = [r for r in rows if r["outcome"] not in causal.OPTIONAL]
    by = {k: [] for k in (causal.IDENTIFIED, causal.NO_EFFECT, causal.WEAK, causal.NOT_IDENTIFIED)}
    for r in main:
        by[causal.reading(r["agents_with_D_variation"], r.get("ci_lo"), r.get("ci_hi"))].append(r)
    return main, by


def _kf_a2(rows: list[dict], s: dict, seq: bool) -> str | None:
    main, by = a2_identified(rows)
    if not main:
        return None
    hit, null = by[causal.IDENTIFIED], by[causal.NO_EFFECT]

    def switch(rs: list[dict]) -> str:
        k = sorted({r["agents_with_D_variation"] for r in rs})
        return f"{k[0]:,}" + (f"–{k[-1]:,}" if len(k) > 1 else "")

    def either(key: str) -> str:
        names = [f"`{x}`" for x in dict.fromkeys(r[key] for r in main)]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" or {names[-1]}"

    def no_effect(r: dict) -> str:
        return (f"`{r['outcome']}` under `{r['treatment']}`: no effect detected ({r['agents_with_D_variation']:,} agents "
                f"switch, 95% CI {r['ci_lo']:+.2f} to {r['ci_hi']:+.2f})")
    fe = "two-way fixed effects (`Y ~ D | agent + item`, SEs clustered by agent)"
    if not hit:
        rest = [r for r in main if r not in null]
        if not null:
            return (f"Its effect is not identified ({switch(main)} agents switch exposure status): no estimate of "
                    f"{either('outcome')} under {either('treatment')} from {fe} rests on enough switching agents with a "
                    "95% CI that excludes 0.")
        return ((f"Not identified for {'most rows' if 2 * len(rest) > len(main) else f'{len(rest)} of the {len(main)} rows'}"
                 f" ({switch(rest)} agents switch exposure status); " if rest else "")
                + "; ".join(map(no_effect, null))
                + f". From {fe}, a row is identified with at least {2 * causal.MIN_SWITCHERS} switching agents and a 95% "
                "CI that excludes 0.")
    by_outcome = {}
    for r in hit:   # an outcome's identified treatments, in the table's order
        by_outcome.setdefault(r["outcome"], []).append(r)
    rest = [r for r in main if r["outcome"] not in by_outcome]
    flat, unseen = [r for r in rest if r in null], [r for r in rest if r not in null]
    others = ", ".join(f"`{o}`" for o in dict.fromkeys(r["outcome"] for r in unseen))

    def cell(rs: list[dict]) -> str:
        r = rs[0]
        return (f"`{r['outcome']}` under `{r['treatment']}` {r['beta']:+.2f} (95% CI {r['ci_lo']:+.2f} to "
                f"{r['ci_hi']:+.2f}, {r['agents_with_D_variation']:,} agents switch"
                + "".join(f"; also under `{x['treatment']}`" for x in rs[1:]) + ")")
    return (f"Identified, from {fe}, where enough agents switch exposure status and the 95% CI excludes 0: "
            + "; ".join(map(cell, by_outcome.values()))
            + (". " + "; ".join(map(no_effect, flat)) if flat else "")
            + (f". Not identified for {others} ({switch(unseen)} agents switch)" if unseen else "") + ".")


def _kf_a3(rows: list[dict], s: dict, seq: bool) -> str:
    top = sorted(rows, key=lambda r: -(r["adoption_share"] if r["adoption_share"] is not None else -1))
    kinds = {r["at_risk_basis"] for r in top[:5]}
    at_risk = ("active in its task families after its first post" if kinds == {"families"} else
               "active after its first post" if "families" not in kinds else
               "active in its task families after its first post, or, for one naming none, every agent active after it")
    out = (f"adoption share = the share of the agents at risk ({at_risk}) that went on to use it: " + ", ".join(
        f"`{r['technique']}`{' (mined)' if r.get('mined') else ''} {pct(r['adoption_share'])} "
        f"({r['adopters']:,} of {r['at_risk']:,})" for r in top[:5]) + ("; lower for the rest" if len(top) > 5 else ""))
    hits = [r["technique"] for r in rows if _contagion_hit(r.get("contagion"))]
    if hits:
        out += (". Consistent with contagion (adoptions landed on a page that already carried the technique more often "
                "than the adopters' own posts did, after Holm's correction; posting on a page is not reading it): "
                + ", ".join(f"`{t}`" for t in hits))
    return out + "."


def _kf_a4(a: dict, s: dict, seq: bool) -> str:
    feat = [r for r in a["rows"] if r.get("featured")]
    if not feat:
        return "no slot carried competing values."
    rule = a["rule"]
    tail = "third" if abs(rule["tail_share"] - 1 / 3) < 1e-9 else pct(rule["tail_share"])
    keep = "holds the majority of all the slot's carriers" if seq else "kept the majority of new carriers"

    def leads(r: dict, but: bool = False) -> str:
        n = sorted(r["tail counts"].values(), reverse=True)
        if len(n) > 1:
            return f"`{r['winner']}` leads the late carriers {' to '.join(map(str, n))}" + (" but" if but else "")
        return (f"all {n[0]} late carriers carry" if n[0] != 1 else "the one late carrier carries") \
            + f" `{r['winner']}`" + (", which" if but else "")

    def slot(r: dict) -> str:
        st, first, win, h = r["status"], r["first variant"], r["winner"], r.get("hours to overtake")
        if st == "held":
            out = f"`{first}` held"
        elif st == "overturned":
            out = f"`{first}` → `{win}`, overturned" + ("" if h is None else " as soon as it appeared" if h == 0 else
                                                         f" {h:.1f} h after it appeared")
        elif r["tie"]:
            out = f"contested: the late carriers tie {' to '.join(map(str, r['tail counts'].values()))}"
        elif r["first seen tie"]:
            out = f"contested: two variants were first seen at the same moment, and {leads(r)}"
        elif st == "contested":
            out = f"`{first}` first, {leads(r, but=True)} {'does not hold' if seq else 'never kept'} the majority of " \
                  f"{'all the slot' + chr(39) + 's' if seq else 'new'} carriers: contested"
        else:
            out = f"`{first}` first, {leads(r)}, fewer than {rule['min_tail']}: {st}"
        return f"{r['slot']}: {out}" + (f" (against the analyst's reference value: {r['outcome']})" if r.get("outcome") else "")
    return (f"winner = the most common variant among the last {tail} of a slot's carriers; overturned = a later winner "
            f"that {keep}. " + "; ".join(map(slot, feat)) + ".")


def _kf_a5(a: dict, s: dict, seq: bool) -> str:
    rows = {r["rule"]: r for r in a.get("attribution") or [] if r["answers"] == "all exposed" and r["n"]}
    if not rows:
        k, e = a["n_nodes"], a["n_relay_edges"]
        return (f"{k:,} agent{'s' * (k != 1)} linked by {e:,} same-page relay hop{'s' * (e != 1)}; no answer had a "
                "first public source, so there is no hub concentration to report.")
    whole, n = top10_range(a)
    parts = []
    if (f := rows.get("first source")) is not None:
        loo = a.get("top10_share_exposure_loo")
        lo, hi = (pct(loo[0]), pct(loo[1])) if loo and loo[0] is not None else (None, None)
        parts.append(f"{pct(f['top10_share'])} credited to the first agent to post the value (out-degree Gini "
                     f"{f['gini']:.2f}" + ("" if lo is None else f"; leaving out any one item leaves it at {lo}" if lo == hi
                                           else f"; leaving out any one item moves it between {lo} and {hi}") + ")")
    if (r := rows.get("latest source")) is not None:
        parts.append(f"{pct(r['top10_share'])} to the latest earlier carrier")
    if (r := rows.get("same-page relay parent")) is not None:
        parts.append(f"{pct(r['top10_share'])} of the {r['n']:,} with a same-page relay parent")
    k = a.get("n_self_originated")
    same = len({pct(r["top10_share"]) for r in rows.values()}) == 1
    return (f"the top 10 sources supplied {whole} of {n:,} exposed answers"
            + (f" ({k:,} more, first posted by the answering agent itself, {'are' if k != 1 else 'is'} credited to "
               "nobody)" if k else "")
            + (" under every attribution rule" if same else ", depending on who is credited")
            + " (nothing records who read whom, so each rule credits one source per answer): " + ", ".join(parts) + ".")


def top10_range(a5: dict) -> tuple[str, int] | None:
    """A5's top-10 share across the attribution rules that credit every exposed answer (first and latest source; the
    relay parent credits only the answers it matches), as text, and how many answers that is; None without one."""
    rows = [r for r in a5.get("attribution") or [] if r["answers"] == "all exposed" and r["n"]]
    if not rows:
        return None
    n = max(r["n"] for r in rows)
    shares = [r["top10_share"] for r in rows if r["n"] == n]
    lo, hi = min(shares), max(shares)
    return (pct(lo) if pct(lo) == pct(hi) else _span(lo, hi)), n


def _kf_a6(a: dict, s: dict, seq: bool) -> str | None:
    def fact(r: dict, what: str) -> str:
        return (f"{r['repeated']:,} of {r['facts']:,} {what} were repeated by other agents"
                + (f", travelling {r['mean_hops_repeated']:.1f} hops on average and at most {r['max_hops']:,}"
                   if r["repeated"] and r.get("mean_hops_repeated") is not None else ""))
    parts = [fact(a[k], w) for k, w in (("task values", "task values"), ("urls", "URLs")) if a.get(k)]
    return ("; ".join(parts) + " (a hop links a carrier to an inferred source: plausible paths, not proven ones).") \
        if parts else None


def _integrity(chk: pd.DataFrame, summary: dict, mapping: str) -> list[str]:
    """One line: how many of ``checks.check_run``'s checks pass, what they compared (and what this run gave them
    nothing to compare), naming those that do not pass."""
    fail, sfx = chk[chk["n_violations"] > 0], suffix(mapping)
    summary["integrity"] = {"n_checks": len(chk), "n_pass": len(chk) - len(fail),
                            "failing": fail[["check", "table", "n_violations", "severity"]].to_dict("records")}
    named = []
    for c, g in fail.groupby("check", sort=False):
        where = ", ".join(f"{n:,} in {t}" for t, n in zip(g["table"], g["n_violations"]))
        named.append(f"`{c}` ({g['severity'].iloc[0]}; {where}: {checks.CHECKS[c][1]})")
    scope, done = list(dict.fromkeys(v[2] for v in checks.CHECKS.values())), {checks.CHECKS[c][2] for c in chk["check"]}

    def series(vs: list[str]) -> str:
        vs = [v.replace("summary.json", f"summary{sfx}.json") for v in vs]
        return ", ".join(vs[:-1]) + (", and " if len(vs) > 2 else " and ") + vs[-1] if len(vs) > 1 else "".join(vs)
    ran, idle = series([v for v in scope if v in done]), series([v for v in scope if v not in done])
    return [f"**Integrity: {len(chk) - len(fail)}/{len(chk)} checks pass** (`checks{sfx}.json`; `swarmprov check RUN"
            + (f" --mapping {mapping}" if sfx else "") + "` reruns them): " + (ran or "none ran") + ". "
            + (f"Not run, as this run has nothing for them to compare: {idle}. " if idle else "")
            + ("Not passing: " + "; ".join(named) + "." if named else "None fails." if len(chk) else "") + "\n"]


def _truth_block(tv: dict, summary: dict) -> list[str]:
    """The ``truth-score`` scorecard: what this synthetic run recovered of what the simulation planted."""
    summary["truth_validation"] = tv
    pct = lambda v: "–" if v is None else f"{v:.0%}"
    d, dv, a2 = tv["D"], tv["D_visible"], tv["A2"]
    c = d["confusion"]
    rows = [("`D` per answer round", pct(d["share"]),
             f"{pct(d['true_public_at_arrival'])} on the board at arrival; {pct(d['true_earlier_copy'])} an earlier "
             "post of the same value",
             f"{pct(d['accuracy_vs_public_at_arrival'])} of rounds with the first (D=1 and public {c['D1_public']}, "
             f"D=1 and not {c['D1_not_public']}, D=0 and public {c['D0_public']}, D=0 and not {c['D0_not_public']}), "
             f"{pct(d['accuracy_vs_earlier_copy'])} with the second")]
    if dv["deleted_posts"]:
        rows.append((f"`D_visible` ({dv['deleted_posts']:,} posts deleted)", pct(dv["share"]),
                     f"{pct(dv['true_visible_copy'])} an earlier post of the value still undeleted",
                     f"{pct(dv['accuracy'])} of rounds"))
    if a2:
        # a point interval (standard error 0: every round has the same outcome) is no estimate
        fit = a2["p"] is not None and a2["ci_lo"] is not None and a2["ci_lo"] < a2["ci_hi"]
        est = f"{a2['beta']:.3f} (95% CI {a2['ci_lo']:.3f} to {a2['ci_hi']:.3f}, " \
            + ("p < 0.001)" if a2["p"] < 0.001 else f"p {a2['p']:.3f})") if fit else a2["note"] or "not estimated"
        rows.append(("A2 `y_instant ~ D_cons`", est,
                     ("–" if a2["true_effect"] is None else f"{a2['true_effect']:.3f}")
                     + " (the share of rounds public at arrival that copied)",
                     {True: "inside the interval", False: "outside the interval", None: "–"}[a2["covers"]]
                     if fit else "–"))
    for r in tv["sources"]:
        rule = r["rule"] + (" (not an A5 rule)" if r["rule"] == "first copy still visible" else "")
        rows.append((f"A5 top-10 share of instant answers, {rule}", pct(r["top10_share"]), pct(r["true_top10_share"])
                     + " of copied answers",
                     f"names the true source of {pct(r['names_true_source'])} of {r['copied_answers']:,} copied answers"
                     + (f" ({r['credits_nobody']:,} credited to nobody)" if r["credits_nobody"] else "")))
    for r in tv["A3"]:
        rows.append((f"A3 `{r['technique']}` adoption share", f"{pct(r['adoption_share'])} of {r['at_risk']}",
                     f"{pct(r['true_adoption_share'])} of {r['true_at_risk']}", ""))
        if r["median_hours_to_adopt"] is not None or r["true_median_hours_to_adopt"] is not None:
            h = lambda v: "not reached" if v is None else f"{v:.1f} h"
            rows.append((f"A3 `{r['technique']}` median time to adopt", h(r["median_hours_to_adopt"]),
                         h(r["true_median_hours_to_adopt"]), ""))
    a4 = tv["A4"]
    rows.append(("A4 disputed value slots", f"{a4['slots']}", f"{a4['true_slots']} (A4's rule on the true answers)",
                 f"in both {a4['in_both']}: status agrees on {a4['status_agrees']}, outcome on {a4['outcome_agrees']}"))
    for m, r in tv["identity"].items():
        rows.append((f"identity, {m} mapping", f"{r['agents']:,} agents", f"{r['true_agents']:,} agents",
                     f"purity {pct(r['purity'])}, {r['true_agents_split']:,} true agents split"))
    return [f"**Synthetic truth** ({tv['label']}; `truth_validation.json` from `swarmprov truth-score`, "
            f"{tv['mapping']} mapping, {tv['rounds']['matched']:,} of {tv['rounds']['truth']:,} true answer rounds "
            "matched): each analysis against what the simulation planted. It shows the pipeline recovers this "
            "simulation's provenance, not that it is right on a real swarm, whose agents need not copy, delete or "
            "sign the way the simulation's do.\n",
            md_table(pd.DataFrame(rows, columns=["measure", "this run", "truth", "agreement"]), index=False)]


def _rounds_cell(s: dict, prov: dict | None) -> str | None:
    """A value slot's "independent / after-public rounds" cell, its variants in the order of *variants (agents)*."""
    if prov is None:
        return None
    by = prov["by_variant"]
    if not any(v["answer_rounds"] for v in by.values()):
        return "no answer rounds"
    return "; ".join(f"{v}: {by[v]['derived_independently']} / {by[v]['adopted_after_public']}" for v in s["agents_per_variant"])


def _zero_clause(k: int, z: int) -> str:
    """How many of the ``k`` answer rounds that state the winner on a split slot score y_consensus = 0 (``z``): all,
    unless a report's own correct flag scores one 1."""
    if z == k:
        return f"the {k:,} answer round{'s' * (k != 1)} there that state{'s' * (k == 1)} the winner score{'s' * (k == 1)} " \
               "y_consensus = 0"
    return f"{z:,} of the {k:,} answer rounds there that state the winner score y_consensus = 0, the rest 1 by their " \
           "report's own correct flag"


def _variant_note(vs: list[dict], split: list[dict], k: int, z: int, prov: dict, mapping: str) -> str:
    """A4's header on the consensus against the later majority, and what the answer-round columns mean."""
    inherited = sum(p["n_inherited"] for p in prov.values())
    dissent = sum(v["independent_dissent"] for p in prov.values() for v in p["by_variant"].values())
    return (f"**{len(split)} of {len(vs)} slots: consensus differs from the later majority ({k:,} rounds).** The consensus "
            "is the value most distinct agents carried over the whole run, the one A2's `y_consensus` scores against; "
            + (f"on {'these value slots' if len(split) != 1 else 'this value slot'} it is not the winner, and "
               f"{_zero_clause(k, z)} (*consensus ≠ winner*). " if split else
               "on no value slot here is it a value other than the winner (*consensus ≠ winner*). ")
            + ("*Independent / after-public rounds* = per variant, the answer rounds (agents' answers to the item) that "
               "state it with nothing carrying it public before the report (D = 0) / with it already public (D = 1). "
               "They count answer rounds, not the carriers in *variants (agents)*, which count every agent that posted "
               f"the value; {inherited:,} round{'s' * (inherited != 1)} that state no value inherit the consensus in "
               "exposure and are left out. D searches captured posts only, so the D = 0 count is an upper bound on "
               f"independent derivation. {dissent:,} round{'s' * (dissent != 1)} derived a value other than the consensus "
               "with nothing carrying it public while the consensus was (D = 0, D_cons = 1): several agents re-deriving "
               "one value while another is public is the strongest sign in A4 that it was computed, not copied. "
               if prov else "")
            + "*Precision variants* = every value is another one rounded (9.69 and 9.7): one quantity at two precisions, so "
              f"the split speaks to provenance, not correctness. `disputes_{mapping}.parquet` lists every carrier of every "
              "variant: when it first carried it, whether it carried two or more, and the slot's status.\n")


def _contagion_cells(techniques: pd.Series, con: dict) -> dict:
    """A3's contagion columns, one cell per technique (``diffusion.contagion``); blank for an untested one."""
    def cells(c):
        if not c or c["excluded"] or not c["n_adopters"]:
            return "not tested" if c and c["excluded"] else None, np.nan, None
        p = c["p"]
        return (f"{c['n_on_carrying']}/{c['n_adopters']} ({c['share_on_carrying']:.0%})", c["baseline"],
                "<0.001" if p < 0.001 else f"{p:.3f}")
    on, base, p = zip(*(cells(con.get(t)) for t in techniques))
    return {"adopted on carrying page": on, "baseline": base, "p (within-agent permutation)": p}


def _contagion_hit(c: dict | None) -> bool:
    """A technique consistent with contagion (``diffusion.contagion``'s entry, as in ``summary.json``): enough
    adopters, more of them on a carrying page than the baseline, and p below ``diffusion.ALPHA`` after Holm's
    correction."""
    return bool(c) and not c["excluded"] and c["n_adopters"] >= diffusion.MIN_ADOPTERS \
        and c["share_on_carrying"] > c["baseline"] and c["p_holm"] < diffusion.ALPHA


def _contagion_note(con: dict) -> str:
    """What the contagion columns mean, which rows are consistent with contagion, and the rows not to read."""
    tested = {t: c for t, c in con.items() if not c["excluded"] and c["n_adopters"]}
    small = [f"{t} ({c['n_adopters']})" for t, c in tested.items() if c["n_adopters"] < diffusion.MIN_ADOPTERS]
    above = {t: c for t, c in tested.items() if c["share_on_carrying"] > c["baseline"]
             and c["n_adopters"] >= diffusion.MIN_ADOPTERS and c["p"] < diffusion.ALPHA}
    hits = [t for t, c in above.items() if _contagion_hit(c)]
    lost = [f"{t} (p = {c['p']:.3f}, {c['p_holm']:.3f} after it)" for t, c in above.items() if t not in hits]
    m = len(tested)
    out = ("**Contagion.** Did adopters adopt where the technique was already posted? *Adopted on carrying page* = the "
           "adopters above (the originator aside) whose first use landed on a page another agent had already posted the "
           "technique on; the adoption post never makes its own page a carrying one. *Baseline* = the share expected if "
           "each adopter's adoption had been any one of its own posts from the technique's first appearance on, in the "
           "adoption post's family, each judged against the pages carrying it at that post's time; *p* = the exact "
           "chance of at least as many adoptions on carrying pages under that within-agent shuffle, per technique. "
           "Within the family, so an agent's move onto a task's pages is not read as contagion. A share above the "
           f"baseline with p < {diffusion.ALPHA:g}"
           + (f" after Holm's correction for the {m} techniques tested" if m > 1 else "")
           + " is **consistent with contagion**" + (f" ({', '.join(hits)})" if hits else " (none here)")
           + ", not proof of it: posting on a page is not reading it, the task can send agents to the pages that carry a "
             "technique, and the regex detection adds false positives. ")
    if lost:
        out += f"Below {diffusion.ALPHA:g} only before the correction: {', '.join(lost)}. "
    for t, c in tested.items():
        if c["n_page_names"]:
            n, k = c["n_page_names"], c["n_adopted_on_named_page"]
            out += (f"{t}'s pattern also matches {n:,} page name{'s' * (n != 1)}, where a post can match because of the "
                    f"page: {f'{k:,}' if k else 'none'} of its {c['n_adopters']:,} adoptions landed on one"
                    + (f", {c['n_carrying_on_named_page']:,} of the {c['n_on_carrying']:,} on a carrying page" if k else "")
                    + ". ")
    for t, c in con.items():
        if c["excluded"]:
            out += f"{t} is not tested: {c['excluded']}. "
    if small:
        out += f"Fewer than {diffusion.MIN_ADOPTERS} adopters, too few to read: {', '.join(small)}."
    return out.rstrip() + "\n"


def _dose_block(fig: Path, ex: pd.DataFrame, ex_alt: pd.DataFrame | None, est: pd.DataFrame, summary: dict) -> list[str]:
    """A2's exploratory dose response among exposed rounds (``causal.estimate_dose``), its figure and
    ``summary['A2_dose']``; kept apart from the table above, which stays A2's reading.  ``ex_alt`` None: the other
    mapping groups the agents the same way, so it has no sensitivity line."""
    dose = causal.estimate_dose(ex)
    causal.dose_figure(dose, fig)
    lat = causal.stated_latency_slope(ex)
    alt = causal.estimate_dose(ex_alt) if ex_alt is not None and len(ex_alt) > 20 else pd.DataFrame()
    summary["A2_dose"] = {"label": "exploratory: dose response among exposed rounds", "ref": causal.DOSE_REF,
                          "min_within_sd": causal.MIN_DOSE_SD, "min_bin_agents": causal.MIN_BIN_AGENTS,
                          "min_informative": causal.MIN_INFORMATIVE, "rows": dose.to_dict("records"),
                          "stated_latency_slope": lat} | ({} if ex_alt is None else {"other_mapping": alt.to_dict("records")})
    first = next(iter(causal.DOSE_OUTCOMES))
    sw = est.set_index(["outcome", "treatment"])["agents_with_D_variation"].get((first, "D_cons"))
    per = dose.drop_duplicates("outcome").set_index("outcome")
    slopes = dose[dose["term"] == causal.SLOPE].set_index("outcome")
    inf = int(slopes.loc[first, "agents_informative"])
    bias = "".join(
        f" `{k}` is shown but {c}, so how often agents state a latency at all enters its slope"
        + (f"; here, within an agent, the share of rounds stating one changes by {lat['beta']:+.3f} per unit of "
           f"log1p(hours) (SE {lat['se']:.3f}), which pushes its slope {'down' if lat['beta'] < 0 else 'up'}." if lat else
           "; here every exposed round states one, so the two rows agree." if per.loc[k, "n"] == per.loc[first, "n"]
           else ".")
        + (f" `{first}` avoids that scoring but keeps only the rounds that state a latency, and whether a round states "
           "one comes after the head start and " + ("moves with it, as just measured, so its sample is" if lat else
                                                     "can move with it, so its sample may be")
           + " selected on the dose: it is the less biased read, not an unbiased one."
           if per.loc[k, "n"] != per.loc[first, "n"] else "") for k, c in causal.DOSE_OUTCOMES.items() if c)
    m, k2 = causal.MIN_INFORMATIVE, 2 * causal.MIN_INFORMATIVE
    L = ["**Exploratory: dose response among exposed rounds.** A narrower question than the table above, asked of the "
         "exposed rounds only (`D_cons` = 1): given the consensus value was public, does instant reuse grow with how "
         "long it had been public? The dose is the consensus value's head start over the report, as `log1p(hours)` "
         "(the slope) and in bins against a head start under 1 h, with agent + item FE and SEs clustered by agent. "
         "Within an agent the head start varies across items" + (" where exposure barely does" if sw is not None and
                                                                  inf > sw else "")
         + f": on `{first}`, {inf:,} agents vary in both the head start and the outcome (the agents that inform the "
         "slope)" + (f", against {int(sw):,} whose exposure status switches (those that identify the table above)"
                     if sw is not None else "")
         + ". It identifies only this intensive margin and says nothing about whether exposure itself matters (the "
         "extensive margin), which this block leaves unidentified; the table above is A2's estimate of that. "
         "Identifying assumption: within an agent, how long before its report others had posted the consensus value "
         "is unrelated to its item-specific propensity to answer instantly.\n",
         "Both outcomes measure the head start to the report, an upper bound on arrival, so a late report gets a "
         "longer one whichever is read." + bias + " An outcome is identified when the within-agent SD of "
         f"log1p(head start in hours) is at least {causal.MIN_DOSE_SD} (a spread of about "
         f"{np.expm1(causal.MIN_DOSE_SD):.0%} in 1 + hours) and at least {causal.MIN_BIN_AGENTS} agents have rounds in "
         f"two or more bins. Each term then needs {m} agents that inform it, else it reads *not identified*: for the "
         "slope, agents whose head start and outcome both vary; for a bin, agents with a round in it and in another bin "
         "whose outcome varies (an agent that answers all its exposed rounds the same way moves an estimate only "
         f"through the item effects). Every bin is measured against the {causal.DOSE_REF} reference, which needs them "
         f"too. An estimate informed by fewer than {k2} (a bin: in it or in the reference) rests on so few agents that "
         "its interval runs narrow and its p-value overstates the evidence: its note reads *weak* (the slope) or *shape "
         "only* (a bin), and it is grey in the figure. Read the bins as the shape and the slope as the test, with its "
         "leave-one-agent-out range below. *beta* is the change in the share answering instantly per unit of "
         "log1p(hours) (a 2.7-fold longer 1 + hours) for the slope, or against the reference for a bin; *mean y* is "
         "the raw share; *agents spanning* counts the agents whose head start varies (slope) or that have a round in "
         "the bin and in another, and *agents informative* those of them whose outcome varies too.\n",
         f"![dose response](figures/{fig.name})\n"]
    tbl = dose.rename(columns={"n_term": "rounds", "mean_y": "mean y", "agents_spanning": "agents spanning",
                               "agents_informative": "agents informative"})
    tbl["term"] = tbl["term"].replace({causal.DOSE_REF: f"{causal.DOSE_REF} (reference)"})
    L.append(md_table(tbl[["outcome", "term", "beta", "se", "p", "rounds", "mean y", "agents spanning",
                           "agents informative", "note"]], index=False, floatfmt="{:.3f}"))
    L.append("Samples (exposed rounds after dropping single-round agents and items): " + "; ".join(
        f"`{r.outcome}` {r.n:,} rounds from {r.agents:,} agents on {r.items:,} items, within-agent SD "
        f"{r.within_sd:.2f}, {r.bin_agents:,} agents in two or more bins, "
        f"{int(slopes.loc[r.outcome, 'agents_informative']):,} informing the slope"
        for r in per.reset_index().itertuples()) + ".\n")
    loo = slopes[slopes["loo_lo"].notna()]
    if len(loo):
        L.append("_Leave one agent out_: dropping one agent at a time moves the slope " + "; ".join(
            f"on `{o}` from {r.beta:+.3f} to between {r.loo_lo:+.3f} and {r.loo_hi:+.3f}, with "
            f"{r.var_share_top2:.0%} of its clustered variance from two agents"
            + (" (its SE rests mostly on them)" if r.var_share_top2 > 0.5 else "") for o, r in loo.iterrows()) + ".\n")
    if len(alt):
        sl = alt[alt["term"] == causal.SLOPE]
        L.append("_Sensitivity (other identity mapping)_: the slope is " + " and ".join(
            (f"{r.beta:+.3f} on `{r.outcome}` (SE {r.se:.3f}, p = {r.p:.2f}, {r.n:,} rounds)" if r.beta == r.beta
             else f"not identified on `{r.outcome}`") for r in sl.itertuples()) + ".\n")
    return L


def _threshold_block(path: Path, ex: pd.DataFrame, visible: bool, summary: dict) -> list[str]:
    """A1's exposure by required head start (``provenance.threshold_curve``), its figure and ``summary['A1']``."""
    curve = provenance.threshold_curve(ex, visible=visible)
    provenance.threshold_figure(ex, path, visible=visible)
    fams = [g for g in curve.index if g in set(ex["family"])]
    summary["A1"]["threshold_curve"] = {
        "label": "public at least t before the report", "hours": list(provenance.HOURS),
        "min_n": provenance.CURVE_MIN_N,
        "curves": [{"answers": g, "n": int(r["n"]), "share": [float(v) for v in r.iloc[1:]]} for g, r in curve.iterrows()]}
    hour = curve.loc["all answers"]
    return ["**Exposure vs required head start.** The share of answers whose value was public *at least t before the "
            "report*, as t grows: t = 0 is the exposed share, 1 h the ≥1 h share. The report time is an upper bound on "
            "when the question arrived, so t is a head start before the report, not before the question. "
            f"Families with {provenance.CURVE_MIN_N} or more answers get a row"
            + (f" ({len(fams)})" if fams else " (none here)")
            + (", and *still-visible copies* counts only copies not yet deleted at the report (D_visible)" if visible else "")
            + f". Where the share falls within hours, answers had been public only briefly (a race); where it holds "
              f"past a day, long before (an archive). Over every answer it falls from {pct(hour['0'])} at t = 0 to "
              f"{pct(hour['24 h'])} at 24 h.\n",
            f"![exposure by head start](figures/{path.name})\n",
            md_table(curve, pct=tuple(curve.columns[1:]))]


def _instant_block(sw: dict, summary: dict) -> list[str]:
    """The instant-answer threshold sweep, one table row (also ``summary['A1']['instant_sweep']``)."""
    summary["A1"]["instant_sweep"] = sw
    k_cfg = sw["k_configured"]
    row = pd.DataFrame([[f"{sw['n_stated_latency']:,} of {sw['n_answers']:,}",
                         *[pct(v) for v in sw["share_instant"]]]],
                       columns=["answers with a stated latency",
                                *[f"k = {k:g} s" + (" (configured)" if k == k_cfg else "") for k in sw["ks"]]])
    fixed = [f"{sw['n_instant_by_wording']:,} called instant in words with no number stay instant at every k"]
    if sw["n_delayed_by_wording"]:
        fixed.append(f"{sw['n_delayed_by_wording']:,} called delayed in words with no number stay delayed")
    differ = sw["n_class_differs_at_configured"]
    return ["**Instant-answer threshold k.** The share of answers counted *instant* if the threshold were k seconds. "
            "Only an answer that states its latency as a number moves; " + ", ".join(fixed)
            + ", and the rest state no latency."
            + (f" At the configured k = {k_cfg:g} s the sweep goes by the number and A1's *instant* column by the class, "
               f"and {differ:,} answer{'s have' if differ > 1 else ' has'} a number on the other side of k from "
               f"{'their' if differ > 1 else 'its'} class (the wording \"instantly\" sets the class whatever the number)."
               if differ else "") + "\n", md_table(row, index=False)]


def _skipped(section: str, reason: str, summary: dict) -> list[str]:
    """A section that cannot run keeps its heading and says why in one line (also ``summary['skipped']``, which the
    key findings repeat)."""
    summary.setdefault("skipped", {})[section] = reason
    return [f"## {SECTIONS[section]}\n", f"_Skipped: {reason}._\n"]


def _needs_clock(what: str) -> str:
    """The line that stands in for a result measured in real time when posts are ordered by message position."""
    return f"{what} need a wall clock and are not shown: posts here are ordered by message position only."


def _when(prof: dict, notes: dict) -> str:
    """The header's span: real times, message positions or none, and the posts no timestamp places."""
    if prof.get("clock") == "sequence":
        k, d = notes.get("timestamps_discarded"), notes.get("n_deleted")
        return ("ordered by message position (no wall clock"
                + (f"; the {k:,} of {notes['n_messages']:,} messages that carried a timestamp were too few to order by, "
                   "so their timestamps were set aside" if k else "")
                + (f"; the deletion times of {d:,} message{'s' * (d != 1)} were set aside too, as a real time cannot be "
                   "ordered against message positions, so D_visible equals D" if d else "") + "). ")
    if not prof.get("first_ts"):
        return "no timestamps. "
    n = prof.get("n_untimed") or 0
    return (f"{prof['first_ts'][:16]} → {prof['last_ts'][:16]} UTC. "
            + (f"{n:,} post{'s have' if n > 1 else ' has'} no timestamp and take{'' if n > 1 else 's'} part in no time "
               "comparison. " if n else ""))


def _family_hint(cfg, roster: bool, llm: bool = False) -> str:
    """Why answer claims have no task family, and the setting that would give them one (``default_family`` only
    where it applies: no families configured and no roster; ``--llm`` unless the run used it)."""
    alt = "" if llm else ", or use `--llm`"
    if cfg.families:
        return ("none has a task family: their posts match no configured `families` pattern"
                + (" and fall in no roster or shared goal" if roster else "") + f" (widen the patterns{alt})")
    if roster:
        return ("none has a task family: their posts fall in no roster or shared goal, and `default_family` does not "
                f"apply with a roster (set `families`{alt})")
    return f"none has a task family (set `families` or `default_family`{alt})"


def _a1_gap(claims: pd.DataFrame, rounds: pd.DataFrame, family_hint: str, llm: bool = False) -> str:
    """Why no agent-round reached exposure, from the agent_rounds filter and exposure's timestamp guard."""
    alt = "" if llm else ", or use `--llm`"
    a = claims[claims["event_type"] == "answer"] if len(claims) else claims
    if not len(a):
        return f"no answer claims were extracted (see the funnel; set `families` and `extra_items`{alt})"
    if not len(rounds):
        has_fam, has_item = a["family"].fillna("").astype(bool), a["item"].notna()
        why = (family_hint if not has_fam.any() else
               f"none names an item (add `extra_items`{alt})" if not has_item.any() else
               "none names both an item and a task family")
        return f"{len(a):,} answer claim{'s' * (len(a) != 1)} extracted, 0 agent-rounds: {why}"
    return f"none of the {len(rounds):,} agent-rounds has a timestamp, so nothing can be ordered against them"


def _framework_a1(fw: dict, n_ind: int, n: int) -> str:
    """A1 on a framework trace: the answers the agent's own tool output, or the task, carried first."""
    out = []
    if "n_own_lookup" in fw:
        look, both = fw["n_own_lookup"] + fw["n_exposed_with_own_lookup"], fw["n_exposed_with_own_lookup"]
        out.append(f"Of {n_ind:,} with no earlier public copy, {fw['n_own_lookup']:,} were preceded by the agent's own tool "
                   f"output (*own lookup preceded report*, `own_lookup_evidence`). Over all {n:,} answers: "
                   f"{fw['n_exposed_no_lookup']:,} ({pct(fw['n_exposed_no_lookup'] / n)}) had a public copy first and no "
                   f"own lookup, {look:,} ({pct(look / n)}) an own lookup first" + (f" ({both:,} of them with an earlier public "
                                                                                     "copy too)" if both else "")
                   + f", {fw['n_neither']:,} ({pct(fw['n_neither'] / n)}) neither. A lookup is evidence that the agent "
                   "fetched the value, not proof that it computed it: a tool can echo a value the agent passed in.")
    if "n_given_in_task" in fw:
        g, g0 = fw["n_given_in_task"], fw["n_given_in_task_no_public_copy"]
        out.append(f"{g:,} answer{'s' * (g != 1)} had the value given in the task (a system or user message carried "
                   "the item and value before the report)" + (f"; {g0:,} of them had no earlier public copy, so the "
                                                              "independent count above includes them" if g0 else "")
                   + ". Task messages are not agents' posts: they are no public copy and no source in A5."
                   if g else "No answer had its value given in the task: no system or user message carried the item and "
                             "value before the report.")
    return " ".join(out)


def _a5_gap(mentions: pd.DataFrame, agent_col: str, ex: pd.DataFrame, untimed: int, joined: bool,
            threading: bool = False) -> str:
    """Why the provenance graph has no edge, from what each edge kind needs: relay edges a fact two agents carried
    in timed posts, exposure edges an exposed answer, citation edges a post naming another agent, reply edges (on a
    threaded source) a reply to another agent's post."""
    n_rep = int((mentions.groupby(["family", "item", "value_key"])[agent_col].nunique() > 1).sum()) if len(mentions) else 0
    facts = ("no fact was carried by two agents" if not n_rep else
             f"{n_rep:,} fact{'s were' if n_rep > 1 else ' was'} carried by two or more agents, but not in posts with a "
             "timestamp (relay edges need an order" + ("; the file joins several message lists, so none is invented"
                                                       if joined else "") + ")")
    answers = ("no answer was exposed" if len(ex) else
               f"none of the {untimed:,} agent-rounds has a timestamp to be exposed against" if untimed else
               "there are no agent-rounds to expose")
    return (f"needs provenance edges; {facts}, {answers} and no post cites another agent"
            + (" or replies to another agent's post" if threading else ""))


def _replies(n: int, unresolved: int) -> str:
    """The A5 graph sentence's reply clause (threaded sources)."""
    return (f", and {n:,} repl{'y' if n == 1 else 'ies'}; a reply is an interaction, not a knowledge transfer, so "
            "replies count toward the brokers below but never toward supply or relays"
            + (f" ({unresolved:,} more repl{'y names' if unresolved == 1 else 'ies name'} a parent that is not an "
               f"agent's post here and {'is' if unresolved == 1 else 'are'} left out)" if unresolved else ""))


def _agents_block(p: pd.DataFrame, mapping: str, summary: dict) -> list[str]:
    """Per-agent counts from ``agents.profile``: who answered with no earlier public copy, and whether the
    suppliers were themselves exposed (also ``summary['agents']``)."""
    ans = p[p["n_answers"] > 0]
    if not len(ans):
        return _skipped("agents", "needs agent-rounds (see A1)", summary)
    k3 = agents_mod.MIN_ROUNDS
    free, src = ans[ans["n_exposed"] == 0], p[p["answers_supplied"] > 0]
    s = summary["agents"] = {
        "mapping": mapping, "n_agents": len(p), "n_answering": len(ans), "n_answers": int(ans["n_answers"].sum()),
        "n_all_no_prior_copy": len(free), f"n_all_no_prior_copy_min{k3}": int((free["n_answers"] >= k3).sum()),
        "n_all_exposed": int((ans["n_no_prior_copy"] == 0).sum()), "n_sources": len(src),
        "n_sources_answering": int((src["n_answers"] > 0).sum()),
        "n_sources_with_no_prior_copy": int((src["n_no_prior_copy"] > 0).sum())}
    first = (p[p["n_no_prior_copy"] > 0].sort_values(["n_no_prior_copy", "n_answers", "label"],
                                                     ascending=[False, False, True]).head(10))
    sup = src.sort_values(["answers_supplied", "label"], ascending=[False, True]).head(10)
    s["top_no_prior_copy"] = [{k: _cell(v) for k, v in r.items()} for r in first[
        ["label", "n_author_strings", "n_answers", "n_no_prior_copy", "n_exposed", "n_self_prepared",
         "answers_supplied"]].to_dict("records")]
    s["top_suppliers"] = [{k: _cell(v) for k, v in r.items()} for r in sup[
        ["label", "n_author_strings", "answers_supplied", "items_first_sourced", "n_answers", "n_no_prior_copy",
         "n_exposed"]].to_dict("records")]
    kn, own = f"exposed (k/n, from {k3} answers)", f"own exposed (k/n, from {k3} answers)"
    L = [f"## {SECTIONS['agents']}\n",
         f"**{len(ans):,} of the {len(p):,} agents with a post gave answers** ({s['n_answers']:,} agent-rounds; median "
         f"{ans['n_answers'].median():g} per agent, at most {ans['n_answers'].max()}). For {len(free):,} of them no answer "
         f"had an earlier public copy ({s[f'n_all_no_prior_copy_min{k3}']:,} with {k3} or more answers), and for "
         f"{s['n_all_exposed']:,} every answer had one. {len(src):,} agent{'s were' if len(src) != 1 else ' was'} the first "
         f"public source for someone's answer: {s['n_sources_answering']:,} of them gave answers of their own, and "
         f"{s['n_sources_with_no_prior_copy']:,} had at least one with no earlier public copy. Exposure is a floor (only "
         "captured posts are searched), so a count of answers with no earlier public copy is a ceiling on answers with no "
         "earlier copy anywhere. It is no bound on computed answers: an exposed answer may still have been computed, "
         "and one with no earlier public copy may copy a post this transcript lacks. One or two answers say little about "
         f"an agent, so the tables give an agent's exposed answers as k/n only from {k3} answers on. *Author strings* = "
         f"how many author strings the {mapping} label joins (`agent_aliases.csv`). "
         f"`agent_profiles_{mapping}.parquet` holds every agent's profile, and `swarmprov agent RUN NAME` prints one.\n",
         "Agents with the most answers that had no earlier public copy:\n",
         md_table(first.rename(columns={"label": "agent", "n_author_strings": "author strings", "n_answers": "answers",
                                        "n_no_prior_copy": "no earlier public copy", "exposed_k_of_n": kn,
                                        "n_self_prepared": "self-prepared", "answers_supplied": "answers supplied"})
                  [["agent", "author strings", "answers", "no earlier public copy", kn, "self-prepared",
                    "answers supplied"]], index=False)]
    if len(sup):
        L += ["\nTop suppliers (first source) and their own answers:\n",
              md_table(sup.rename(columns={"label": "agent", "n_author_strings": "author strings",
                                           "answers_supplied": "answers supplied", "items_first_sourced": "items",
                                           "n_answers": "own answers", "n_no_prior_copy": "own: no earlier public copy",
                                           "exposed_k_of_n": own})
                       [["agent", "author strings", "answers supplied", "items", "own answers",
                         "own: no earlier public copy", own]], index=False)]
    return L


def _funnel_counts(events: pd.DataFrame, tags: pd.DataFrame, claims: pd.DataFrame, rounds: pd.DataFrame,
                   catch_all: str | None = None) -> dict:
    """``summary['funnel']`` (all zeros for a run with no posts)."""
    a = claims[claims["event_type"] == "answer"] if len(claims) else pd.DataFrame(columns=["item", "value_norm", "family"])
    kinds = events["source_kind"].value_counts() if "source_kind" in events else pd.Series(dtype=int)
    private = {"task_messages": int(kinds.get("task", 0)), "tool_results": int(kinds.get("tool_result", 0))}
    return {"posts": len(events), **(private if any(private.values()) else {}),
            "posts_with_family": int(events["family"].fillna("").astype(bool).sum()) if len(events) else 0,
            "answer_posts": int(tags["is_answer"].fillna(False).astype(bool).sum()) if len(tags) else 0,
            "answer_claims": len(a), "answer_claims_no_item": int(a["item"].isna().sum()),
            "answer_claims_no_value": int(a["value_norm"].isna().sum()),
            "answer_claims_no_family": int((a["family"].fillna("") == "").sum()),
            "agent_rounds": len(rounds), "agent_rounds_untimed": int(rounds["t_report"].isna().sum()) if len(rounds) else 0,
            "catch_all_family": catch_all}


def _funnel(events, tags, claims, rounds, mapping: str, catch_all: str | None, summary: dict) -> list[str]:
    """Where posts dropped out on the way to an agent-round (also ``summary['funnel']``)."""
    f = summary["funnel"] = _funnel_counts(events, tags, claims, rounds, catch_all)
    c = {k: f"{v:,}" for k, v in f.items() if isinstance(v, int)}
    t, r = f.get("task_messages", 0), f.get("tool_results", 0)
    private = (f"posts; {t:,} task message{'s' * (t != 1)} (system or user) and {r:,} tool result{'s' * (r != 1)} among "
               "them are not agents' posts and give no claims") if "tool_results" in f else "posts"
    rows = [("posts", c["posts"], private),
            ("with a task family", c["posts_with_family"],
             f"posts; the catch-all family `{catch_all}`, because no families are configured" if catch_all else "posts"),
            ("with an answer report", c["answer_posts"], "posts saying answered, submitted, replied or responded"),
            ("answer claims", c["answer_claims"], f"claims: {c['answer_claims_no_item']} name no item, "
             f"{c['answer_claims_no_value']} no value, {c['answer_claims_no_family']} no task family"),
            ("agent-rounds", c["agent_rounds"], f"(agent, family, item) under the {mapping} mapping, from claims with an "
             "item and a family" + (f"; {c['agent_rounds_untimed']} {'have' if f['agent_rounds_untimed'] > 1 else 'has'} "
                                    "no timestamp" if f["agent_rounds_untimed"] else ""))]
    return ["## Pipeline funnel\n",
            "Where the posts went on the way to an answer. The rows count posts, claims and agent-rounds, so they need not "
            "shrink from one row to the next: a post can carry several answer claims, and an agent's repeated reports of "
            "one item are one agent-round.\n",
            md_table(pd.DataFrame(rows, columns=["stage", "count", "unit"]), index=False)]


def _window_block(ws: pd.DataFrame, n_urls: int) -> list[str]:
    """A6's sensitivity of the relay trees to the relay window W (``structure.window_sensitivity``)."""
    t = ws.assign(W=["none (the graph above)" if pd.isna(w) else provenance.hour_label(w / 3600)
                     for w in ws["W_s"]]).set_index("W").drop(columns="W_s")
    t = t.rename(columns={"relay": "same-page hops", "relay_xchannel": "cross-page hops",
                          "unattributed": "unattributed carriers", "mean_linked_repeaters": "linked repeaters per fact",
                          "mean_max_depth": "mean deepest hop (facts with ≥2 carriers)"})
    return ["**Sensitivity to the relay window W.** The relay trees link each carrier to the latest earlier carrier on "
            "its page, else the originator, at any lag. Here a parent counts only if it posted the fact at most W "
            "before the carrier: a carrier whose parent is older is left *unattributed*, a root of depth 0 that later "
            "carriers can still descend from, and every fact keeps all its carriers. *Linked repeaters* are the "
            "carriers with a parent inside W, per fact. Task values only"
            + (f": the {n_urls:,} URL trees would take seconds to re-infer for each window" if n_urls else "")
            + ". The rows show how much of the depth rests on long hops; they do not pick a W.\n",
            md_table(t, floatfmt="{:.2f}")]


def _depth_sentence(r5: dict, r6: dict, edges: pd.DataFrame, ws: pd.DataFrame | None = None) -> str:
    if r5.get("mean_chain_depth") == r5.get("mean_chain_depth"):
        hour = ws.loc[ws["W_s"] == 3600, "mean_max_depth"] if ws is not None else pd.Series(dtype=float)
        return (f"Facts carried by ≥2 agents travelled {r5['mean_chain_depth']:.1f} hops on average"
                + (f" ({hour.iloc[0]:.1f} if a parent must have posted within 1 h; A6 varies that window)"
                   if len(hour) and hour.iloc[0] == hour.iloc[0] else "") + ".\n")
    urls = r6.get("urls") or {}
    if not urls.get("repeated"):
        return "No task value or URL was carried by two or more agents.\n"
    url = edges[edges["family"] == "url"]
    relay, xch = int((url["kind"] == "relay").sum()), int((url["kind"] == "relay_xchannel").sum())
    return (f"No task value was carried by two or more agents, but {urls['repeated']:,} URLs were, over {relay + xch:,} hops "
            f"({relay:,} same-page relay, {xch:,} cross-page attributed to the originator; "
            f"{urls['mean_hops_repeated']:.1f} deep on average, at most {urls['max_hops']}).\n")


def _hub_headline(r5: dict) -> str:
    """First-source concentration beside the per-item null, and its range across the attribution rules."""
    a = r5["attribution"]
    share = {sub: a.loc[(a["answers"] == sub) & (a["n"] > 0), "top10_share"] for sub in ("all exposed", "instant only")}
    inst, n_items, one = share["instant only"], r5["items_exposed"], r5["items_one_source"]

    def span(s: pd.Series) -> str:
        lo, hi = pct(s.min()), pct(s.max())
        return lo if lo == hi else f"{lo} to {hi}"

    def moves(what: str, lo: str, hi: str) -> tuple[str, str]:
        return ("leaves", f"the {what} at {lo}") if lo == hi else ("moves", f"the {what} between {lo} and {hi}")
    whole = span(share["all exposed"])
    n_src = r5["exposure_sources"]
    (s_lo, s_hi), (g_lo, g_hi) = r5["top10_share_exposure_loo"], r5["gini_exposure_outdegree_loo"]
    v_share, share_loo = moves("share", pct(s_lo), pct(s_hi))
    v_gini, gini_loo = moves("Gini", f"{g_lo:.2f}", f"{g_hi:.2f}")
    loo = f"{v_share} {share_loo} and " + (gini_loo if v_gini == v_share else f"{v_gini} {gini_loo}")
    k = r5["n_self_originated"]
    return (f"{n_src} agent{'s were' if n_src != 1 else ' was'} the first public source for someone's answer; "
            f"the top 10 supplied **{pct(r5['top10_share_exposure'])}** of "
            + (f"the {r5['n_exposure_edges']:,} exposed answers credited to a source " if k else "all exposed answers ")
            + f"(out-degree Gini {r5['gini_exposure_outdegree']:.2f}; the exposure counts of the {n_items} "
            f"item{'s' * (n_items != 1)} alone have a Gini of {r5['gini_item_exposures']:.2f}, which is what first-source "
            "credit gives if each item has one source of its own). "
            + (f"{k:,} more exposed answer{'s were' if k != 1 else ' was'} self-originated: the answering agent had "
               "posted the value before any other agent did, so no source is credited. " if k else "")
            + (f"Leaving out any one item {loo}; concentration gets no bootstrap interval, since resampling items would "
               "repeat their sources and inflate it. " if s_lo == s_lo else "")
            + ("**That share depends on who is credited**: across the "
               if " to " in whole or (len(inst) and " to " in span(inst)) else "Across the ")
            + ("three attribution rules below" if len(share["all exposed"]) == 3 else
               "two attribution rules below that credit any answer")
            + (f" it runs from {whole}" if " to " in whole else f" it is {whole} under each")
            + (f" ({span(inst)} on instant answers)" if len(inst) else "")
            + ". First-source credit measures who first posted the value on an item: on "
            + ("the only item" if n_items == 1 else f"all {n_items} items" if one == n_items else f"{one} of the {n_items} items")
            + " with exposed answers, one agent was the first source of every one. ")


def _attribution_block(r5: dict, summary: dict) -> list[str]:
    a = r5["attribution"]
    summary["A5"]["attribution"] = [{k: _cell(v) for k, v in r.items()} for r in a.to_dict("records")]
    no_instant = not a.loc[a["answers"] == "instant only", "n"].any()
    L = ["**Hub concentration by attribution rule.** Each rule credits an exposed answer to one agent. None is observed "
         "(nothing records what an agent read), so the rows are a sensitivity range, not competing estimates. "
         "*First source* = the first other agent to post the item and value before the report (the exposure edges); "
         "*latest source* = the last one before the report; *same-page relay parent* = the graph's parent of the agent's "
         "first post of the value, matched on the value key exposure used, and *unmatched* when that post arrived by a "
         "cross-page hop, the agent has no relay edge for the value, or the parent's post is not before the report. "
         "*Instant only* keeps answers reported as instant, the copy signature. "
         + (f"No rule credits the {r5['n_self_originated']:,} self-originated answer"
            f"{'s' * (r5['n_self_originated'] != 1)} (the answering agent had posted the value before any other agent "
            f"did), so {'they are' if r5['n_self_originated'] != 1 else 'it is'} in no row. "
            if r5["n_self_originated"] else "")
         + "*One item left out* = the lowest and "
         "highest value with each item's answers dropped in turn: how much a single item moves the concentration. "
         "On synthetic swarms the rule that copies the way the simulated agents do finds the true "
         "source (latest when they copy the latest post, first when they copy the first), so a rule's success there "
         "describes how the simulated agents copy, not how a real swarm does.\n"]
    if no_instant:
        L.append("No exposed answer was reported as instant, so the copy-signature subset is empty and has no concentration.\n")

    def cell(lo: float, hi: float, fmt: str) -> str:   # one value where the range collapses
        return "" if lo != lo else fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else _span(lo, hi, fmt=fmt)
    t = a.assign(**{"top-10 share, one item left out": [cell(lo, hi, "{:.0%}") for lo, hi in
                                                       zip(a["top10_share_loo_lo"], a["top10_share_loo_hi"])],
                    "Gini, one item left out": [cell(lo, hi, "{:.2f}") for lo, hi in
                                                zip(a["gini_loo_lo"], a["gini_loo_hi"])]})
    t = t.drop(columns=["top10_share_loo_lo", "top10_share_loo_hi", "gini_loo_lo", "gini_loo_hi"])
    L.append(md_table((t[t["answers"] == "all exposed"] if no_instant else t)
                      .rename(columns={"top10_share": "top-10 share", "gini": "Gini"}), index=False, pct=("top-10 share",)))
    return L


def _origins_block(exs: dict[str, pd.DataFrame], mapping: str, summary: dict,
                   same_as: str | None = None) -> tuple[list[str], pd.DataFrame]:
    """Origin of each exposed answer's source under the first- and latest-source rules and both mappings, the
    roots relayed sources lead to, and the sources with no answer round observed (for audit).  Also returns the
    first-source origins under ``mapping`` for the top-sources table.  ``same_as`` names a mapping that groups the
    authors as ``mapping`` does: the summary repeats its counts under that name, so both mappings' summaries keep
    the same keys."""
    tr = {(rule, m): origins.trace_root(e, col) for rule, col in (("first source", "src_agent"), ("latest source", "src_latest"))
          for m, e in exs.items()}
    own = {m: int(((e["D"] == 1) & (e["self_originated"] == 1)).sum()) for m, e in exs.items()}
    labels = origins.LABELS + ((origins.SELF,) if any(own.values()) else ())
    tbl = pd.DataFrame({f"{rule}, {m} ({len(t):,})": [f"{n:,} ({pct(n / len(t))})" for n in
                                                      t["origin"].value_counts().reindex(origins.LABELS, fill_value=0)]
                        + ([f"{own[m]:,}"] if len(labels) > len(origins.LABELS) else [])
                        for (rule, m), t in tr.items()}, index=pd.Index(labels, name="credited source"))
    rules = ("first source", "latest source")
    root_labels = [k for k in origins.LABELS if k != "relayed"] + [origins.POSTED_FIRST, origins.CUT]
    roots = pd.DataFrame({r: tr[(r, mapping)]["root_origin"].value_counts().reindex(root_labels, fill_value=0) for r in rules})
    roots.index.name = "root"
    relayed = {r: tr[(r, mapping)].loc[tr[(r, mapping)]["origin"] == "relayed", "depth"] for r in rules}
    first = tr[("first source", mapping)]
    audit = origins.by_source(first[first["origin"] == "no answer round observed"], exs[mapping])
    audit = audit[["answers", "items", "answer rounds (any item)"]].head(10)
    names = {m: m for m in exs} | ({same_as: mapping} if same_as else {})
    summary["A5"]["origins"] = {
        "by_mapping": {m: {r: tr[(r, src)]["origin"].value_counts().reindex(origins.LABELS, fill_value=0).to_dict()
                           for r in rules} for m, src in names.items()},
        "self_originated": {m: own[src] for m, src in names.items()},
        "mapping": mapping, "roots": roots.to_dict(), "max depth": {r: int(tr[(r, mapping)]["depth"].max()) for r in rules},
        "no answer round observed, top sources": audit.reset_index().to_dict("records")}
    L = ["**Where did copied answers come from?** Each exposed answer's credited source (under the two rules that credit "
         "every answer" + (", and both identity mappings" if len(exs) > 1 else "") + "; answers in brackets), judged by "
         "the source's own answer round on the "
         "same item: **computed** = it answered the same value with nothing matching public before its report (D = 0); "
         "**no answer round observed** = no answer round of its own on the item was extracted, which fits an agent that "
         "only published values (an answer list, a sequence scout) but also a missed answer or an identity split, so audit "
         "the sources below before calling them compilers; **dissent** = its answer round matched a different value (its "
         "own, or the consensus when it stated none); **relayed** = it answered the same value but was itself exposed"
         + ("; **self-originated** = no source is credited, as the answering agent had posted the value before any other "
            "agent did (outside the count in brackets)" if any(own.values()) else "")
         + ". Being the first public source is not proof of being read.\n",
         md_table(tbl),
         f"Roots ({mapping} mapping): a relayed source is followed through its own source under the same rule until one "
         "is not relayed" + (" (hops from a relayed answer to its root, median / max: " + ", ".join(
             f"{r} {d.median():g} / {d.max()}" for r, d in relayed.items() if len(d)) + ")" if any(map(len, relayed.values())) else "")
         + (f". *{origins.POSTED_FIRST.capitalize()}* = the agent had posted the value no later than any other agent "
            "(the walk stops there, or the chain leads back to it), so it counts as exposed only because another agent "
            "posted the value between its own post and its report" if roots.loc[origins.POSTED_FIRST].any() else "")
         + (f". *{origins.CUT.capitalize()}* = the agent was exposed before its own post of the value, but the latest "
            "carrier before its report posted after that post, so the latest rule names no earlier source for it"
            if roots.loc[origins.CUT].any() else "")
         + ". Relayed means exposed, not copied: a chain can run past an agent that computed the value while a copy was "
         "already public, so a root can sit earlier than the true one.\n", md_table(roots)]
    if len(audit):
        L.append("Sources of answers with no answer round observed (first source, top 10, for audit; *answer rounds "
                 "(any item)* = 0 fits an agent that only publishes, more suggests a missed answer or an identity split on "
                 "these items):\n")
        L.append(md_table(audit))
    return L, first


def _clock_note(notes: dict, ex: pd.DataFrame, events: pd.DataFrame) -> str:
    """Grades counted over the analysed posts (after ignored authors are dropped), so the total matches the header."""
    from .graph import CLOCK_RES_S
    pg = events["ts_grade"].value_counts().to_dict() if "ts_grade" in events else {}
    n_posts = len(events) or 1
    unc = notes.get("clock_uncertainty_s") or []
    lag = notes.get("delete_request_lag_s") or {}
    parts = [f"Clock quality: post timestamps are the save request's wall-clock second, corroborated at grade "
             + ", ".join(f"{k} for {v:,}" for k, v in sorted(pg.items(), key=lambda kv: -kv[1])) + f" of {n_posts:,} posts"
             + (f" (stated uncertainty {', '.join(f'{u:g}' for u in unc)} s)" if unc else "") + "."]
    if lag:
        n_del = sum(lag.values())
        one = lag.get("1", 0)
        parts.append(f" Deletion times are the deletion's success second ({one:,} of {n_del:,} one second after the request).")
    if len(ex) and "gap_s" in ex and ex["gap_s"].notna().any():
        gs, r = ex["gap_s"].dropna(), CLOCK_RES_S
        parts.append(f" Minimum exposure head start {gs.min():.0f} s; {int((gs <= r).sum())} of {len(gs):,} exposed answers within the "
                     f"{r:g} s summed uncertainty, {int(((gs - 600).abs() <= r).sum())} within {r:g} s of the 10 min threshold, "
                     f"{int(((gs - 3600).abs() <= r).sum())} within {r:g} s of the 1 h threshold.")
    return "".join(parts)


def _identity_merges(agents: pd.DataFrame, explicit: bool) -> str:
    """How many merged ids join more than one author string, the largest five, how many of those only join case
    variants of one handle (which the strict mapping joins too), and a warning on explicit-author sources."""
    g = agents.groupby("agent_merged")
    k = g["author_raw"].nunique()
    multi = (k[k > 1].reset_index(name="n").assign(handles=lambda d: d["agent_merged"].map(g["agent_strict"].nunique()))
             .sort_values(["n", "agent_merged"], ascending=[False, True]))
    if not len(multi):
        ids = "the only merged id is" if len(k) == 1 else f"each of the {len(k):,} merged ids is"
        return f"Identity merges: none; {ids} a single author string (`agent_aliases.csv` lists both mappings)."
    n, case = len(multi), int((multi["handles"] == 1).sum())
    other = n - case
    top = ", ".join(f"`{m}` ({c})" for m, c in zip(multi["agent_merged"].head(5), multi["n"].head(5)))
    out = (f"Identity merges: {n:,} of {len(k):,} merged ids join{'s' if n == 1 else ''} more than one author string "
           f"({int(multi['n'].sum()):,} strings in all); largest: {top}. "
           + (f"Of these, {case:,} only join{'s' if case == 1 else ''} case variants of one handle, which the strict "
              "mapping folds together too. " if case else "")
           + "`agent_aliases.csv` lists every author string with its strict and merged id.")
    if not explicit:
        return out
    return (out + " **Warning:** this source names its authors explicitly. "
            + (f"{other:,} merge{'' if other == 1 else 's'} join{'s' if other == 1 else ''} handles it keeps apart "
               "(a shared cohort date token and task family, or a roster); compare the strict-mapping sensitivity lines. "
               if other else "")
            + ("Author strings that differ only in case are one agent under both mappings, so no sensitivity line "
               "separates them." if case else "")).rstrip()


def _visibility_counts(ex: pd.DataFrame) -> dict:
    exposed = ex[ex["D"] == 1]
    gone = exposed[exposed["src_deleted_before_report"] == 1]
    return {"n_independent_visible": int((ex["D_visible"] == 0).sum()),
            "n_src_deleted_before_report": int(len(gone)),
            "n_only_deleted_copies": int((gone["D_visible"] == 0).sum())}


def _visibility_note(ex: pd.DataFrame) -> str:
    exposed = ex[ex["D"] == 1]
    gone = exposed[exposed["src_deleted_before_report"] == 1]
    if not len(gone):
        return ("Honouring deletions: no exposed answer's first public copy had been deleted before its report, "
                "so D_visible equals D.")
    other = int((gone["D_visible"] == 1).sum())
    n_ind, n_ind_vis = int((ex["D"] == 0).sum()), int((ex["D_visible"] == 0).sum())
    age = (pd.to_datetime(gone["t_report"], utc=True) - pd.to_datetime(gone["t_public"], utc=True)).dt.total_seconds().div(3600)
    since = (pd.to_datetime(gone["t_report"], utc=True) - pd.to_datetime(gone["t_src_deleted"], utc=True)).dt.total_seconds().div(3600)
    return (f"Honouring deletions: {len(gone):,} of {len(exposed):,} exposed answers were preceded by a public copy that had been "
            f"deleted before the report (posted a median {age.median():.1f} h before it, deleted a median {since.median():.1f} h before it); "
            f"for {other:,} another copy was still visible, for {len(gone) - other:,} "
            f"nothing visible carried the value, so the independent count would be {n_ind_vis:,} ({pct(n_ind_vis / len(ex))}) "
            f"under D_visible instead of {n_ind:,} ({pct(n_ind / len(ex))}).")


def _relay_tie_note(edges: pd.DataFrame) -> str:
    if not len(edges) or "within_clock_res" not in edges:
        return ""
    rel = edges[edges["kind"].isin(graph.RELAY_KINDS)]
    url = rel["family"].astype(str).str.startswith("url")
    tie = rel["within_clock_res"].fillna(False).astype(bool)
    return (f" {int((tie & ~url).sum()):,} of {int((~url).sum()):,} answer relay hops (and {int((tie & url).sum()):,} of "
            f"{int(url.sum()):,} URL hops) fall within 2 s of their source and carry no reliable direction.")
