"""Run the analyses on a RunDir and write ``report.md`` + figures + summary.json."""

from __future__ import annotations

import json
import math
from fractions import Fraction

import numpy as np
import pandas as pd

from . import graph, identity, lifecycle as lifecycle_mod, roster as roster_mod, village as village_mod
from .adapters.base import read_capabilities
from .adapters.chat import no_author_warning
from .analysis import agents as agents_mod, causal, diffusion, errors, origins, provenance, structure
from .claims import agent_rounds, public
from .schema import TABLES, RunDir


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
                      "no deletions are recorded, so D_visible equals D"),
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


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def _roster_block(run: RunDir, prof: dict, summary: dict) -> list[str]:
    ros = run.read("roster")
    ev = run.read("events") if prof["n_events"] else None
    ann = run.read("roster_events") if run.has("roster_events") else None
    s = roster_mod.summary(ros, ev, ann)
    summary["roster"] = _jsonable(s)
    return roster_mod.section(s, md_table, prof.get("roster_notes", {}).get("timestamp_issues"))


def _cell(v):
    """A summary.json cell: missing values as null, timestamps as text."""
    if v is None or v is pd.NaT or (isinstance(v, float) and math.isnan(v)):
        return None
    return v if isinstance(v, (int, float, str, bool)) else str(v)


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
    from .pipeline import _lists
    directory = run.read("directory") if run.has("directory") else None
    channels = _lists(run.read("channels")) if run.has("channels") else None
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


def _write(run: RunDir, L: list[str], summary: dict) -> str:
    text = "\n".join(L)
    (run.path / "report.md").write_text(text)
    (run.path / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[swarmprov] wrote {run.path / 'report.md'}")
    return text


def build(run: RunDir, mapping: str = "merged") -> str:
    prof = json.loads((run.path / "profile.json").read_text())
    from .pipeline import load_config
    cfg = load_config(run)
    caps = read_capabilities(prof)
    if not prof["n_events"]:
        summary = {"profile": {k: prof[k] for k in ("adapter", "source", "n_events")},
                   "funnel": _funnel_counts(*[pd.DataFrame()] * 4)}
        notes = prof.get("notes") or {}
        if notes.get("swarmtraces"):
            L = _swarmtraces_block(prof, notes["swarmtraces"], summary)
            return _write(run, L, summary)
        L = ["# Context report (no posts)\n",
             f"Source: `{prof['source']}` (adapter `{prof['adapter']}`) holds no posts, so there is no provenance to "
             "analyse. The tables below become task families, cohorts, channel lifecycle and audiences once a "
             "message table sits next to them (or is attached with `--roster`).\n"] + _warnings(prof, summary)
        L += _village_block(run, prof, summary)
        if run.has("roster"):
            L += _roster_block(run, prof, summary)
        return _write(run, L, summary)
    agent_col = "agent_merged" if mapping == "merged" else "agent_strict"
    ex = run.read(f"exposures_{mapping}")
    alt = "strict" if mapping == "merged" else "merged"
    ex_alt = run.read(f"exposures_{alt}")
    missing = [c for c in TABLES["exposures"] if c not in ex.columns or c not in ex_alt.columns]
    if missing and len(ex.columns):   # a 0x0 table from a run with no agent-rounds has nothing to miss
        raise SystemExit(f"{run.path}: the exposure tables predate this version (no {', '.join(missing)}); "
                         f"rerun `swarmprov expose {run.path}`, then `graph` and `report`")
    claims, tags, mentions = run.read("claims"), run.read("tags"), public(run.read("mentions"))
    edges, chains = run.read(f"edges_{mapping}"), run.read(f"chains_{mapping}")
    events = run.read("events")
    notes = prof.get("notes") or {}
    seq = prof.get("clock") == "sequence"   # posts ordered by message position: nothing measured in hours
    figs, summary = {}, {"mapping": mapping, "profile": {k: prof.get(k) for k in ("adapter", "n_events", "n_authors", "first_ts",
                                                                                   "last_ts", "clock")}}
    L: list[str] = []
    L.append(f"# Copied or computed? — provenance report\n")
    L.append(f"Source: `{prof['source']}` (adapter `{prof['adapter']}`), {prof['n_events']:,} posts from "
             f"{prof['n_authors']:,} author strings, " + _when(prof, notes)
             + f"Agent identity mapping: **{mapping}** (sensitivity under the other mapping below).\n")
    L.append("## Data capabilities\n")
    L.append("| capability | available | consequence |\n|---|---|---|")
    n_untimed = prof.get("n_untimed") or 0
    for k, v in caps.items():
        yes, no = CONSEQUENCES.get(k, ("", ""))
        if k == "has_wall_clock" and prof.get("first_ts") and n_untimed:   # too few timestamps for a clock, but real ones
            no = (f"only the {prof['n_events'] - n_untimed:,} of {prof['n_events']:,} posts with a timestamp are ordered "
                  "by real time")
        L.append(f"| {k} | {'yes' if v else 'no'} | {yes if v else no} |")
    L.append("")
    explicit = bool(caps.get("has_explicit_author"))
    L.append(_identity_merges(run.read("agents"), explicit) + "\n")
    L += _warnings(prof, summary)
    if notes.get("clock_grades"):
        L.append((_needs_clock("Clock quality notes") if seq else _clock_note(notes, ex, events)) + "\n")
    rounds = agent_rounds(claims, agent_col) if len(claims) else pd.DataFrame()
    roster = run.has("roster") or run.has("eras")
    catch_all = identity.catch_all_family(cfg, roster)
    catch_all = catch_all if catch_all and (events["family"] == catch_all).any() else None
    L += _funnel(events, tags, claims, rounds, mapping, catch_all, summary)
    L += _village_block(run, prof, summary)
    if run.has("roster"):
        L += _roster_block(run, prof, summary)
    if run.has("lifecycle"):
        lc = run.read("lifecycle")
        dels = lc[lc["action"] == "delete"] if len(lc) else lc
        # page-level deletions (the wiki dump records each with an event id); room deletions are covered in the Village block
        if len(dels) and "event_id" in dels and dels["event_id"].notna().any() and seq:
            L.append(_needs_clock("Deletions and recreations (deletion times are real times)") + "\n")
        elif len(dels) and "event_id" in dels and dels["event_id"].notna().any():
            ds = lifecycle_mod.deletion_summary(lc, events, notes.get("recreation_check"))
            summary["deletions"] = _jsonable({k: v for k, v in ds.items() if k != "check"}
                                             | {"check": {k: v for k, v in (ds.get("check") or {}).items() if k != "restored_event_ids"}})
            L += lifecycle_mod.deletion_section(ds, _pct)
    if run.has("probes"):
        ps = lifecycle_mod.probe_summary(run.read("probes"))
        summary["probes"] = _jsonable(ps)
        L += lifecycle_mod.probe_section(ps)

    # A1
    untimed = int(rounds["t_report"].isna().sum()) if len(rounds) else 0
    if len(ex):
        r1 = provenance.summarize(ex, wall_clock=not seq)
        figs["provenance"] = provenance.figure(ex, r1, run.figure("a1_provenance.png"), wall_clock=not seq)
        summary["A1"] = {k: v for k, v in r1.items() if not isinstance(v, pd.DataFrame)} | (
            {"n_rounds_untimed": untimed} if untimed else {})
        L.append(f"## {SECTIONS['A1']}\n")
        L.append(f"**{r1['n_answers']:,} answers given by {r1['n_agents']:,} agents; "
                 f"{r1['n_independent']:,} independent lookups** (nothing carrying that answer was public when the agent reported it). "
                 f"{_pct(r1['share_exposed'])} of answers were already public before the report"
                 + (". " if seq else
                    f", {_pct(r1['share_exposed_60m'])} at least an hour before (robust to the unknown lag between a question's arrival and its report). "
                    f"Median head start: {r1['median_gap_h']:.1f} h. ")
                 + f"In {_pct(r1['share_self_prepared'])} of rounds the agent itself had posted the answer in advance.\n")
        kinds = set(events["source_kind"].dropna())
        if kinds & {"tool_result", "task"}:
            fw = provenance.framework_split(ex, tools="tool_result" in kinds, task="task" in kinds)
            summary["A1"].update(fw)
            L.append(_framework_a1(fw, r1["n_independent"], r1["n_answers"]) + "\n")
        if seq:
            L.append(_needs_clock("The ≥10 min and ≥1 h shares and the head start in hours") + "\n")
        if untimed:
            L.append(f"{untimed:,} agent-round{'s have' if untimed > 1 else ' has'} no timestamp and "
                     f"{'are' if untimed > 1 else 'is'} left out: nothing can be ordered against "
                     f"{'them' if untimed > 1 else 'it'}.\n")
        if "D_visible" in ex and caps.get("has_lifecycle") and seq:
            L.append(_needs_clock("Deletion-aware shares (D_visible)") + "\n")
        elif "D_visible" in ex and caps.get("has_lifecycle"):
            L.append(_visibility_note(ex) + "\n")
            summary["A1"].update(_visibility_counts(ex))
        L.append(f"![provenance](figures/{run.figure('a1_provenance.png').name})\n")
        L.append(md_table(r1["by_family"], pct=("exposed", "exposed_60m", "self_prepared", "instant")))
        L.append("\nBy round:\n")
        L.append(md_table(r1["by_episode"], pct=("exposed", "exposed_60m", "instant")))
        if len(ex_alt):
            L.append(f"\n_Sensitivity_: under the other identity mapping, {len(ex_alt):,} agent-rounds, "
                     f"{_pct(ex_alt['D'].mean())} exposed" + ("" if seq else f", {_pct(ex_alt['D_w60m'].mean())} exposed ≥1 h") + ".\n")
    else:
        L += _skipped("A1", "needs agent-rounds (an agent's answer to an item in a task family); "
                      + _a1_gap(claims, rounds, _family_hint(cfg, roster)))

    # A2
    if len(ex) > 20:
        est = causal.estimate(ex, wall_clock=not seq)
        figs["causal"] = causal.figure(est, run.figure("a2_causal.png"))
        summary["A2"] = est.to_dict("records")
        L.append(f"## {SECTIONS['A2']}\n")
        L.append("Two-way fixed effects on agent-rounds, `Y ~ D_cons | agent + item`, SEs clustered by agent "
                 "(agents and items with a single round dropped). Item FE absorb \"same question, same answer\"; "
                 "agent FE absorb ability. Treatment `D_cons` = the consensus value for the item was public before the agent's report; "
                 "it is defined without reference to the agent's own answer, so a wrong answer cannot mechanically produce D=0. "
                 "Identifying assumption: an agent's place in the run order is unrelated to its ability. "
                 + ("On a message-order run only the consensus outcome under `D_cons` is estimated: the ≥1 h treatment "
                    "needs a wall clock, and the latency outcome (an instant answer, as the post states it) is left out "
                    "with it, although A1's *instant* column and A5's *instant only* rows, read from the same text, are "
                    "shown." if seq else
                    "Latency would be the discriminating outcome (a common cause explains the same answer, not a 1-second answer on a 14-second timer); "
                    "check `agents_with_D_variation` before reading it — when few agents switch exposure status the latency estimate is uninformative.")
                 + "\n")
        L.append(f"![causal](figures/{run.figure('a2_causal.png').name})\n")
        cols = [c for c in ["outcome", "treatment", "beta", "se", "p", "n", "agents", "items",
                            "agents_with_D_variation", "mean_y_D0", "mean_y_D1", "note"] if c in est]
        L.append(md_table(est[cols], index=False, floatfmt="{:.3f}"))
        if len(ex_alt) > 20:
            est_alt = causal.estimate(ex_alt, wall_clock=not seq)
            if len(est_alt):
                L.append("\n_Sensitivity (other identity mapping)_:\n")
                L.append(md_table(est_alt[[c for c in ["outcome", "treatment", "beta", "se", "p", "n",
                                                       "agents_with_D_variation", "note"] if c in est_alt]],
                                  index=False, floatfmt="{:.3f}"))
    else:
        L += _skipped("A2", f"needs more than 20 {'timed ' if untimed else ''}agent-rounds; this run has {len(ex):,}"
                      + (f" ({untimed:,} {'have' if untimed > 1 else 'has'} no timestamp)" if untimed else ""))

    # A3
    r3 = diffusion.analyze(tags, cfg.techniques, agent_col) if len(tags) and cfg.techniques else {}
    if r3:
        figs["diffusion"] = None if seq else diffusion.figure(r3, run.figure("a3_diffusion.png"),
                                                              {t.name: t.families for t in cfg.techniques})
        tbl = pd.DataFrame([{k: v for k, v in r.items() if k != "km"} | {"technique": n} for n, r in r3.items()])
        tbl = tbl[[c for c in ["technique", "first_seen", "originator", "n_posts", "at_risk", "adopters", "adoption_share",
                               "median_hours_to_adopt", "median_hours_among_adopters"]
                   if not seq or not (c == "first_seen" or "hours" in c)]]
        summary["A3"] = tbl.to_dict("records")
        L.append(f"## {SECTIONS['A3']}\n")
        L.append(("At-risk set: every agent active after the technique's first post, because no post has a task family "
                  "(a larger set than one task's agents, so adoption shares are lower)" if diffusion.familyless(tags) else
                  "At-risk set: agents active in the technique's task families after its first post"
                  + (f" (every post is in the catch-all family `{catch_all}`, so for a technique without families of its "
                     "own that is every agent active after its first post, which lowers adoption shares)" if catch_all else ""))
                 + "; adoption = first post mentioning or using it"
                 + ("." if seq else "; non-adopters censored at their last post (Kaplan–Meier).") + "\n")
        if seq:
            L.append(_needs_clock("Times to adoption (hours, the Kaplan–Meier median) and the first-seen time") + "\n")
        if figs["diffusion"]:
            L.append(f"![diffusion](figures/{run.figure('a3_diffusion.png').name})\n")
        L.append(md_table(tbl, index=False, pct=("adoption_share",), floatfmt="{:.1f}"))
    else:
        L += _skipped("A3", "needs configured techniques (config `techniques`: a name and a regex each)" if not cfg.techniques else
                      f"none of the {len(cfg.techniques)} configured techniques appears in a post with a timestamp")

    # A4
    n_untimed_mn = int(mentions["ts"].isna().sum()) if len(mentions) else 0
    if len(mentions) > n_untimed_mn:
        rule = {"bin_h": None} if seq else {}   # no hours: the whole run is one majority bin
        vs = errors.value_slots(mentions, agent_col, **rule)
        ss = errors.sequence_slots(claims, agent_col, **rule) if len(claims) else []
        ds = errors.regex_slots(events, run.read("agents"), cfg.disputes, agent_col, **rule) if cfg.disputes else []
        feat = errors.pick_featured(vs, ss, cfg.featured_disputes, declared=ds)
        figs["errors"] = None if seq else errors.figure(feat, run.figure("a4_errors.png"))
        refs = errors.reference_keys(cfg.reference_values)
        slots = feat + [s for s in vs[:12] + ss[:6] if s not in feat]
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
            rows.append(r)
        unknown, outside = errors.unused_references(refs, set(zip(mentions["family"], mentions["item"])),
                                                    *({(s["family"], s["item"]) for s in g} for g in (vs + ss + ds, slots)))
        summary["A4"] = {"n_value_slots_disputed": len(vs), "n_sequence_slots_disputed": len(ss),
                         "rule": {"bin_hours": None if seq else errors.BIN_H, "tail_share": errors.TAIL_SHARE,
                                  "min_tail": errors.MIN_TAIL},
                         "featured": [{k: _cell(v) for k, v in r.items()}
                                      | {"tail counts": s["tail_counts"], "tie": s["tie"], "first seen tie": s["first_tie"],
                                         "corrected": s["status"] == "overturned"}   # deprecated alias of status
                                      for r, s in zip(rows, feat)]}
        if refs:
            summary["A4"] |= {"references_unknown": unknown, "references_outside_table": outside}
        if n_untimed_mn:
            summary["A4"]["n_mentions_untimed"] = n_untimed_mn
        tail = f"{Fraction(errors.TAIL_SHARE).limit_denominator(10)}"
        keep, never, kept = (("holds the majority of all the slot's carriers", "does not hold that majority", "holds that majority")
                             if seq else ("kept that majority", "never kept the majority", "kept the majority"))
        L.append(f"## {SECTIONS['A4']}\n")
        L.append(f"{len(vs)} (family, item) slots carried two or more values, each repeated by ≥3 agents; "
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
                    f"The {errors.BIN_H:g}-hour bin and the last-{tail} rule are conventions, not measurements.\n"))
        if seq:
            L.append(_needs_clock(f"The {errors.BIN_H:g}-hour majority bins, the time a winner took the majority and the hours "
                                  "to overtake") + "\n")
        if n_untimed_mn:
            L.append(f"{n_untimed_mn:,} of {len(mentions):,} item–value and URL mentions have no timestamp and are left out: "
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
        if figs["errors"]:
            L.append(f"![errors](figures/{run.figure('a4_errors.png').name})\n")
        L.append(md_table(pd.DataFrame(rows), index=False, floatfmt="{:.1f}"))
        if unknown:
            L.append("Reference values that name no item or disputed slot in this run (a mistyped `family|item` key, or an item "
                     "nobody posted): " + ", ".join(f"`{k}`" for k in unknown) + ".\n")
        if outside:
            L.append("Reference values for disputed slots outside this table (no outcome shown): "
                     + ", ".join(f"`{k}`" for k in outside) + ".\n")
    elif len(mentions):
        k = mentions["event_id"].nunique()
        L += _skipped("A4", f"needs posts with a timestamp; none of the {k:,} post{'s' * (k != 1)} with an item–value pair "
                      "or URL has one" + (", and the file joins several message lists, so no order is invented for them"
                                          if notes.get("clock") == "none" else ""))
    else:
        L += _skipped("A4", "needs item–value pairs or URLs in the posts; none was found")

    # A5 / A6
    r6 = structure.reach(chains)
    agents = run.read("agents")
    r5 = structure.analyze(edges, chains, ex=ex if len(ex) else None, events=events) if len(edges) else None
    profiles = agents_mod.profile(agents, events, tags, ex, edges, chains, mapping, wall_clock=not seq,
                                  btw=r5["betweenness"] if r5 else None)
    threading = bool(caps.get("has_threading"))
    if r5:
        figs["structure"] = structure.figure(r5, chains, run.figure("a5_structure.png"))
        summary["A5"] = {k: v for k, v in r5.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
        L.append(f"## {SECTIONS['A5']}\n")
        L.append(f"Graph: {r5['n_nodes']:,} agents, {r5['n_edges']:,} weighted links "
                 f"({r5['n_relay_edges']:,} same-page relay hops, {r5['n_xchannel_edges']:,} cross-page hops attributed to the originator, "
                 f"{r5['n_exposure_edges']:,} first-source exposures, "
                 f"{r5['n_citation_edges']:,} explicit citations"
                 + (_replies(r5["n_reply_edges"], graph.reply_edges(events, agents, agent_col)[1]) if threading else "")
                 + "). "
                 + (_hub_headline(r5) if r5.get("n_exposure_edges") else
                    "No answer had a first public source here, so there is no exposure concentration to report. ")
                 + _depth_sentence(r5, r6, edges))
        if figs["structure"]:
            L.append(f"![structure](figures/{run.figure('a5_structure.png').name})\n")
        top = r5["top_sources"].rename("answers").to_frame()
        if r5.get("n_exposure_edges"):
            figs["supply"] = structure.supply_figure(edges, profiles.set_index("label"), r5,
                                                     run.figure("a7_supply_network.png"))
            L.append("Who supplied whom under one attribution rule, *first source*, which credits whoever first posted "
                     "the value before the report: each line joins an agent with an exposed answer (a dot, coloured by "
                     "task family) to that source (a grey disc sized by the answers it supplied). The rows below give "
                     "the other rules, and nothing records who read whom.\n")
            L.append(f"![supply network](figures/{run.figure('a7_supply_network.png').name})\n")
            L += _attribution_block(r5, summary)
            lines, first = _origins_block({m: e for m, e in ((mapping, ex), (alt, ex_alt)) if len(e) and e["D"].any()},
                                         mapping, summary)
            L += lines
            top = top.join(origins.by_source(first, ex)[["items", "origin"]])
        L.append("Top first-sources (answers they were first public source for"
                 + (", on how many items, and the origin of those answers" if "origin" in top else "") + "):\n")
        L.append(md_table(top))
        L.append(f"\nTop brokers (betweenness on relay + citation{' + reply' if r5['n_reply_edges'] else ''} graph"
                 + (f", estimated from {r5['betweenness_sampled_sources']} sampled sources" if r5["betweenness_sampled_sources"] else "")
                 + "):\n")
        L.append(md_table(r5["top_brokers"].rename("betweenness").to_frame(), floatfmt="{:.3f}"))
    else:
        L += _skipped("A5", _a5_gap(mentions, agent_col, ex, untimed, notes.get("clock") == "none", threading))
    L += _agents_block(profiles, mapping, summary)
    if r5 and r6:
        figs["reach"] = structure.reach_figure(chains, run.figure("a6_reach.png"))
        summary["A6"] = r6
        L.append(f"## {SECTIONS['A6']}\n")
        L.append(md_table(pd.DataFrame(r6).T, floatfmt="{:.2f}"))
        L.append(f"\n![reach](figures/{run.figure('a6_reach.png').name})\n")
    else:
        L += _skipped("A6", "needs facts (an item–value pair or a URL) to follow; none was posted" if r5 else
                      "needs the provenance graph (see A5)")

    # Validation
    L.append("## Validation\n")
    rc = notes.get("recreation_check")
    if rc:
        L.append(f"**Recreation edges** (source vs pipeline): the source marks {rc['dump_edges']} first recreations after a deletion "
                 f"({rc['dump_edges_with_revision']} with a stored revision); the pipeline's rule finds {rc['pipeline_edges']}, "
                 f"{rc['overlap']} identical, {rc['dump_only']} of the {rc['dump_edges_with_revision']} with a revision missed "
                 f"({rc['dump_edges_without_revision']} have no revision to match), {rc['pipeline_only']} extra"
                 + (f" ({rc['pipeline_only_after_cutoff']} after the source's cutoff)" if rc.get("cutoff") else "")
                 + f". {rc['restored_posts']} of {rc['posts_on_recreations']} posts on those revisions restore pre-deletion text.\n")
    vfiles = sorted(run.path.glob("validation*.json"))
    if vfiles:
        summary["validation"] = {}
        for vpath in vfiles:
            v = json.loads(vpath.read_text())
            summary["validation"][vpath.stem] = v
            L.append(f"**{vpath.stem}** — {v['n']} labelled posts ({v.get('split') or 'labels'}; labeller: {v.get('labeller')}); "
                     f"{v.get('gold_answers', '?')} gold answers, {v.get('predicted_answers', '?')} predicted.\n")
            L.append(md_table(pd.DataFrame(v["fields"]).T, floatfmt="{:.2f}"))
    else:
        L.append("No gold labels yet. `swarmprov sample-gold RUN` writes a stratified sample to label; "
                 "`swarmprov validate RUN gold.jsonl` scores the extractor against it.\n")

    L.append("## Limitations\n")
    if run.has("roster"):
        identity_note = ("- **Identity**: the roster fixes cohort (assignment batch) and family (goal) for matched authors; "
                         f"unmatched authors fall back to {'their author field' if explicit else 'parsed signatures'}.\n")
    elif explicit:
        identity_note = ("- **Identity**: authors come from the source's author field, case-folded (`Planner` and `planner` are "
                         "one agent under both mappings); the merged mapping also joins handles that share a cohort date token "
                         "and a task family (see Identity merges). Both mappings are reported.\n")
    else:
        identity_note = ("- **Identity**: names are parsed from signatures; the merged mapping assumes one agent per "
                         "(cohort date, task family). Both mappings are reported.\n")
    coverage = cfg.coverage_note or ("Only the posts in this transcript are searched for earlier copies of an answer, so a "
                                     "copy made anywhere else is invisible: the exposed share is a floor and the "
                                     "independent count a ceiling.")
    L.append("- **Exposure is inferred, not observed**: no page-view logs, so D means \"was public\", not \"was read\". "
             + ("t_report is the report's message position, an upper bound on when the question arrived; the ≥1 h variant "
                "needs a wall clock.\n" if seq else
                "t_report is an upper bound on question arrival; the ≥1 h variant guards against report lag.\n")
             + f"- **Exposure is a lower bound.** {coverage}\n"
             + identity_note +
             "- **Extraction**: rule-based on templated posts; unrestated answers (\"answered same second\") inherit the consensus value.\n"
             "- **Inferred relay edges** link each carrier to the latest earlier carrier; they are plausible paths, not proven ones."
             + ("" if seq else _relay_tie_note(edges)) + "\n"
             + ("- **Message order is not time.** With no wall clock, posts are ordered by their position in the file, so every "
                "\"before\" above means earlier in the file, and is only as good as the order the file was written in.\n" if seq else "")
             + (f"- **Catch-all family.** No task families are configured, so every post is in one family, `{catch_all}`: answers "
                "to unrelated questions that name the same item (a country, a state) count as one fact, which can overstate "
                "exposure. Set `families` to separate the tasks.\n" if catch_all else "")
             + ("- **Deletion ends visibility, not knowledge.** D_visible treats a copy deleted before the report as never public; "
                "an agent that read it earlier, or a copy on an uncaptured surface, is not affected, so D stays the headline.\n"
                if "D_visible" in ex and len(ex) and caps.get("has_lifecycle") and not seq else "")
             + ("- **The request log is narrow.** Only script-injection probe rows are included, with no page views, so exposure "
                "remains inferred.\n" if run.has("probes") else "")
             + ("- **Derived text.** The posts are an LLM's daily digest of what agents did (one line per event, Pacific time), "
                "not the agents' own messages: who-did-what-when and technique mentions are plausible, the provenance of specific "
                "values is not, and anything the digest's author chose to omit is invisible.\n" if caps.get("derived_text") else ""))
    return _write(run, L, summary)


def _skipped(section: str, reason: str) -> list[str]:
    return [f"## {SECTIONS[section]}\n", f"_Skipped: {reason}._\n"]


def _needs_clock(what: str) -> str:
    """The line that stands in for a result measured in real time when posts are ordered by message position."""
    return f"{what} need a wall clock and are not shown: posts here are ordered by message position only."


def _when(prof: dict, notes: dict) -> str:
    """The header's span: real times, message positions or none, and the posts no timestamp places."""
    if prof.get("clock") == "sequence":
        k = notes.get("timestamps_discarded")
        return ("ordered by message position (no wall clock"
                + (f"; the {k:,} of {notes['n_messages']:,} messages that carried a timestamp were too few to order by, "
                   "so their timestamps were set aside" if k else "") + "). ")
    if not prof.get("first_ts"):
        return "no timestamps. "
    n = prof.get("n_untimed") or 0
    return (f"{prof['first_ts'][:16]} → {prof['last_ts'][:16]} UTC. "
            + (f"{n:,} post{'s have' if n > 1 else ' has'} no timestamp and take{'' if n > 1 else 's'} part in no time "
               "comparison. " if n else ""))


def _family_hint(cfg, roster: bool) -> str:
    """Why answer claims have no task family, and the setting that would give them one (``default_family`` only
    where it applies: no families configured and no roster)."""
    if cfg.families:
        return ("none has a task family: their posts match no configured `families` pattern"
                + (" and fall in no roster or shared goal" if roster else "") + " (widen the patterns, or use `--llm`)")
    if roster:
        return ("none has a task family: their posts fall in no roster or shared goal, and `default_family` does not "
                "apply with a roster (set `families`, or use `--llm`)")
    return "none has a task family (set `families` or `default_family`, or use `--llm`)"


def _a1_gap(claims: pd.DataFrame, rounds: pd.DataFrame, family_hint: str) -> str:
    """Why no agent-round reached exposure, from the agent_rounds filter and exposure's timestamp guard."""
    a = claims[claims["event_type"] == "answer"] if len(claims) else claims
    if not len(a):
        return "no answer claims were extracted (see the funnel; set `families` and `extra_items`, or use `--llm`)"
    if not len(rounds):
        has_fam, has_item = a["family"].fillna("").astype(bool), a["item"].notna()
        why = (family_hint if not has_fam.any() else
               "none names an item (add `extra_items`, or use `--llm`)" if not has_item.any() else
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
                   f"{fw['n_exposed_no_lookup']:,} ({_pct(fw['n_exposed_no_lookup'] / n)}) had a public copy first and no "
                   f"own lookup, {look:,} ({_pct(look / n)}) an own lookup first" + (f" ({both:,} of them with an earlier public "
                                                                                     "copy too)" if both else "")
                   + f", {fw['n_neither']:,} ({_pct(fw['n_neither'] / n)}) neither. A lookup is evidence that the agent "
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
        return _skipped("agents", "needs agent-rounds (see A1)")
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
         "captured posts are searched), so an answer with no earlier public copy is not proof of computation and its "
         "count is a ceiling on independent answers. One or two answers say little about an agent, so the tables give "
         f"an agent's exposed answers as k/n only from {k3} answers on. *Author strings* = how many author strings the "
         f"{mapping} label joins (`agent_aliases.csv`). "
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


def _depth_sentence(r5: dict, r6: dict, edges: pd.DataFrame) -> str:
    if r5.get("mean_chain_depth") == r5.get("mean_chain_depth"):
        return f"Facts carried by ≥2 agents travelled {r5['mean_chain_depth']:.1f} hops on average.\n"
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
        lo, hi = _pct(s.min()), _pct(s.max())
        return lo if lo == hi else f"{lo} to {hi}"
    whole = span(share["all exposed"])
    n_src = r5["exposure_sources"]
    return (f"{n_src} agent{'s were' if n_src != 1 else ' was'} the first public source for someone's answer; "
            f"the top 10 supplied **{_pct(r5['top10_share_exposure'])}** of all exposed answers "
            f"(out-degree Gini {r5['gini_exposure_outdegree']:.2f}; the exposure counts of the {n_items} "
            f"item{'s' * (n_items != 1)} alone have a Gini of {r5['gini_item_exposures']:.2f}, which is what first-source "
            "credit gives if each item has one source of its own). **That share depends on who is credited**: across the "
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
         "*Instant only* keeps answers reported as instant, the copy signature. On synthetic swarms the rule that copies the way the simulated agents do finds the true "
         "source (latest when they copy the latest post, first when they copy the first), so a rule's success there "
         "describes how the simulated agents copy, not how a real swarm does.\n"]
    if no_instant:
        L.append("No exposed answer was reported as instant, so the copy-signature subset is empty and has no concentration.\n")
    L.append(md_table((a[a["answers"] == "all exposed"] if no_instant else a)
                      .rename(columns={"top10_share": "top-10 share", "gini": "Gini"}), index=False, pct=("top-10 share",)))
    return L


def _origins_block(exs: dict[str, pd.DataFrame], mapping: str, summary: dict) -> tuple[list[str], pd.DataFrame]:
    """Origin of each exposed answer's source under the first- and latest-source rules and both mappings, the
    roots relayed sources lead to, and the sources with no answer round observed (for audit).  Also returns the
    first-source origins under ``mapping`` for the top-sources table."""
    tr = {(rule, m): origins.trace_root(e, col) for rule, col in (("first source", "src_agent"), ("latest source", "src_latest"))
          for m, e in exs.items()}
    tbl = pd.DataFrame({f"{rule}, {m} ({len(t):,})": [f"{n:,} ({_pct(n / len(t))})" for n in
                                                      t["origin"].value_counts().reindex(origins.LABELS, fill_value=0)]
                        for (rule, m), t in tr.items()}, index=pd.Index(origins.LABELS, name="credited source"))
    rules = ("first source", "latest source")
    root_labels = [k for k in origins.LABELS if k != "relayed"] + [origins.POSTED_FIRST, origins.CUT]
    roots = pd.DataFrame({r: tr[(r, mapping)]["root_origin"].value_counts().reindex(root_labels, fill_value=0) for r in rules})
    roots.index.name = "root"
    relayed = {r: tr[(r, mapping)].loc[tr[(r, mapping)]["origin"] == "relayed", "depth"] for r in rules}
    first = tr[("first source", mapping)]
    audit = origins.by_source(first[first["origin"] == "no answer round observed"], exs[mapping])
    audit = audit[["answers", "items", "answer rounds (any item)"]].head(10)
    summary["A5"]["origins"] = {
        "by_mapping": {m: {r: tr[(r, m)]["origin"].value_counts().reindex(origins.LABELS, fill_value=0).to_dict()
                           for r in rules} for m in exs},
        "mapping": mapping, "roots": roots.to_dict(), "max depth": {r: int(tr[(r, mapping)]["depth"].max()) for r in rules},
        "no answer round observed, top sources": audit.reset_index().to_dict("records")}
    L = ["**Where did copied answers come from?** Each exposed answer's credited source (under the two rules that credit "
         "every answer, and both identity mappings; answers in brackets), judged by the source's own answer round on the "
         "same item: **computed** = it answered the same value with nothing matching public before its report (D = 0); "
         "**no answer round observed** = no answer round of its own on the item was extracted, which fits an agent that "
         "only published values (an answer list, a sequence scout) but also a missed answer or an identity split, so audit "
         "the sources below before calling them compilers; **dissent** = its answer round matched a different value (its "
         "own, or the consensus when it stated none); **relayed** = it answered the same value but was itself exposed. "
         "Being the first public source is not proof of being read.\n",
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
            f"nothing visible carried the value, so the independent count would be {n_ind_vis:,} ({_pct(n_ind_vis / len(ex))}) "
            f"under D_visible instead of {n_ind:,} ({_pct(n_ind / len(ex))}).")


def _relay_tie_note(edges: pd.DataFrame) -> str:
    if not len(edges) or "within_clock_res" not in edges:
        return ""
    rel = edges[edges["kind"].isin(["relay", "relay_xchannel"])]
    url = rel["family"].astype(str).str.startswith("url")
    tie = rel["within_clock_res"].fillna(False).astype(bool)
    return (f" {int((tie & ~url).sum()):,} of {int((~url).sum()):,} answer relay hops (and {int((tie & url).sum()):,} of "
            f"{int(url.sum()):,} URL hops) fall within 2 s of their source and carry no reliable direction.")
