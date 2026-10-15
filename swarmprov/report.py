"""Run the analyses on a RunDir and write ``report.md`` + figures + summary.json."""

from __future__ import annotations

import json
import math
from fractions import Fraction

import numpy as np
import pandas as pd

from . import lifecycle as lifecycle_mod, roster as roster_mod, village as village_mod
from .adapters.base import read_capabilities
from .analysis import causal, diffusion, errors, origins, provenance, structure
from .schema import RunDir


# capability -> (consequence when the source has it, consequence when it does not)
CONSEQUENCES = {
    "has_wall_clock": ("ordering by real time is possible", "no reliable wall clock, so posts cannot be ordered by real time"),
    "has_explicit_author": ("authors come from fields, not parsed signatures", "authors are parsed from signatures in the text"),
    "has_reads": ("read events are recorded; exposure is still inferred from what was public",
                  "no read events, so exposure is inferred from what was public"),
    "has_lifecycle": ("deletions are recorded, posts carry visible_until, and A1 states D_visible beside D",
                      "no deletions are recorded, so D_visible equals D"),
    "has_request_log": ("request-level rows exist; still no page views, so exposure stays inferred",
                        "no request-level rows"),
    "derived_text": ("posts are an LLM digest of what agents did, not their words: values and quotes are second-hand",
                     "posts are the agents' own text"),
    "has_threading": ("reply links are recorded but not used; edges are still inferred from timing",
                      "no reply links: edges are inferred from timing"),
    "has_episodes": ("rounds are fields", "rounds are parsed from text"),
    "has_activity": ("a presence log says when each agent was running", "no presence log; activity is inferred from posts"),
}


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
        summary = {"profile": {k: prof[k] for k in ("adapter", "source", "n_events")}}
        notes = prof.get("notes") or {}
        if notes.get("swarmtraces"):
            L = _swarmtraces_block(prof, notes["swarmtraces"], summary)
            return _write(run, L, summary)
        L = ["# Context report (no posts)\n",
             f"Source: `{prof['source']}` (adapter `{prof['adapter']}`) holds no posts, so there is no provenance to "
             "analyse. The tables below become task families, cohorts, channel lifecycle and audiences once a "
             "message table sits next to them (or is attached with `--roster`).\n"]
        L += _village_block(run, prof, summary)
        if run.has("roster"):
            L += _roster_block(run, prof, summary)
        return _write(run, L, summary)
    agent_col = "agent_merged" if mapping == "merged" else "agent_strict"
    ex = run.read(f"exposures_{mapping}")
    alt = "strict" if mapping == "merged" else "merged"
    ex_alt = run.read(f"exposures_{alt}")
    claims, tags, mentions = run.read("claims"), run.read("tags"), run.read("mentions")
    edges, chains = run.read(f"edges_{mapping}"), run.read(f"chains_{mapping}")
    figs, summary = {}, {"mapping": mapping, "profile": {k: prof[k] for k in ("adapter", "n_events", "n_authors", "first_ts", "last_ts")}}
    L: list[str] = []
    L.append(f"# Copied or computed? — provenance report\n")
    L.append(f"Source: `{prof['source']}` (adapter `{prof['adapter']}`), {prof['n_events']:,} posts from "
             f"{prof['n_authors']:,} author strings, "
             + (f"{prof['first_ts'][:16]} → {prof['last_ts'][:16]} UTC. " if prof.get("first_ts") else "no timestamps. ")
             + f"Agent identity mapping: **{mapping}** (sensitivity under the other mapping below).\n")
    L.append("## Data capabilities\n")
    L.append("| capability | available | consequence |\n|---|---|---|")
    for k, v in caps.items():
        yes, no = CONSEQUENCES.get(k, ("", ""))
        L.append(f"| {k} | {'yes' if v else 'no'} | {yes if v else no} |")
    L.append("")
    explicit = bool(caps.get("has_explicit_author"))
    L.append(_identity_merges(run.read("agents"), explicit) + "\n")

    notes = prof.get("notes") or {}
    if notes.get("clock_grades"):
        L.append(_clock_note(notes, ex, run.read("events")) + "\n")
    L += _village_block(run, prof, summary)
    if run.has("roster"):
        L += _roster_block(run, prof, summary)
    if run.has("lifecycle"):
        lc = run.read("lifecycle")
        dels = lc[lc["action"] == "delete"] if len(lc) else lc
        # page-level deletions (the wiki dump records each with an event id); room deletions are covered in the Village block
        if len(dels) and "event_id" in dels and dels["event_id"].notna().any():
            ds = lifecycle_mod.deletion_summary(lc, run.read("events"), notes.get("recreation_check"))
            summary["deletions"] = _jsonable({k: v for k, v in ds.items() if k != "check"}
                                             | {"check": {k: v for k, v in (ds.get("check") or {}).items() if k != "restored_event_ids"}})
            L += lifecycle_mod.deletion_section(ds, _pct)
    if run.has("probes"):
        ps = lifecycle_mod.probe_summary(run.read("probes"))
        summary["probes"] = _jsonable(ps)
        L += lifecycle_mod.probe_section(ps)

    # A1
    if len(ex):
        r1 = provenance.summarize(ex)
        figs["provenance"] = provenance.figure(ex, r1, run.figure("a1_provenance.png"))
        summary["A1"] = {k: v for k, v in r1.items() if not isinstance(v, pd.DataFrame)}
        L.append("## A1. Provenance split\n")
        L.append(f"**{r1['n_answers']:,} answers given by {r1['n_agents']:,} agents; "
                 f"{r1['n_independent']:,} independent lookups** (nothing carrying that answer was public when the agent reported it). "
                 f"{_pct(r1['share_exposed'])} of answers were already public before the report, "
                 f"{_pct(r1['share_exposed_60m'])} at least an hour before (robust to the unknown lag between a question's arrival and its report). "
                 f"Median head start: {r1['median_gap_h']:.1f} h. In {_pct(r1['share_self_prepared'])} of rounds the agent itself had posted the answer in advance.\n")
        if "D_visible" in ex and caps.get("has_lifecycle"):
            L.append(_visibility_note(ex) + "\n")
            summary["A1"].update(_visibility_counts(ex))
        L.append(f"![provenance](figures/{run.figure('a1_provenance.png').name})\n")
        L.append(md_table(r1["by_family"], pct=("exposed", "exposed_60m", "self_prepared", "instant")))
        L.append("\nBy round:\n")
        L.append(md_table(r1["by_episode"], pct=("exposed", "exposed_60m", "instant")))
        if len(ex_alt):
            L.append(f"\n_Sensitivity_: under the other identity mapping, {len(ex_alt):,} agent-rounds, "
                     f"{_pct(ex_alt['D'].mean())} exposed, {_pct(ex_alt['D_w60m'].mean())} exposed ≥1 h.\n")

    # A2
    if len(ex) > 20:
        est = causal.estimate(ex)
        figs["causal"] = causal.figure(est, run.figure("a2_causal.png"))
        summary["A2"] = est.to_dict("records")
        L.append("## A2. Does availability cause copying?\n")
        L.append("Two-way fixed effects on agent-rounds, `Y ~ D_cons | agent + item`, SEs clustered by agent "
                 "(agents and items with a single round dropped). Item FE absorb \"same question, same answer\"; "
                 "agent FE absorb ability. Treatment `D_cons` = the consensus value for the item was public before the agent's report; "
                 "it is defined without reference to the agent's own answer, so a wrong answer cannot mechanically produce D=0. "
                 "Identifying assumption: an agent's place in the run order is unrelated to its ability. "
                 "Latency would be the discriminating outcome (a common cause explains the same answer, not a 1-second answer on a 14-second timer); "
                 "check `agents_with_D_variation` before reading it — when few agents switch exposure status the latency estimate is uninformative.\n")
        L.append(f"![causal](figures/{run.figure('a2_causal.png').name})\n")
        cols = [c for c in ["outcome", "treatment", "beta", "se", "p", "n", "agents", "items",
                            "agents_with_D_variation", "mean_y_D0", "mean_y_D1", "note"] if c in est]
        L.append(md_table(est[cols], index=False, floatfmt="{:.3f}"))
        if len(ex_alt) > 20:
            est_alt = causal.estimate(ex_alt)
            if len(est_alt):
                L.append("\n_Sensitivity (other identity mapping)_:\n")
                L.append(md_table(est_alt[[c for c in ["outcome", "treatment", "beta", "se", "p", "n",
                                                       "agents_with_D_variation", "note"] if c in est_alt]],
                                  index=False, floatfmt="{:.3f}"))

    # A3
    if len(tags) and cfg.techniques:
        r3 = diffusion.analyze(tags, cfg.techniques, agent_col)
        if r3:
            figs["diffusion"] = diffusion.figure(r3, run.figure("a3_diffusion.png"),
                                                 {t.name: t.families for t in cfg.techniques})
            tbl = pd.DataFrame([{k: v for k, v in r.items() if k != "km"} | {"technique": n} for n, r in r3.items()])
            tbl = tbl[["technique", "first_seen", "originator", "n_posts", "at_risk", "adopters", "adoption_share",
                       "median_hours_to_adopt", "median_hours_among_adopters"]]
            summary["A3"] = tbl.to_dict("records")
            L.append("## A3. Technique diffusion\n")
            L.append("At-risk set: agents active in the technique's task families after its first post; adoption = first post "
                     "mentioning or using it; non-adopters censored at their last post (Kaplan–Meier).\n")
            if figs["diffusion"]:
                L.append(f"![diffusion](figures/{run.figure('a3_diffusion.png').name})\n")
            L.append(md_table(tbl, index=False, pct=("adoption_share",), floatfmt="{:.1f}"))

    # A4
    if len(mentions):
        vs = errors.value_slots(mentions, agent_col)
        ss = errors.sequence_slots(claims, agent_col) if len(claims) else []
        ds = errors.regex_slots(run.read("events"), run.read("agents"), cfg.disputes, agent_col) if cfg.disputes else []
        feat = errors.pick_featured(vs, ss, cfg.featured_disputes, declared=ds)
        figs["errors"] = errors.figure(feat, run.figure("a4_errors.png"))
        refs = errors.reference_keys(cfg.reference_values)
        slots = feat + [s for s in vs[:12] + ss[:6] if s not in feat]
        rows = []
        for s in slots:
            r = {"slot": s["slot"], "kind": s["kind"],
                 "variants (agents)": ", ".join(f"{k} ({v})" for k, v in s["agents_per_variant"].items()),
                 "agents carrying both": s["overlap_agents"], "first variant": s["first_variant"],
                 "winner": s["winner"], "status": s["status"],
                 "late carriers (winner/total)": f"{max(s['tail_counts'].values())}/{s['n_tail']}"
                                                 + (" tie" if s["tie"] else ""),
                 "winner first seen": s["t_winner_first"], "winner majority from": s["t_winner_majority"],
                 "hours to overtake": s["hours_to_overtake"]}
            if refs:
                ref = refs.get((s["family"], s["item"]))
                named = None if ref is None else errors.match_reference(s["variants"], ref) or ref
                r |= {"reference (analyst input)": named, "outcome": errors.outcome(s, ref) if ref is not None else None}
            rows.append(r)
        unknown, outside = errors.unused_references(refs, set(zip(mentions["family"], mentions["item"])),
                                                    *({(s["family"], s["item"]) for s in g} for g in (vs + ss + ds, slots)))
        summary["A4"] = {"n_value_slots_disputed": len(vs), "n_sequence_slots_disputed": len(ss),
                         "rule": {"bin_hours": errors.BIN_H, "tail_share": errors.TAIL_SHARE,
                                  "min_tail": errors.MIN_TAIL},
                         "featured": [{k: _cell(v) for k, v in r.items()}
                                      | {"tail counts": s["tail_counts"], "tie": s["tie"], "first seen tie": s["first_tie"],
                                         "corrected": s["status"] == "overturned"}   # deprecated alias of status
                                      for r, s in zip(rows, feat)]}
        if refs:
            summary["A4"] |= {"references_unknown": unknown, "references_outside_table": outside}
        tail = f"{Fraction(errors.TAIL_SHARE).limit_denominator(10)}"
        L.append("## A4. Error propagation\n")
        L.append(f"{len(vs)} (family, item) slots carried two or more values, each repeated by ≥3 agents; "
                 f"{len(ss)} round slots had competing *items* discussed by ≥2 agents who carried both (e.g. a cracked-seed forecast vs the observed question; "
                 "slots whose variants have disjoint carriers are different task versions and are excluded). "
                 f"Winner = the most common variant among the last {tail} of carriers, none if the top two tie; "
                 "\"late carriers\" = the winner's count among them. "
                 f"\"Majority from\" = start of the first {errors.BIN_H:g}-hour bin after which the winner never lost the majority of new carriers; "
                 "\"hours to overtake\" runs from a later winner's first appearance to that point. "
                 "Status: **held** = the first variant is still the winner; "
                 f"**overturned** = a later variant won and kept that majority, with at least {errors.MIN_TAIL} late carriers in all; "
                 "**contested** = the late carriers tie, a later winner never kept the majority, or two variants were first seen "
                 "at the same moment (first variant blank); "
                 f"**contested (small n)** = a later winner kept the majority, with fewer than {errors.MIN_TAIL} late carriers in all. "
                 f"The {errors.BIN_H:g}-hour bin and the last-{tail} rule are conventions, not measurements.\n")
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

    # A5 / A6
    if len(edges):
        r5 = structure.analyze(edges, chains, ex=ex if len(ex) else None)
        figs["structure"] = structure.figure(r5, chains, run.figure("a5_structure.png"))
        summary["A5"] = {k: v for k, v in r5.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
        L.append("## A5. Structure\n")
        L.append(f"Graph: {r5['n_nodes']:,} agents, {r5['n_edges']:,} weighted links "
                 f"({r5['n_relay_edges']:,} same-page relay hops, {r5['n_xchannel_edges']:,} cross-page hops attributed to the originator, "
                 f"{r5['n_exposure_edges']:,} first-source exposures, "
                 f"{r5['n_citation_edges']:,} explicit citations). "
                 + (_hub_headline(r5) if r5.get("n_exposure_edges") else
                    "No answer had a first public source here, so there is no exposure concentration to report. ")
                 + (f"Facts carried by ≥2 agents travelled {r5['mean_chain_depth']:.1f} hops on average.\n"
                    if r5.get("mean_chain_depth") == r5.get("mean_chain_depth") else "No fact was carried by two or more agents.\n"))
        if figs["structure"]:
            L.append(f"![structure](figures/{run.figure('a5_structure.png').name})\n")
        top = r5["top_sources"].rename("answers").to_frame()
        if r5.get("n_exposure_edges"):
            L += _attribution_block(r5, summary)
            lines, first = _origins_block({m: e for m, e in ((mapping, ex), (alt, ex_alt)) if len(e) and e["D"].any()},
                                         mapping, summary)
            L += lines
            top = top.join(origins.by_source(first, ex)[["items", "origin"]])
        L.append("Top first-sources (answers they were first public source for"
                 + (", on how many items, and the origin of those answers" if "origin" in top else "") + "):\n")
        L.append(md_table(top))
        L.append("\nTop brokers (betweenness on relay + citation graph"
                 + (f", estimated from {r5['betweenness_sampled_sources']} sampled sources" if r5["betweenness_sampled_sources"] else "")
                 + "):\n")
        L.append(md_table(r5["top_brokers"].rename("betweenness").to_frame(), floatfmt="{:.3f}"))
        r6 = structure.reach(chains)
        if r6:
            figs["reach"] = structure.reach_figure(chains, run.figure("a6_reach.png"))
            summary["A6"] = r6
            L.append("## A6. Reach\n")
            L.append(md_table(pd.DataFrame(r6).T, floatfmt="{:.2f}"))
            L.append(f"\n![reach](figures/{run.figure('a6_reach.png').name})\n")

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
             "t_report is an upper bound on question arrival; the ≥1 h variant guards against report lag.\n"
             f"- **Exposure is a lower bound.** {coverage}\n"
             + identity_note +
             "- **Extraction**: rule-based on templated posts; unrestated answers (\"answered same second\") inherit the consensus value.\n"
             "- **Inferred relay edges** link each carrier to the latest earlier carrier; they are plausible paths, not proven ones."
             + _relay_tie_note(edges) + "\n"
             + ("- **Deletion ends visibility, not knowledge.** D_visible treats a copy deleted before the report as never public; "
                "an agent that read it earlier, or a copy on an uncaptured surface, is not affected, so D stays the headline.\n"
                if "D_visible" in ex and len(ex) and caps.get("has_lifecycle") else "")
             + ("- **The request log is narrow.** Only script-injection probe rows are included, with no page views, so exposure "
                "remains inferred.\n" if run.has("probes") else "")
             + ("- **Derived text.** The posts are an LLM's daily digest of what agents did (one line per event, Pacific time), "
                "not the agents' own messages: who-did-what-when and technique mentions are plausible, the provenance of specific "
                "values is not, and anything the digest's author chose to omit is invisible.\n" if caps.get("derived_text") else ""))
    return _write(run, L, summary)


def _hub_headline(r5: dict) -> str:
    """First-source concentration beside the per-item null, and its range across the attribution rules."""
    a = r5["attribution"]
    share = {sub: a.loc[(a["answers"] == sub) & (a["n"] > 0), "top10_share"] for sub in ("all exposed", "instant only")}
    inst = share["instant only"]
    return (f"{r5['exposure_sources']} agents were the first public source for someone's answer; "
            f"the top 10 supplied **{_pct(r5['top10_share_exposure'])}** of all exposed answers "
            f"(out-degree Gini {r5['gini_exposure_outdegree']:.2f}; the exposure counts of the {r5['items_exposed']} items "
            f"alone have a Gini of {r5['gini_item_exposures']:.2f}, which is what first-source credit gives if each item "
            "has one source of its own). **That share depends on who is credited**: across the three attribution rules "
            f"below it runs from {_pct(share['all exposed'].min())} to {_pct(share['all exposed'].max())}"
            + (f" ({_pct(inst.min())} to {_pct(inst.max())} on instant answers)" if len(inst) else "")
            + ". First-source credit measures who answered first on an item: on "
            + (f"all {r5['items_exposed']}" if r5["items_one_source"] == r5["items_exposed"] else
               f"{r5['items_one_source']} of the {r5['items_exposed']}")
            + " items with exposed answers, one agent was the first source of every one. ")


def _attribution_block(r5: dict, summary: dict) -> list[str]:
    a = r5["attribution"]
    summary["A5"]["attribution"] = [{k: _cell(v) for k, v in r.items()} for r in a.to_dict("records")]
    no_instant = not a.loc[a["answers"] == "instant only", "n"].any()
    L = ["**Hub concentration by attribution rule.** Each rule credits an exposed answer to one agent. None is observed "
         "(nothing records what an agent read), so the rows are a sensitivity range, not competing estimates. "
         "*First source* = the first other agent to post the item and value before the report (the exposure edges); "
         "*latest source* = the last one before the report; *same-page relay parent* = the graph's parent of the agent's "
         "first post of the value, matched on the value key exposure used, and *unmatched* when that post arrived by a "
         "cross-page hop or the agent has no relay edge for the value. *Instant only* keeps answers reported as instant, "
         "the copy signature. On synthetic swarms the rule that copies the way the simulated agents do finds the true "
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
    root_labels = [k for k in origins.LABELS if k != "relayed"] + [origins.POSTED_FIRST]
    roots = pd.DataFrame({r: tr[(r, mapping)]["root_origin"].value_counts().reindex(root_labels, fill_value=0) for r in rules})
    roots.index.name = "root"
    relayed = {r: tr[(r, mapping)].loc[tr[(r, mapping)]["origin"] == "relayed", "depth"] for r in rules}
    first = tr[("first source", mapping)]
    audit = origins.by_source(first[first["origin"] == "no answer round observed"], exs[mapping])
    audit = audit[["answers", "items", "answer rounds (any item)"]].head(10)
    summary["A5"]["origins"] = {
        m: {r: tr[(r, m)]["origin"].value_counts().reindex(origins.LABELS, fill_value=0).to_dict() for r in rules} for m in exs} | {
        "roots": roots.to_dict(), "max depth": {r: int(tr[(r, mapping)]["depth"].max()) for r in rules},
        "no answer round observed, top sources": audit.reset_index().to_dict("records")}
    L = ["**Where did copied answers come from?** Each exposed answer's credited source (under the two rules that credit "
         "every answer, and both identity mappings; answers in brackets), judged by the source's own answer round on the "
         "same item: **computed** = it answered the same value with nothing matching public before its report (D = 0); "
         "**no answer round observed** = no answer round of its own on the item was extracted, which fits an agent that "
         "only published values (an answer list, a sequence scout) but also a missed answer or an identity split, so audit "
         "the sources below before calling them compilers; **dissent** = it answered a different value; **relayed** = it "
         "answered the same value but was itself exposed. Being the first public source is not proof of being read.\n",
         md_table(tbl),
         f"Roots ({mapping} mapping): a relayed source is followed through its own source under the same rule until one "
         "is not relayed" + (" (hops from a relayed answer to its root, median / max: " + ", ".join(
             f"{r} {d.median():g} / {d.max()}" for r, d in relayed.items() if len(d)) + ")" if any(map(len, relayed.values())) else "")
         + (f". *{origins.POSTED_FIRST.capitalize()}* = the agent had posted the value no later than the post it is "
            "credited to (or the chain leads back to the answering agent), so it counts as exposed only because another "
            "agent posted the value between its own post and its report" if roots.loc[origins.POSTED_FIRST].any() else "")
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
