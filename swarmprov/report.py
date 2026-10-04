"""Run the analyses on a RunDir and write ``report.md`` + figures + summary.json."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import lifecycle as lifecycle_mod, roster as roster_mod, village as village_mod
from .analysis import causal, diffusion, errors, provenance, structure
from .schema import RunDir


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
        return str(v).replace("|", "\\|")

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
    summary["roster"] = {k: v for k, v in s.items() if not isinstance(v, pd.DataFrame)}
    return roster_mod.section(s, md_table, prof.get("roster_notes", {}).get("timestamp_issues"))


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
    s = village_mod.summary(directory, channels, ev, activity, roster, notes.get("unrecognised"), eras, era_ann,
                            notes.get("village"), load_config(run).schedule_tz, summaries, digest)
    summary["village"] = {k: v for k, v in s.items() if not isinstance(v, pd.DataFrame)}
    return village_mod.section(s, md_table)


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
    caps = prof["capabilities"]
    if not prof["n_events"]:
        summary = {"profile": {k: prof[k] for k in ("adapter", "source", "n_events")}}
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
    ex_alt = run.read(f"exposures_{'strict' if mapping == 'merged' else 'merged'}")
    claims, tags, mentions = run.read("claims"), run.read("tags"), run.read("mentions")
    edges, chains = run.read(f"edges_{mapping}"), run.read(f"chains_{mapping}")
    figs, summary = {}, {"mapping": mapping, "profile": {k: prof[k] for k in ("adapter", "n_events", "n_authors", "first_ts", "last_ts")}}
    L: list[str] = []
    L.append(f"# Copied or computed? — provenance report\n")
    L.append(f"Source: `{prof['source']}` (adapter `{prof['adapter']}`), {prof['n_events']:,} posts from "
             f"{prof['n_authors']:,} author strings, {prof['first_ts'][:16]} → {prof['last_ts'][:16]} UTC. "
             f"Agent identity mapping: **{mapping}** (sensitivity under the other mapping below).\n")
    L.append("## Data capabilities\n")
    L.append("| capability | available | consequence |\n|---|---|---|")
    conseq = {
        "has_wall_clock": "ordering by real time is possible",
        "has_explicit_author": "authors come from fields, not parsed signatures",
        "has_reads": "exposure is *observed*; otherwise it is inferred from what was public",
        "has_lifecycle": "deletions are recorded, posts carry visible_until, and A1 reports D_visible beside D",
        "has_request_log": "request-level rows exist (script-injection probes only); still no page views",
        "derived_text": "posts are an LLM digest of what agents did, not their words: values and quotes are second-hand",
        "has_threading": "reply links usable as explicit edges",
        "has_episodes": "rounds are fields; otherwise parsed from text",
        "has_activity": "a presence log says when each agent was running",
    }
    for k, v in caps.items():
        L.append(f"| {k} | {'yes' if v else 'no'} | {conseq.get(k, '')} |")
    L.append("")

    notes = prof.get("notes") or {}
    if notes.get("clock_grades"):
        L.append(_clock_note(notes, ex) + "\n")
    L += _village_block(run, prof, summary)
    if run.has("roster"):
        L += _roster_block(run, prof, summary)
    if run.has("lifecycle"):
        lc = run.read("lifecycle")
        dels = lc[lc["action"] == "delete"] if len(lc) else lc
        # page-level deletions (the wiki dump records each with an event id); room deletions are covered in the Village block
        if len(dels) and "event_id" in dels and dels["event_id"].notna().any():
            ds = lifecycle_mod.deletion_summary(lc, run.read("events"), notes.get("recreation_check"))
            summary["deletions"] = {k: v for k, v in ds.items() if k != "check"} | {"check": {k: v for k, v in (ds.get("check") or {}).items() if k != "restored_event_ids"}}
            L += lifecycle_mod.deletion_section(ds, _pct)
    if run.has("probes"):
        ps = lifecycle_mod.probe_summary(run.read("probes"))
        summary["probes"] = ps
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
        if "D_visible" in ex and ex["src_deleted_before_report"].sum():
            L.append(_visibility_note(ex) + "\n")
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
        rows = [{"slot": s["slot"], "kind": s["kind"], "variants (agents)": ", ".join(f"{k} ({v})" for k, v in s["agents_per_variant"].items()),
                 "agents carrying both": s["overlap_agents"],
                 "first variant": s["first_variant"], "winner": s["winner"], "corrected": s["corrected"],
                 "winner first seen": s["t_winner_first"], "winner majority from": s["t_winner_majority"],
                 "hours to overtake": s["hours_to_overtake"]} for s in (feat + [s for s in vs[:12] + ss[:6] if s not in feat])]
        summary["A4"] = {"n_value_slots_disputed": len(vs), "n_sequence_slots_disputed": len(ss),
                         "featured": [{k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
                                       for k, v in r.items()} for r in rows[:len(feat)]]}
        L.append("## A4. Error propagation\n")
        L.append(f"{len(vs)} (family, item) slots carried two or more values, each repeated by ≥3 agents; "
                 f"{len(ss)} round slots had competing *items* discussed by ≥2 agents who carried both (e.g. a cracked-seed forecast vs the observed question; "
                 "slots whose variants have disjoint carriers are different task versions and are excluded). "
                 "Winner = majority among the last third of carriers; \"majority from\" = start of the first 3-hour bin after which the winner never lost the majority of new carriers.\n")
        if ds:
            L.append("Analyst-declared disputes (config `disputes`: a context pattern plus one pattern per variant) are listed first.\n")
        if figs["errors"]:
            L.append(f"![errors](figures/{run.figure('a4_errors.png').name})\n")
        L.append(md_table(pd.DataFrame(rows), index=False, floatfmt="{:.1f}"))

    # A5 / A6
    if len(edges):
        r5 = structure.analyze(edges, chains)
        figs["structure"] = structure.figure(r5, chains, run.figure("a5_structure.png"))
        summary["A5"] = {k: v for k, v in r5.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
        L.append("## A5. Structure\n")
        L.append(f"Graph: {r5['n_nodes']:,} agents, {r5['n_edges']:,} weighted links "
                 f"({r5['n_relay_edges']:,} same-page relay hops, {r5['n_xchannel_edges']:,} cross-page hops attributed to the originator, "
                 f"{r5['n_exposure_edges']:,} first-source exposures, "
                 f"{r5['n_citation_edges']:,} explicit citations). "
                 + (f"{r5['exposure_sources']} agents were the first public source for someone's answer; "
                    f"the top 10 supplied **{_pct(r5['top10_share_exposure'])}** of all exposed answers "
                    f"(out-degree Gini {r5['gini_exposure_outdegree']:.2f}). " if r5.get("n_exposure_edges") else
                    "No answer had a first public source here, so there is no exposure concentration to report. ")
                 + (f"Facts carried by ≥2 agents travelled {r5['mean_chain_depth']:.1f} hops on average.\n"
                    if r5.get("mean_chain_depth") == r5.get("mean_chain_depth") else "No fact was carried by two or more agents.\n"))
        if figs["structure"]:
            L.append(f"![structure](figures/{run.figure('a5_structure.png').name})\n")
        L.append("Top first-sources (answers they were first public source for):\n")
        L.append(md_table(r5["top_sources"].rename("answers").to_frame()))
        L.append("\nTop brokers (betweenness on relay + citation graph):\n")
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
                 f"{rc['overlap']} identical, {rc['dump_only']} missed, {rc['pipeline_only']} extra"
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
    identity_note = (
        "- **Identity**: the roster fixes cohort (assignment batch) and family (goal) for matched authors; "
        "unmatched authors fall back to parsed signatures.\n" if run.has("roster") else
        "- **Identity**: names are parsed from signatures; the merged mapping assumes one agent per (cohort date, task family). "
        "Both mappings are reported.\n")
    L.append("- **Exposure is inferred, not observed**: no page-view logs, so D means \"was public\", not \"was read\". "
             "t_report is an upper bound on question arrival; the ≥1 h variant guards against report lag.\n"
             "- **Exposure is a lower bound.** Only the captured surfaces are searched for earlier copies of an answer; "
             "the collectors' coverage tables list 143 surfaces the swarm touched, most of them (Discord, 12 uncrawled wikis, "
             "relays) not captured. An \"independent\" answer may have been relayed through one of them, so the exposed "
             "share is a floor and the independent count a ceiling.\n"
             + identity_note +
             "- **Extraction**: rule-based on templated posts; unrestated answers (\"answered same second\") inherit the consensus value.\n"
             "- **Inferred relay edges** link each carrier to the latest earlier carrier; they are plausible paths, not proven ones."
             + _relay_tie_note(edges) + "\n"
             + ("- **Deletion ends visibility, not knowledge.** D_visible treats a copy deleted before the report as never public; "
                "an agent that read it earlier, or a copy on an uncaptured surface, is not affected, so D stays the headline.\n"
                if "D_visible" in ex and len(ex) else "")
             + ("- **The request log is narrow.** Only script-injection probe rows are included, with no page views, so exposure "
                "remains inferred.\n" if run.has("probes") else "")
             + ("- **Derived text.** The posts are an LLM's daily digest of what agents did (one line per event, Pacific time), "
                "not the agents' own messages: who-did-what-when and technique mentions are plausible, the provenance of specific "
                "values is not, and anything the digest's author chose to omit is invisible.\n" if caps.get("derived_text") else ""))
    return _write(run, L, summary)


def _clock_note(notes: dict, ex: pd.DataFrame) -> str:
    g, pg = notes.get("clock_grades", {}), notes.get("posts_by_clock_grade", {})
    n_posts = sum(pg.values()) or 1
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
        gs = ex["gap_s"].dropna()
        parts.append(f" Minimum exposure head start {gs.min():.0f} s; {int((gs < 2).sum())} of {len(gs):,} exposed answers within the 2 s "
                     f"summed uncertainty, {int(((gs >= 598) & (gs < 602)).sum())} within 2 s of the 10 min threshold, "
                     f"{int(((gs >= 3598) & (gs < 3602)).sum())} within 2 s of the 1 h threshold.")
    return "".join(parts)


def _visibility_note(ex: pd.DataFrame) -> str:
    exposed = ex[ex["D"] == 1]
    gone = exposed[exposed["src_deleted_before_report"] == 1]
    other = int((gone["D_visible"] == 1).sum())
    n_ind, n_ind_vis = int((ex["D"] == 0).sum()), int((ex["D_visible"] == 0).sum())
    age = (gone["t_report"] - gone["t_public"]).dt.total_seconds().div(3600)
    since = (gone["t_report"] - pd.to_datetime(gone["t_src_deleted"], utc=True)).dt.total_seconds().div(3600)
    return (f"Honouring deletions: {len(gone):,} of {len(exposed):,} exposed answers cite a public copy that had been deleted before "
            f"the report (posted a median {age.median():.1f} h before it, deleted a median {since.median():.1f} h before it); "
            f"for {other:,} another copy was still visible, for {len(gone) - other:,} "
            f"nothing visible carried the value, so the independent count would be {n_ind_vis:,} ({_pct(n_ind_vis / len(ex))}) "
            f"under D_visible instead of {n_ind:,} ({_pct(n_ind / len(ex))}).")


def _relay_tie_note(edges: pd.DataFrame) -> str:
    if not len(edges) or "within_clock_res" not in edges:
        return ""
    rel = edges[edges["kind"].isin(["relay", "relay_xchannel"])]
    url = rel["family"].astype(str).str.startswith("url")
    tie = rel["within_clock_res"].fillna(False).astype(bool)
    return (f" {int((tie & ~url).sum()):,} of {int((~url).sum()):,} answer relay hops (and {int((tie & url).sum()):,} URL hops) "
            f"fall within 2 s of their source and carry no reliable direction.")
