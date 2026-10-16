"""Stage orchestration.  Each stage reads/writes tables in a RunDir, so stages
can be re-run independently (``swarmprov extract`` after editing a regex
doesn't re-ingest)."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd

from . import adapters, claims as claims_mod, exposure, graph, identity, remote, roster as roster_mod, rules, village
from .adapters.base import check_fields, read_capabilities
from .analysis import agents as agents_mod, errors
from .analysis.agents import AGENT_COLS
from .schema import RunDir, conform, empty


def _log(msg: str) -> None:
    print(f"[swarmprov {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config(run: RunDir):
    """The adapter's built-in config, overridden by a --config file if one was given at ingest."""
    prof = json.loads((run.path / "profile.json").read_text())
    cfg = adapters.get(prof["adapter"]).config()
    if prof.get("config_path"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # ingest logged them, and the report lists them
            cfg = adapters.AdapterConfig.from_file(prof["config_path"], base=cfg)
    return cfg


def ingest(path: str | Path, run: RunDir, adapter: str = "auto", config: str | None = None,
           roster: str | Path | None = None, fields: dict | None = None) -> dict:
    """``fields`` (canonical field -> source key) overrides the config file's ``fields``; both reach the chat adapter,
    also while detecting."""
    source = str(path)
    if remote.is_remote(path):
        _log(f"fetch {path}")
        path = remote.resolve(path)
    user = adapters.AdapterConfig.read_file(config) if config else {}
    fields = check_fields({**(user.get("fields") or {}), **(fields or {})})
    a = adapters.detect(path, fields) if adapter == "auto" else adapters.get(adapter, fields)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = adapters.AdapterConfig.from_dict(user, base=a.config())
    cfg.fields = fields
    warns = [str(w.message) for w in caught]
    if fields and not a.reads_fields:
        warns.append(f"`fields` reach the chat reader only (chat transcripts and an AI Village message table); the "
                     f"`{a.name}` adapter reads its own schema and ignores them.")
    _log(f"ingest {path} with adapter '{a.name}'" + (f", fields {fields}" if fields else ""))
    b = a.load(Path(path))
    warns += b.notes.pop("warnings", [])
    if cfg.digest_as_posts and b.digest is not None and not len(b.events):
        _log(f"  no message table: using the {len(b.digest):,}-event digest parsed from the daily summaries as posts (derived text)")
        b.events = village.digest_to_events(b.digest)
        b.capabilities.derived_text = True
        b.capabilities.has_explicit_author = True
    ev = b.events.sort_values("ts").reset_index(drop=True)
    if cfg.ignore_authors:
        ev = ev[~ev["author_raw"].isin(cfg.ignore_authors)].reset_index(drop=True)
    run.clear()  # stage outputs of an earlier run in this directory would otherwise survive a re-ingest
    run.write("events", conform(ev, "events"))
    if b.lifecycle is not None:
        run.write("lifecycle", b.lifecycle)
    if b.reads is not None:
        run.write("reads", b.reads)
    ros, roster_notes = b.roster, b.notes if b.roster is not None else {}
    if roster is not None:  # a roster attached to a transcript
        rpath = remote.resolve(roster)
        _log(f"roster {roster}")
        rb = adapters.get("roster").load(Path(rpath))
        ros, roster_notes = rb.roster, rb.notes
    if ros is not None:
        run.write("roster", ros)
    for name in ("directory", "channels", "activity", "probes", "eras", "summaries", "digest"):
        if getattr(b, name) is not None:
            run.write(name, getattr(b, name))
    # wall: real timestamps; sequence: message positions (no real times to print); none: only timed posts are ordered
    clock = b.notes.get("clock") or ("wall" if b.capabilities.has_wall_clock else "none")
    has_ts = clock != "sequence" and len(ev) and ev["ts"].notna().any()
    profile = {
        "adapter": a.name, "config": cfg.name, "source": source,
        "roster": str(roster) if roster is not None else (source if b.roster is not None else None),
        "roster_notes": roster_notes,
        "config_path": str(Path(config).resolve()) if config else None, "fields": fields,
        "capabilities": b.capabilities.as_dict(),
        "clock": clock, "n_untimed": int(ev["ts"].isna().sum()),
        "n_events": len(ev), "n_authors": int(ev["author_raw"].nunique()),
        "first_ts": str(ev["ts"].min()) if has_ts else None, "last_ts": str(ev["ts"].max()) if has_ts else None,
        "events_per_day": ev["ts"].dt.strftime("%Y-%m-%d").value_counts().sort_index().to_dict() if has_ts else {},
        "notes": b.notes, "warnings": warns,
    }
    (run.path / "profile.json").write_text(json.dumps(profile, indent=2, default=str))
    if ros is not None:
        _log(f"  roster: {len(ros)} goals for {ros['agent_id'].nunique()} agents in {ros['role'].nunique()} roles")
    _log(f"  {len(ev)} posts from {profile['n_authors']} author strings")
    for w in warns:
        _log(f"  warning: {w}")
    return profile


def extract(run: RunDir, llm: str | None = None) -> None:
    cfg = load_config(run)
    ev = run.read("events")
    if not len(ev):
        _log("no posts: nothing to extract (a roster alone gives a roster report; add a transcript for provenance)")
        return
    _log("resolve identities + families")
    fam = identity.post_families(ev, cfg, roster=run.has("roster") or run.has("eras"))
    ann = None
    # family precedence: the agent's own goal while it was in force > text/channel families > the shared goal (era)
    if run.has("roster"):
        ros = run.read("roster")
        ann = roster_mod.annotate(ev, ros, cfg.roster_aliases)
        hit = ann["roster_agent"].notna()
        if cfg.family_from_roster:
            fam = fam.where(~(hit & ann["in_window"].fillna(False).astype(bool)), ann["role"])
        _log(f"  roster matched {int(hit.sum())}/{len(ev)} posts by {int(ev.loc[hit, 'author_raw'].nunique())} authors")
    if run.has("eras"):
        era_ann = village.annotate_eras(ev, run.read("eras"))
        if cfg.family_from_roster:
            fam = fam.where(fam.astype(bool) | era_ann["era"].isna(), era_ann["era"])
        ann = era_ann if ann is None else pd.concat([ann, era_ann], axis=1)
        _log(f"  shared goals cover {int(era_ann['era'].notna().sum())}/{len(ev)} posts")
    if ann is not None:
        run.write("roster_events", ann)
    caps = read_capabilities(json.loads((run.path / "profile.json").read_text()))
    agents = identity.resolve(ev, fam, cfg, explicit_author=caps.get("has_explicit_author", False))
    if ann is not None and "roster_agent" in ann:
        agents = roster_mod.merge_agents(agents, ev, ann, ros)
    ev["family"] = fam   # a task message or tool result without one is read under every family (index_mentions)
    run.write("events", ev)
    run.write("agents", agents)
    run.write_csv("agent_aliases", identity.alias_table(agents))
    learned = rules.learn_items(ev, fam, cfg)
    for f, items in cfg.extra_items.items():
        learned.setdefault(f, set()).update(items)
    (run.path / "items.json").write_text(json.dumps({k: sorted(v) for k, v in learned.items()}, indent=1))
    im = _matcher(learned)
    _log("rule extraction")
    own = ~ev["source_kind"].isin(claims_mod.PRIVATE_KINDS)   # task messages and tool results are no agent's claims
    cl, tags = rules.extract(ev[own], fam[own], agents, cfg, im)
    if llm:
        from . import extract_llm
        _log(f"LLM extraction with {llm}")
        llm_cl = extract_llm.run(ev[own], fam[own], agents, model=llm, cache_dir=run.path / "llm_cache")
        cl = extract_llm.merge(cl, llm_cl)
    seqs = rules.learn_sequences(ev, fam, im)
    cl = claims_mod.fill_episodes(cl, sequences=seqs) if len(cl) else cl
    run.write("claims", conform(cl, "claims") if len(cl) else conform(pd.DataFrame(), "claims"))
    run.write("tags", conform(tags, "tags"))
    counts = cl["event_type"].value_counts().to_dict() if len(cl) else {}
    _log(f"  {len(agents)} authors -> {agents['agent_merged'].nunique()} merged agents; claims: {counts}")


def _lists(channels: pd.DataFrame) -> pd.DataFrame:
    """Parquet stores the allow/deny lists as text; turn them back into lists."""
    import ast
    ch = channels.copy()
    for c in ("allow", "deny"):
        ch[c] = ch[c].map(lambda v: ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else (v if isinstance(v, list) else []))
    return ch


def _matcher(learned: dict) -> rules.ItemMatcher:
    im = rules.ItemMatcher()
    for f, items in learned.items():
        im.add(f, set(items))
    return im


def expose(run: RunDir) -> None:
    if not run.has("claims"):
        return
    ev, agents, cl = run.read("events"), run.read("agents"), run.read("claims")
    learned = json.loads((run.path / "items.json").read_text())
    im = _matcher(learned)
    targets = set(zip(cl["family"], cl["item"])) if len(cl) else set()
    _log("index item-value mentions")
    mn = claims_mod.index_mentions(ev, ev["family"].fillna(""), agents, im, targets)
    if mn.empty:
        mn = pd.DataFrame(columns=["event_id", "ts", "agent_strict", "agent_merged", "channel", "family",
                                   "item", "value_norm", "source_kind", "value_key"])
    if "visible_until" in ev and len(mn):  # a deleted copy stops being public: exposure reports D_visible beside D
        vu = ev[["event_id"]].assign(visible_until=pd.to_datetime(ev["visible_until"], utc=True, errors="coerce"))
        mn = mn.merge(vu.drop_duplicates("event_id"), on="event_id", how="left")
    run.write("mentions", mn)
    cons = claims_mod.consensus(cl, mn) if len(cl) else {}
    channels = run.read("channels") if run.has("channels") else None
    directory = run.read("directory") if run.has("directory") else None
    wall = json.loads((run.path / "profile.json").read_text()).get("clock") != "sequence"
    for name, col in AGENT_COLS.items():
        rounds = claims_mod.agent_rounds(cl, col) if len(cl) else pd.DataFrame()
        aud = village.audience(_lists(channels), agents, col, directory) if channels is not None else None
        if aud:
            _log(f"  [{name}] {len(aud)} restricted channels limit who could read a mention")
        # A4's overturned value slots, the ones y_winner scores (the report finds the same slots from the same mentions)
        slots = errors.value_slots(claims_mod.public(mn), col, **({} if wall else {"bin_h": None})) if len(rounds) else []
        winners = {(s["family"], s["item"]): s["winner"] for s in slots if s["status"] == "overturned"}
        ex = exposure.build(rounds, mn, cons, col, audience=aud, wall_clock=wall, winners=winners) \
            if len(rounds) else pd.DataFrame()
        if len(rounds) and rounds["t_report"].isna().any():
            _log(f"  [{name}] {int(rounds['t_report'].isna().sum())} agent-rounds have no timestamp: left out of exposure")
        run.write(f"exposures_{name}", ex if len(ex) else empty("exposures"))
        if len(ex):
            _log(f"  [{name}] {len(ex)} agent-rounds, exposed {ex['D'].mean():.1%}")


def build_graph(run: RunDir) -> None:
    """Edges and fact chains, the agent profiles, and the graph as GraphML and node-link JSON, per mapping.  Reply
    edges only when the source has threading: elsewhere ``parent_id`` is a revision or summary id."""
    if not run.has("mentions"):
        return
    import networkx as nx
    cfg = load_config(run)
    prof = json.loads((run.path / "profile.json").read_text())
    mn, agents, tags, events = run.read("mentions"), run.read("agents"), run.read("tags"), run.read("events")
    seq = prof.get("clock") == "sequence"
    threading = read_capabilities(prof).get("has_threading", False)
    # message positions are 1 s apart, so the 2 s clock resolution would flag every adjacent hop
    res = None if seq else graph.CLOCK_RES_S
    pub = claims_mod.public(mn)   # a tool result or task message relays nothing: no other agent saw it
    posters = agents[agents["author_raw"].isin(claims_mod.public(events)["author_raw"])]   # citable; never `task`
    for name, col in AGENT_COLS.items():
        ex = run.read(f"exposures_{name}")
        rel, chains = graph.relay_edges(pub, col, res) if len(pub) else (pd.DataFrame(), pd.DataFrame())
        parts = [rel, graph.exposure_edges(ex) if len(ex) else pd.DataFrame(),
                 graph.citation_edges(tags, posters, col, cfg.cohort_regex)]
        note = ""
        if threading:
            rep, unresolved = graph.reply_edges(events, agents, col)
            parts.append(rep)
            note = f", {len(rep)} repl{'y' if len(rep) == 1 else 'ies'}" + (
                f" ({unresolved} name{'s' if unresolved == 1 else ''} a parent that is not an agent's post here)"
                if unresolved else "")
        edges = conform(pd.concat([p for p in parts if len(p)], ignore_index=True), "edges") \
            if any(len(p) for p in parts) else empty("edges")
        chains = chains if len(chains) else empty("chains")
        run.write(f"edges_{name}", edges)
        run.write(f"chains_{name}", chains)
        profiles = agents_mod.profile(agents, events, tags, ex, edges, chains, name, wall_clock=not seq)
        run.write(f"agent_profiles_{name}", profiles)
        if len(edges):
            G = graph.to_networkx(edges, events=events, nodes=profiles.set_index("label", drop=False))
            G.graph["clock"] = "sequence" if seq else "wall"
            if seq:
                G.graph["clock_note"] = ("message order, no wall clock: a time is the message's position "
                                         "(2000-01-01 + i s) and median_dt_s a gap in messages")
            nx.write_graphml(G, run.path / f"provenance_{name}.graphml")
            (run.path / f"provenance_{name}.json").write_text(json.dumps(nx.node_link_data(G)))
        _log(f"  [{name}] {len(edges)} edges{note}, {len(chains)} fact chains, {len(profiles)} agent profiles")


def score_gold(run: RunDir, paths) -> None:
    """Score the extractor against each label file (a directory gives its ``*.jsonl``) as ``validation_<split>.json``;
    a file whose labelled posts are none of this run's is skipped, not scored as zero."""
    from . import validate
    files = [f for p in map(Path, paths) for f in (sorted(p.glob("*.jsonl")) if p.is_dir() else [p])]
    if not files:
        _log(f"gold labels: no *.jsonl label file in {', '.join(map(str, paths))}")
    for f in files:
        res = validate.score(run, f, f"validation_{validate.split_name(f)}.json")
        if res is None:
            _log(f"gold labels {f}: none of its labelled posts is in this run (labels of another source?): skipped")
            continue
        ap, cov = res["fields"]["answer_post"], res["coverage"]
        _log(f"gold labels {f}: {cov['in_run']} of {cov['labelled']} labelled posts in the run; answer posts "
             f"precision {ap['precision']:.2f}, recall {ap['recall']:.2f} -> validation_{validate.split_name(f)}.json")


def run_all(path, out, adapter="auto", llm=None, config=None, roster=None, fields=None, gold=None) -> RunDir:
    """``gold``: label files or directories, scored before the report (``score_gold``)."""
    run = RunDir(out)
    ingest(path, run, adapter, config, roster, fields)
    extract(run, llm=llm)
    expose(run)
    build_graph(run)
    if gold:
        score_gold(run, gold)
    from . import report
    report.build(run)
    return run
