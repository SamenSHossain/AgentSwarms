"""Stage orchestration.  Each stage reads/writes tables in a RunDir, so stages
can be re-run independently (``swarmprov extract`` after editing a regex
doesn't re-ingest)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import adapters, claims as claims_mod, exposure, graph, identity, remote, roster as roster_mod, rules, village
from .schema import RunDir, conform

AGENT_COLS = {"merged": "agent_merged", "strict": "agent_strict"}


def _log(msg: str) -> None:
    print(f"[swarmprov {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config(run: RunDir):
    """The adapter's built-in config, overridden by a --config file if one was given at ingest."""
    prof = json.loads((run.path / "profile.json").read_text())
    cfg = adapters.get(prof["adapter"]).config()
    if prof.get("config_path"):
        cfg = adapters.AdapterConfig.from_file(prof["config_path"], base=cfg)
    return cfg


def ingest(path: str | Path, run: RunDir, adapter: str = "auto", config: str | None = None,
           roster: str | Path | None = None) -> dict:
    source = str(path)
    if remote.is_remote(path):
        _log(f"fetch {path}")
        path = remote.resolve(path)
    a = adapters.detect(path) if adapter == "auto" else adapters.get(adapter)
    _log(f"ingest {path} with adapter '{a.name}'")
    b = a.load(Path(path))
    cfg = a.config()
    if config:
        cfg = adapters.AdapterConfig.from_file(config, base=cfg)
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
    has_ts = len(ev) and ev["ts"].notna().any()
    profile = {
        "adapter": a.name, "config": cfg.name, "source": source,
        "roster": str(roster) if roster is not None else (source if b.roster is not None else None),
        "roster_notes": roster_notes,
        "config_path": str(Path(config).resolve()) if config else None,
        "capabilities": b.capabilities.as_dict(),
        "n_events": len(ev), "n_authors": int(ev["author_raw"].nunique()),
        "first_ts": str(ev["ts"].min()) if has_ts else None, "last_ts": str(ev["ts"].max()) if has_ts else None,
        "events_per_day": ev["ts"].dt.strftime("%Y-%m-%d").value_counts().sort_index().to_dict() if has_ts else {},
        "notes": b.notes,
    }
    (run.path / "profile.json").write_text(json.dumps(profile, indent=2, default=str))
    if ros is not None:
        _log(f"  roster: {len(ros)} goals for {ros['agent_id'].nunique()} agents in {ros['role'].nunique()} roles")
    _log(f"  {len(ev)} posts from {profile['n_authors']} author strings")
    return profile


def extract(run: RunDir, llm: str | None = None) -> None:
    cfg = load_config(run)
    ev = run.read("events")
    if not len(ev):
        _log("no posts: nothing to extract (a roster alone gives a roster report; add a transcript for provenance)")
        return
    _log("resolve identities + families")
    fam = identity.post_families(ev, cfg)
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
    ev["family"] = fam
    run.write("events", ev)
    agents = identity.resolve(ev, fam, cfg)
    if ann is not None and "roster_agent" in ann:
        agents = roster_mod.merge_agents(agents, ev, ann, ros)
    run.write("agents", agents)
    learned = rules.learn_items(ev, fam, cfg)
    for f, items in cfg.extra_items.items():
        learned.setdefault(f, set()).update(items)
    (run.path / "items.json").write_text(json.dumps({k: sorted(v) for k, v in learned.items()}, indent=1))
    im = _matcher(learned)
    _log("rule extraction")
    cl, tags = rules.extract(ev, fam, agents, cfg, im)
    if llm:
        from . import extract_llm
        _log(f"LLM extraction with {llm}")
        llm_cl = extract_llm.run(ev, fam, agents, model=llm, cache_dir=run.path / "llm_cache")
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
    _log("index public mentions")
    mn = claims_mod.index_mentions(ev, ev["family"].fillna(""), agents, im, targets)
    if mn.empty:
        mn = pd.DataFrame(columns=["event_id", "ts", "agent_strict", "agent_merged", "channel", "family",
                                   "item", "value_norm", "value_key"])
    if "visible_until" in ev and len(mn):  # a deleted copy stops being public: exposure reports D_visible beside D
        vu = ev[["event_id"]].assign(visible_until=pd.to_datetime(ev["visible_until"], utc=True, errors="coerce"))
        mn = mn.merge(vu.drop_duplicates("event_id"), on="event_id", how="left")
    run.write("mentions", mn)
    cons = claims_mod.consensus(cl, mn) if len(cl) else {}
    channels = run.read("channels") if run.has("channels") else None
    directory = run.read("directory") if run.has("directory") else None
    for name, col in AGENT_COLS.items():
        rounds = claims_mod.agent_rounds(cl, col) if len(cl) else pd.DataFrame()
        aud = village.audience(_lists(channels), agents, col, directory) if channels is not None else None
        if aud:
            _log(f"  [{name}] {len(aud)} restricted channels limit who could read a mention")
        ex = exposure.build(rounds, mn, cons, col, audience=aud) if len(rounds) else pd.DataFrame()
        run.write(f"exposures_{name}", ex)
        if len(ex):
            _log(f"  [{name}] {len(ex)} agent-rounds, exposed {ex['D'].mean():.1%}")


def build_graph(run: RunDir) -> None:
    if not run.has("mentions"):
        return
    cfg = load_config(run)
    mn, agents, tags = run.read("mentions"), run.read("agents"), run.read("tags")
    for name, col in AGENT_COLS.items():
        ex = run.read(f"exposures_{name}")
        rel, chains = graph.relay_edges(mn, col) if len(mn) else (pd.DataFrame(), pd.DataFrame())
        parts = [rel, graph.exposure_edges(ex) if len(ex) else pd.DataFrame(),
                 graph.citation_edges(tags, agents, col, cfg.cohort_regex)]
        edges = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) \
            else pd.DataFrame(columns=["src", "dst", "family", "item", "value_norm", "kind", "dt_s", "event_id"])
        run.write(f"edges_{name}", edges)
        run.write(f"chains_{name}", chains)
        if name == "merged" and len(edges):
            import networkx as nx
            G = graph.to_networkx(edges, kinds=("relay", "exposure", "citation"))
            nx.write_graphml(G, run.path / "provenance_merged.graphml")
        _log(f"  [{name}] {len(edges)} edges, {len(chains)} fact chains")


def run_all(path, out, adapter="auto", llm=None, config=None, roster=None) -> RunDir:
    run = RunDir(out)
    ingest(path, run, adapter, config, roster)
    extract(run, llm=llm)
    expose(run)
    build_graph(run)
    from . import report
    report.build(run)
    return run
