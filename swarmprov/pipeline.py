"""Stage orchestration.  Each stage reads/writes tables in a RunDir, so stages
can be re-run independently (``swarmprov extract`` after editing a regex
doesn't re-ingest)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import adapters, claims as claims_mod, exposure, graph, identity, remote, rules
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


def ingest(path: str | Path, run: RunDir, adapter: str = "auto", config: str | None = None) -> dict:
    source = str(path)
    if remote.is_remote(path):
        _log(f"fetch {path}")
        path = remote.resolve(path)
    a = adapters.detect(path) if adapter == "auto" else adapters.get(adapter)
    _log(f"ingest {path} with adapter '{a.name}'")
    b = a.load(Path(path))
    ev = b.events.sort_values("ts").reset_index(drop=True)
    cfg = a.config()
    if config:
        cfg = adapters.AdapterConfig.from_file(config, base=cfg)
    if cfg.ignore_authors:
        ev = ev[~ev["author_raw"].isin(cfg.ignore_authors)].reset_index(drop=True)
    run.write("events", conform(ev, "events"))
    if b.lifecycle is not None:
        run.write("lifecycle", b.lifecycle)
    if b.reads is not None:
        run.write("reads", b.reads)
    profile = {
        "adapter": a.name, "config": cfg.name, "source": source,
        "config_path": str(Path(config).resolve()) if config else None,
        "capabilities": b.capabilities.as_dict(),
        "n_events": len(ev), "n_authors": int(ev["author_raw"].nunique()),
        "first_ts": str(ev["ts"].min()), "last_ts": str(ev["ts"].max()),
        "events_per_day": ev["ts"].dt.strftime("%Y-%m-%d").value_counts().sort_index().to_dict(),
        "notes": b.notes,
    }
    (run.path / "profile.json").write_text(json.dumps(profile, indent=2, default=str))
    _log(f"  {len(ev)} posts from {profile['n_authors']} author strings")
    return profile


def extract(run: RunDir, llm: str | None = None) -> None:
    cfg = load_config(run)
    ev = run.read("events")
    _log("resolve identities + families")
    fam = identity.post_families(ev, cfg)
    ev["family"] = fam
    run.write("events", ev)
    agents = identity.resolve(ev, fam, cfg)
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


def _matcher(learned: dict) -> rules.ItemMatcher:
    im = rules.ItemMatcher()
    for f, items in learned.items():
        im.add(f, set(items))
    return im


def expose(run: RunDir) -> None:
    ev, agents, cl = run.read("events"), run.read("agents"), run.read("claims")
    learned = json.loads((run.path / "items.json").read_text())
    im = _matcher(learned)
    targets = set(zip(cl["family"], cl["item"])) if len(cl) else set()
    _log("index public mentions")
    mn = claims_mod.index_mentions(ev, ev["family"].fillna(""), agents, im, targets)
    if mn.empty:
        mn = pd.DataFrame(columns=["event_id", "ts", "agent_strict", "agent_merged", "channel", "family",
                                   "item", "value_norm", "value_key"])
    run.write("mentions", mn)
    cons = claims_mod.consensus(cl, mn) if len(cl) else {}
    for name, col in AGENT_COLS.items():
        rounds = claims_mod.agent_rounds(cl, col) if len(cl) else pd.DataFrame()
        ex = exposure.build(rounds, mn, cons, col) if len(rounds) else pd.DataFrame()
        run.write(f"exposures_{name}", ex)
        if len(ex):
            _log(f"  [{name}] {len(ex)} agent-rounds, exposed {ex['D'].mean():.1%}")


def build_graph(run: RunDir) -> None:
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


def run_all(path, out, adapter="auto", llm=None, config=None) -> RunDir:
    run = RunDir(out)
    ingest(path, run, adapter, config)
    extract(run, llm=llm)
    expose(run)
    build_graph(run)
    from . import report
    report.build(run)
    return run
