"""swarmprov command line.

    swarmprov run      INPUT -o runs/wiki [--config cfg.json]   # all stages + report
    swarmprov ingest   INPUT -o runs/wiki [--adapter wiki|chat|auto]
    swarmprov extract  runs/wiki [--llm claude-haiku-4-5]
    swarmprov expose   runs/wiki
    swarmprov graph    runs/wiki
    swarmprov report   runs/wiki [--mapping merged|strict]
    swarmprov sample-gold runs/wiki [-n 120]
    swarmprov validate runs/wiki gold.jsonl
    swarmprov crosssite runs/wiki data/raw2 -o runs/crosssite
    swarmprov synth    -o data/synth.jsonl            # synthetic swarm with known provenance
"""

from __future__ import annotations

import argparse
import sys

from . import pipeline
from .schema import RunDir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swarmprov", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="ingest -> extract -> expose -> graph -> report")
    s.add_argument("input")
    s.add_argument("-o", "--out", default="runs/latest")
    s.add_argument("--adapter", default="auto")
    s.add_argument("--llm", default=None, help="model id for optional LLM extraction (needs ANTHROPIC_API_KEY)")
    s.add_argument("--config", default=None, help="JSON/TOML file overriding the adapter config (families, techniques, ...)")

    s = sub.add_parser("ingest")
    s.add_argument("input")
    s.add_argument("-o", "--out", default="runs/latest")
    s.add_argument("--adapter", default="auto")
    s.add_argument("--config", default=None)

    s = sub.add_parser("extract")
    s.add_argument("run")
    s.add_argument("--llm", default=None)

    for name in ("expose", "graph"):
        s = sub.add_parser(name)
        s.add_argument("run")

    s = sub.add_parser("report")
    s.add_argument("run")
    s.add_argument("--mapping", choices=["merged", "strict"], default="merged")

    s = sub.add_parser("sample-gold", help="write a stratified sample of posts to hand-label")
    s.add_argument("run")
    s.add_argument("-n", type=int, default=120)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--exclude", nargs="*", default=[], help="label files whose posts to skip (e.g. a dev set)")
    s.add_argument("--name", default="gold_template.jsonl")

    s = sub.add_parser("validate", help="score the extractor against hand labels")
    s.add_argument("run")
    s.add_argument("gold")
    s.add_argument("--out-name", default="validation.json", help="file name in the run dir for the scores")

    s = sub.add_parser("crosssite", help="technique spread between surfaces, timeline, coverage bounds")
    s.add_argument("wiki_run", help="run directory of the primary wiki (e.g. runs/wiki)")
    s.add_argument("corpus", help="directory with records.jsonl, shortener-logs.json, other-wikis.json, site-coverage.csv")
    s.add_argument("-o", "--out", default="runs/crosssite")

    s = sub.add_parser("synth", help="generate a synthetic swarm transcript with known provenance")
    s.add_argument("-o", "--out", default="data/synth/transcript.jsonl")
    s.add_argument("--agents", type=int, default=60)
    s.add_argument("--copy-rate", type=float, default=0.7)
    s.add_argument("--seed", type=int, default=0)

    a = p.parse_args(argv)
    if a.cmd == "run":
        pipeline.run_all(a.input, a.out, a.adapter, a.llm, a.config)
    elif a.cmd == "ingest":
        pipeline.ingest(a.input, RunDir(a.out), a.adapter, a.config)
    elif a.cmd == "extract":
        pipeline.extract(RunDir(a.run), llm=a.llm)
    elif a.cmd == "expose":
        pipeline.expose(RunDir(a.run))
    elif a.cmd == "graph":
        pipeline.build_graph(RunDir(a.run))
    elif a.cmd == "report":
        from . import report
        report.build(RunDir(a.run), mapping=a.mapping)
    elif a.cmd == "sample-gold":
        from . import validate
        validate.sample(RunDir(a.run), a.n, a.seed, a.exclude, a.name)
    elif a.cmd == "validate":
        from . import validate
        validate.score(RunDir(a.run), a.gold, a.out_name)
    elif a.cmd == "crosssite":
        from .crosssite_pipeline import run_crosssite
        run_crosssite(a.wiki_run, a.corpus, a.out)
    elif a.cmd == "synth":
        from . import synth
        synth.write(a.out, n_agents=a.agents, copy_rate=a.copy_rate, seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
