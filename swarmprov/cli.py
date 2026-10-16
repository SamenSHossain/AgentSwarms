"""swarmprov command line.

    swarmprov run      INPUT -o runs/wiki [--config cfg.json]   # all stages + report
    swarmprov run      chat.jsonl.gz -o runs/village --roster agent_goals.jsonl  # with a goal roster
    swarmprov run      autogen.jsonl -o runs/agents --field author=source        # name a chat field's source key
    swarmprov inspect  INPUT [--field author=source]  # what each adapter and chat field finds; suggests a mapping
    swarmprov ingest   INPUT -o runs/wiki [--adapter auto|wiki|corpus|swarmtraces|village|roster|chat]
    swarmprov extract  runs/wiki [--llm claude-haiku-4-5]
    swarmprov expose   runs/wiki
    swarmprov graph    runs/wiki
    swarmprov report   runs/wiki [--mapping merged|strict] [--boot 1000]   # strict: report_strict.md beside report.md
    swarmprov sample-gold runs/wiki [-n 120]
    swarmprov validate runs/wiki gold.jsonl
    swarmprov run      data/raw -o runs/wiki --gold validation/   # also score the extractor on each label file
    swarmprov check    runs/wiki                          # do the tables agree? exit 1 on an error
    swarmprov truth-score runs/synth data/synth           # a synthetic run against what synth planted
    swarmprov crosssite runs/wiki data/raw2 -o runs/crosssite
    swarmprov compare  runs/wiki runs/wiki2 -o runs/cmp                    # what changed: setup, summary, rows
    swarmprov compare  runs/wiki runs/wiki --mapping-b strict              # the two identity mappings of one run
    swarmprov trace    runs/wiki --agent "Mar16|oecd-equity" --item "Czech Republic"   # one answer's sources
    swarmprov trace    runs/wiki --fact "oecd-equity|Hungary|9.9" --png hungary.png     # one fact's relay tree
    swarmprov agent    runs/wiki May28                                                 # one agent's profile and links
    swarmprov export   runs/wiki -o out/ [--tables exposures,adoptions] [--format csv|parquet] [--mapping strict]
    swarmprov synth    -o data/synth.jsonl            # synthetic swarm with known provenance
    swarmprov synth    -o data/ag/autogen.jsonl --format autogen   # the same as an AutoGen stream with tool calls
    swarmprov synth    -o data/ramp/transcript.jsonl --copy-ramp-h 4   # copying grows with the answer's head start
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from . import pipeline
from .adapters import ADAPTERS
from .adapters.base import FIELD_NAMES, check_fields
from .schema import RunDir


def _positive(v: str) -> int:
    if not v.isdigit() or int(v) < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {v!r}")
    return int(v)


def _count(v: str) -> int:
    if not v.isdigit():
        raise argparse.ArgumentTypeError(f"expected a whole number (0 or more), got {v!r}")
    return int(v)


def _hours(v: str) -> float:
    try:
        h = float(v)
    except ValueError:
        h = float("nan")
    if not 0 < h < float("inf"):
        raise argparse.ArgumentTypeError(f"expected a positive, finite number of hours, got {v!r}")
    return h


def _fields(v: str) -> dict[str, str]:
    """``author=source``, or several at once: ``author=source,text=content``.  Only a comma before another
    ``FIELD=`` separates, so a key may hold one (``author=Last, First``)."""
    pairs = [p.split("=", 1) for p in re.split(r",(?=\s*\w+\s*=)", v) if p.strip()]
    if not pairs or any(len(p) != 2 for p in pairs):
        raise argparse.ArgumentTypeError(f"expected FIELD=KEY (e.g. author=source), got {v!r}")
    try:
        return check_fields({k.strip(): key for k, key in pairs})
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _tolerance(v: str) -> float:
    try:
        t = float(v)
    except ValueError:
        t = float("nan")
    if not 0 <= t < 1:
        raise argparse.ArgumentTypeError(f"expected a relative tolerance from 0 up to (not including) 1, got {v!r}")
    return t


def _tables(v: str) -> list[str]:
    from .export import EXPORTS
    names = [t.strip() for t in v.split(",") if t.strip()]
    bad = [t for t in names if t not in EXPORTS]
    if bad or not names:
        raise argparse.ArgumentTypeError(f"unknown table{'s' * (len(bad) > 1)} {', '.join(bad) or repr(v)}; "
                                         f"choose from {', '.join(EXPORTS)}")
    return names


def _add_fields(s: argparse.ArgumentParser) -> None:
    s.add_argument("--field", type=_fields, action="append", default=[], metavar="FIELD=KEY",
                   help=f"chat adapter: read FIELD ({', '.join(FIELD_NAMES)}) from the source's KEY (a dotted KEY such as "
                        "author.name reaches into nested objects); repeatable, overrides the config's fields")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swarmprov", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    adapter = dict(choices=["auto", *ADAPTERS], default="auto")
    s = sub.add_parser("run", help="ingest -> extract -> expose -> graph -> report")
    s.add_argument("input")
    s.add_argument("-o", "--out", default="runs/latest")
    s.add_argument("--adapter", **adapter)
    s.add_argument("--llm", default=None, help="model id for optional LLM extraction (needs ANTHROPIC_API_KEY)")
    s.add_argument("--config", default=None, help="JSON/TOML file overriding the adapter config (families, techniques, ...)")
    s.add_argument("--roster", default=None, help="agent-goal table (AI Village agent_goals) to attach: goals become families and cohorts")
    s.add_argument("--gold", nargs="+", default=None, metavar="FILE_OR_DIR",
                   help="hand-label files (a directory: its *.jsonl) to score the extractor on before the report, as "
                        "validation_<split>.json; a file with no labelled post in this run is skipped")
    _add_fields(s)

    s = sub.add_parser("ingest")
    s.add_argument("input")
    s.add_argument("-o", "--out", default="runs/latest")
    s.add_argument("--adapter", **adapter)
    s.add_argument("--config", default=None)
    s.add_argument("--roster", default=None)
    _add_fields(s)

    s = sub.add_parser("inspect", help="each adapter's sniff, the chat fields found and their coverage; suggests a mapping")
    s.add_argument("input")
    s.add_argument("--config", default=None, help="take the fields from this config file")
    _add_fields(s)

    s = sub.add_parser("extract")
    s.add_argument("run")
    s.add_argument("--llm", default=None)

    for name in ("expose", "graph"):
        s = sub.add_parser(name)
        s.add_argument("run")

    s = sub.add_parser("report")
    s.add_argument("run")
    s.add_argument("--mapping", choices=["merged", "strict"], default="merged")
    s.add_argument("--boot", type=_count, default=None, metavar="N",
                   help="bootstrap resamples behind the confidence intervals (seeded; default: the config's "
                        "bootstrap_reps, 1000); 0 leaves the intervals out")

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

    s = sub.add_parser("check", help="integrity checks: do the run's tables agree with each other? (exit 1 on an error)")
    s.add_argument("run")
    s.add_argument("--mapping", choices=["merged", "strict"], default="merged",
                   help="whose report summary to check against the tables (strict: summary_strict.json)")

    s = sub.add_parser("truth-score", help="score a synthetic run against the truth synth wrote beside the transcript")
    s.add_argument("run")
    s.add_argument("truth", help="the directory holding the transcript's truth.csv and config.json")
    s.add_argument("--mapping", choices=["merged", "strict"], default="merged")

    s = sub.add_parser("crosssite", help="technique spread between surfaces, timeline, coverage bounds")
    s.add_argument("wiki_run", help="run directory of the primary wiki (e.g. runs/wiki)")
    s.add_argument("corpus", help="directory with records.jsonl, shortener-logs.json, other-wikis.json, site-coverage.csv")
    s.add_argument("-o", "--out", default="runs/crosssite")

    s = sub.add_parser("compare", help="what changed between two runs, configs or identity mappings: setup, summary, rows")
    s.add_argument("run_a")
    s.add_argument("run_b", help="another run, or RUN_A again to compare its two mappings")
    s.add_argument("--mapping-a", choices=["merged", "strict"], default="merged")
    s.add_argument("--mapping-b", choices=["merged", "strict"], default="merged")
    s.add_argument("-o", "--out", default="runs/compare", help="directory for compare.md and compare.json")
    s.add_argument("--limit", type=_positive, default=None, help="at most N examples per change (default 10)")
    s.add_argument("--tol", type=_tolerance, default=1e-6,
                   help="relative tolerance within which summary numbers count as equal (default 1e-6)")

    s = sub.add_parser("trace", help="one answer's sources, one fact's relay tree, one post's extraction")
    s.add_argument("run")
    q = s.add_mutually_exclusive_group(required=True)
    q.add_argument("--agent", help="agent label or author string (case-insensitive; lists candidates when ambiguous)")
    q.add_argument("--answer", metavar="CLAIM_ID", help="an answer claim id")
    q.add_argument("--fact", metavar="FAMILY|ITEM[|VALUE]", help="a fact; without a value, every value of the item")
    q.add_argument("--event", metavar="EVENT_ID", help="a post: author, family, tags, claims, mentions, edges")
    q.add_argument("--list", action="store_true", help="facts ranked by carriers (--family url for URLs)")
    s.add_argument("--item", help="with --agent: one item")
    s.add_argument("--family", help="with --agent or --list: one task family")
    s.add_argument("--mapping", choices=["merged", "strict"], default="merged")
    s.add_argument("--json", metavar="FILE", help="also write the result as JSON")
    s.add_argument("--png", metavar="FILE", help="timeline of one fact (--fact with a value, or a single answer)")
    s.add_argument("--limit", type=_positive, default=None,
                   help="at most N answers or listed facts (default 25), N hops printed per fact (JSON keeps all)")

    s = sub.add_parser("agent", help="one agent's profile, answers, and whom it learned from and supplied")
    s.add_argument("run")
    s.add_argument("name", help="agent label or author string (case-insensitive; lists candidates when ambiguous)")
    s.add_argument("--mapping", choices=["merged", "strict"], default="merged")
    s.add_argument("--json", metavar="FILE", help="also write the result as JSON")

    s = sub.add_parser("export", help="write a run's tables (exposures, edges, adoptions, disputes, ...) as CSV or Parquet")
    s.add_argument("run")
    s.add_argument("-o", "--out", required=True, help="directory to write the files to")
    s.add_argument("--tables", type=_tables, default=None, metavar="T1,T2,...",
                   help="comma-separated (default: exposures,edges_exposure_citation,adoptions,disputes,a1_by_family)")
    s.add_argument("--format", choices=["csv", "parquet"], default="csv")
    s.add_argument("--mapping", choices=["merged", "strict"], default="merged")
    s.add_argument("--all-columns", action="store_true",
                   help="exposures and edges_exposure_citation with every column, not only those of the results/ mirrors")

    s = sub.add_parser("synth", help="generate a synthetic swarm transcript with known provenance")
    s.add_argument("-o", "--out", default="data/synth/transcript.jsonl")
    s.add_argument("--agents", type=int, default=60)
    s.add_argument("--copy-rate", type=float, default=0.7)
    s.add_argument("--copy-from", choices=["latest", "first", "random"], default="latest",
                   help="which earlier post a copying agent takes its value from")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--copy-ramp-h", type=_hours, default=None, metavar="H",
                   help="copy probability grows linearly with the hours the answer has been public, reaching "
                        "--copy-rate at H hours (default: constant)")
    s.add_argument("--format", choices=["chat", "autogen"], default="chat",
                   help="autogen: an untimed AutoGen message stream with each computed answer's own tool call")
    s.add_argument("--delete-rate", type=float, default=0.0, metavar="P",
                   help="share of posts deleted (a deleted_at 15 min to 6 h after posting); a deleted post leaves the board")
    s.add_argument("--technique", default=None, metavar="TOKEN",
                   help="a token such as TRICK-A that one agent uses from its first post and every agent of its family "
                        "once it has read a post carrying it (truth_techniques.csv, and a techniques pattern in config.json)")
    s.add_argument("--boards", type=_positive, default=1, metavar="N",
                   help="boards per family: each agent posts on its own and reads them all (cross-page hops)")
    s.add_argument("--alias-rate", type=float, default=0.0, metavar="P",
                   help="share of posts signed with the agent's variant handle (truth gains true_agent)")
    s.add_argument("--confound", type=float, default=0.0, metavar="P",
                   help="share of agents, the latest to start, that never copy and answer half their questions "
                        "instantly, whatever is on the board (a null with outcome variance)")

    a = p.parse_args(argv)
    fields = {k: v for d in getattr(a, "field", []) for k, v in d.items()}
    if a.cmd == "trace" and ((a.item is not None and a.agent is None)
                             or (a.family is not None and a.agent is None and not a.list)):
        p.error("trace: --item goes with --agent, --family with --agent or --list")
    if a.cmd == "run":
        pipeline.run_all(a.input, a.out, a.adapter, a.llm, a.config, a.roster, fields, gold=a.gold)
    elif a.cmd == "ingest":
        pipeline.ingest(a.input, RunDir(a.out), a.adapter, a.config, a.roster, fields)
    elif a.cmd == "inspect":
        from . import inspect
        return inspect.inspect(a.input, a.config, fields)
    elif a.cmd == "extract":
        pipeline.extract(RunDir(a.run), llm=a.llm)
    elif a.cmd == "expose":
        pipeline.expose(RunDir(a.run))
    elif a.cmd == "graph":
        pipeline.build_graph(RunDir(a.run))
    elif a.cmd == "report":
        from . import report
        report.build(RunDir(a.run), mapping=a.mapping, boot=a.boot)
    elif a.cmd == "sample-gold":
        from . import validate
        validate.sample(RunDir(a.run), a.n, a.seed, a.exclude, a.name)
    elif a.cmd == "validate":
        from . import validate
        res = validate.score(RunDir(a.run), a.gold, a.out_name)
        if res is None:
            print(f"none of the labelled posts in {a.gold} is in {a.run}: nothing to score", file=sys.stderr)
            return 1
        print(json.dumps(res, indent=2))
    elif a.cmd == "check":
        from . import checks
        return checks.main(RunDir(a.run), a.mapping)
    elif a.cmd == "truth-score":
        from . import truthscore
        return truthscore.main(RunDir(a.run), a.truth, a.mapping)
    elif a.cmd == "crosssite":
        from .crosssite_pipeline import run_crosssite
        run_crosssite(a.wiki_run, a.corpus, a.out)
    elif a.cmd == "compare":
        from . import compare
        return compare.main(a.run_a, a.run_b, a.mapping_a, a.mapping_b, a.out, a.limit, a.tol)
    elif a.cmd == "trace":
        from . import query
        return query.trace(a.run, agent=a.agent, item=a.item, family=a.family, answer=a.answer, fact=a.fact,
                           event=a.event, mapping=a.mapping, json_path=a.json, png_path=a.png, limit=a.limit)
    elif a.cmd == "agent":
        from . import query
        return query.agent(a.run, a.name, mapping=a.mapping, json_path=a.json)
    elif a.cmd == "export":
        from . import export
        return export.export(a.run, a.out, a.tables or export.DEFAULT, a.format, a.mapping, a.all_columns)
    elif a.cmd == "synth":
        from . import synth
        synth.write(a.out, n_agents=a.agents, copy_rate=a.copy_rate, seed=a.seed, copy_from=a.copy_from, fmt=a.format,
                    copy_ramp_h=a.copy_ramp_h, delete_rate=a.delete_rate, technique=a.technique, n_boards=a.boards,
                    alias_rate=a.alias_rate, confound=a.confound)
    return 0


if __name__ == "__main__":
    sys.exit(main())
