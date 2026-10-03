# AgentSwarms — *Copied or computed?*

`swarmprov` turns a raw multi-agent transcript into a **provenance graph** (who learned what, from whom, when).
From that graph it measures:
- how much of a swarm's knowledge was computed independently and how much was relayed;
- how fast techniques and errors spread;
- which agents were the hubs.

The demo dataset is the OpenAI wiki swarm: 14,591 wiki revisions, which become 15,987 posts. Chat transcripts such as AI Village, and any JSON/JSONL message log, go through the same pipeline via the chat adapter.

```
transcript ──adapter──▶ events ──rules/LLM──▶ claims ──▶ mentions ──▶ exposure table ──▶ A1–A6 + report
 (wiki dump,            (posts)   (answers,            (every       (agent-round:        (figures,
  chat JSON)                       predictions,         item+value   was it public?)      markdown,
                                   round reports)       in public)                        GraphML)
```

## Quick start

```bash
pip install -e .            # pandas, pyarrow, networkx, matplotlib, pyfixest
# optional: pip install -e '.[llm]'  for Claude-based extraction

# the wiki dump: a directory or the .zip with revisions/events/pages/labels.jsonl
swarmprov run data/raw -o runs/wiki                    # ≈1 min; writes runs/wiki/report.md
swarmprov run full-wiki-logs.zip -o runs/wiki          # zip works too

# any chat transcript (AI Village village-transcript.json, Slack/Discord exports, framework logs)
swarmprov run village-transcript.json -o runs/village --adapter chat --config my_config.json

# synthetic swarm with known provenance (used by the tests)
swarmprov synth -o data/synth/transcript.jsonl --agents 60 --seed 2
swarmprov run data/synth/transcript.jsonl -o runs/synth --config data/synth/config.json
```

Stages can be rerun on their own; each one reads and writes Parquet tables in the run directory:

| command | reads → writes |
|---|---|
| `swarmprov ingest INPUT -o RUN` | source → `events`, `lifecycle`, `profile.json` |
| `swarmprov extract RUN [--llm claude-haiku-4-5]` | `events` → `agents`, `claims`, `tags`, `items.json` |
| `swarmprov expose RUN` | → `mentions`, `exposures_{merged,strict}` |
| `swarmprov graph RUN` | → `edges_*`, `chains_*`, `provenance_merged.graphml` |
| `swarmprov report RUN [--mapping strict]` | → `report.md`, `summary.json`, `figures/` |
| `swarmprov sample-gold RUN -n 100` / `swarmprov validate RUN labels.jsonl` | hand-label sample / precision–recall |

## Results on the OpenAI wiki swarm

The full report, with all tables and figures, is in [`results/openai-wiki/report.md`](results/openai-wiki/report.md).

| | finding |
|---|---|
| **A1 Provenance** | **1,108 answers by 617 agents; only 83 were independent lookups.** For 93% of answers, the same item and value were already on the wiki before the agent reported answering; for 83%, at least an hour before. The median head start was 11.6 h. In 19% of rounds the agent had posted the answer itself before the question arrived. |
| **A2 Does availability cause copying?** | **Not identified in this swarm.** The consensus value was public before almost every report, so only 3–37 agents ever switch between exposed and unexposed rounds. The fixed-effects estimates are about zero with wide intervals, and the tool reports *"not identified"* when fewer than 10 agents switch. On a synthetic swarm with known copying the same estimator recovers the true effect (0.70 vs 0.70 true), so the null result comes from the data, not the method. |
| **A3 Diffusion** | The **blob-hostname bypass** was first posted 2026-06-20 05:10 UTC by the Mar30 OECD agent. 17 of 73 at-risk OECD agents (23%) adopted it, a median of 0.8 h after first exposure. The *live-tooltip value* it enabled reached 35 of 74 (47%); Kaplan–Meier median 5.7 h. Heartbeat beacons spread slowly across the whole swarm (96 of 1,114 agents, median 81 h among adopters). |
| **A4 Errors** | **Cracked-seed G5 forecast**: "Maryland" (RNG prediction) was carried by 30 agents before the observed **Montana** (9 agents) appeared. Montana took the majority of *new* carriers 12 h later. **OECD precision fight**: 16.40→**16.38** (Poland) and 14.60→**14.59** (Slovak Republic) overtook within the first 3-hour bin after the tooltip evidence. **9.90→9.91 (Hungary) never won**: 69 agents kept 9.90, against 17. |
| **A5 Structure** | 49 agents were the first public source for someone else's answer. **The top 10 supplied 73%** of all exposed answers (out-degree Gini 0.69). Facts repeated by ≥2 agents travelled 2.0 hops on average, with a maximum of 10. |
| **A6 Reach** | Of 365 task values, 159 were repeated by other agents (mean 3.6 repeaters, p90 10). Of 23,560 URLs, 5,373 were repeated. |

### Validation of the extractor

Labels were made by Claude reading posts (`validation/*.jsonl`, event ids plus labels). They should be **replaced or confirmed by human labellers** before anyone cites them. There are three samples:

| split | posts | answer-post P / R | item | value | round | latency |
|---|---|---|---|---|---|---|
| dev (used for tuning) | 60 | 1.00 / 0.97 | 1.00 | 1.00 | 0.97 | 1.00 |
| held-out, scored **before** fixes, then used for tuning | 48 | 1.00 / 1.00 | 0.86 | 0.86 | 0.86 | 0.78 |
| **untouched test** (labelled after the last extractor change) | 40 | **1.00 / 0.91** | **0.96** | **1.00** | **0.96** | **1.00** (n=9) |

The misses are answers reported without the verb "answered", e.g. *"our submission was wrong"* or *"answer receipt 14:27:23"*.

## How it generalises

All source-specific knowledge lives in one adapter plus an `AdapterConfig`, which can be overridden with `--config file.json|toml`:

```json
{
  "name": "village",
  "families": {"fundraiser": "donat|fundrais", "event": "venue|RSVP"},
  "family_from_channel": false,
  "techniques": [{"name": "shared-doc", "pattern": "docs\\.google\\.com", "description": "..."}],
  "disputes": [{"family": "fundraiser", "slot": "total raised",
                "context": "raised|total", "variants": {"$1,481": "1,?481", "$2,000": "2,?000"}}]
}
```

- **Capabilities are detected and reported**: wall clock, explicit authors, read logs, deletions, threading. An analysis that needs a missing capability is skipped or labelled.
- **Items are learned from the transcript.** Phrases that follow a round marker in posts by ≥2 authors become items, so no task list has to be written by hand. US states and countries are built in.
- **Round sequences are learned from chains** like `MA -> CT -> MI -> WV`.
- **URLs are indexed as facts**, so reach and relay chains work on free-form chat with no task structure.
- **Optional LLM extraction** (`--llm claude-haiku-4-5`) uses structured outputs and caches results by event id. The LLM wins on semantic fields and the rules fill in numbers.

## Design notes and limitations

- **Unit of authorship.** A wiki revision stores the whole page, so the unit is the *paragraph a revision added*. A paragraph counts as new the first time its normalised text appears on the page since the page was last deleted. Line diffs would over-attribute, because repeated mis-decoding re-encodes every line on each save. The author is the trailing `-- Signature`, not the account that saved the page.
- **Exposure is inferred.** The dump contains saves, deletions and probes but **no page views**, so "public" does not mean "read". `t_report` is an upper bound on arrival; the ≥1 h variant is robust to report lag. Only wiki timestamps are used as a clock, because the agents' task clocks are dilated.
- **The causal treatment `D_cons`** asks whether the *consensus* value was public, regardless of what this agent answered. A first version used the agent's own value, which mechanically ties wrong answers to "unexposed" and produced a spurious +0.43 effect.
- **Identity.** `merged` = (cohort date, task family), where the cohort date comes from names like `OpenAIOECDNov27` or the post's opening token. `strict` = the signature itself. Every headline number is reported under both.
- **Relay edges are plausible paths, not proven ones.** Each carrier is linked to the latest earlier carrier on the same page, and to the originator if there was none.

## Repository layout

```
swarmprov/
  adapters/   base.py (Adapter, AdapterConfig, Capabilities)  wiki.py  chat.py
  segment.py  identity.py  rules.py  gazetteer.py  claims.py  exposure.py  graph.py
  extract_llm.py  validate.py  synth.py  report.py  pipeline.py  cli.py  plotting.py
  analysis/   provenance.py (A1)  causal.py (A2)  diffusion.py (A3)  errors.py (A4)  structure.py (A5, A6)
tests/        rules, segmentation, chat adapter, LLM merge (fake client), synthetic end-to-end recovery
validation/   labels (event ids only) for the three validation splits
results/      the committed report + figures for the wiki dump
docs/         PLAN.md (general pipeline plan), DATA.md (what the dump actually contains)
```

Run the tests with `python -m pytest -q` (24 tests, about 10 s). The raw data and run directories are git-ignored.
