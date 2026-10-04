# AgentSwarms — *Copied or computed?*

`swarmprov` turns a raw multi-agent transcript into a **provenance graph** (who learned what, from whom, when).
From that graph it measures:
- how much of a swarm's knowledge was computed independently and how much was relayed;
- how fast techniques and errors spread;
- which agents were the hubs.

The demo dataset is the OpenAI wiki swarm (the German message board): 14,591 wiki revisions, which become 15,987 posts. Chat transcripts such as AI Village, and any JSON/JSONL message log, go through the same pipeline via the chat adapter.

**What it found on the wiki swarm** (full report: [`results/openai-wiki/report.md`](results/openai-wiki/report.md); every number is computed from the dump, and every share is a floor because only captured surfaces are searched):

| question | answer |
|---|---|
| How much was copied? | **93% of 1,108 answers were already public before the agent reported them**; at most 83 were independent lookups. Median head start 11.6 h; in 19% of rounds the agent had posted the answer itself before the question arrived. |
| Who supplied it? | **The top 10 agents were first public source for 73% of exposed answers** (out-degree Gini 0.69). Facts repeated by two or more agents travelled 2.0 hops on average, up to 10. |
| How fast did tricks spread? | The blob-hostname sandbox bypass reached **23% of at-risk OECD agents, a median 0.8 h after first exposure**; the `markdown.new` fetch proxy reached 6 other sites within a day (median 5.8 h to adoption), while every task-level trick stayed on one wiki. |
| Did errors win? | The cracked-seed forecast **"Maryland" was carried by 30 agents before the observed "Montana" (9) appeared**; the padded `16.40` lost to the live `16.38` within the first 3 h bin, but `9.90` beat the correct `9.91` 69 agents to 17 and never lost. |
| Does availability *cause* copying? | **Not identified in this swarm**: the answer was public before almost every report, so only 3–37 agents ever switch exposure status. On a synthetic swarm with known copying the same estimator recovers the true effect (0.70 vs 0.70), so the null is the data, not the method. |

```
transcript ──adapter──▶ events ──rules/LLM──▶ claims ──▶ mentions ──▶ exposure table ──▶ A1–A6 + report
 (wiki dump,            (posts)   (answers,            (every       (agent-round:        (figures,
  chat JSON)                       predictions,         item+value   was it public?)      markdown,
                                   round reports)       in public)                        GraphML)
```

## Quick start

**Install:**
```bash
pip install -e .            # pandas, pyarrow, networkx, matplotlib, pyfixest
pip install -e '.[llm]'     # optional: Claude-based extraction
pip install -e '.[hf]'      # optional: hf:// dataset URIs
```

**Wiki dump** (directory or .zip):
```bash
swarmprov run data/raw -o runs/wiki           # ≈1 min; writes runs/wiki/report.md
swarmprov run full-wiki-logs.zip -o runs/wiki # zip works too
```

**Chat transcript** (AI Village, Slack, Discord, framework logs):
```bash
swarmprov run village-transcript.json -o runs/village --adapter chat --config my_config.json
swarmprov run hf://datasets/aidigestorg/ai-village/chat_messages.jsonl.gz -o runs/village --adapter chat
```

**AI Village tables** (directory with agents, chat_rooms, agent_goals, etc.):
```bash
swarmprov run data/raw_village -o runs/village
swarmprov run data/village/agent_goals.jsonl -o runs/village_goals  # roster only
```

**Cross-site analysis** (wiki + corpus):
```bash
swarmprov run data/raw2 -o runs/corpus                      # corpus alone
swarmprov crosssite runs/wiki data/raw2 -o runs/crosssite   # wiki + corpus: technique spread
```

**Swarm traces** (redacted payload release: reconstruction corpus, no clock/identity):
```bash
swarmprov run redacted.jsonl.gz -o runs/swarmtraces
```

**Synthetic swarm** (known provenance, used by tests):
```bash
swarmprov synth -o data/synth/transcript.jsonl --agents 60 --seed 2
swarmprov run data/synth/transcript.jsonl -o runs/synth --config data/synth/config.json
```

Stages can be rerun on their own; each one reads and writes Parquet tables in the run directory:

| command | reads → writes |
|---|---|
| `swarmprov ingest INPUT -o RUN [--roster GOALS]` | source → `events`, `lifecycle`, `roster`, `profile.json` |
| `swarmprov extract RUN [--llm claude-haiku-4-5]` | `events` (+ `roster`) → `agents`, `claims`, `tags`, `roster_events`, `items.json` |
| `swarmprov expose RUN` | → `mentions`, `exposures_{merged,strict}` |
| `swarmprov graph RUN` | → `edges_*`, `chains_*`, `provenance_merged.graphml` |
| `swarmprov report RUN [--mapping strict]` | → `report.md`, `summary.json`, `figures/` |
| `swarmprov sample-gold RUN -n 100` / `swarmprov validate RUN labels.jsonl` | hand-label sample / precision–recall |
| `swarmprov crosssite WIKI_RUN CORPUS_DIR -o RUN` | combined events → `technique_spread`, `technique_site_summary`, timeline, coverage bounds |

## Where the data comes from

The raw inputs are git-ignored (they are large and carry their publishers' terms); fetch them into the paths below and every command above runs as written. Only the small AI Village side tables (`data/raw_village`) and the roster (`data/village`) are committed.

| dataset | get it from | put it in | notes |
|---|---|---|---|
| OpenAI wiki swarm / German message board (`revisions`, `events`, `pages`, `labels`, `manifest`) | [collusion.wiki/explorer/download](https://collusion.wiki/explorer/download/) | `data/raw/` (or keep the `.zip`) | `SHA256SUMS` ships with it; `swarmprov run data/raw -o runs/wiki`. Redistribution rights unconfirmed, so do not commit it. |
| Cross-site batch (`records`, `links`, `shortener-logs`, `other-wikis`, coverage CSVs) | same download page | `data/raw2/` | `swarmprov run data/raw2 -o runs/corpus`; `swarmprov crosssite runs/wiki data/raw2 -o runs/crosssite` |
| AI Village transcript database (`chat_messages`, `agents`, `chat_rooms`, `agent_goals`, `village_goals`, `villages`, `claude_code_sessions`, `summaries`) | [huggingface.co/datasets/aidigestorg/ai-village](https://huggingface.co/datasets/aidigestorg/ai-village) (gated: request access, approved by hand) | `data/raw_village/` or read directly with `hf://datasets/aidigestorg/ai-village/<file>` | Research terms: no training, no re-identification, attribute AI Village. The side tables here are committed; the message table is not, so the committed Village reports are context reports. |
| Swarm traces redacted payload release (`redacted.jsonl.gz`, 189,579 records) | [swarmtraces.org/viewer](https://swarmtraces.org/viewer/) → "Download dataset" (`/data/final/redacted.jsonl.gz`) | anywhere | A reconstruction corpus with no clock and no agent identity; `swarmprov run redacted.jsonl.gz -o runs/swarmtraces` writes a reconstruction report, not a provenance report. Check the viewer page for terms before redistributing. |
| Synthetic swarm with known provenance | generated: `swarmprov synth -o data/synth/transcript.jsonl --agents 60 --seed 2` | `data/synth/` | writes `transcript.jsonl`, `truth.csv` and `config.json`; the tests build their own copy |

## Results on the OpenAI wiki swarm

The full report, with all tables and figures, is in [`results/openai-wiki/report.md`](results/openai-wiki/report.md).

| | finding |
|---|---|
| **A1 Provenance** | **1,108 answers by 617 agents; at most 83 were independent lookups** (104 under `D_visible`, which drops the 21 exposed answers whose only public copies had been deleted before the report; 171 were preceded by a deleted copy, but for 150 of them another copy was still visible). For 93% of answers, the same item and value were already on the wiki before the agent reported answering; for 83%, at least an hour before. The median head start was 11.6 h. In 19% of rounds the agent had posted the answer itself before the question arrived. These are floors: only captured surfaces are searched (see coverage bounds below). |
| **A2 Does availability cause copying?** | **Not identified in this swarm.** The consensus value was public before almost every report, so only 3–37 agents ever switch between exposed and unexposed rounds. The fixed-effects estimates are about zero with wide intervals, and the tool reports *"not identified"* when fewer than 10 agents switch. On a synthetic swarm with known copying the same estimator recovers the true effect (0.70 vs 0.70 true), so the null result comes from the data, not the method. |
| **A3 Diffusion** | The **blob-hostname bypass** was first posted 2026-06-20 05:10 UTC by the Mar30 OECD agent. 17 of 73 at-risk OECD agents (23%) adopted it, a median of 0.8 h after first exposure. The *live-tooltip value* it enabled reached 35 of 74 (47%); Kaplan–Meier median 5.7 h. Heartbeat beacons spread slowly across the whole swarm (96 of 1,114 agents, median 81 h among adopters). |
| **A4 Errors** | **Cracked-seed G5 forecast**: "Maryland" (RNG prediction) was carried by 30 agents before the observed **Montana** (9 agents) appeared. Montana took the majority of *new* carriers 12 h later. **OECD precision fight**: 16.40→**16.38** (Poland) and 14.60→**14.59** (Slovak Republic) overtook within the first 3-hour bin after the tooltip evidence. **9.90→9.91 (Hungary) never won**: 69 agents kept 9.90, against 17. |
| **A5 Structure** | 49 agents were the first public source for someone else's answer. **The top 10 supplied 73%** of all exposed answers (out-degree Gini 0.69). Facts repeated by ≥2 agents travelled 2.0 hops on average, with a maximum of 10. |
| **A6 Reach** | Of 365 task values, 159 were repeated by other agents (mean 3.6 repeaters, p90 10). Of 23,560 URLs, 5,373 were repeated. |

### Cross-site results (wiki run + corpus, [`results/crosssite/report.md`](results/crosssite/report.md))

17,283 posts across 20 surfaces (15,987 from the wiki dump, 871 timed corpus posts after dropping the corpus's redacted copies of wikis already in the dump, plus the rmn.re shortener log and the sandbox wikis).

| finding | detail |
|---|---|
| **Fetch-proxy tricks crossed platforms in hours; task tricks never did.** | `markdown.new` first appeared on dse at 09:50 on May 26 and was on **6 other surfaces within 7 h** (user/milk 4.1 h, ludism 4.7, paste.linuxiarz 5.4, probier 5.8, fractal 5.9, wiki4d 6.8). The `allorigins` proxy originated on the **pastebin**, two days before it reached dse. All six task-level techniques (blob bypass, live tooltip, heartbeat beacons, seed cracking, clock.wait, ZZZ pages) stayed on dse. |
| **The earliest proxy use predates the main wiki by nine days.** | A May 17 publictestwiki sandbox edit already chains three proxies (Facebook redirect → web2md → `r.jina.ai`). |
| **Three activity waves.** | Sandbox tests from May 11; the main wiki from May 24 with a May 26 burst on every surface; the June 16–22 spike (peak 4,834 dse posts on June 18) with the shortener burst (235 links on June 18) and the pastebin "Iowa" channel (119 posts on June 16). |
| **Coverage bounds.** | Of 143 surfaces the collectors inventoried, 76 were never searched or yielded no agent text, 23 relays and 23 shorteners were deliberately not visited, and 491 Discord links point into these sites with no Discord messages captured. Every exposure share above is a floor. |

Running the standard pipeline on the corpus alone (`results/corpus/`) exercises the generic path on an unsigned, mixed-format source: 1,296 posts, 126 author strings, 15 answer claims (the pastebin channel is unsigned, so per-agent provenance is not possible there), and 1,569 URL facts for the reach analysis.

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
  "family_from_roster": true,
  "roster_aliases": {"Claude Opus 4.5": "169ea37e-c664-4012-acba-cb583aaab1f3"},
  "techniques": [{"name": "shared-doc", "pattern": "docs\\.google\\.com"}],
  "disputes": [{"family": "fundraiser", "slot": "total raised", "variants": {"$1,481": "1,?481"}}]
}
```

**Capabilities** — Detected and reported: wall clock, explicit authors, read logs, deletions, threading, presence logs, request logs. Analyses needing missing capabilities are skipped or labelled.

**Lifecycle** — Not just recorded, but used. From wiki events: deletions (with cycle tracking), recreations, and probes. The report adds clock-quality notes, deletion/recreation blocks, probing blocks, and exposure gains `D_visible` (ignoring deleted copies).

**Items and rounds** — Learned from the transcript: phrases following round markers become items (no manual task list needed). US states and countries are built in. Round sequences learned from chains like `MA → CT → MI → WV`.

**AI Village tables** — The `village` adapter recognises: `agents`, `chat_rooms`, `agent_goals`, `village_goals` (eras), `villages` (metadata), message tables, activity logs (`claude_code_sessions`), and `summaries`. Posts get agent/room names, rooms give channel lifecycle, and allow/deny lists become **audiences** for per-audience exposure judgement. Set `schedule_tz` in config for timezone-aware post timing.

**Rosters** — Replace guessed identities and families. Any table with agent id + role/goal + start time works (JSON, JSONL, CSV, TSV). Family precedence: agent's own goal → text/channel families → shared era goal. Report shows batches, comparable tasks, goal changes, and coverage gaps.

**URLs as facts** — Indexed for reach and relay chains on free-form chat without task structure.

**Optional LLM extraction** — `--llm claude-haiku-4-5` uses structured outputs and caches by event id. LLM handles semantics, rules fill in numbers.

## Design notes and limitations

- **Unit of authorship.** A wiki revision stores the whole page, so the unit is the *paragraph a revision added*. A paragraph counts as new the first time its normalised text appears on the page since the page was last deleted. Line diffs would over-attribute, because repeated mis-decoding re-encodes every line on each save. The author is the trailing `-- Signature`, not the account that saved the page.
- **Exposure is inferred.** The dump contains saves, deletions and probes but **no page views**, so "public" does not mean "read". `t_report` is an upper bound on arrival; the ≥1 h variant is robust to report lag. Only wiki timestamps are used as a clock, because the agents' task clocks are dilated.
- **The causal treatment `D_cons`** asks whether the *consensus* value was public, regardless of what this agent answered. A first version used the agent's own value, which mechanically ties wrong answers to "unexposed" and produced a spurious +0.43 effect.
- **Identity.** `merged` = (cohort date, task family), where the cohort date comes from names like `OpenAIOECDNov27` or the post's opening token. `strict` = the signature itself. Every headline number is reported under both.
- **Relay edges are plausible paths, not proven ones.** Each carrier is linked to the latest earlier carrier on the same page, and to the originator if there was none.

## Repository layout

| path | contents |
|---|---|
| `swarmprov/adapters/` | Adapter interface + implementations (wiki, chat, corpus, swarmtraces, roster, village) |
| `swarmprov/` | Core modules: segment, identity, roster, village, lifecycle, rules, claims, exposure, graph, extract_llm, validate, synth, report, pipeline, cli, plotting, remote |
| `swarmprov/analysis/` | A1–A6 analyses: provenance, causal, diffusion, errors, structure, crosssite |
| `tests/` | 125 tests (≈50 s): rules, segmentation, adapters, roster, village, wiki events, LLM merge, synthetic recovery |
| `validation/` | Labels (event ids) for dev, held-out, and test splits |
| `results/` | Committed reports: openai-wiki, crosssite, corpus, synthetic, village-goals, village-context |
| `data/village/` | AI Village roster (33 goal assignments) |
| `data/raw_village/` | AI Village side tables: agents, chat_rooms, agent_goals, villages, claude_code_sessions, summaries |
| `docs/` | PLAN.md, DATA.md |

**Git-ignored:** `data/raw/` (wiki dump), `data/raw2/` (corpus), run directories. See *Where the data comes from* for download links. Small AI Village exports in `data/raw_village/` are committed.

Run tests: `python -m pytest -q`
