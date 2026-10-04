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
# optional: pip install -e '.[llm]'  for Claude-based extraction;  '.[hf]'  for hf:// inputs

# the wiki dump: a directory or the .zip with revisions/events/pages/labels.jsonl
swarmprov run data/raw -o runs/wiki                    # ≈1 min; writes runs/wiki/report.md
swarmprov run full-wiki-logs.zip -o runs/wiki          # zip works too

# any chat transcript (AI Village, Slack/Discord exports, framework logs); hf:// URIs are fetched first
swarmprov run village-transcript.json -o runs/village --adapter chat --config my_config.json
swarmprov run hf://datasets/aidigestorg/ai-village/chat_messages.jsonl.gz -o runs/village --adapter chat

# AI Village: a directory holding the exported tables (agents, chat_rooms, agent_goals, chat_messages ...)
# is one source: names from `agents`, room names + lifecycle + audiences from `chat_rooms`,
# families + cohorts from `agent_goals`; without a message table it writes a context report
swarmprov run data/raw_village -o runs/village
swarmprov run data/village/agent_goals.jsonl -o runs/village_goals          # a roster on its own
swarmprov run hf://datasets/aidigestorg/ai-village/chat_messages.jsonl.gz -o runs/village \
    --adapter chat --roster hf://datasets/aidigestorg/ai-village/agent_goals.jsonl.gz

# the cross-site batch (records.jsonl, links.jsonl, shortener-logs.json, other-wikis.json, coverage CSVs)
swarmprov run data/raw2 -o runs/corpus                 # standard pipeline on the corpus alone (reach, techniques)
swarmprov crosssite runs/wiki data/raw2 -o runs/crosssite   # wiki run + corpus: technique spread between surfaces

# synthetic swarm with known provenance (used by the tests)
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

## Results on the OpenAI wiki swarm

The full report, with all tables and figures, is in [`results/openai-wiki/report.md`](results/openai-wiki/report.md).

| | finding |
|---|---|
| **A1 Provenance** | **1,108 answers by 617 agents; at most 83 were independent lookups** (104 under `D_visible`, which discounts the 171 exposed answers whose cited copy had been deleted before the report; for 150 of those another copy was still visible). For 93% of answers, the same item and value were already on the wiki before the agent reported answering; for 83%, at least an hour before. The median head start was 11.6 h. In 19% of rounds the agent had posted the answer itself before the question arrived. These are floors: only captured surfaces are searched (see coverage bounds below). |
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
  "techniques": [{"name": "shared-doc", "pattern": "docs\\.google\\.com", "description": "..."}],
  "disputes": [{"family": "fundraiser", "slot": "total raised",
                "context": "raised|total", "variants": {"$1,481": "1,?481", "$2,000": "2,?000"}}]
}
```

- **Capabilities are detected and reported**: wall clock, explicit authors, read logs, deletions, threading, presence logs, request logs. An analysis that needs a missing capability is skipped or labelled.
- **Lifecycle is used, not just recorded.** From the wiki dump's events table the adapter keeps every deletion (with whether the page ever had a published revision and its delete/recreate cycle), one `recreate` row per first-recreation edge, and the script-injection probes as a `probes` table. The report adds a clock-quality note (grades and uncertainty of post timestamps, how many exposure gaps sit inside the clock resolution), a *Deletions and recreations* block (sweeps, re-saves after deletion, restored vs fresh text), a *Probing* block (co-timing with saves under a strict rule, never attribution), and a validation row comparing the source's recreation edges with the pipeline's own rule. Exposure gains `D_visible`, which ignores copies deleted before the report, reported beside `D` and never replacing it; relay edges inside the 2 s clock resolution are flagged as direction-free.
- **Items are learned from the transcript.** Phrases that follow a round marker in posts by ≥2 authors become items, so no task list has to be written by hand. US states and countries are built in.
- **Round sequences are learned from chains** like `MA -> CT -> MI -> WV`.
- **The AI Village side tables are one source.** The `village` adapter takes a directory and recognises tables by their columns: `agents` (id → name, model, join date), `chat_rooms` (room id → name, created/deleted, `whitelisted_agent_names` / `blacklisted_agent_names`) and `agent_goals`, plus `village_goals` (the shared goal everyone was given, one contiguous window per week → the `eras` table), any message table, and presence logs such as `claude_code_sessions` (agent_id + created_at → the `activity` table). Posts get agent names and room names; rooms give channel lifecycle; a room's allow/deny list is its **audience**, and exposure is judged per audience: an answer posted only where an agent could not read it is not "public" for that agent (`exposure.build(..., audience=...)`). The report's Village block lists the directory by vendor, every room with its lifetime and access, what the activity log covers (and says so when it covers none of the agents under study), and any file in the directory no adapter recognised.
- **A roster replaces guessed identities and families.** AI Village `agent_goals` (or any table with an agent id, a role or goal, and a start time; JSON, JSONL, CSV or TSV, so a spreadsheet export works) is detected by the `roster` adapter. Timestamps a spreadsheet has reduced to `37:49.4` are reported as unreadable rather than parsed as a time today. Attached with `--roster`, each post is matched to a roster agent (by `agent_id` in the transcript, a `roster_aliases` entry, or the role name) and to the goal window it falls in; the goal becomes the post's task family, the goal's assignment batch its cohort, and two agents given the same goal stay distinct. Family precedence is: the agent's own goal while it was in force, then text or channel families from the config, then the shared goal (era) of the post's time, so a post before an agent's goal started is not back-dated onto it. The report gains a Roster block: batches, roles held by several agents (the comparable tasks), goal changes (reworded vs reassigned), and coverage (unmatched authors, posts outside every goal window, goals with no posts).
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
  adapters/   base.py (Adapter, AdapterConfig, Capabilities)  wiki.py  chat.py  corpus.py (records/shortener/other-wikis)
              roster.py (agent_goals tables)  village.py (a directory of AI Village tables)
  segment.py  identity.py  roster.py (goal windows -> families, cohorts)  village.py (directory, rooms, audiences)
  lifecycle.py (deletion sweeps, recreation check, probes)  rules.py  gazetteer.py  claims.py  exposure.py  graph.py
  extract_llm.py  validate.py  synth.py  report.py  pipeline.py  cli.py  plotting.py  remote.py (hf:// inputs)
  analysis/   provenance.py (A1)  causal.py (A2)  diffusion.py (A3)  errors.py (A4)  structure.py (A5, A6)
              crosssite.py (technique spread between surfaces, timeline, coverage bounds)
  crosssite_pipeline.py   the `crosssite` stage
tests/        rules, segmentation, chat adapter, roster, village tables + audience exposure, wiki events (recreations, probes, D_visible), LLM merge (fake client), synthetic end-to-end recovery
validation/   labels (event ids only) for the three validation splits
results/      committed reports + figures: openai-wiki/, crosssite/, corpus/, synthetic/, village-goals/, village-context/
data/village/ agent_goals.jsonl (AI Village roster, 33 goal assignments); agent_goals_sheet.tsv (the same after a spreadsheet round-trip)
data/raw_village/  AI Village side tables as exported: agents, chat_rooms, agent_goals, village_goals, claude_code_sessions (.jsonl.gz)
docs/         PLAN.md (general pipeline plan), DATA.md (what the dump actually contains)
```

Run the tests with `python -m pytest -q` (101 tests, about 30 s). The raw data and run directories are git-ignored.
