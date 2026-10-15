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
| Who supplied it? | **It depends on who gets the credit.** Credited to the first agent who posted the value, the top 10 supplied 73% of exposed answers (out-degree Gini 0.69); credited to the latest earlier carrier or the same-page relay parent, 20%. First-source credit mostly measures who answered first on an item (on 47 of 61 items one agent was first source of every exposed answer), so the report gives the range, not one number. Only 21% of exposed answers trace to a first source that computed the value itself; 63% to one with no answer round observed on the item. Facts repeated by two or more agents travelled 2.0 hops on average, up to 10. |
| How fast did tricks spread? | The blob-hostname sandbox bypass reached **23% of at-risk OECD agents, a median 0.8 h after first exposure**; the `markdown.new` fetch proxy reached 6 other sites within a day (median 5.8 h to adoption), while every task-level trick stayed on one wiki. |
| Did errors win? | The cracked-seed forecast **"Maryland" was carried by 30 agents before the observed "Montana" (11) appeared**, 12 h after the forecast; Montana overturned it as soon as it appeared (11 of the last 14 carriers). The live `16.38` overturned the rounded `16.40` within the first 3 h bin, but Hungary stays **contested**: `9.90` kept 69 agents to 17 for the live `9.91`, which led the late carriers only 16 to 13 and never held a lasting majority. |
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
| `swarmprov extract RUN [--llm claude-haiku-4-5]` | `events` (+ `roster`) → `agents`, `agent_aliases.csv`, `claims`, `tags`, `roster_events`, `items.json` |
| `swarmprov expose RUN` | → `mentions`, `exposures_{merged,strict}` |
| `swarmprov graph RUN` | → `edges_*`, `chains_*`, `provenance_merged.graphml` |
| `swarmprov report RUN [--mapping strict]` | → `report.md`, `summary.json`, `figures/` |
| `swarmprov sample-gold RUN -n 100` / `swarmprov validate RUN labels.jsonl` | hand-label sample / precision–recall |
| `swarmprov crosssite WIKI_RUN CORPUS_DIR -o RUN` | combined events → `technique_spread`, `technique_site_summary`, timeline, coverage bounds |
| `swarmprov trace RUN (--agent A [--item I] [--family F] \| --answer ID \| --fact 'F\|I[\|V]' \| --event ID \| --list)` | stage tables → one answer's sources, one fact's relay tree or one post's extraction (`--json`, `--png`); see below |

## Asking questions of a run

`swarmprov trace` reads a finished run directory (every stage up to `graph`; it names the tables a partial run lacks) and answers one question at a time. It always prints the identity mapping it uses (`--mapping merged`, the default, or `strict`).

```bash
swarmprov trace runs/wiki --agent "Mar16|oecd-equity" --item "Czech Republic"  # where did this answer come from?
swarmprov trace runs/wiki --answer d50220c9ec0f653b                              # the same answer, by claim id
swarmprov trace runs/wiki --fact "oecd-equity|Hungary|9.9" --png hungary.png     # how did this value travel?
swarmprov trace runs/wiki --fact "oecd-equity|Hungary"                           # every value posted for the item
swarmprov trace runs/wiki --event b847fc1cbc646343                               # what was extracted from this post, and why not more?
swarmprov trace runs/wiki --list --limit 50                                      # facts ranked by carriers (--family url for URLs)
```

`--agent` matches agent ids and author strings, ignoring case (`march16oecdscout` finds `Mar16|oecd-equity`). When several agents match, `--item` and `--family` keep the ones with a matching answer; if more than one is left, it lists them. `--item` works only with `--agent`, `--family` with `--agent` or `--list`. `--answer` takes an answer's claim id; a later report of the same agent-round shows the round and says so, and a claim that is not an answer (a prediction, a round sequence) names the agent's answer for the item instead. `--json FILE` also writes the result as JSON. `--png FILE` draws one fact as a timeline: x is time, each row is a carrier in order of first post, same-page relay hops are blue and start at a small dot on the parent's row (its latest earlier post on the page), and cross-page hops are grey. `--limit N` (at least 1) caps the answers or listed facts and the hops printed per fact.

An answer on the wiki run, abridged:

```
mapping: merged (one agent per cohort date and task family; --mapping strict for one per author string)
answer d50220c9ec0f653b: Mar16|oecd-equity, oecd-equity | Czech Republic = 9.7
  report post: b847fc1cbc646343, 2026-06-19 23:26:04, dse/OECDEducationEquitySequence, by March16OECDScout
  public before the report: yes (D); at least 60 min before: yes (D_w60m); a copy still visible at the report: yes (D_visible)
  first public copy: Oct04|oecd-equity, post 2a057dd82c900e32, 2026-06-17 09:46:10, dse/OECDEducationEquitySequence, by OECDEquityOct04Agent; head start 61.7 h; page deleted 2026-06-19 23:40:50 (after the report)
  own earlier post: none
  inferred relay parent: Oct04|oecd-equity (relay), parent post c86deee978462c75 (2026-06-17 09:46:10), 61.7 h before this agent's first post of the value (b847fc1cbc646343, 2026-06-19 23:26:04)
  inferred path from the originator (1 hop): Oct04|oecd-equity -> Mar16|oecd-equity
  competing values: 9.7 (27 carriers) <- this answer, 9.69 (17 carriers), ...
```

| field | meaning |
|---|---|
| report post | the post carrying the agent's first answer report for the item. Its time is `t_report`, an upper bound on when the question arrived. An answer that stated no value of its own reads `(no value; matched on the consensus V)`: exposure matched it on the item's consensus value. |
| `D`, `D_w60m`, `D_visible` | the A1 exposure flags: the value was public before the report, at least 60 min before it, and in a copy still visible (not deleted) at the report. |
| first public copy | the earliest post by another agent with the same item and value before the report, and the head start `t_report − t_public`. Copies posted in the same second are taken in event-id order. Public is not read: this is what was available, not what the agent saw. |
| first copy still visible | printed only when the first copy's page had been deleted by the report. |
| own earlier post | the agent posted the value itself before reporting it (`self_prepared`). |
| inferred relay parent | the graph's guess at the source: the latest earlier carrier of the value on the same page (`relay`). With no earlier carrier on that page the hop goes to the originator and reads *cross-page hop: source unknown, attributed to originator* (`relay_xchannel`). A hop inside the 2 s clock resolution reads *direction unreliable*. The parent post is the parent's latest post of the value on that page; the delay runs from it to this agent's first post of the value. When the parent posted only after the report it reads *parent post after the report: not this answer's source*, and no path is printed. *None* says why: the agent originated the value, or it has no item–value mention of the matched value (the extractor read its answer report but found no mention, so the relay graph has no hop for it), or it stated no value and never posted the consensus one. |
| inferred path | relay parents back to the value's originator. Its length is the hop depth A5 and A6 count. |
| competing values | the other values posted for the item, each with its number of carrying agents. The matched value is marked `<- this answer`, or `<- matched (consensus)` for an answer without a value. |

A fact (`--fact`) prints its origin post, then every relay hop as an indented tree (depth, kind, time, delay, page, post, excerpt). The counts match `chains_*`: `oecd-equity|Hungary|9.9` has 69 carriers, 68 hops (42 of them cross-page) and depth 10. Values match on the comparison key, so `9.90`, `9.9` and `9.9%` are one fact, and a multi-number value matches on its first number. A post (`--event`) shows its author under both mappings, its family and tags, the claims and item–value mentions extracted from it, the URLs it carries, and the edges and exposures that point at it. When no claim was extracted, it says why. Excerpts are one line of 160 characters around the item, with an ellipsis at a cut end; `--event` prints the post's text the same way, cut to 600. Agent text is untrusted (the wiki holds a script payload), so escape it before putting it into a web page.

The posts behind these answers are stored at expose time: `exposures_*` carries `report_event_id`, `src_event_id` and `src_channel` (the first public copy), `src_latest`, `t_latest` and `latest_event_id` (the latest copy before the report, A5's second attribution rule), `src_visible_event_id` and `self_event_id`.

## Where the data comes from

The raw inputs are git-ignored (they are large and carry their publishers' terms); fetch them into the paths below and every command above runs as written. Only the small AI Village side tables (`data/raw_village`) and the roster (`data/village`) are committed.

| dataset | get it from | put it in | notes |
|---|---|---|---|
| OpenAI wiki swarm / German message board (`revisions`, `events`, `pages`, `labels`, `manifest`) | [collusion.wiki/explorer/download](https://collusion.wiki/explorer/download/) | `data/raw/` (or keep the `.zip`) | `SHA256SUMS` ships with it; `swarmprov run data/raw -o runs/wiki`. Redistribution rights unconfirmed, so do not commit it. |
| Cross-site batch (`records`, `links`, `shortener-logs`, `other-wikis`, coverage CSVs) | same download page | `data/raw2/` | `swarmprov run data/raw2 -o runs/corpus`; `swarmprov crosssite runs/wiki data/raw2 -o runs/crosssite` |
| AI Village transcript database (`chat_messages`, `agents`, `chat_rooms`, `agent_goals`, `village_goals`, `villages`, `claude_code_sessions`, `summaries`) | [huggingface.co/datasets/aidigestorg/ai-village](https://huggingface.co/datasets/aidigestorg/ai-village) (gated: request access, approved by hand) | `data/raw_village/` or read directly with `hf://datasets/aidigestorg/ai-village/<file>` | Research terms: no training, no re-identification, attribute AI Village. The side tables here are committed; the message table is not, so the committed Village reports are context reports. |
| Swarm traces redacted payload release (`redacted.jsonl.gz`, 189,579 records) | [swarmtraces.org/viewer](https://swarmtraces.org/viewer/) → "Download dataset" (`/data/final/redacted.jsonl.gz`) | anywhere | A reconstruction corpus with no clock and no agent identity; `swarmprov run redacted.jsonl.gz -o runs/swarmtraces` writes a reconstruction report, not a provenance report. Check the viewer page for terms before redistributing. |
| Synthetic swarm with known provenance | generated: `swarmprov synth -o data/synth/transcript.jsonl --agents 60 --seed 2` | `data/synth/` | writes `transcript.jsonl`, `truth.csv` and `config.json` (which carries the true values as `reference_values`). `truth.csv` names the author of the post each copying agent took (`src_agent`); `--copy-from latest\|first\|random` picks which post that is (the default, `latest`, writes the same transcript as before the option existed). The tests build their own copy. |

## Results on the OpenAI wiki swarm

The full report, with all tables and figures, is in [`results/openai-wiki/report.md`](results/openai-wiki/report.md).

| | finding |
|---|---|
| **A1 Provenance** | **1,108 answers by 617 agents; at most 83 were independent lookups** (104 under `D_visible`, which drops the 21 exposed answers whose only public copies had been deleted before the report; 171 were preceded by a deleted copy, but for 150 of them another copy was still visible). For 93% of answers, the same item and value were already on the wiki before the agent reported answering; for 83%, at least an hour before. The median head start was 11.6 h. In 19% of rounds the agent had posted the answer itself before the question arrived. These are floors: only captured surfaces are searched (see coverage bounds below). |
| **A2 Does availability cause copying?** | **Not identified in this swarm.** The consensus value was public before almost every report, so only 3–37 agents ever switch between exposed and unexposed rounds. The fixed-effects estimates are about zero with wide intervals, and the tool reports *"not identified"* when fewer than 10 agents switch. On a synthetic swarm with known copying the same estimator recovers the true effect (0.70 vs 0.70 true), so the null result comes from the data, not the method. |
| **A3 Diffusion** | The **blob-hostname bypass** was first posted 2026-06-20 05:10 UTC by the Mar30 OECD agent. 17 of 73 at-risk OECD agents (23%) adopted it, a median of 0.8 h after first exposure. The *live-tooltip value* it enabled reached 35 of 74 (47%); Kaplan–Meier median 5.7 h. Heartbeat beacons spread slowly across the whole swarm (96 of 1,114 agents, median 81 h among adopters). |
| **A4 Errors** | Each disputed slot is **held**, **overturned** (a later value won the last third of carriers, which numbered at least 5 in all, and kept the majority of new carriers) or **contested**. **Cracked-seed G5 forecast**: "Maryland" (RNG prediction) was carried by 30 agents before the observed **Montana** (11 agents) appeared, 12 h after the forecast. Montana took the majority of *new* carriers as soon as it appeared and 11 of the last 14: overturned. **OECD precision fight**: 16.40→**16.38** (Poland, 21 of 28 late carriers) and 14.60→**14.59** (Slovak Republic, 31 of 35) were overturned within the first 3-hour bin after the tooltip evidence. **Hungary is contested**: 69 agents kept 9.90 against 17 for 9.91, which led the late carriers only 16 to 13 and never held a lasting majority, so against the live-tooltip value (analyst input) the outcome is unresolved. Of the 15 slots in the table, 9 were overturned, 2 held and 4 contested (one a 2–2 tie, one on 4 late carriers, one whose two values were first posted in the same second). |
| **A5 Structure** | **Hub concentration is reported as a range across attribution rules**, because nothing records what an agent read. Credited to the *first* agent who posted the value before the report, 49 agents supplied someone's answer and the top 10 supplied **73%** (out-degree Gini 0.69, against 0.56 for the 61 items' exposure counts alone, the value if each item had one source of its own). Credited to the *latest* earlier carrier, 327 agents and 20%; to the *same-page relay parent*, 20% of the 222 answers that have one. On instant answers only, the range is 17–80%. First-source credit mostly measures who answered first on an item: on 47 of 61 items one agent was the first source of every exposed answer. **Where copied answers came from** (first source, merged mapping): 21% from an agent that computed the value itself, 63% from one with no answer round observed on the item (67% under strict; scouts and answer lists such as `clothingsequencescout`, but possibly missed answers or identity splits, so the report lists them for audit), 7% from one that answered a different value, and 9% from one that was itself exposed; followed back, every such chain ends within two hops at an agent that had posted the value before its own first public copy. Credited to the latest carrier instead, 63% of answers have a source that was itself exposed. Facts repeated by ≥2 agents travelled 2.0 hops on average, with a maximum of 10. |
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
  "disputes": [{"family": "fundraiser", "slot": "total raised", "variants": {"$1,481": "1,?481"}}],
  "reference_values": {"fundraiser|total raised": "$1,481"}
}
```

**Capabilities** — Detected and reported: wall clock, explicit authors, read logs, deletions, threading, presence logs, request logs. The report's capability table states what each value, yes or no, means for the analyses (reply links are recorded but not yet used as edges). Analyses needing missing capabilities are skipped or labelled.

**Identity** — An explicit author field is used as given; otherwise authors are parsed from signatures. Case variants of one handle (`OpenAIResearcher`, `OpenAiResearcher`) are resolved together, so each strict id has one merged id (variants naming different cohorts stay split and are flagged in the audit). The cohort date comes from the handle (`OpenAIOECDNov27`), and from the date a handle's posts open with (`Mar16 R3 …`) only when the source has no explicit author field; set `cohort_from_text` to override. `extract` writes `agent_aliases.csv` (every author string with its strict and merged id, cohort and family, and the size of both groups; on a roster run the family is the roster role behind the merged id). The report counts the merges, lists the largest, says how many only join case variants (which both mappings fold together), and warns when they join authors that an explicit-author source keeps apart.

**Coverage** — `coverage_note` is the Limitations bullet on what the exposure search could not see. The wiki and corpus configs cite the collectors' inventory of 143 surfaces (the corpus note adds that a corpus run does not search the wiki dump); without a note, the report says that only the transcript's own posts were searched, so the exposed share is a floor.

**Disputes** — A4 finds (family, item) slots with competing values, plus any declared in `disputes`, and gives each a status: *held*, *overturned* or *contested*, with the late-carrier counts behind it. `reference_values` (`{"family|item": value}`) is analyst input: it fills a separate outcome column and never changes the status. A held or overturned slot reads truth held, error corrected, error persisted or truth displaced by error (*precision variant* when one value is the other rounded); a contested slot reads *unresolved*. A reference matches a variant by its text or its comparison key (`1,481` matches `$1,481`); one that matches no variant says so, and the report lists references that name no item or disputed slot in the run, so a mistyped key shows. The wiki config sets the live-tooltip values for the three OECD disputes; the synthetic config carries the true values.

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
- **Identity.** `merged` = (cohort date, task family), where the cohort date comes from names like `OpenAIOECDNov27` or, for signature-parsed sources, the post's opening token. `strict` = the signature itself, case-folded; case variants share one merged id. Every headline number is reported under both, and `agent_aliases.csv` lists every merge for audit.
- **A4 status is a convention.** The winner is the most common value among the last third of a slot's carriers (none on a tie). If it is the first value, the slot is *held*. A later winner counts as *overturned* only if it also held the majority of new carriers in every 3-hour bin from some point on, with at least 5 late carriers in all; otherwise the slot is *contested*. Two values first seen at the same moment leave no first value, so that slot is contested too, whichever way the labels sort. The report prints the bin width and the tail share.
- **Relay edges are plausible paths, not proven ones.** Each carrier is linked to the latest earlier carrier on the same page, and to the originator if there was none. `trace` labels the relay parent as inferred and keeps it apart from the first public copy. On the synthetic swarm, where copiers take the latest post, the relay parent names the true source for at least 98% of copied answers and the first public copy for about a quarter. When copiers take the first post (`--copy-from first`), the first public copy is right every time and the relay parent about a fifth of the time.
- **Hub concentration depends on the attribution rule.** Exposure says a value was public, not whom an agent took it from, so A5 credits each exposed answer three ways: the first other agent to post the value before the report, the latest one, and the same-page relay parent (joined on the same value key exposure matched, with the unmatched count printed). It reports all three, on every exposed answer and on instant answers only, plus the Gini of the items' exposure counts, which is what first-source credit gives if each item has a single source. On synthetic swarms (seeds 0–7) the rule that matches how the agents copy names the true source for 98–100% of copied answers and reproduces the true top-10 share to within 0.01, while the other rule names it for 19–28% and misses the share by 0.36–0.55 (first-source credit reports 64–76% where copiers take the latest post and the truth is 26–28%). With no copying at all, first-source credit still puts 68–80% of exposed answers on its top 10, and with `--copy-from random` neither rule names the source for more than 57%. So the rules are a sensitivity range: a rule that wins on the synthetic swarm only matches how the simulation copies. `analysis/origins.py` labels each credited source by its own answer round on the item (computed, no answer round observed, dissent, relayed) and follows relayed sources to a root; `summary.json` carries both as `A5.attribution` and `A5.origins`. *Relayed* means exposed, not copied, so on the copy-latest swarm those chains end at the true root for only 51–68% of copied answers.
- **Every edge names its posts.** `edges_*.event_id` is always a row of `events`: the carrier's first post of the value (relay), the answer's report post (exposure) or the citing post (citation). `src_event_id` is the source's post: the parent's latest post on the page, the originator's first post for a cross-page hop, or the first public copy. Exposure edges also carry `claim_id`. Earlier runs stored the claim id in an exposure edge's `event_id`; read `claim_id` for it now. `value_norm` holds the comparison key for every kind (the first number of a multi-number value), so edges join `chains_*.value_key`.

## Repository layout

| path | contents |
|---|---|
| `swarmprov/adapters/` | Adapter interface + implementations (wiki, chat, corpus, swarmtraces, roster, village) |
| `swarmprov/` | Core modules: segment, identity, roster, village, lifecycle, rules, claims, exposure, graph, query, extract_llm, validate, synth, report, pipeline, cli, plotting, remote |
| `swarmprov/analysis/` | A1–A6 analyses: provenance, causal, diffusion, errors, structure, origins, crosssite |
| `tests/` | 242 tests (≈100 s): rules, segmentation, adapters, identity, roster, village, wiki events, LLM merge, synthetic recovery, A4 status and outcomes, A5 brokers, A5 attribution rules and origins against synthetic truth, trace against synthetic truth, report integrity |
| `validation/` | Labels (event ids) for dev, held-out, and test splits |
| `results/` | Committed reports: openai-wiki, crosssite, corpus, synthetic, village-goals, village-context |
| `data/village/` | AI Village roster (33 goal assignments) |
| `data/raw_village/` | AI Village side tables: agents, chat_rooms, agent_goals, villages, claude_code_sessions, summaries |
| `docs/` | PLAN.md, DATA.md |

**Git-ignored:** `data/raw/` (wiki dump), `data/raw2/` (corpus), run directories. See *Where the data comes from* for download links. Small AI Village exports in `data/raw_village/` are committed.

Run tests: `python -m pytest -q`
