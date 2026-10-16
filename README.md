# AgentSwarms — *Copied or computed?*

`swarmprov` turns a raw multi-agent transcript into a **provenance graph** (who posted which value, when, and who could have taken it from whom) and measures from it how much of a swarm's knowledge was computed independently and how much relayed, how fast techniques and errors spread, and which agents were the hubs. It reads wiki dumps and chat logs (AI Village, Slack, Discord, OpenAI, LangChain or AutoGen). Nothing records what an agent read, so *public* never means *read*: every exposure share is a floor.

## What it found

**On the OpenAI wiki swarm** (the German message board: 14,591 wiki revisions, which become 15,987 posts; full report: [`results/openai-wiki/report.md`](results/openai-wiki/report.md), its key findings alone in [`headline.md`](results/openai-wiki/headline.md); every number is computed from the dump, and every share is a floor because only captured surfaces are searched):

| question | answer |
|---|---|
| How much was already public? | **93% (95% CI 88–96%) of 1,108 answers by 617 agents were already public before the agent reported them**, 83% at least an hour before, with a median head start of 11.6 h. At most 89 were independent lookups: 83 with nothing public before the report and 6 the agent itself had posted before any other copy (at most 110 under `D_visible`, which counts a copy deleted before the report as never public). |
| Who supplied it? | **It depends on who gets the credit.** The top 10 sources supplied 20–74% of the 1,019 exposed answers credited to a source: 74% credited to the first agent to post the value, 20% to the latest earlier carrier, 21% of the 218 that have a same-page relay parent; first-source credit mostly measures who first posted on an item (on 52 of 61 items one agent was first source of every exposed answer). |
| How fast did tricks spread? | The blob-hostname sandbox bypass reached **23% of at-risk OECD agents (17 of 73), a median 0.8 h after its first post**; the `markdown.new` fetch proxy reached 6 other sites within 7 h of its first post, while every task-level trick stayed on one wiki. |
| Did errors win? | The cracked-seed forecast **"Maryland" was carried by 30 agents before the observed "Montana" (11) appeared**, and Montana overturned it as soon as it appeared (11 of the last 14 carriers). Hungary stays **contested**: `9.90` kept 69 agents to 17 for the live `9.91`, which led the late carriers only 16 to 13 and never held a lasting majority. |
| Does availability *cause* copying? | **Not identified for most rows**: the answer was public before almost every report, so most rows have only 3–17 agents that switch exposure status; where 37 do, no effect is detected (95% CI −0.34 to 0.11). On a synthetic swarm with known copying the same estimator recovers the true effect (0.70 vs 0.70), so the null is the data, not the method. |

## Quick start

### Where the data comes from

The raw inputs are git-ignored (they are large and carry their publishers' terms): fetch them into the paths below and the commands that read `data/` run as written. A path that does not exist stops the run with one line pointing here. Only the small AI Village side tables (`data/raw_village`) and the roster (`data/village`) are committed.

| dataset | get it from | put it in | notes |
|---|---|---|---|
| OpenAI wiki swarm / German message board (`revisions`, `events`, `pages`, `labels`, `manifest`) | [collusion.wiki/explorer/download](https://collusion.wiki/explorer/download/) | `data/raw/` (or keep the `.zip`) | `SHA256SUMS` ships with it; redistribution rights unconfirmed, so do not commit it ([what it holds](docs/DATA.md)) |
| Cross-site batch (`records`, `links`, `shortener-logs`, `other-wikis`, coverage CSVs) | same download page | `data/raw2/` | the corpus the cross-site analysis joins to the wiki run |
| AI Village transcript database (`chat_messages` and its side tables) | [huggingface.co/datasets/aidigestorg/ai-village](https://huggingface.co/datasets/aidigestorg/ai-village) (gated, approved by hand) | `data/raw_village/`, or read as `hf://datasets/aidigestorg/ai-village/<file>` | research terms: no training, no re-identification, attribute AI Village; only the side tables are committed |
| Swarm traces redacted payload release (`redacted.jsonl.gz`) | [swarmtraces.org/viewer](https://swarmtraces.org/viewer/) → "Download dataset" | anywhere | a reconstruction corpus with no clock and no agent identity; check the viewer's terms before redistributing |
| Synthetic swarm with known provenance | generated: `swarmprov synth -o data/synth/transcript.jsonl --agents 60 --seed 2` | `data/synth/` | `truth.csv` (who copied from whom) and `config.json` (the true values) beside it ([options](docs/USAGE.md#synth)) |

### Install and run

```bash
pip install -e .            # pandas, pyarrow, networkx, matplotlib, pyfixest
pip install -e '.[llm]'     # optional: Claude-based extraction
pip install -e '.[hf]'      # optional: hf:// dataset URIs
pip install -e '.[dev]'     # optional: pytest, to run the tests (python -m pytest -q)

swarmprov run data/raw -o runs/wiki --gold validation/   # ≈2 min: every stage, scoring the extractor on the hand labels
swarmprov run full-wiki-logs.zip -o runs/wiki            # a zip works too
```

`runs/wiki/report.md` opens with **Key findings**, one bullet per analysis, also written alone to `headline.md`. `runs/wiki/report.html` is the same report as one offline page with the headline numbers as tiles and a **fact explorer**: pick a value or a disputed slot to see its carriers over time and the post behind each, or look an agent up. Without the dump, download the committed [`results/openai-wiki/report.html`](results/openai-wiki/report.html) and open it in a browser.

**Chat transcripts** (AI Village, Slack, Discord, framework logs, spreadsheets):

```bash
swarmprov run village-transcript.json -o runs/village --adapter chat --config my_config.json
swarmprov run messages.json -o runs/agents --adapter chat   # no timestamps, no config: message order, one catch-all family
swarmprov run autogen.jsonl -o runs/agents                  # OpenAI, LangChain or AutoGen roles: task turns and tool results are not posts
swarmprov run slack-export/ -o runs/slack                   # a Slack export directory; a Discord export or a CSV works the same way
swarmprov inspect steps.jsonl                               # which fields the chat reader finds, and a mapping to paste
swarmprov suggest runs/agents -o runs/agents/suggested_config.json   # techniques, families and items mined from a run, for review
```

### What the rule extractor recognises

A1 and the analyses built on it start from *answer reports*, in one of two forms. The verb form says `answered`, `submitted`, `replied` or `responded` followed by the value, with the item in the same sentence (`R1 CONFIRMED: Poland arrived; answered 2.9 at +5s`); the item must be a built-in US state or country, a phrase two or more authors write after round markers (`R1`, `G3`, `#4`), or one listed under `extra_items`. The `=` form is `Answer` or `CONFIRMED`, an optional colon and `ITEM = value` (`Answer: Hungary = 9.90%`), whose item is the capitalised phrase before `=`.

Task-value facts (A4, the relay trees, A6) count only items some post answered, so a free-form chat ("Poland's rate was 2.8% in 2023, per Eurostat") yields URL facts and nothing at A1: `families` and `extra_items` create no answer reports, and other wording needs `--llm`. When no claim is extracted, A1's skip note names the wording ([the full rules](docs/USAGE.md#what-the-rule-extractor-recognises)).

### Other sources

```bash
swarmprov run data/raw_village -o runs/village                      # AI Village side tables (agents, rooms, goals, ...)
swarmprov run data/village/agent_goals.jsonl -o runs/village_goals  # a roster alone
swarmprov run data/raw2 -o runs/corpus                              # the cross-site corpus alone
swarmprov crosssite runs/wiki data/raw2 -o runs/crosssite           # wiki + corpus: technique spread across sites
swarmprov run redacted.jsonl.gz -o runs/swarmtraces                 # Swarm traces: a reconstruction corpus
swarmprov synth -o data/synth/transcript.jsonl --agents 60 --seed 2
swarmprov run data/synth/transcript.jsonl -o runs/synth --config data/synth/config.json
swarmprov truth-score runs/synth data/synth                         # every analysis against what synth planted
```

Every command, its options and what it writes, and the synthetic scenarios, are in [docs/USAGE.md](docs/USAGE.md#commands).

## Results on the OpenAI wiki swarm

The headline above has the main numbers. The full report, with every table and figure, is [`results/openai-wiki/report.md`](results/openai-wiki/report.md); the rows below add what the headline leaves out. How each analysis works, and where it stops, is in [docs/METHODS.md](docs/METHODS.md).

| | finding |
|---|---|
| **A1 Provenance** | The share falls with the head start required: 21% of answers had been public a day before the report, none of datausa-sector61-state's at 6 h but 64% of oecd-equity's at 24 h, so the threshold separates races from archives. 171 answers were preceded by a deleted copy, and for 150 of them another copy was still visible. In 19% of rounds the agent had posted the answer itself before reporting it. |
| **A2 Does availability cause copying?** | The tool reports *not identified* when fewer than 10 agents switch exposure status and *weakly identified* when fewer than 20 do; with 20 or more it reads *identified* where the 95% CI excludes zero and *no effect detected* where it covers zero. |
| **A3 Diffusion** | The *live-tooltip value* the blob bypass enabled reached 35 of 74 (47%); Kaplan–Meier median 5.7 h (Greenwood 95% CI 4.8–8.5 h). Heartbeat beacons spread slowly across the whole swarm (96 of 1,113 agents, median 81 h among adopters). Seed-crack and `clock.wait` adoptions landed on a page that already carried them more often than the adopters' own baseline (45% and 37% against 30%, after Holm's correction): consistent with contagion, not proof. |
| **A4 Errors** | Of the 16 disputed slots, 9 were overturned, 3 held and 4 contested: Hungary, a 2–2 tie, one on 4 late carriers and one whose two values were first posted in the same second. The live values overturned the rounded ones within the first 3-hour bin: 16.38 for Poland (21 of 28 late carriers), 14.59 for the Slovak Republic (31 of 35). On 6 of the 13 value slots the consensus, which A2's `y_consensus` scores against, is not the later majority. |
| **A5 Structure** | Credited to the first source, 46 agents supplied someone's answer (out-degree Gini 0.68, against 0.56 for the 61 items' exposure counts alone); to the latest earlier carrier, 325. On instant answers only, the top 10 supplied 17–80%. Of the answers credited to a first source, 21% came from one that had computed the value itself, 63% from one with no answer round observed on the item (67% under strict), 7% from one that answered a different value and 9% from one that had posted the value before any other agent. |
| **A6 Reach** | Of 365 task values, 159 were repeated by other agents (mean 3.6 repeaters, 95% CI 2.7–4.7, p90 10), and of 23,560 URLs, 5,373. A repeated value travelled 2.0 hops on average, at most 10, when a carrier's parent may have posted at any time before it, but 1.3 when the parent must have posted within the hour (949 of 1,331 carriers then have none): most of the depth rests on hops longer than an hour. |

### Cross-site results (wiki run + corpus, [`results/crosssite/report.md`](results/crosssite/report.md))

16,858 posts across 17 surfaces: 15,987 from the wiki dump and 871 timed corpus posts (the rmn.re shortener log and the sandbox wikis among them), after dropping 425 corpus rows that are redacted copies of wikis already in the dump.

| finding | detail |
|---|---|
| **Fetch-proxy tricks crossed platforms in hours; task tricks never did.** | `markdown.new` first appeared on dse at 09:50 on May 26 and was on **6 other surfaces within 7 h** (user/milk 4.1 h, ludism 4.7, paste.linuxiarz 5.4, probier 5.8, fractal 5.9, wiki4d 6.8). The `allorigins` proxy originated on the **pastebin**, two days before it reached dse. All six task-level techniques (blob bypass, live tooltip, heartbeat beacons, seed cracking, clock.wait, ZZZ pages) stayed on dse. |
| **The earliest proxy use predates the main wiki by a week.** | A May 17 publictestwiki sandbox edit already chains three proxies (Facebook redirect → web2md → `r.jina.ai`); the first dse post is from May 24, its first proxy post (`markdown.new`) from May 26. |
| **Coverage bounds.** | Of 143 surfaces the collectors inventoried, 76 were never searched or yielded no agent text, 23 relays and 23 shorteners were deliberately not visited, and 491 Discord links point into these sites with no Discord messages captured. Every exposure share above is a floor. |

Running the standard pipeline on the corpus alone (`results/corpus/`) exercises the generic path on an unsigned, mixed-format source: 1,296 posts, 126 author strings, 15 answer claims that name no known item, so A1 and A2 skip (the pastebin channel is unsigned besides, so per-agent provenance is not possible there), the corpus config's seven URL techniques for A3, and 1,569 URL facts for the reach analysis.

### Validation

Labels were made by Claude reading posts (`validation/*.jsonl`, event ids plus labels). They should be **replaced or confirmed by human labellers** before anyone cites them. There are three samples:

| split | posts | answer-post P / R | item | value | round | latency |
|---|---|---|---|---|---|---|
| dev (used for tuning) | 60 | 1.00 / 0.97 | 1.00 | 1.00 | 0.97 | 1.00 |
| held-out, scored **before** fixes, then used for tuning | 48 | 1.00 / 1.00 | 0.86 | 0.86 | 0.86 | 0.78 |
| **untouched test** (labelled after the last extractor change) | 40 | **1.00 / 0.91** | **0.96** | **1.00** | **0.96** | **1.00** (n=9) |

The misses are answers reported without the verb "answered" (*"our submission was wrong"*). The Quick start run scores all three splits in the report's Validation section; the held-out split, tuned on since, now scores 1.00 on every field.

That section opens with an integrity line from `swarmprov check`, which re-derives every exposure column from the mentions with a loop of its own and checks the graph and the summary against the tables ([the checks](docs/METHODS.md#integrity-checks)). On the wiki it prints *19 of 19 checks pass: 0 errors, 0 warnings*.

### What the committed `results/` hold

Each directory under `results/` is a run's report, summary and linked figures, written by the current code from the repository root with the commands [results/README.md](results/README.md) lists; `results/openai-wiki` also keeps the CSV mirrors and `report.html`. `swarmprov compare results/openai-wiki runs/wiki` checks a fresh run against the committed summary: after the Quick start command it reports 0 changes.

## Asking questions of a run

A finished run answers one question at a time; [docs/USAGE.md](docs/USAGE.md#asking-questions-of-a-run) shows example output and what every printed field means.

```bash
swarmprov trace runs/wiki --agent "Mar16|oecd-equity" --item "Czech Republic"    # where did this answer come from?
swarmprov trace runs/wiki --fact "oecd-equity|Hungary|9.9" --png runs/wiki/hungary.png   # how did this value travel?
swarmprov trace runs/wiki --event b847fc1cbc646343        # what was extracted from this post, and why not more?
swarmprov trace runs/wiki --list --limit 50               # facts ranked by carriers (--family url for URLs)
swarmprov agent runs/wiki "May28|datausa-cashiers-masters"   # one agent: its answers, whom it learned from and supplied
swarmprov check runs/wiki                                 # do the run's tables agree with each other? exit 1 on an error
swarmprov compare runs/wiki runs/wiki --mapping-b strict -o runs/cmp   # what changes under the other identity mapping
swarmprov export runs/wiki -o runs/wiki_csv               # the exposure and edge tables, adoptions and disputes as CSV
swarmprov html runs/wiki                                  # report.html again, after editing report.md
```

`trace` prints an answer's first public copy and, apart from it, the graph's inferred relay parent: the copy shows what was available, not what the agent saw.

## How it works

```
transcript ──adapter──▶ events ──rules/LLM──▶ claims ──▶ mentions ──▶ exposure table ──▶ A1–A6 + report
 (wiki dump,            (posts)   (answers,            (every       (agent-round:        (figures,
  chat JSON)                       predictions,         item+value   was it public?)      markdown,
                                   round reports)       in public)                        GraphML)
```

`run` chains five stages; each reads and writes Parquet tables in the run directory, so any can be rerun alone ([options and outputs](docs/USAGE.md#commands)):

| stage | does |
|---|---|
| `swarmprov ingest INPUT -o RUN` | reads the source through its adapter into posts, deletions, a framework trace's tool results and a roster, and records the source's capabilities and clock in `profile.json` |
| `swarmprov extract RUN` | resolves authors under both identity mappings and extracts claims (answer reports, predictions, round sequences) and technique tags |
| `swarmprov expose RUN` | indexes every item–value and URL mention and writes one row per agent-round: when its value and the consensus were first public, the head starts and the treatment flags |
| `swarmprov graph RUN` | relay, exposure, citation and reply edges, fact chains, agent profiles and the GraphML/JSON graph |
| `swarmprov report RUN` | A1–A6 with figures, the key findings, `summary.json`, the integrity checks and `report.html` |

**A1** is the share of answers already public before their report, **A2** whether that availability causes copying (agent and item fixed effects), **A3** how techniques spread among the agents at risk, **A4** which of a slot's competing values won, **A5** how concentrated the sources were, and **A6** how far task values and URLs travelled.

### How it generalises

All source-specific knowledge lives in one adapter plus an `AdapterConfig`, which `--config file.json|toml` overrides ([every setting](docs/USAGE.md#adapters-and-the-config)):

```json
{
  "name": "village",
  "families": {"fundraiser": "donat|fundrais", "event": "venue|RSVP"},
  "techniques": [{"name": "shared-doc", "pattern": "docs\\.google\\.com"}],
  "reference_values": {"fundraiser|total raised": "$1,481"}
}
```

- **Items and rounds** are learned from the transcript (phrases after round markers; US states and countries are built in).
- **Families** map each task to a regex on text and channel names; without any, one catch-all family, which the report labels.
- **Identity** has two mappings, *strict* (one agent per author string) and *merged* (one per cohort date and task family), and every headline number is reported under both.
- **Deletions** stop a copy being public, which gives `D_visible`; **URLs** are facts too, so free-form chat still gives reach and relay chains.
- **Chat logs** are read as JSON, JSONL, CSV, Slack and Discord exports or directories, with fields found by alias or mapped with `--field`; a mostly untimed message list runs on message order.
- **Framework traces**: task turns and tool results are not public posts, so neither becomes a source; an own tool lookup before an answer is reported apart.
- **Rosters and AI Village tables** replace guessed identities and families; `--llm` reads other answer wording.

[docs/USAGE.md](docs/USAGE.md#inputs) has the details. Around the stages, `inspect` shows what the chat reader finds in a file, `suggest` mines a starter config for review, `synth` writes a swarm with known provenance and `truth-score` scores every analysis against it, and `compare` says what changed between two runs.

## Design notes and limitations

[docs/METHODS.md](docs/METHODS.md) has each analysis's rules, its checks against synthetic truth and these limits in full.

- **Public is not read.** The dump has no page views, so exposure is inferred from what was public, and a copy on a surface the collectors missed is invisible: exposure is a floor and the independent count a ceiling.
- **The unit of authorship** is the paragraph a revision added, and its author the trailing `-- Signature`, not the account that saved the page.
- **`D_cons`, the causal treatment,** asks whether the *consensus* value was public, whatever the agent answered: the agent's own value ties wrong answers to "unexposed" and gave a spurious +0.43 effect.
- **Attribution is a range**: nothing records whom an agent copied, so A5 credits each exposed answer three ways and reports all three. Relay edges are plausible paths, not proven ones.
- **A4 status and A3 contagion are conventions**: a later value must keep a lasting majority of new carriers to overturn, and an adoption on a page that carried the technique is consistent with contagion, never proof.
- **Intervals cover sampling noise across items only**, not identity or extraction error, and the synthetic scorecard holds under the simulation's own model only.
- **Agent-written text is data**: ingest strips control characters from names, `report.html` escapes all text and the console prints escapes.
- **The headline is generated** from `summary.json`, and `tests/test_results.py` ties every number this README quotes to the committed summaries.

## Repository layout

| path | contents |
|---|---|
| `swarmprov/adapters/` | Adapter interface + implementations (wiki, chat with framework roles in `framework.py`, corpus, swarmtraces, roster, village) |
| `swarmprov/` | Core modules: the stages (`pipeline`, `exposure`, `graph`, `report`, `html`), extraction (`segment`, `rules`, `claims`, `identity`, `extract_llm`) and the commands around them (`query` for `trace` and `agent`, `checks`, `compare`, `suggest`, `synth`, `truthscore`, `export`, `inspect`, `validate`) |
| `swarmprov/analysis/` | A1–A6 analyses: provenance, causal, diffusion, errors, structure, origins, agents (per-agent profiles), uncertainty (bootstrap and Greenwood intervals), crosssite |
| `tests/` | 643 tests (≈5.5 min with the wiki dump present); [tests/README.md](tests/README.md) says what each file checks |
| `validation/` | Labels (event ids) for dev, held-out, and test splits |
| `results/` | Committed reports ([results/README.md](results/README.md)) |
| `data/village/` | AI Village roster (33 goal assignments) |
| `data/raw_village/` | AI Village side tables: agents, chat_rooms, agent_goals, villages, claude_code_sessions, summaries |
| `docs/` | [USAGE.md](docs/USAGE.md) (commands, inputs, outputs), [METHODS.md](docs/METHODS.md) (how each analysis works and where it stops), [DATA.md](docs/DATA.md) (what the dumps hold), [PLAN.md](docs/PLAN.md) (the original plan and its status) |

**Git-ignored:** the datasets (`data/raw/`, `data/raw2/`), the synthetic swarms the examples write (`data/synth/`, `data/ag/`, `data/ramp/`, `data/scen/`) and `runs/`, so following the examples leaves `git status` clean.
