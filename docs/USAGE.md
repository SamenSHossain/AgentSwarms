# Using swarmprov

The README's [Quick start](../README.md#quick-start) gets a run going. This page holds the rest: every command and what it reads and writes, how to read `trace` and `agent`, the inputs the adapters read and the files a run writes. What the analyses compute, and where they stop, is in [METHODS.md](METHODS.md).

## Commands

Stages can be rerun on their own; each one reads and writes Parquet tables in the run directory (`run` chains them, `synth` writes a transcript to run them on). A missing input, `--config` or `--roster` file, a malformed config, or an input its adapter cannot read stops `run`, `ingest`, `inspect` and the other stages with one `swarmprov: error:` line and exit status 2, before anything is written for a missing file. Every command that reads a run does the same when the directory holds none (no `profile.json`, or one that is not a JSON object, as a write cut short leaves it), and never creates it. `--json` and `--png` create the directory they name.

### run

`swarmprov run INPUT -o RUN [--adapter A] [--config FILE] [--roster GOALS] [--field FIELD=KEY] [--gold FILE_OR_DIR ...] [--llm MODEL]`

Runs every stage below on the source in order (`ingest`, `extract`, `expose`, `graph`, `report`), scoring the `--gold` label files before the report.

```bash
swarmprov run hf://datasets/aidigestorg/ai-village/chat_messages.jsonl.gz -o runs/village --adapter chat   # an hf:// URI
swarmprov run steps.jsonl -o runs/steps --field ts=step_time --field author=agent_handle --field text=output
```

### synth

`swarmprov synth -o FILE [--agents N] [--seed S] [--copy-rate P] [--copy-from latest|first|random] [--copy-ramp-h H] [--format chat|autogen] [--delete-rate P] [--technique TOKEN] [--boards N] [--alias-rate P] [--confound P]`

Writes a synthetic swarm with known provenance: `FILE`, with `truth.csv` and `config.json` beside it; `config.json` carries the true values as `reference_values`.

`truth.csv` names the author of the post each copying agent took (`src_agent`); `--copy-from latest|first|random` picks which post that is (the default, `latest`, writes the same transcript as before the option existed).

`--format autogen` writes the same swarm as an untimed AutoGen stream with each computed answer's own tool call; its tool rows draw on a separate random stream, so the default transcript is unchanged.

`--copy-ramp-h H` (a positive, finite number of hours) makes copying grow with how long the answer has been on the board: probability `copy-rate × min(1, hours since its first post by another agent / H)`, a dose effect for A2's exploratory block (it changes only the threshold of the same draw, so without it the transcript is unchanged).

Five scenario options exercise the other analyses, each off by default and on its own random stream, so without them the transcript and truth are unchanged: `--delete-rate P` (a share of posts get a `deleted_at` 15 min to 6 h after posting, and a deleted post leaves the board; shares run from 0 to 1), `--technique TOKEN` (the first agent of `alpha-task` to start uses the token from its first post, and every other agent of the family from the round in which it reads, among the posts on its question, one carrying it; `config.json` gains its pattern and `truth_techniques.csv` the adoptions), `--boards N` (each family posts on N boards, every agent on its own, and agents read them all, so copies cross pages), `--alias-rate P` (a share of posts signed with the agent's variant handle, `AlphaRunner-Mar16`; `truth.csv` `agent` is the handle signed and `true_agent` the agent) and `--confound P` (that share of agents, the latest to start, never copy and answer each question instantly with probability 0.5: a null whose instant answers vary, and go with exposure between agents but not within one).

`--delete-rate` and `--boards` need the chat format: the AutoGen stream has no timestamps or channels to carry them, so `synth` refuses them there. `swarmprov truth-score` scores a run against this truth. The tests build their own copy.

```bash
swarmprov synth -o data/ag/autogen.jsonl --seed 2 --format autogen   # the same swarm as an untimed AutoGen stream with tool calls
swarmprov run data/ag/autogen.jsonl -o runs/ag --adapter chat --config data/ag/config.json
swarmprov synth -o data/ramp/transcript.jsonl --agents 120 --seed 1 --copy-ramp-h 4   # copying grows with the answer's time on the board
swarmprov synth -o data/scen/transcript.jsonl --seed 2 --copy-rate 0.3 --copy-from first --boards 3 \
    --alias-rate 0.2 --delete-rate 0.1 --technique TRICK-A                           # boards, aliases, deletions, a technique
swarmprov run data/scen/transcript.jsonl -o runs/scen --config data/scen/config.json
swarmprov truth-score runs/scen data/scen      # every analysis against what synth planted; rebuilds the report
```

### inspect

`swarmprov inspect INPUT [--config FILE] [--field FIELD=KEY]`

Prints, and writes nothing: each adapter's sniff (with the error a sniff raised), the first record's keys, the key each field resolved to and how many records hold it, and coverage over the first 500 records (timestamps read, authors known, text non-empty, channels), and the framework roles it finds. It then suggests a `{"fields": ...}` snippet, guessed from key names and values, to paste into `--config` ([Field mapping](#field-mapping-and-ingest-warnings)); nothing is applied.

### ingest

`swarmprov ingest INPUT -o RUN [--roster GOALS] [--field FIELD=KEY]`

Writes `events`, `lifecycle`, `reads` (a framework trace's tool results), `roster`, `profile.json` (capabilities, the run's `clock`, the `fields` used, ingest's `warnings`, and the settings the stages ran with: `config_settings` as the latest stage read them, a hash per stage in `config_stages`).

### extract

`swarmprov extract RUN [--llm claude-haiku-4-5]`

Reads `events` (and `roster`) and writes `agents`, `agent_aliases.csv`, `claims`, `tags`, `roster_events`, `items.json`, and under `auto_techniques` with no techniques configured `mined_techniques.json`.

### expose

`swarmprov expose RUN`

Writes `mentions`, `exposures_{merged,strict}` (one row per agent-round: when its value and the consensus value were first public, the head starts, the treatment flags and outcomes).

### graph

`swarmprov graph RUN`

Writes `edges_*`, `chains_*`, `agent_profiles_*`, `provenance_{merged,strict}.graphml` and `.json` (attributed graph for Gephi or networkx).

### report

`swarmprov report RUN [--mapping strict] [--boot N]`

Writes `report.md` (opening with the key findings), `headline.md` (that block alone), `report.html` (see `html` below), `summary.json` (strict JSON: ISO times, `null` for missing; with the pipeline `funnel`, `agents`, every A4 row and their statuses counted (`A4.status_counts`), A2's exploratory dose response as `A2_dose`, the integrity checks as `integrity`, the key findings as `headline`, one string per bullet, and the reason each skipped section gives as `skipped`), `checks.json` (the checks the Validation section counts, as `swarmprov check` writes them), `figures/`, and per mapping the detail behind A1, A3 and A4: `a1_by_family_*` (every family, no size cut), `adoptions_*` (one row per at-risk agent and per user of each technique) and `disputes_*` (one row per carrier of each disputed variant).

`--boot` sets the seeded bootstrap resamples behind the intervals (default the config's `bootstrap_reps`, 1000; `0` leaves them out).

`--mapping strict` writes `report_strict.md`, `headline_strict.md`, `report_strict.html`, `summary_strict.json`, `checks_strict.json` and `figures/*_strict.png` beside the merged files, which it leaves as they were.

The detail tables hold: `a1_by_family_*` (every family; the printed table drops those under 10 answers and says how many), `adoptions_*` (technique, agent, `in_at_risk`, `is_originator`, `t_first_use`, `hours_from_first_seen`, `adopted`, `censored_at_h` (both hour columns empty under a sequence clock), and for each adopter the page, whether it carried the technique, its baseline and `t_exposed`, its first earlier post on such a page; the at-risk rows are exactly the Kaplan–Meier inputs) and `disputes_*` (slot, kind, family, item, variant, agent, `t_first`, `is_first_variant`, `is_winner`, `carried_both`, `status`; a value slot's carriers come from item–value mentions, a round slot's from round claims, a declared slot's from its patterns).

### html

`swarmprov html RUN [--mapping strict]`

Reads `report.md`, `summary.json`, `figures/` and the graph tables and writes `report.html` (`report_strict.html`), which `report` and `crosssite` also write: the report as one self-contained page (figures as data URIs, no external scripts or fonts, light and dark, phone width) with headline tiles, a table of contents, sortable tables and the fact explorer (a directory without the graph tables, such as a committed `results/` one, gets one line in its place saying so); exit status 1 when the run has no such report.

`report.html` (`html.py`) converts exactly the Markdown the reports write, escaping all text before applying any formatting (the wiki report quotes a probe's `<script>` payload, which the page shows as text); its hero tiles read the same summary values, and `swarmprov html` reads a summary written before a value was added (the independent tile without `n_self_originated`).

The fact explorer carries, as JSON in the page, the task facts carried by two or more agents (the 200 most carried), the 3 most carried URLs (traced in that order until their hops and posts take 1 MB of the page; when that is not every fact two or more agents carried, the page says how many of each kind it picked out of how many, and how many of those are not in it, a fact a featured slot brings in counting as in) and A4's featured slots in full, each traced by `query.fact_lineage` as `trace --fact` does (on that fact's own rows: the explorer groups the edges and mentions by fact once), with each post's excerpt once per item it carries (centred on that item, cut to 120 characters) and the agent profiles (`agent_profiles_*`) of the carriers, the answering agents and the first sources with their links: the agents each learned from and supplied, by edge kind (at most 6 each way, replies left out as interactions).

Its timeline draws a carrier per row in order of first post (against time, or whole message positions under a sequence clock): same-page relays solid from the parent's post, cross-page hops grey (source unknown, attributed to the originator), a ring where the hop falls within the 2 s clock resolution, the variants of a slot in their own colours; a row's post, page and delay show on hover, tap or arrow keys, in a box that stays on screen below the timeline however long it is (where rows carry no label, at phone width or past 100 rows, the active row's band names its carrier), and every row is also in a table. The paths are inferred, not proven, and the page says so beside the timeline.

Figures go into the page as 256-colour palette PNGs, about a third of their bytes and as sharp (the files in `figures/` stay as drawn). On the wiki the page is 1.4 MB (0.6 MB of it figures) and builds in about 4 s; on a 30,000-post chat it is 1.9 MB and builds in about 4 s (it was 4.4 MB and 41 s when every fact was traced on the whole tables), so `report` and `crosssite` always write it.

### export

`swarmprov export RUN -o DIR [--tables T1,T2,...] [--format csv|parquet] [--mapping strict] [--all-columns]`

Writes stage and report tables as `DIR/<table>_<mapping>.csv` (or `.parquet`; `events`, `agents`, `claims`, `tags` and `mentions` carry no mapping).

The default tables are `exposures`, `edges_exposure_citation` (exposure and citation edges), `adoptions`, `disputes` and `a1_by_family`; the first two keep the columns of the CSVs under `results/openai-wiki` (exposure: every column but the own-lookup and task-message columns, which come from framework traces; edges: the post behind each end and the answer's `claim_id`) unless `--all-columns`. Also `edges`, `chains`, `agent_profiles` and the mapping-free tables.

An empty table still gets the full header. A detail table exists only for the mapping the report was built under (`run` builds `merged`), and `export --mapping strict` says to run `report --mapping strict` first.

### sample-gold and validate

`swarmprov sample-gold RUN -n 100` / `swarmprov validate RUN labels.jsonl`

A hand-label sample, and precision–recall on the labelled posts the run holds (`validation.json`, with that `coverage`; labels none of whose posts are in the run are not scored, exit 1).

`swarmprov run ... --gold FILE_OR_DIR ...` scores each label file (a directory: its `*.jsonl`) before the report as `validation_<split>.json` (`test_labels.jsonl` → `test`) and skips a file with no labelled post in the run, and a `*.jsonl` in the directory that holds no labels (each row needs an `event_id` and a `gold` block); a missing path, or a named file that is no label file, stops the run before any stage. The report shows every split.

### check

`swarmprov check RUN [--mapping strict]`

Reads the stage tables (and `summary.json`, or `summary_strict.json`) and writes `checks.json` (`checks_strict.json`) and one line per check: every exposure column re-derived from the mentions (`D`, `D_cons`, `D_visible`, `self_prepared`, `self_originated`, their times, posts and sources, the head starts and windowed flags; a failing round is listed with the columns that are off), flag nesting, the graph against the chains and rounds (a self-originated round must carry no exposure edge), the posts behind edges and exposures, the strict mapping inside the merged one (errors: exit status 1), the author handles whose case variants carry conflicting cohorts or roster ids, which identity keeps apart (`agent_aliases.csv`), and `summary.json` against the tables (warnings); exit status 2 when the directory holds no exposure or edge table to check; [Integrity checks](METHODS.md#integrity-checks) says what each one tests.

### truth-score

`swarmprov truth-score RUN TRUTH_DIR [--mapping strict]`

Reads a run of `swarmprov synth` output and the truth beside its transcript, and writes `truth_validation.json`, the scorecard of every analysis against what the simulation planted, and each report the run has (`report.md`, `report_strict.md`) rebuilt with its mapping and resamples so its Validation section shows it.

A run missing a table, or a truth directory without `truth.csv` or `config.json`, gets the stage or file named (exit 1, nothing written); where every round has the same `y_instant` (copy rate 0 without `--confound`) A2's fit is a point and the scorecard says *not estimated*, claiming no coverage, and on a message-order run (`synth --format autogen`) its A2 row and printed line say *not estimated: the latency outcome needs a wall clock*.

### crosssite

`swarmprov crosssite WIKI_RUN CORPUS_DIR -o RUN`

Joins the wiki run's and the corpus's events and writes `technique_spread`, `technique_site_summary` (Parquet, and CSV beside them), timeline, coverage bounds (`report.md`, `report.html`).

### suggest

`swarmprov suggest RUN [-o FILE]`

Reads `events`, `agents`, `items.json` and the settings in `profile.json`, and writes a printed evidence table (each URL domain's agents, posts, distinct URLs, the share of them that wrap another URL, its first post, and what it suggests: a technique, near the threshold, short of agents, a data source) and `FILE` (default `suggested_config.json`) with `families`, `techniques` and `extra_items`, and the run's other settings where they differ from the adapter's built-in config, to review and pass to `--config` in place of the run's config; nothing is applied.

`swarmprov suggest RUN` mines a starter config from a run's posts (`suggest.py`). *Techniques*: a URL *wraps* another when, after its scheme and host and percent-decoded (inner URLs are often encoded, sometimes twice), its path or query holds a scheme (`http:`, `https:`, `://`), its path begins with another hostname (`markdown.new/example.com/page`; a first segment is a host only when its last label is a top-level domain and no file extension, so `wiki.cgi`, `README.md` and `report.xlsx` are files and `user.name` a name), or a query value does and goes on with a path (`?url=www.sec.gov/files`). A segment that is a real hostname, such as a GitHub user called `socket.io`, still reads as one, which is why a technique needs most of a domain's URLs to wrap. A `.tld/` further into the path is not enough, or nearly every URL would wrap. A domain (the host as posted, percent-decoded, lower case, no port) whose distinct URLs wrap another in at least 80% of cases and which 3 or more agents posted becomes the technique `proxy:<domain>`, matching the domain wherever a post names it. Domains from 10% up to 80% (shorteners, redirectors, viewers, archives) are printed as near the threshold and left out, a domain only two agents posted whose URLs wrap in 10% of cases or more as short of agents, and data sources (under 10%) by agents; a run with no domain two agents posted says how many URLs and domains it has, and under message order a first post is shown as its position (`message 12`).

*Families*: one per channel, matching its name exactly, when two or more channels each carry 10 posts by two or more authors; else none, so the chat config's catch-all family applies.

*Items*: per family, capitalised phrases with a number within 60 characters after them that two or more authors posted, at most 10, leaving out round words, months, cohort tokens, trailing punctuation (`Poland.` is Poland, a built-in country) and items the run already knows.

Proposed families come with items found under them: each post gets the family the file gives it (a roster goal or shared-goal era it had stays), the known items are learned again as `extract` would under those families, items the config filed under the catch-all family go to each, and settings that name that family (reference values, disputes, technique families) are printed to re-file.

What the run's config sets is kept: its families, and its techniques and items before the new ones (a mined domain an existing technique already matches is printed as such). The settings are those the run's stages recorded in `profile.json` (a run recorded before that reads its config as it is now), and every other one that differs from the adapter's built-in config (field mapping, thresholds, reference values, ...) carries over, so the file replaces the run's config; the printed command reruns the source with it.

On the wiki run the wrap share separates the fetch proxies (`jqp.vercel.app` 613 agents 99%, `md.succ.ai` 411 99%, `markdown.new` 370 99%, `r.jina.ai` 214 99%, `pure.md` 162 100%) from the data sources (`api.datausa.io` 909 agents, `www.sec.gov` 539, `wikiservice.at` 715, each 0.1% or less); the 32 mined techniques include the six proxies the corpus config lists by hand, and `webcrawlerapi.com` (73%), `httpbin.org` (47%), `vanderbi.lt` (23%) and `docs.google.com` (22%, its document viewer) sit near the threshold. The item heuristic is a starting point only: it proposes 264 phrases on the wiki, many of them column names.

With `"auto_techniques": true` (off by default for every adapter) and no `techniques` configured, `extract` mines the same URL techniques before tagging, writes them to `mined_techniques.json`, and A3 runs on them with each row marked *(mined)* (`summary.json` A3 rows carry `"mined": true`). A3's skip line points to both.

A6's `url domains` row counts the distinct agents that posted any URL on each domain, beside the exact URLs, so agents who fetched different pages of one source count together although no exact URL repeats; it has no hops, its interval (with enough domains) resamples domains, and the sentence below the table names the three most shared (`summary.json` `A6` `url domains` `top`).

### compare

`swarmprov compare RUN_A RUN_B [--mapping-a M] [--mapping-b M] [-o DIR] [--limit N] [--tol T]`

Reads two runs, or one run under both mappings, and writes `DIR/compare.md`, `DIR/compare.json` (default `runs/compare/`) and a printed summary: the setup and config settings that differ, the `summary.json` values that differ (counts exactly, other numbers beyond a relative tolerance: `--tol`, default 1e-4), and per table the rows added, dropped and changed on their stable ids, with at most `--limit` examples each (default 10).

```bash
swarmprov compare runs/wiki runs/wiki_cfg2 -o runs/cmp                 # setup, summary values and rows that differ
swarmprov compare runs/wiki runs/wiki --mapping-b strict -o runs/cmp   # one run's two identity mappings
swarmprov compare results/openai-wiki runs/wiki -o runs/cmp            # the Quick start run against the committed summary: 0 changes
```

It says what changed between two runs: another config, LLM against rule extraction, a roster attached or not, a code change, or (`RUN_B` = `RUN_A` with `--mapping-b strict`) the other identity mapping. It counts every difference once (`n_changes`, 0 for a run against itself) in three layers.

*Setup*: adapter, source, config file and every config setting that differs, capabilities, posts, roster, clock and mapping. The settings are those each run's stages recorded in `profile.json` as they ran (`config_settings`, with a hash per stage in `config_stages`), not the config file as it reads now, so a file edited in place between two runs shows; a stage rerun after an edit is noted, and a run recorded before stages kept their settings is read from its config as it is now, with a note.

*Summary*: `summary.json` (or `summary_strict.json`) flattened to one value per path, with list rows keyed by what they describe (A2 by outcome and treatment, A3 by technique, A4 by slot, A5 attribution by rule, top sources by agent, the threshold curves by answer group, the window sweep by W), so a reordered or removed row reads as such; sections and rows only one side has are listed apart, counts (whole numbers) must match exactly while other numbers count as equal within a relative tolerance (`--tol`, default 1e-4), a missing value (null, or NaN in an older summary) equals a missing one and a time spelt two ways (`2026-06-20 05:10:21+00:00`, `2026-06-20T05:10:21+00:00`) is one value, and the report prints the tolerance and the largest difference it hid. Fixed-effects estimates move by up to 6e-5 relative between processes on identical tables (the wiki's A2 and dose rows), which the default hides and `--tol 0` reports.

*Rows*, on the stable ids: posts by `event_id` (added, dropped), claims by `claim_id` (added, dropped, or a changed item, value, round, family or type), agent-rounds by `claim_id` (added, dropped, flips of `D`, `D_w60m`, `D_visible`, `D_cons`, `y_instant` and `y_consensus`, a changed source or agent), author strings by `author_raw` (a changed label, and *regrouped*: its agent joins a different set of author strings) and facts by (family, item, value key) (carriers, originator, depth). Across mappings, the strict side's labels are read as merged ones (merged is a function of strict) before sources, originators and the top-source rows compare, so what remains is more than the mapping's relabelling.

A chat event id hashes the file path, so a moved transcript changes every id: when fewer than half the posts share one, posts are aligned on (time, author, text hash), claims inside paired posts by type, item, value, round and order, and a key two posts share on either side is a collision, counted and left unpaired. A committed `results/` directory holds a summary and no tables, so only the setup and summary are compared there.

On the wiki, merged against strict gives 1,108 → 1,164 agent-rounds (56 only under strict, none only under merged), one `D` flip and 366 author strings regrouped; a rerun with `{"disputes": []}` lists the declared G5 slot as the one A4 row only in A (another slot is featured in its place), with no claim or exposure change.

## Asking questions of a run

### trace

`swarmprov trace RUN (--agent A [--item I] [--family F] | --answer ID | --fact 'F|I[|V]' | --event ID | --list)`

`swarmprov trace` reads a finished run directory (every stage up to `graph`; it names the tables a partial run lacks) and answers one question at a time. It always prints the identity mapping it uses (`--mapping merged`, the default, or `strict`).

```bash
swarmprov trace runs/wiki --agent "Mar16|oecd-equity" --item "Czech Republic"  # where did this answer come from?
swarmprov trace runs/wiki --answer d50220c9ec0f653b                              # the same answer, by claim id
swarmprov trace runs/wiki --fact "oecd-equity|Hungary|9.9" --png runs/wiki/hungary.png   # how did this value travel?
swarmprov trace runs/wiki --fact "oecd-equity|Hungary"                           # every value posted for the item
swarmprov trace runs/wiki --event b847fc1cbc646343                               # what was extracted from this post, and why not more?
swarmprov trace runs/wiki --list --limit 50                                      # facts ranked by carriers (--family url for URLs)
```

`--agent` matches agent ids and author strings, ignoring case (`march16oecdscout` finds `Mar16|oecd-equity`). When several agents match, `--item` and `--family` keep the ones with a matching answer; if more than one is left, it lists them. `--item` works only with `--agent`, `--family` with `--agent` or `--list`.

`--answer` takes an answer's claim id; a later report of the same agent-round shows the round and says so, and a claim that is not an answer (a prediction, a round sequence) names the agent's answer for the item instead.

`--json FILE` also writes the result as JSON, with the run's `clock`. On a run with a sequence clock (see [Clock](#clock)), `trace` says so under the mapping line, counts head starts and hop gaps in messages, and says the 60-min flag needs a wall clock.

`--png FILE` draws one fact as a timeline: x is time (message position on a sequence clock), each row is a carrier in order of first post, same-page relay hops are blue and start at a small dot on the parent's row (its latest earlier post on the page), and cross-page hops are grey. `--limit N` (at least 1) caps the answers or listed facts and the hops printed per fact.

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
| own earlier post | the agent posted the value itself before reporting it (`self_prepared`). A tool result is not a post, so it never counts here. |
| posted first by this agent | replaces the two lines above when the agent posted the value before any other agent did (`self_originated`, at the time given): `D` stays 1, since the value was public before the report, but no source is credited with the answer, so the other agent's copy prints as *a later repeat by another agent* and the answer gives no exposure edge, no A5 credit and no supply. |
| own tool output before the report | framework traces only: the agent's own tool result carried the item and value before the report (`t_own_lookup`). It reads *own lookup preceded report* when no public copy came first (`own_lookup_evidence`), else says a public copy was also there before the report and whether it came before or after the lookup. A tool can echo a value the agent passed in, so this is a lookup, not proof of computation. |
| value given in the task | framework traces only: a system or user message carried the item and value before the report (`t_task`). A task message is no agent's post and no public copy. |
| inferred relay parent | the graph's guess at the source: the latest earlier carrier of the value on the same page (`relay`). With no earlier carrier on that page the hop goes to the originator and reads *cross-page hop: source unknown, attributed to originator* (`relay_xchannel`). A hop inside the 2 s clock resolution reads *direction unreliable*. The parent post is the parent's latest post of the value on that page; the delay runs from it to this agent's first post of the value. When the parent's post is not before the report (the rule exposure uses for a public copy) it reads *parent post not before the report: not this answer's source*, and no path is printed. *None* says why: the agent originated the value, or it has no item–value mention of the matched value (the extractor read its answer report but found no mention, so the relay graph has no hop for it), or it stated no value and never posted the consensus one. |
| inferred path | relay parents back to the value's originator. Its length is the hop depth A5 and A6 count. |
| competing values | the other values posted for the item, each with its number of carrying agents. The matched value is marked `<- this answer`, or `<- matched (consensus)` for an answer without a value. |

A fact (`--fact`) prints its origin post, then every relay hop as an indented tree (depth, kind, time, delay, page, post, excerpt). The counts match `chains_*`: `oecd-equity|Hungary|9.9` has 69 carriers, 68 hops (42 of them cross-page) and depth 10. Values match on the comparison key, so `9.90`, `9.9` and `9.9%` are one fact, and a multi-number value matches on its first number.

A post (`--event`) shows its author under both mappings, its family and tags, the claims and item–value mentions extracted from it, the URLs it carries, and the edges and exposures that point at it. When no claim was extracted, it says why; a task message or tool result is marked as such and gives no claims.

Excerpts are one line of 160 characters around the item, with an ellipsis at a cut end; `--event` prints the post's text the same way, cut to 600. Agent text is untrusted (the wiki holds a script payload), so escape it before putting it into a web page.

The posts behind these answers are stored at expose time: `exposures_*` carries `report_event_id`, `src_event_id` and `src_channel` (the first public copy), `src_latest`, `t_latest` and `latest_event_id` (the latest copy before the report, A5's second attribution rule), `src_visible_event_id` and `self_event_id`, and on framework traces `own_lookup_event_id` and `task_event_id`.

### agent

`swarmprov agent RUN NAME [--mapping strict] [--json FILE]`

`swarmprov agent RUN NAME` answers the per-agent questions: did this agent answer with no earlier public copy, did it supply others, whom did it learn from? It matches `NAME` the way `--agent` does (when several agents match it lists them) and prints the agent's profile, its answers (value, `D`, first public source and head start) and the agents its edges link it to, by kind: *learned from* (edges into it) and *supplied* (edges out of it), with replies apart as *replied to* and *replied by* because a reply is an interaction, not a knowledge transfer; at most 10 agents per kind (`--json FILE` keeps all). The top first source on the wiki run, abridged:

```
agent May28|datausa-cashiers-masters: 3 author strings (CashierCoordAgentX, CashierCoordOurRun, CashierSequenceAgentMay28); cohort May28; family datausa-cashiers-masters
  answers: 3; no earlier public copy 3, exposed 0 (0/3), at least 1 h before 0, median head start -; self-prepared 3; wrong 0
  first public source for 164 answers on 4 items (origin of those answers: computed 132, no answer round observed 32)
  same-page relays: 1 in, 8 out; task values posted first: 5, carried by 68 other agents (269 summed over the values); betweenness 0.0003; techniques: clock-wait-fastforward
answers (3):
  datausa-cashiers-masters | Business = 5269: no earlier public copy (D=0)  [6333d6e4d5884095]
supplied (edges out of this agent):
  exposure (164 edges, 65 agents): Apr02|datausa-cashiers-masters 4, Dec02|datausa-cashiers-masters 4, ...
```

The profile is one row of `agent_profiles_{merged,strict}.parquet`, which `graph` writes for every agent with a public post; the report's Agents section and the GraphML node attributes read the same table (`analysis/agents.py`), so a count has one definition:

| column | meaning |
|---|---|
| `n_author_strings`, `cohort`, `family`, `families` | how many author strings the label joins (as in `agent_aliases.csv`), its cohort, its task family (the label's family part when the label was built as `cohort|family`, else its most frequent post family; a `|` in a plain author name is no family) and every family it posted in, most frequent first |
| `n_posts`, `first_seen`, `last_seen` | public posts and the first and last of them (message positions under a sequence clock) |
| `n_answers`, `n_no_prior_copy`, `n_exposed`, `exposed_k_of_n` | agent-rounds; those with no earlier public copy (D = 0) and with one (D = 1); `k/n` exposed, given only from 3 answers on, because one or two say little about an agent |
| `n_exposed_60m`, `median_head_h` | exposed at least an hour before the report, and the median head start of exposed answers; empty without a wall clock |
| `n_self_prepared`, `n_wrong` | answers the agent had posted itself before reporting them, and answers its report calls wrong or a guess |
| `answers_supplied`, `items_first_sourced`, `origin` | exposed answers it was the first public source for (its exposure out-degree, A5's first-source count; a self-originated answer credits nobody), on how many items, and those answers' origins (A5's *where did copied answers come from*) |
| `relay_in`, `relay_out` | same-page relay hops into and out of it (cross-page hops are not counted: their source is unknown) |
| `facts_originated`, `reach_of_originated`, `agents_reached` | task values it posted first; their other carriers summed over those values (an agent that carried three of them counts three times); and the distinct other agents that carried any of them |
| `betweenness`, `techniques_used` | A5's broker score, and the configured techniques its posts used |

Exposure is a floor, so `n_no_prior_copy` is a ceiling on an agent's answers with no earlier copy anywhere (a copy on a surface the run lacks is invisible). It is no bound on computed answers: an exposed answer may still have been computed, and one with no earlier public copy may copy a post the run lacks.

## Inputs

### Adapters and the config

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

### What the rule extractor recognises

A1 and the analyses built on it start from *answer reports*, and the rules read two forms. The verb form is a post saying `answered`, `submitted`, `replied` or `responded` followed by the value, with the item named in the same sentence (`R1 CONFIRMED: Poland arrived; answered 2.9 at +5s`); the item is recognised only when it is a built-in US state or country, a phrase two or more authors write after round markers (`R1`, `G3`, `#4`), or one listed under `extra_items`.

The `=` form is `Answer` or `CONFIRMED` (or `confirmed`), an optional colon and `ITEM = value` (`Answer: Hungary = 9.90%`, `G5 CONFIRMED: Montana = 8553`); its item is the capitalised phrase before `=` (up to 40 characters; a known item inside the phrase is used instead), so it needs no item list. It is read in posts that hold no verb-form answer (`Answered: ITEM = value` is one, so its item must be one the verb form recognises).

Task-value facts (A4, the relay trees, A6) count only items some post answered, so a free-form chat ("Poland's rate was 2.8% in 2023, per Eurostat") yields URL facts and nothing at A1. `families` and `extra_items` create no answer reports; for other wording use `--llm`. The report's funnel counts the posts saying one of those verbs and the answer claims of both forms; when there are no claims, A1's skip note names the wording.

### Chat formats

The chat adapter reads JSON (a message list, `{"messages": [...]}`, `{room: [...]}`), JSONL and CSV/TSV with a header row (the delimiter is the one the header uses most: comma, tab, semicolon or pipe), each optionally gzipped and with any UTF-8 byte-order mark dropped, and two kinds of directory.

A Slack export (`users.json` and `channels.json` beside one folder of day files per channel) gives one post per message: the channel is the folder, the author the user's `real_name` (or `name`; a bot its `username`) and the id the message's `ts`, so a reply's `thread_ts` names its thread's first message; `<url|label>` becomes the URL, `<@U…>` becomes `@name`, `&amp;`, `&lt;` and `&gt;` are unescaped (in links too), system messages such as channel joins are skipped, and messages are ordered by `ts` across files.

Any other directory of JSON/JSONL files is joined (a file holding no messages, such as an agent list, is skipped), and each file's name is the channel of the messages that name none. When joined lists reuse ids (`1, 2, …` in every file, or a `ts` two Slack channels share), every id and parent is prefixed with its list's name (`planning/2`, `general/1777885200.000100`), so a reply finds its parent in its own list; ids that never repeat stay as given. A Discord export (DiscordChatExporter JSON) keeps its `channel.name`, and its replies point at `reference.messageId`.

AutoGen's `source` names the speaker only when no author alias and no agent id (`agent_id`, `user_id`, …) holds a value, because elsewhere it often names a data source (`web`, `api`); a `source` that looks like a URL or a path is never taken for a speaker. The wiki, corpus, swarm-traces, AI Village and roster adapters are tried first, so the files and directories they recognise are not read as chat.

### Field mapping and ingest warnings

Every field is found through aliases (`ALIASES` in `adapters/chat.py`). When a source uses other keys, map them in the config (`"fields": {"ts": "step_time", "author": "agent_handle"}`) or with `--field FIELD=KEY` (repeatable, or comma-separated where a comma comes before another `FIELD=`, so a key may contain one; it overrides the config). The fields are `ts`, `author`, `author_alt`, `text`, `channel`, `id`, `parent` and `visible_until` (a message's deletion time, found as `visible_until`, `deleted_at` or `deletedAt`: posts that carry one stop being public then, the run records `has_lifecycle` and A1 states `D_visible` (under message order a real deletion time cannot be ordered against positions, so the times are set aside, the header says so and `has_lifecycle` stays off); an AI Village message keeps its own time when its room was deleted later), and a dotted key such as `author.nickname` reaches into nested objects.

The mapping applies during detection too, so a file whose text key no alias knows is auto-detected once `text` is mapped, and to an AI Village directory's message table, which is read by the same reader (`inspect` then shows that table).

Ingest warns in the log, in `profile.json` (`warnings`) and in the report's Data capabilities block when a field looks missing: posts that name no author (with the first record's keys when they are more than half), one author for more than 10 posts, more than 10% of posts without a readable timestamp (no field, or an unreadable value, with an example; a log with no time field at all is warned about only when a key holds times, which the warning names, and otherwise the report's clock note covers it), or more than 10 posts all in one channel (A5's relay rule then takes every hop as same-page).

A config key that is not a setting is ignored with a warning naming the nearest setting (`technques` → `techniques`); `"strict_config": true` makes it an error. `--adapter` takes `auto` or an adapter name, and when nothing recognises the input the error lists each sniff that failed and why.

### Framework traces

When messages carry a role (`adapters/framework.py`: an OpenAI or Anthropic `role`, a LangChain `type` of `human`/`ai`/`tool`/`system`, also in `messages_to_dict`'s `{"type", "data"}` form, or an AutoGen `type` such as `TextMessage`, `ToolCallRequestEvent` or `ToolCallExecutionEvent`), only the agents' own turns are public posts. Of AutoGen's types only the chat messages (`TextMessage`, `ToolCallSummaryMessage`, `HandoffMessage`, any other `*Message`) are posts; its other events (`ThoughtEvent`, streaming chunks, `ToolCallRequestEvent`, `SelectSpeakerEvent`, …) are steps inside one agent that AgentChat passes to no other agent, so ingest drops them (`profile.json` `notes.n_inner_dropped`; a call request still opens its calls).

A system or developer message, and a user or human turn that names no other speaker (AutoGen: source `user`), is the **task**: author `task`, `source_kind` `task`. A user turn that names a speaker keeps it, because AutoGen 0.2 group chats and LangGraph supervisors pass agents' turns on as named user messages; a human correcting the agents that way counts as an agent too.

A tool result (an OpenAI or LangChain `tool`/`function` message, an AutoGen `ToolCallExecutionEvent` or `CodeExecutionEvent` (its `result.output`), an autogen-core `FunctionExecutionResultMessage`, an Anthropic `tool_result` block) is the calling agent's **private read**: `source_kind` `tool_result`, attributed to the agent whose call it answers by `tool_call_id`, `call_id` or `tool_use_id` (autogen-core: the ids of an `AssistantMessage`'s function calls), else, for an AutoGen event, to its `source` (the agent whose tool ran; an OpenAI tool message's `name` is the function's, never taken), else to the one agent with calls still pending (an agent's next turn closes its earlier calls; a result naming an id those calls do not have is not theirs), else to `tool`, with a warning.

Each attributed result is also a `reads` row (`method` `tool_result`) and sets `has_reads`. A task message or tool result that names no task family is read under every family, since a prompt or a tool's output names the item but rarely the family: a family-less prompt and an agent's lookups in any of its families still match the rounds they precede.

Task messages and tool results give no claims or tags, never count as a public copy, a relay hop, a consensus vote or an A4 carrier (declared disputes included), and `task` cannot be cited, so neither the task nor unattributed tool output becomes a source or a hub; the report header counts the task apart (*60 author strings plus `task`*). An assistant turn that names no speaker is still pooled as `unknown`, with the no-author warning.

Exposure reads them for two columns: `t_own_lookup` (the agent's own tool output carried the item and value before its report; `own_lookup_evidence` is that with no earlier public copy, i.e. *own lookup preceded report*) and `t_task` (a task message carried them). A1 then says how many of the answers with no earlier public copy were preceded by the agent's own tool output, splits all answers into a public copy first and no own lookup, an own lookup first, and neither, and counts the answers whose value was given in the task (`summary.json` `A1.n_own_lookup`, `n_exposed_no_lookup`, `n_given_in_task`, ...). `swarmprov synth --format autogen` writes the synthetic swarm as an untimed AutoGen stream with a tool call before every computed answer and none before a copied one.

### Clock

The chat adapter reads ISO times, epoch numbers and epoch strings such as Slack's `1777885200.000100` (9 to 13 digits, seconds or milliseconds by magnitude; only the timestamp field is read this way, so numeric ids stay ids). A time before 1990 or from 2100 on (a mis-scaled number, a 1970 placeholder) counts as no timestamp.

Framework logs (OpenAI message lists, AutoGen results, LangGraph state) often carry no timestamps. When fewer than 90% of a file's messages have one and the file is a single message list (a JSON array, JSONL, one wrapper such as `{"messages": [...]}`, or a single `{room: [...]}`), the whole file moves to a *sequence clock*: message *i* is posted at 2000-01-01 + *i* s, `profile.json` records `clock: sequence`, and the timestamps the file did carry are set aside and counted in the header, never mixed with positions (deletion times too, so `D_visible` needs a wall clock).

A1's exposed share, A2's consensus estimate, A4's winners and status (the whole run is one majority bin) and A5–A6 run on that order. In the report, every number measured in time is replaced by a line saying it needs a wall clock: the ≥10 min and ≥1 h shares, head starts and the exposure-by-head-start curve, A2's ≥1 h treatment and its dose response, A3's times to adoption and first-seen time, A4's bins and times, A6's relay-window sweep, and the clock-quality and deletion notes. A2 also leaves out its latency outcome, although latency is read from the post text, so A1's *instant* column and A5's *instant only* rows still show. The exposure tables hold `gap_s`, `D_w10m`, `D_w60m`, `D_cons_w60m` and `gap_cons_s` empty, `adoptions_*` its hour columns, and `trace` prints a post's position as its time (2000-01-01 + *i* s), counts head starts and hop gaps in messages and says the 60-min flag needs a wall clock.

A file that joins several lists (`{room: [...]}` with two rooms or more), and any directory, has no order between them, so it gets no sequence clock (`has_order` is no): untimed posts keep no time, exposure drops the agent-rounds without one, and the report says how many posts, agent-rounds and mentions each analysis left out (`summary.json` keeps `A1.n_rounds_untimed` and `A4.n_mentions_untimed`; A4 and A5 skip with that reason when nothing is timed).

### Families

`families` maps each task family to a regex on post text and channel names. With none configured, and no roster or channel family, the chat config's `default_family` (`all`) puts every post in one catch-all family, so a transcript with no config still reaches A1 when its posts report answers in the wording the rules read ([What the rule extractor recognises](#what-the-rule-extractor-recognises)). The report labels it, because answers to unrelated questions that name the same item then count as one fact, which can overstate exposure. A configured swarm's unmatched posts stay family-less. When no post has a family, A3's at-risk set is every agent active after a technique's first post; under the catch-all family the same holds for a technique without families of its own. Either way the set is larger than one task's agents, so adoption shares are lower. `diffusion.analyze` decides which set applies once, per technique (`at_risk_basis` in each `summary.json` A3 row: `all_active`, `catch_all` or `families`), and the A3 text and its key finding both read it.

### Identity

An explicit author field is used as given; otherwise authors are parsed from signatures. Case variants of one handle (`OpenAIResearcher`, `OpenAiResearcher`) are resolved together, so each strict id has one merged id (variants naming different cohorts stay split and are flagged in the audit). The cohort date comes from the handle (`OpenAIOECDNov27`), and from the date a handle's posts open with (`Mar16 R3 …`) only when the source has no explicit author field; set `cohort_from_text` to override.

`extract` writes `agent_aliases.csv` (every author string with its strict and merged id, cohort and family, and the size of both groups; on a roster run the family is the roster role behind the merged id). The report counts the merges, lists the largest, says how many only join case variants (which both mappings fold together), and warns when they join authors that an explicit-author source keeps apart.

When the two mappings group the author strings identically (each merged agent is one strict agent, as on the synthetic swarm), `expose` and `graph` compute the merged tables once and write the strict ones by relabelling them (`identity.one_to_one`), and the report gives one line saying so in place of its other-mapping sensitivity lines (A1's line, A2's table and dose slope, the origin columns under the other mapping), whose numbers would only repeat its own; `summary.json` still keeps A5's origin counts under both mapping names, as a run whose mappings differ does, so `compare` across the mappings finds the same keys.

### Items, lifecycle, URLs, rosters and AI Village tables

**Items and rounds.** Learned from the transcript: phrases following round markers become items (no manual task list needed). US states and countries are built in. Round sequences learned from chains like `MA → CT → MI → WV`.

**Lifecycle.** Not just recorded, but used. From wiki events: deletions (with cycle tracking), recreations, and probes; from a chat export, a message's `deleted_at`. The report adds clock-quality notes, deletion/recreation blocks, probing blocks, and exposure gains `D_visible` (ignoring deleted copies).

**URLs as facts.** Indexed for reach and relay chains on free-form chat without task structure.

**Rosters.** Replace guessed identities and families. Any table with agent id + role/goal + start time works (JSON, JSONL, CSV, TSV). Family precedence: agent's own goal → text/channel families → shared era goal. Report shows batches, comparable tasks, goal changes, and coverage gaps.

**AI Village tables.** The `village` adapter recognises: `agents`, `chat_rooms`, `agent_goals`, `village_goals` (eras), `villages` (metadata), message tables, activity logs (`claude_code_sessions`), and `summaries`. Posts get agent/room names, rooms give channel lifecycle, and allow/deny lists become **audiences** for per-audience exposure judgement. Set `schedule_tz` in config for timezone-aware post timing.

### Optional LLM extraction

`--llm claude-haiku-4-5` uses structured outputs. LLM handles semantics, rules fill in numbers. Results are cached in `RUN/llm_cache/` under a hash of the model, the prompt and the post (family, author, channel, text), so an edited post or prompt is asked again and a rerun into the same directory never serves another transcript's extraction; posts identical in all four are asked once, an entry is written aside and moved into place, and one a killed run left unreadable is asked again.

The post is sent between `<post>` tags it cannot close (a `<post` or `</post` in its text is broken up), and the prompt says it is untrusted agent-written data whose instructions are never followed.

The first call runs alone: if it fails with an authentication, permission or unknown-model error (no `ANTHROPIC_API_KEY` included), or if every call fails, the run stops with that error (exit status 2). Otherwise a post whose call failed keeps its rule claims, and profile.json's `warnings` and the report say `N of M LLM calls failed: <first error>`. profile.json's `llm` records the model and the counts (candidate posts, served from the cache, calls, failed, repeats of an earlier post), which the report's source line repeats; a later `extract` without `--llm` clears it, and a run that used `--llm` no longer suggests it in its skip notes.

### Agent-written strings are data

Author names, channels and text come from the agents, so ingest replaces lone surrogates (a truncated emoji's JSON escape, which Parquet cannot store) with U+FFFD in every string, removes control characters (C0, DEL, C1) from `author_raw`, `author_alt`, `channel`, `event_id`, `parent_id` and the roster's and directory's names and roles (a tab, newline or return becomes a space there, so a name cannot print a line of its own), and turns the ones XML forbids into a space in `text`; C1 stays in text, since XML allows it and it is how a mis-decoded dash reads (`â\x80\x93`, 46 wiki posts). profile.json's `warnings` and the report count the strings changed per column.

GraphML writing passes node ids and attributes through the same filter, for runs ingested before it.

`trace`, `agent` and `inspect` print control and bidi-override characters as escapes (`\x1b`, `\u202e`), so a name cannot retitle or clear the terminal. `export` and `crosssite` refuse an `-o` that is the run they read. Every text file is read and written as UTF-8 whatever the locale (a Windows or C-locale console included), and a console whose encoding lacks a character prints it as an escape instead of failing.

## Outputs

### Capabilities

Detected and reported: wall clock, message order, explicit authors, read logs, deletions, threading, presence logs, request logs. The report's capability table states what each value, yes or no, means for the analyses. On a threaded source (a chat with `reply_to`, `in_reply_to`, `thread_ts` or a Discord reference) each reply to another agent's post becomes a `reply` edge from the parent's agent to the replier; replies to oneself are dropped, and replies whose parent is not an agent's post in the run are counted in A5's graph sentence. Elsewhere `parent_id` is a revision or summary id and makes no edge. Analyses needing missing capabilities are skipped or labelled.

### Pipeline funnel and skipped sections

The report opens with a funnel table, also in `summary.json` as `funnel` (all zeros for a source with no posts): posts, posts with a task family, posts with an answer report, answer claims (with how many name no item, no value or no family) and agent-rounds. Each row has its own unit (a post can carry several claims, and repeated reports of one item are one agent-round), so the counts need not shrink row by row.

An analysis that cannot run keeps its heading and says why in one line, from the condition the report tests: A1 needs agent-rounds (and says whether no answer was extracted, none had a family or an item, or none had a timestamp; it suggests `default_family` only where that setting applies), A2 more than 20 timed agent-rounds, A3 configured techniques that appear in a timed post, A4 item–value pairs or URLs in posts with a timestamp, A5 provenance edges (and says which kind is missing: a fact two agents carried in timed posts, an exposed answer, a citation). With no task value repeated, A5 counts the URLs that travelled instead, and how many of their hops were same-page relays.

### The exported graph

`provenance_{merged,strict}.graphml` and `.json` (networkx node-link data) have one edge per (source, destination) pair over every kind: `kinds` lists them, `n_relay`, `n_xchannel`, `n_exposure`, `n_citation` and `n_reply` count them, and `weight` counts the knowledge transfers, `n_relay + n_exposure + n_citation`: a cross-page hop's source is unknown, and a reply is an interaction. Each edge also carries `n_facts` (distinct item–value facts), `first_ts` and `last_ts` (the first and last destination post), `median_dt_s` and `n_within_clock_res`; nodes carry the agent profile.

Every attribute has one type, as GraphML needs: a missing number is -1 and a missing time or text is empty. The graph attribute `clock` is `wall` or `sequence`; under a sequence clock (with a `clock_note` saying so) a time is the message's position, 2000-01-01 + i s, and `median_dt_s` a gap in messages.

`figures/a7_supply_network.png` draws the exposure edges alone, under the first-source rule its title states: sources are grey discs sized by the answers they supplied (the top 10 labelled), and each agent with an exposed answer is a dot coloured by its task family. Its layout is seeded and independent of the hash seed, so a rerun draws the same picture.

### Every edge names its posts

`edges_*.event_id` is always a row of `events`: the carrier's first post of the value (relay), the answer's report post (exposure), the citing post (citation) or the reply (reply). `src_event_id` is the source's post: the parent's latest post on the page, the originator's first post for a cross-page hop, the first public copy, or the post replied to. Exposure edges also carry `claim_id`. Earlier runs stored the claim id in an exposure edge's `event_id`; read `claim_id` for it now. `value_norm` holds the comparison key for every kind (the first number of a multi-number value), so edges join `chains_*.value_key`.

### Every JSON file is strict JSON

`summary*.json`, `checks*.json`, `validation*.json`, `truth_validation.json`, `compare.json`, `trace --json`, `agent --json` and the fact explorer's data in `report.html` all go through one encoder (`fmt.dumps`): NaN, infinities and NaT become `null`, times ISO text and numpy scalars plain numbers, so a strict parser reads every file.
