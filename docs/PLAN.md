# swarmprov: general pipeline plan

> **Implementation status (this branch).** Every step below is implemented in `swarmprov/`. Where it differs from the plan:
> - **Step 0**: the dump has **no read logs** (see `docs/DATA.md`), so `D_observed` is unused for this source. Capability flags are written to `profile.json` and printed in the report.
> - **Step 0 / clocks and families (new)**: `profile.json` records the run's clock. A chat file that is one message list with fewer than 90% of its messages timed runs on a *sequence clock* (message position; the few real timestamps are set aside, not mixed in), and every number measured in hours is replaced by a "needs a wall clock" line; a file of several lists keeps its untimed posts out of every time comparison (`has_order` no), and exposure drops untimed rounds. Slack-style epoch strings parse. A chat with no families configured falls into the catch-all family `all` (`default_family`, chat config only). The report opens with a pipeline funnel and gives a one-line reason for every analysis it skips.
> - **Step 0 / chat ingest (new)**: the chat adapter also reads CSV/TSV, Slack export directories (thread replies resolve to their thread's first message), other directories of message files (a channel per file), Discord exports (channel name, reply references) and AutoGen streams (`source` as the speaker when no other author key or agent id holds one); joined lists that reuse ids get ids prefixed with their list's name. Instead of the planned `--map map.yaml`, the config's `fields` and a repeatable `--field FIELD=KEY` map any canonical field to a source key (dotted keys reach nested objects), during detection too. `swarmprov inspect` shows each adapter's sniff, the key each field resolved to and its coverage, and suggests a `fields` snippet without applying it. Ingest warns (log, `profile.json`, report) when authors, timestamps or channels look unmapped, and on config keys that are not settings (`strict_config` makes those an error).
> - **Step 0 / framework traces (new)**: the planned framework-trace adapter is the chat adapter's role parsing (`adapters/framework.py`: OpenAI/Anthropic `role`, LangChain `type`, AutoGen message types). System and unnamed user turns become the task (author `task`), AutoGen's inner events (thoughts, call requests, streaming chunks) are dropped, and tool results become the calling agent's private reads (matched by call id, else an AutoGen event's `source`, else one agent's pending calls, else `tool` with a warning; one naming no family is read under every family), written as `reads` rows with `method="tool_result"` rather than the planned `context_window` (no trace on disk logs prompts). Neither is a public post. Exposure adds `t_own_lookup` / `own_lookup_evidence` (*own lookup preceded report*, deliberately not "computed") and `t_task`, and A1 splits answers by a public copy and an own lookup. `synth --format autogen` writes the synthetic swarm as an untimed AutoGen stream with tool calls before computed answers, and `tests/test_framework_roles.py` checks the split against its truth.
> - **Step 1**: wiki events are *new paragraphs since the last deletion* rather than revision diffs (`segment.py`), because line diffs over-attribute on re-encoded pages.
> - **Step 2**: no fuzzy name matching. `strict` is the case-folded handle; `merged` is (cohort, family) resolved per strict handle, so case variants share one id. A cohort is read from a post's opening date only where authors are parsed from signatures (`cohort_from_text` overrides). `extract` writes `agent_aliases.csv`, and the report counts the merges.
> - **Step 3/4**: the rules reach P = 1.00, R = 0.91 on an untouched 40-post test set (labelled by Claude; needs human confirmation). LLM extraction (`--llm`) is built and unit-tested with a fake client but was not run (no API key in this environment).
> - **Step 5**: items are *learned* (phrases after round markers, ≥2 authors) and rounds are learned from `A -> B -> C` chains.
> - **Step 6**: added `D_cons` (consensus value public) as the causal treatment; the value-specific `D` mechanically biases the consensus-match outcome.
> - **Step 8 / A2**: the wiki swarm has too little within-agent variation in exposure, so the estimator reports "not identified". Its validity is shown on the synthetic swarm (`tests/test_synthetic_recovery.py`).
> - **Step 8 / A2 dose response (new)**: exposure stores the consensus value's head start (`t_cons`, `gap_cons_s`), and A2 adds a secondary, exploratory block: among exposed rounds only, `causal.estimate_dose` regresses the instant outcomes on `log1p(head start in hours)` and on head-start bins (<1 h reference, 1–6 h, 6–24 h, >24 h) with agent + item FE. It identifies only the intensive margin (given the answer was public, does reuse grow with how long it had been?), gates on the within-agent spread of the head start, on agents spanning the bins and, per term, on the agents whose outcome varies too (not on exposure switchers), gives the slope's exact leave-one-agent-out range and the share of its clustered variance its two largest agents carry, prefers `y_instant|known` (`y_instant` scores an unstated latency 0; both measure the head start to the report, and `y_instant|known` keeps only rounds that state a latency, so it is the less biased read, not an unbiased one), and stays out of A2's reading. `synth --copy-ramp-h H` plants a dose effect, and `tests/test_dose.py` checks the slope finds it on three seeds and finds none without it.
> - **A4**: in addition to auto-detected disputes, configs can declare disputes as regex variants (used for the cracked-seed G5 forecast). Each slot is *held*, *overturned* or *contested* (a tie, a later winner with no lasting majority or fewer than 5 late carriers in all, or two values first seen at the same moment); ground truth enters as optional analyst `reference_values`, which only label the outcome and never change the status.
> - **A5**: hub concentration is reported under three attribution rules (first source, latest source, same-page relay parent), as a sensitivity range with the per-item exposure Gini as a null, because exposure does not say whom an agent copied. Synthetic swarms that copy the latest post, the first post or nothing show each rule recovering the source only when it matches the simulation. `analysis/origins.py` labels each credited source by its own answer round on the item and traces relayed sources to a root.
> - **Step 8 / figures**: each figure draws what its table says. A4 counts every carrier (an axis that starts after a stray early post counts the carriers before it and gives the first one's time), draws the first variant with the winner or the tied leaders, marks the winner and the start of its lasting majority, and gives the winner's share of new carriers per 3-hour bin, sparse bins faint; A2 draws every row, *not identified* ones at zero, with the switching agents in each label; A5 marks where the top 10 first sources begin; A6 pools 20 or more repeaters into one labelled point.
> - **Step 8 / uncertainty (new)**: the report gives seeded 95% intervals (`analysis/uncertainty.py`; `bootstrap_reps`, `report --boot N`, 0 for none): A1's shares and median head start resample whole items, as do family and round rows with at least 5 items; A6's mean repeaters resample facts; A3's Kaplan–Meier median carries Greenwood's interval where it is reached. A5's top-10 share and Gini get a leave-one-item-out range instead, because resampling items repeats their sources and inflates concentration. The intervals cover sampling noise only, not identity or extraction error.
> - **Step 7 and Step 10 / sensitivity to W and k**: the inferred relay edges keep no window (a parent at any lag), and A6 reports how the task-value trees change with a window W of 10 min to 24 h (`graph.relay_edges(max_gap_s=W)`: an older parent leaves the carrier unattributed, a root of depth 0). A1 adds the exposed share for a required head start of 0 to 48 h, overall and per family, and the instant share for a threshold k of 1, 2, 5 and 10 s. Hour-based sweeps need a wall clock.
> - **Step 7 / trace (new)**: exposure rows keep the posts behind their times (report, first public copy, first copy still visible, the agent's own earlier post; same-second copies in event-id order), and every edge's `event_id` is a post, with `claim_id` and `src_event_id` as their own columns. `swarmprov trace` (`query.py`) answers one answer, one fact or one post from these tables, labelling the first public copy and the inferred relay parent separately. The synthetic truth names the post each copier took (`--copy-from latest|first|random`), and the tests check both fields against it.
> - **Step 8 / detail and contagion (new)**: `report` keeps the per-agent rows behind A1 (every family), A3 (one row per at-risk agent and user; the Kaplan–Meier inputs are exactly its at-risk rows) and A4 (one row per carrier of each disputed variant, with the slot's status) as `a1_by_family_*`, `adoptions_*` and `disputes_*`, `summary.json` is strict JSON with every A4 row, and `swarmprov export` writes these and the stage tables as CSV or Parquet. A3 adds a contagion test (adoptions on a page another agent had already posted the technique on, against each adopter's own posts in the adoption's family, with an exact within-agent permutation p-value, Holm-corrected across the techniques tested; no rate ratio, which rising activity on task pages would inflate). A4 splits each value slot's variants by answer round into derived with nothing public (D = 0, an upper bound) and adopted after public, flags slots where the consensus is not the later majority, and exposure adds `y_winner` (overturned slots only) beside `y_consensus`, an extra A2 row.
> - **Step 10 / scorecard and integrity checks (new)**: `synth` gains scenario options (deletions, a technique spread by reading, several boards per family, aliased handles, a confounded null of fast agents), each on its own random stream so the default swarm is byte-identical, and the chat adapter reads a message's `deleted_at` / `visible_until`. `swarmprov truth-score` scores a run against the planted truth (exposure, `D_visible`, A2 against the true effect, each A5 attribution rule, A3, A4, identity purity) as `truth_validation.json`, labelled as holding under the synthetic generative model only, and `tests/test_truthscore.py` checks three scenarios with thresholds from seeds 0–7. `swarmprov check` (`checks.py`, also run by the report) re-derives every exposure column from the mentions independently of `exposure.build` and checks the graph, chains, post references and identity mappings against each other. `run --gold DIR` scores the hand labels before the report, with their coverage of the run.
> - **Step 10 / comparing runs (new)**: a strict report writes `report_strict.md`, `summary_strict.json` and `figures/*_strict.png` beside the merged files. `swarmprov compare` (`compare.py`) diffs two runs, or one run's two mappings: the setup and config settings, `summary.json` keyed by meaning (A3 by technique, A4 by slot, A5 by rule and source) within a relative tolerance with NaN equal to null, and the rows on their stable ids (posts, claims, agent-rounds with flag flips, author strings regrouped, facts), aligning posts on (time, author, text hash) when a moved chat file changed its event ids and leaving duplicate keys unpaired.
> - **Step 5 / suggested configs (new)**: `swarmprov suggest` (`suggest.py`) writes a reviewable config mined from a run: URL techniques (domains 3+ agents posted whose distinct URLs wrap another URL in 80% of cases: fetch proxies, recovering the corpus config's six by hand and more on the wiki), channel families and items, with the evidence printed and nothing applied. `auto_techniques` (off by default) mines the same techniques at extract, and A3 marks their rows *(mined)*; A6 adds a `url domains` row (distinct agents per domain).
> - **Cross-site (new)**: a `corpus` adapter loads the second batch (records, shortener log, sandbox wikis) and `swarmprov crosssite` measures technique spread *between surfaces*, draws the activity timeline and prints the collectors' coverage bounds. The AI Village stretch stays open (no transcript provided); the chat adapter is tested on a synthetic sample and on small OpenAI-, AutoGen-, Slack-export-, Discord- and CSV-shaped fixtures.

**Goal.** Take any multi-agent transcript (a wiki dump, a chat log, a framework trace) and output a provenance graph showing who learned what, from whom, and when. Use it to report how much of the swarm's knowledge was computed independently and how much was relayed, how techniques and errors spread, and which agents acted as hubs.

The original plan was written for the OpenAI wiki swarm. This document separates the steps that work on **any transcript** from the parts that are **specific to one source**. Each source-specific part goes behind an adapter and a config file. If a step is listed under "Core", it must not contain any wiki-specific logic.

---

## 0. Generalizing the concepts

| Wiki-specific concept | General concept | Notes |
|---|---|---|
| Wiki page / site | **Channel**: a shared surface where posts appear (page, chat room, thread, blackboard key, shared file) | |
| Revision / post | **Event**: a single timestamped write by one author to one channel | |
| `-- Name Mar23` signature | **Author identity**, resolved to an `agent_id` | Usually explicit in chat and framework logs |
| Task family (OECDEquity*, UEFA*…) | **Task family**: groups items that share a procedure | From an adapter rule or an LLM label |
| Round R1–R6 | **Episode**: a period of time in which an agent works on a set of items | Can be absent (then you use a sliding window) |
| (task, round, item) | **Item key**: the question being answered | |
| Answer value | **Claim**: (item key, value), asserted by an agent at time t | |
| Blob-hostname bypass, ZZZ trick | **Technique**: a procedural claim, i.e. how to do something | |
| 16.40 vs 16.38 | **Variant**: competing values for the same item key | |
| "Question arrived" | **Task-start event** | |
| Page-view logs | **Read events**: observed exposure | Rarely available. Framework traces give you the agent's context window, which is better still |
| Wiki revision time | **Wall clock** | Never use a clock that the agents report themselves |

---

## 1. Canonical data contracts (Parquet tables)

Every stage reads and writes these tables, and adapters only need to produce `events`.

```python
# events: one row per write (post, revision, message, tool output shared with others)
event_id: str          # stable hash
ts: datetime64[UTC]    # wall clock
channel: str           # page / room / thread / blackboard key
author_raw: str        # as it appears in the source
agent_id: str | None   # filled in by identity resolution
text: str
parent_id: str | None  # reply-to / previous revision of the same page
visibility: str        # "broadcast" | "channel" | "direct:<ids>" | "private"
meta: dict             # adapter-specific extras (revision no., IP, tool name, ...)

# reads (optional): observed exposure
agent_id, ts, channel, event_id | None, method  # "pageview" | "context_window" | "tool_result"

# extractions: one row per (event, extracted fact); output of Stage 4
event_id, agent_id, ts, task_family, episode, item_key, value_raw, value_norm,
event_type,            # task_start | claim | confirm | technique_share | technique_use
                       # | prediction | correction | request | other
claimed_latency_s, timer_s, cites: list[str], independent_claim: bool,
disputes: bool, extractor: str  # "rule" | "llm:<model>" | "gold"

# exposures: one row per agent-episode-item
agent_id, episode, item_key, t_start, t_public, t_read | None,
D_inferred: bool, D_observed: bool | None, y_instant: bool, y_match: bool

# edges: provenance graph
src_agent, dst_agent, item_key, kind, dt_s, evidence  # "citation" | "inferred" | "read"
```

---

## 2. Steps

### Step 0: Profile the input (any source, about 1h)
1. Find out the file format (JSON, JSONL, CSV, HTML, SQLite) and write `adapters/<name>.py` with a `sniff()` function.
2. Check which fields exist. The **capability flags** below decide which analyses can run later:
   - `has_wall_clock`: real timestamps for each write
   - `has_explicit_author`: author is a field, not text to parse
   - `has_reads`: page views, retrieval logs, or context-window dumps
   - `has_episodes`: explicit task, round, or turn boundaries
   - `has_ground_truth`: a known-correct value for each item
   - `has_threading`: reply or parent links
3. Count events per channel, per author, and per day, and plot activity over time to find the spikes.
4. Save the result to `profile.json`. The report will print these flags so readers know which conclusions the data supports.

### Step 1: Adapter, i.e. ingest into `events` (2h per new source)
Adapters implement a single interface:

```python
class Adapter(Protocol):
    name: str
    def sniff(self, path: Path) -> bool: ...
    def iter_events(self, path: Path) -> Iterator[Event]: ...
    def iter_reads(self, path: Path) -> Iterator[Read]: ...      # may yield nothing
    def rules(self) -> RuleSet: ...                              # Step 3 regexes
    def config(self) -> AdapterConfig: ...                       # task families, aliases, timers
```

Adapters to plan for:
- **wiki**: one event per revision. The event's text is the diff against the previous revision, not the whole page, so a claim is attributed to the revision that added it. Keep the full page text in `meta` so `t_public` can be computed.
- **chat** (AI Village, Slack, Discord exports): one event per message, with visibility set to the room. (Built: a chat post is public to every agent from its timestamp; only an AI Village room table restricts who can read a room, and channels are the pages of A5's relay rule.)
- **framework trace** (AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Claude Agent SDK): one event per message. Each LLM call's prompt becomes a `reads` row with `method="context_window"`, which gives observed exposure for free. (Built inside the chat adapter, with tool results as `method="tool_result"` reads; see the status note above.)
- **generic JSONL**: columns are mapped through a YAML file, for quick one-off sources.

Handling rules that apply to every adapter: normalize all times to UTC, deduplicate exact repeats, and keep raw IDs in `meta`.

### Step 2: Identity resolution (1h)
1. If the source has explicit author IDs, use them.
2. Otherwise, use the adapter's `rules().signature` regex to get (name, cohort token).
3. Fallback key: (cohort token, task family). Merge aliases from the adapter config plus fuzzy name matching (rapidfuzz ≥ 90).
4. Write `agent_aliases.csv` so a human can audit the merges.
5. Keep **both** mappings, strict and merged. Every headline number gets reported under each, as a sensitivity check.

### Step 3: Rule-based tagging (2h; the core is generic, the patterns come from the adapter)
The generic engine applies the adapter's `RuleSet` to each event and writes partial `extractions` rows with `extractor="rule"`:
- episode markers (`R\d`, `G\d`, `#\d`, "turn 7")
- event-type keywords (`CONFIRMED`, `arrived`, `due`, `answered`, `instantly`, `same second`)
- latency and timer tokens (`14s`, `30s timer`, `slow-tier`)
- mentions and citations (`@Name`, "X's report", "independently reproduced")
- channel name → task family mapping
- value patterns (numbers, place names, enumerations from config)

Generic rules that ship with the core: URLs, numbers with units, `@mentions`, quoted strings, and code blocks (likely technique carriers).

### Step 4: LLM structured extraction (3h; runs in the background)
1. Run one call per event with a small model and a JSON schema matching the `extractions` table. The prompt contains: the event text, its channel, the author, the previous 1–2 events in the same channel, and the Step 3 tags as hints.
2. The task family vocabulary and the item-key format come from the adapter config, so the prompt template stays generic.
3. Build in batching, retries, caching on `event_id`, and a cost estimate printed before the run.
4. Merge the results: where both the rules and the LLM fill a field, the rule wins for fields it parses deterministically (timestamps, latency) and the LLM wins for semantic fields (event_type, independent_claim).
5. **Validation**: sample 100–200 events stratified by channel and event_type, hand-label them into `gold.jsonl`, and report precision and recall for each field. This step is the same for every source.

### Step 5: Canonicalize claims (2h)
This step is not in the original plan, but general inputs need it.
1. **Item keys**: normalize (task_family, episode, item). When episodes are missing, cluster claims by item text embedding within a task family.
2. **Values**: convert to numbers (keeping the rounding precision), resolve place names to codes, and lowercase and strip strings. Record `precision`, so that 16.40 and 16.4 are the same value but 16.38 is a different one.
3. **Techniques**: cluster `technique_share` and `technique_use` texts with embeddings plus an LLM pass that assigns names. Each cluster becomes a technique ID, and a human approves the cluster list.
4. **Variants**: for each item key, the set of distinct `value_norm` values, each with its first-seen time and author.

### Step 6: Exposure table (3h)
For each (agent, episode, item):
- `t_start`: the agent's task-start event. If none exists, use the agent's first event in that episode, or the time of an announcement such as "due in N min" plus N.
- `t_public`: the earliest event that carries the same (item_key, value_norm) and that this agent could see, based on its visibility.
- `D_inferred = t_public < t_start`
- `D_observed`: only when `has_reads` is true. It is true if a read of the carrying channel or event happened between `t_public` and the agent's answer. For context-window reads, check that the value actually appears in the prompt.
- Outcomes: `y_instant` means the claimed latency is at most k seconds, or the wall-clock gap from t_start is at most k. `y_match` means the agent's value equals the earliest public value.
- Rule: use **only** the wall clock. Clocks that agents report are kept as features but are never used for ordering.

### Step 7: Provenance graph (2h)
- Nodes are agents and carry attributes such as cohort, number of active episodes, and first-seen time. Implemented as the agent profile (`analysis/agents.py`, `agent_profiles_{merged,strict}.parquet`): cohort, family, posts with first- and last-seen time, answers and how many had no earlier public copy, answers supplied, relays in and out, facts originated, betweenness.
- Edges come from three sources, each tagged with `kind`:
  1. `citation`: an explicit mention or reference.
  2. `read`: observed exposure, when the source has reads.
  3. `inferred`: the consumer states the same (item, value) within window W after the source posted it, makes no independent-work claim, and the source is the **latest earlier** poster of that value. Report how results change with W.
  4. `reply` (added): an explicit reply link on a threaded source. It is an interaction, not a knowledge transfer, so it counts for brokers (betweenness) and never for supply.
- Store the graph as both NetworkX and a Parquet edge list, and export GraphML for Gephi. Implemented for both mappings as `provenance_{merged,strict}.graphml` and node-link `.json`, with per-kind edge counts and first and last post times on every edge.

### Step 8: Analyses (4h)
Each analysis is a function `analysis(tables, profile) -> (DataFrame, Figure)` that does nothing when the capabilities it needs are missing.

| # | Question | Needs | Method |
|---|---|---|---|
| A1 | Provenance split: what fraction of answers were computed independently vs relayed | wall clock | Share with D=1; break down by task family and episode; headline "N answers, M independent" |
| A2 | Does availability cause copying | wall clock, repeated agents and items | TWFE: `Y ~ D \| agent + item_key` with SEs clustered by agent (`pyfixest`); repeat with D_observed where available; Y = instant answer, value match; exploratory: among exposed rounds, Y on how long the answer had been public |
| A3 | Technique diffusion | technique clusters | Kaplan–Meier time from first share to adoption, among agents active afterwards; median time to reproduction; share of adoptions on a page already carrying the technique against a within-agent, within-family baseline (permutation p) |
| A4 | Error propagation | variants (ground truth optional) | Share of each variant over time per item; time for the correction to reach 50%; per variant, answer rounds derived with nothing public vs adopted after public; if ground truth exists, label which variant is the error |
| A5 | Structure | graph | Out-degree Gini and top-k share under three attribution rules (first source, latest source, same-page relay parent; all and instant answers) beside the per-item exposure Gini; origin of each credited source (computed, no answer round observed, dissent, relayed) and the root of relayed chains; betweenness (exact up to 5,000 agents, sampled from 300 sources above), distribution of relay chain lengths |
| A6 | Reach (from the stretch goal; works on chat) | graph | For each announced fact: number of agents who repeated it, and the hop depth of its cascade |

Assumptions to state in the report: run order is unrelated to ability, and visibility is approximated by the channel model.

### Step 9: CLI and packaging (2h)
```
swarmprov ingest   <path> [--adapter auto|wiki|chat|trace|jsonl --map map.yaml]
swarmprov profile
swarmprov resolve  [--strict|--merged]
swarmprov tag
swarmprov extract  [--model ... --limit N --dry-run]
swarmprov canon
swarmprov expose   [--window 600]
swarmprov graph
swarmprov report   [--out report/]      # markdown + PNG/SVG figures
swarmprov validate gold.jsonl
```
Proposed layout:
```
swarmprov/
  core/  schema.py  identity.py  tagging.py  extract.py  canon.py
         exposure.py  graph.py  analyses/  report.py  cli.py
  adapters/  wiki.py  chat.py  trace.py  jsonl.py
  configs/   openai_wiki.yaml  ai_village.yaml
tests/       fixtures/ (tiny synthetic swarm with known provenance)
```
Each stage caches its output under `.swarmprov/<run>/` so stages can be rerun on their own.

### Step 10: Validation and tests (ongoing)
- **Synthetic swarm fixture**: a script that simulates N agents, some of which copy from a blackboard with known probability. The pipeline must recover the true provenance split and the copy effect. This is the main way to show the tool generalizes.
- Labelled extraction (Step 4) and an audit of identity merges (Step 2).
- Sensitivity checks on the strict vs merged identity mapping, the inferred-edge window W, and the instant-answer threshold k.

### Step 11: Demo datasets
1. The OpenAI wiki swarm (primary): wiki adapter plus `openai_wiki.yaml`.
2. AI Village chat (stretch): chat adapter, running A1, A5, and A6.
3. An optional framework trace that has context windows, to show observed exposure.

### Step 12: Write-up and video
Order: problem → tool → capability matrix → five findings with figures → validation numbers → limitations (identity noise, the visibility approximation, inferred edges) → how to add an adapter.

---

## 3. Build order for a hackathon
1. Step 0 → Step 1 (wiki adapter) → Step 2 → Step 3 gets the end-to-end path working on rules alone.
2. Start Step 4 in the background while building Steps 5–7.
3. Run A1 and A5 first; they need the least. Then A2, A3, and A4.
4. Add the synthetic fixture and the CLI, then the chat adapter if there is time.
