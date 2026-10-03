# swarmprov: general pipeline plan

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
agent_id, ts, channel, event_id | None, method  # "pageview" | "context_window" | "tool_read"

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
- **chat** (AI Village, Slack, Discord exports): one event per message, with visibility set to the room.
- **framework trace** (AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Claude Agent SDK): one event per message. Each LLM call's prompt becomes a `reads` row with `method="context_window"`, which gives observed exposure for free.
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
- Nodes are agents and carry attributes such as cohort, number of active episodes, and first-seen time.
- Edges come from three sources, each tagged with `kind`:
  1. `citation`: an explicit mention or reference.
  2. `read`: observed exposure, when the source has reads.
  3. `inferred`: the consumer states the same (item, value) within window W after the source posted it, makes no independent-work claim, and the source is the **latest earlier** poster of that value. Report how results change with W.
- Store the graph as both NetworkX and a Parquet edge list, and export GraphML for Gephi.

### Step 8: Analyses (4h)
Each analysis is a function `analysis(tables, profile) -> (DataFrame, Figure)` that does nothing when the capabilities it needs are missing.

| # | Question | Needs | Method |
|---|---|---|---|
| A1 | Provenance split: what fraction of answers were computed independently vs relayed | wall clock | Share with D=1; break down by task family and episode; headline "N answers, M independent" |
| A2 | Does availability cause copying | wall clock, repeated agents and items | TWFE: `Y ~ D \| agent + item_key` with SEs clustered by agent (`pyfixest`); repeat with D_observed where available; Y = instant answer, value match |
| A3 | Technique diffusion | technique clusters | Kaplan–Meier time from first share to adoption, among agents active afterwards; median time to reproduction |
| A4 | Error propagation | variants (ground truth optional) | Share of each variant over time per item; time for the correction to reach 50%; if ground truth exists, label which variant is the error |
| A5 | Structure | graph | Out-degree Gini, share of relays from the top-k agents, betweenness, distribution of relay chain lengths |
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
