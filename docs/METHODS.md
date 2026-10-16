# Methods and limitations

What each analysis computes, the rules behind its labels, how it was checked against synthetic swarms with known provenance, and where it stops. The commands, inputs and outputs are in [USAGE.md](USAGE.md); the README has the results.

## Posts, exposure and identity

### Unit of authorship

A wiki revision stores the whole page, so the unit is the *paragraph a revision added*. A paragraph counts as new the first time its normalised text appears on the page since the page was last deleted. Line diffs would over-attribute, because repeated mis-decoding re-encodes every line on each save. The author is the trailing `-- Signature`, not the account that saved the page.

### Exposure is inferred

The dump contains saves, deletions and probes but **no page views**, so "public" does not mean "read". `t_report` is an upper bound on arrival; the ≥1 h variant is robust to report lag. Only wiki timestamps are used as a clock, because the agents' task clocks are dilated.

### Identity

`merged` = (cohort date, task family), where the cohort date comes from names like `OpenAIOECDNov27` or, for signature-parsed sources, the post's opening token. `strict` = the signature itself, case-folded; case variants share one merged id. Every headline number is reported under both (where both group the author strings identically, one line says so instead), and `agent_aliases.csv` lists every merge for audit.

### Coverage

`coverage_note` is the Limitations bullet on what the exposure search could not see. The wiki and corpus configs cite the collectors' inventory of 143 surfaces (the corpus note adds that a corpus run does not search the wiki dump); without a note, the report says that only the transcript's own posts were searched, so the exposed share is a floor.

### An own lookup is not a computation

A tool result before the answer shows the agent fetched the value, but a tool can echo what the agent passed in (a search for a value it had already read), so the flag is `own_lookup_evidence` and the report says *own lookup preceded report*, never *computed*. On the synthetic AutoGen swarm (seeds 0–7, on message order) an own lookup before the report marks exactly the computed rounds, and a public copy with no own lookup gives the true copied share (55–62%); `own_lookup_evidence` itself, which also needs no earlier public copy, flags 44–56% of the computed answers (those whose value was not yet public) and no copied one. Attribution without call ids is guesswork under parallel calls, so a result that neither an id, an AutoGen `source` nor one agent's pending calls settles stays with `tool`. Treating unnamed user turns as the task can mislabel a human's correction as task input.

### Message order is a clock, not a time

A sequence clock preserves what the exposure test needs, which post came first. On the synthetic swarm with its timestamps stripped, the exposed share (80.7%), the top-10 source share and the A2 consensus estimate equal the wall-clock run's; A4 uses a different rule: a later winner must hold the majority of all the slot's carriers, not of every 3-hour bin from some point on. Nothing measured in hours is converted into messages, and the order is only as good as the order the file was written in.

## Intervals and sensitivity sweeps

The report's intervals are seeded, and `bootstrap_reps` (default 1000, a whole number; `report --boot N` overrides it, `0` turns them off, and anything else stops the report with a message) sets how many resamples they take (`analysis/uncertainty.py`). A1's shares and median head start get 95% percentile intervals from resampling whole items (`item_fe`), and so do the family and round rows that have at least 5 items, each resampling its own; the median head start needs 5 items with an exposed answer, and a statistic undefined in more than 5% of the resamples gets no interval rather than one taken over the rest.

A6's `mean_repeaters` resamples facts (5 or more); A3's Kaplan–Meier median carries Greenwood's interval (log-log) where the Kaplan–Meier curve falls to one half (non-adopters are censored, so this can happen with fewer than half adopting). A5's top-10 share and Gini get no bootstrap interval: resampling items repeats their sources and inflates concentration, so the report gives the lowest and highest value with each item left out in turn, beside the range across attribution rules.

Three sweeps show what the analyst-chosen thresholds do. A1's *exposure vs required head start* gives the share of answers public at least *t* before the report for *t* from 0 to 48 h, over every answer (and under `D_visible` when deletions are recorded) and for each family with at least 40 answers, as a table, `figures/a1_threshold.png` and `summary.json` `A1.threshold_curve`. A1's instant-answer row recounts the instant share for a threshold *k* of 1, 2, 5 and 10 s and the configured `instant_threshold_s` (only answers that state their latency as a number move; at the configured *k* the report counts the answers whose number and class disagree, since the wording "instantly" sets A1's *instant* class whatever the number). A6's *sensitivity to the relay window W* re-infers the task-value relay trees with a parent allowed only within W (10 min, 1 h, 6 h, 24 h, none): a carrier whose parent posted earlier is left unattributed, a root of depth 0 (`graph.relay_edges(max_gap_s=W)`; the graph itself keeps no window). URLs are left out of that sweep, because their trees take seconds to re-infer for each window. Under a sequence clock the hour-based sweeps are replaced by a needs-a-wall-clock line.

### Intervals cover sampling noise across items only

One first poster supplies most of an item's exposures, so answers to one item are not independent: on the wiki, resampling single answers gives the ≥1 h share an interval about a third as wide as resampling items (0.05 against 0.13 wide), and resampling agents hardly widens it (0.06). Items are the cluster.

Families are too few to resample (23 on the wiki), and a family row from fewer than 5 items gets no interval. No interval covers identity-mapping or extraction error, which the strict-mapping sensitivity lines and the validation section address instead.

On the synthetic swarm, the threshold curve and the instant-threshold sweep reproduce the truth at every threshold, and A1's exposed share equals D's target computed from the truth (the share of answers whose value another agent had reported earlier), so its interval contains that target by construction; it does not contain `public_at_arrival` (0.833 against 0.807, interval 0.783–0.828), which counts any earlier post on the item, whatever its value.

Coverage itself is checked by simulation (`tests/test_uncertainty.py`): over 200 draws of 40 items whose answers mostly agree, the item interval covers the population share 93% of the time (a percentile interval runs a little narrow on few items), and one from resampling single answers 52%.

## A2 Does availability cause copying?

### The causal treatment `D_cons`

`D_cons` asks whether the *consensus* value was public, regardless of what this agent answered. A first version used the agent's own value, which mechanically ties wrong answers to "unexposed" and produced a spurious +0.43 effect.

### The reading rule

Only agents whose exposure status switches between their rounds identify an effect (`agents_with_D_variation`), and in this kind of swarm the answer is public before almost every report, so few do. A row with fewer than 10 switching agents is *not identified*; with 10–19 it is *weakly identified* (its note says so and the figure draws it grey); with 20 or more it reads *identified* where its 95% CI excludes 0 and *no effect detected* where the CI covers 0. `causal.reading` is that one rule, and the table's notes, the key findings and the `report.html` tile all read it. On the wiki most rows have 3–17 switchers; the one with 37, `y_instant` under `D_cons_w60m`, detects no effect.

### A dose response identifies only the intensive margin

Exposure stores how long the consensus value had been public before each report (`t_cons`, `gap_cons_s`), and A2 adds an exploratory block below its table: among exposed rounds only, the instant outcome on `log1p(head start in hours)` (the slope) and on bins against a head start under 1 h, with agent + item FE. Within an agent the head start varies across items where exposure barely does, but it answers a narrower question, whether reuse grows with how long the answer had been public given that it was, and leaves whether exposure matters unidentified.

Its assumption is that, within an agent, how long before its report others had posted the consensus is unrelated to its item-specific propensity to answer instantly. Both outcomes measure the head start to the report, so a late report gets a longer one. `y_instant`, which scores a round with no stated latency 0, is biased besides: on the wiki, within an agent, rounds with longer head starts state a latency less often (−0.18 per unit of log1p hours). `y_instant|known` avoids that scoring but keeps only the rounds that state a latency, so its sample is selected on the dose: the less biased read, not an unbiased one.

Only agents whose outcome varies inform an estimate (one that answers every exposed round the same way moves it only through the item effects), so the block counts them per term. An outcome is identified when the within-agent SD of the log head start is at least 0.25 and at least 10 agents have rounds in two or more bins; each term then needs 10 informative agents (the slope: head start and outcome both vary; a bin: a round in it and in another bin, and an outcome that varies), and so does the reference; an estimate on fewer than 20 (a bin: in it or in the reference) is noted *weak* or *shape only* and drawn grey. The slope also gives its exact leave-one-agent-out range and the share of its clustered variance that its two largest agents carry.

On the wiki, among rounds with a stated latency, 127 agents' head starts vary but only 26 also vary in the outcome (against 4 that switch exposure status); the slope is −0.010, but two agents carry 87% of its clustered variance and leaving one agent out moves it between −0.073 and −0.002, so its SE (0.013) understates the uncertainty, and only 2 agents inform the reference bin, so no bin is identified.

On synthetic swarms of 120 agents (seeds 0–7) with copying that ramps up over 4 h (`--copy-ramp-h 4`) the slope is positive on all 8 and significant on 6, over 12 h on all 8 and 5, and its 95% interval covers the design's own target (the same regression on each round's true copy probability) in 23 of 24 runs including 8 with no ramp, where it is never significant. The bins are less reliable: 2–14 agents inform the reference there, so on most runs no bin is identified and elsewhere each is shape only; with no ramp none of the 6 bins estimated differs from zero at the 5% level, with a 4 h ramp 7 of 12 do. This block stays out of A2's reading.

### The consensus is not the later majority

`y_consensus` scores against the value most distinct agents carried over the whole run, which on the wiki is often the older, more-copied value; A4's winner is the majority among the last third of carriers. A2 keeps `y_consensus` as its outcome, adds `y_winner` for overturned slots as an extra row, and warns how many answer rounds the two disagree on.

## A3 Technique diffusion

### Techniques and contagion

`techniques` are a name and a regex each (optionally `families`, which restrict the at-risk set). Beside the Kaplan–Meier curve, A3 asks whether adopters adopted where the technique was already posted (`diffusion.contagion`, same at-risk set): the share of adoptions on a page another agent had posted it on strictly before, against each adopter's own posts from the first appearance on in the adoption post's family, with an exact within-agent permutation p-value per technique. A share above the baseline with p < 0.05 after Holm's correction for the techniques tested (`p_holm` in `summary.json`) reads *consistent with contagion*, and one below 0.05 only uncorrected is named as such; fewer than 10 adopters get a warning; a technique with a post that matches its pattern only by naming a page (a link to `AgentLinksZZZ2` under `ZZZ`) is not tested, one whose pattern merely also matches page names is tested and the report says how many of its adoptions landed on such a page, and the test needs posts on two or more pages or channels.

### Contagion is a page test, not a reading test

A3's contagion columns ask whether an adoption landed on a page that already carried the technique more often than the adopter's other posts did. Posting on a page is not reading it, and a task can send agents to the pages that carry a technique (a common cause), so the baseline stays within the adoption post's family and the result reads *consistent with contagion*, never proof, and only after Holm's correction for the techniques tested. A per-agent-hour rate ratio before and after exposure was left out: agents post more once they reach a task's pages, which inflates it.

## A4 Error propagation

### Disputes

A4 finds (family, item) slots with competing values, plus any declared in `disputes`, and gives each a status: *held*, *overturned* or *contested*, with the late-carrier counts behind it. `reference_values` (`{"family|item": value}`) is analyst input: it fills a separate outcome column and never changes the status. A held or overturned slot reads truth held, error corrected, error persisted or truth displaced by error (*precision variant* when one value is the other rounded); a contested slot reads *unresolved*. A reference matches a variant by its text or its comparison key (`1,481` matches `$1,481`); one that matches no variant says so, and the report lists references that name no item or disputed slot in the run, so a mistyped key shows. The wiki config sets the live-tooltip values for the three OECD disputes; the synthetic config carries the true values.

Each value slot's variants are also split by answer round (`errors.variant_provenance`): rounds that state the variant with nothing carrying it public before the report (D = 0, an upper bound on independent derivation) against rounds that adopted it after it was public (D = 1); rounds that state no value inherit the consensus in exposure and are left out, and these are answer rounds, not the mention-level carriers in *variants (agents)*. The table flags slots whose consensus (the value most distinct agents carried, which `y_consensus` scores against) is not the winner, and *precision variants*, where every value is another one rounded.

`exposures_*` carry `y_winner` beside `y_consensus`: whether the stated value is the winner, on overturned slots only (NaN elsewhere, and for a round that states no value, since a report's own correct or wrong flag grades the answer against the task, not the winner); A2 adds it as an extra row and warns when the consensus and the winner disagree on slots with answer rounds.

### A4 status is a convention

The winner is the most common value among the last third of a slot's carriers (none on a tie). If it is the first value, the slot is *held*. A later winner counts as *overturned* only if it also held the majority of new carriers in every 3-hour bin from some point on, with at least 5 late carriers in all; otherwise the slot is *contested*. Two values first seen at the same moment leave no first value, so that slot is contested too, whichever way the labels sort. The report prints the bin width and the tail share.

## A5 Structure

### Hub concentration depends on the attribution rule

Exposure says a value was public, not whom an agent took it from, so A5 credits each exposed answer three ways: the first other agent to post the value before the report, the latest one, and the same-page relay parent (joined on the same value key exposure matched; a parent whose post is not before the report credits nobody, as in `trace`; the unmatched count is printed). It reports all three, on every exposed answer and on instant answers only, plus the Gini of the items' exposure counts, which is what first-source credit gives if each item has a single source.

On synthetic swarms (seeds 0–7) the rule that matches how the agents copy names the true source for 98–100% of copied answers and reproduces the true top-10 share to within 0.01, while the other rule names it for 19–28% and misses the share by 0.36–0.55 (first-source credit reports 64–76% where copiers take the latest post and the truth is 26–28%). With no copying at all, first-source credit still puts 68–80% of exposed answers on its top 10, and with `--copy-from random` neither rule names the source for more than 57%. So the rules are a sensitivity range: a rule that wins on the synthetic swarm only matches how the simulation copies.

### Self-originated answers credit no source

No rule credits a *self-originated* answer, one whose agent had posted the value before any other agent did (`exposure.self_originated`, `t_self_prepared < t_public`): `D` stays 1, as the value was public before the report, but the other agent's copy is a later repeat, not its source, so the answer gives no exposure edge, no supply in the agent profiles or `a7_supply_network.png`, and its own row in the origins table. A1 counts these apart (`n_self_originated`) and bounds independent lookups by their sum with `n_independent` (`n_independent_at_most`); on the wiki that is 6 answers under each mapping.

### Where copied answers came from

`analysis/origins.py` labels each credited source by its own answer round on the item (computed, no answer round observed, dissent, relayed, or *posted before its own source* when that round is self-originated, so a source that posted the value first is its origin, not a relay of the later copy) and follows relayed sources to a root, stopping at a source that had posted the value before the post it is credited to (*posted before its own source* if no other agent had posted it earlier, else a *chain cut*, which only the latest rule produces); `summary.json` carries both as `A5.attribution` and `A5.origins` (origin counts under `by_mapping`, roots for the report's `mapping`). *Relayed* means exposed, not copied, so on the copy-latest swarm those chains end at the true root for only 51–68% of copied answers.

### Replies are interactions, not knowledge transfers

A reply edge says one agent answered another's post, not that it took a value from it, so A5 counts replies for brokers (betweenness) and the GraphML only: never for the exposure out-degree, the Gini or top-10 shares, the attribution rules or the relay counts.

## A6 Reach

### Relay edges are plausible paths, not proven ones

Each carrier is linked to the latest earlier carrier on the same page, and to the originator if there was none, at any lag; A6's window sweep shows how much of the depth needs hops longer than W (it does not pick a W). `trace` labels the relay parent as inferred and keeps it apart from the first public copy. On the synthetic swarm, where copiers take the latest post, the relay parent names the true source for at least 98% of copied answers and the first public copy for about a quarter. When copiers take the first post (`--copy-from first`), the first public copy is right every time and the relay parent about a fifth of the time.

## Validation

### The synthetic scorecard holds under the simulation's own model only

`swarmprov truth-score` compares a run of `synth` output with the truth beside its transcript (`truth_validation.json`, shown in the report's Validation section). Over seeds 0–7 of three 60-agent scenarios: copying the latest post at rate 0.7 on one board, `D` agrees per round with the answer being on the board at arrival for 95–97% of rounds (D is value-specific: a wrong computed answer has no earlier copy, and a copy posted between arrival and report counts) and with an earlier post of the same value for 100%; A2's 95% interval covers the true effect (the share of rounds public at arrival that copied) in 8 of 8; the latest-source rule names the true source of 98–100% of copied answers, with a top-10 share on instant answers within 0.005 of the truth; and A4's status and outcome match the same rule run on the true answers on every slot.

Copying the first post at rate 0.3 on three boards, with 20% of posts under an alias, 10% deleted and a planted technique: `D` agrees for 90–93%, `D_visible` for 100% against an earlier copy still undeleted at the report; A2 covers the truth in 8 of 8 (significant in 7); first-source credit names the source of 87–100%, every miss a copy of a first post deleted before the copier read the board, where the first copy still visible names it every time; 80–85 cross-page hops; A3's at-risk set, adoption share (83–100%) and Kaplan–Meier median equal the planted ones; and the merged mapping joins every alias (purity 100%, no agent split) where the strict one splits 36–41 agents.

With no copying and 30% fast agents (the latest to start, instant on half their questions), instant answers run 14–22 points more often in exposed rounds when agents are not held fixed, yet A2's estimate is significant in none of seeds 0–9 (p 0.06–0.89, |β| ≤ 0.10, its interval covering 0 each time). The pipeline recovers what this simulation plants; a real swarm's agents need not copy, delete or sign the way it does.

### Integrity checks

The report's Validation section opens with an integrity line: `swarmprov check` (which the report runs) re-derives every exposure column from the mentions with a plain loop of its own (the first and latest earlier copy by another agent, the first still undeleted at the report, the agent's own earlier post and the first copy of the consensus value, and the flags, times, sources and head starts built from them), checks that the flags nest (`D_w60m ≤ D_w10m ≤ D`, `D_visible ≤ D`, `D_cons_w60m ≤ D_cons`), that every multi-carrier fact has one relay hop per carrier after the first, every exposed round one exposure edge from its source and a self-originated one (its agent had posted the value before the first public copy) none, that every edge and exposure names posts the run holds, that the strict mapping refines the merged one (a handle whose case variants carry conflicting cohorts or roster ids, which identity keeps apart by design, is a warning), and that `summary.json` agrees with the tables. The line names the checks that ran and those the run gave nothing to compare (a run with no answer rounds has no exposure checks). On the wiki all 19 pass.

### Extractor labels

The hand labels behind the extractor's scores, and the scores on its three samples, are in the README's [Validation](../README.md#validation). A label file is scored only on the labelled posts the run holds, and one with none of them (labels of another source) is skipped.

## How the report is written

### The key findings are generated

`report.key_findings` writes the block that opens every report with posts: one bullet per analysis that ran, or the line a skipped one gives (kept in `summary.json` as `skipped`), each stating the definition it rests on. Every number in it is a `summary.json` value formatted, never recomputed (`tests/test_headline.py` parses each back), so the block, `headline.md` (`headline_strict.md` under strict) and `summary['headline']` cannot drift from the report.

A1 gives the exposed share with its interval, the share an hour before, the median head start, the independent count (with the self-originated answers counted apart, so the bound is at most their sum) and its `D_visible` variant (with the self-originated answers `D_visible` does not already count added, `n_independent_at_most_visible`), as floors; A2 reads each headline row by one rule (`causal.reading`, which the table's notes and the tile read too): *identified* with at least twice the table's minimum of switching agents (20) and a 95% CI that excludes 0, *no effect detected* with as many and a CI that covers 0 (each such row with its switchers and interval), and otherwise *not identified* with the range of agents that switch (a weakly identified row, 10–19 switchers, is not identified here), naming the outcomes and treatments it tried when no row is either (`y_winner`, an extra row, is no headline), and the tile reads *mostly not identified* when most headline rows are not identified, naming the row with no effect detected beside them; A3 the largest adoption shares (and the techniques consistent with contagion); A4 each featured slot's first variant, winner, status and outcome; A5 the top-10 share's range across the attribution rules that credit every exposed answer (first and latest source), then each rule's share, the relay parent's of the answers it matches, with first-source credit's leave-one-item-out range (or, with no source credited, how many exposed answers are self-originated); A6 the repeated facts and the longest chain. A report with no posts has none. `swarmprov compare` leaves `headline` out, since it restates values compared one by one.

The README's results tables are written by hand, so they can drift from the code; the report's key findings, `headline.md` and the HTML tiles are formatted from `summary.json` alone, each bullet names its definition and says *floor* where exposure is one, and a test parses every number back. Quote `headline.md` rather than copying numbers. The README's own numbers are checked the same way: `tests/test_results.py` parses each row of the headline table, the wiki results table, the integrity line and the cross-site and corpus text, and the cross-site table in `docs/DATA.md`, and formats each number from the committed summaries (the wiki's, the synthetic swarm's scorecard, the cross-site and corpus ones), the cross-site report and its `technique_spread.csv`, so a regenerated summary that moves a number fails the suite until the README is updated; the A2 row's wording is checked against `causal.reading`. It also requires each committed key-findings block, `headline.md` and `summary['headline']` to be what `report.key_findings` writes from that summary, and the wiki's `report.html` to render the committed `report.md`, so a wording change fails until `results/` is regenerated.

### Figures draw what the tables say

The A4 figure counts every carrier, so each line ends at its variant's agents; when the earliest 3% of carriers would take more than a third of the time axis (a post days before the dispute), the axis starts after them and the legend counts them and gives the first one's time. It draws at most three variants: the first to appear, the winner (or every variant tied for the lead of the late carriers) and the next to appear. The legend marks the winner (none on a tie) and when its lasting majority began, also drawn as a dashed line, and the lower row gives the winner's share of new carriers in each 3-hour bin (each drawn variant's on a tie), leaving empty bins out and drawing bins with fewer than 3 new carriers faint.

The A2 figure draws every row: one with fewer than 10 agents whose exposure status switches reads *not identified* at zero, each label gives that number, and an estimate from fewer than 20 is grey; its dose figure (`a2_dose.png`) draws each head-start bin against the reference (hollow at zero), reads *not identified* where under 10 agents inform a bin, greys a bin under 20 (in it or in the reference) and gives each slope's leave-one-agent-out range.

The A5 Lorenz curve marks where the top 10 first sources begin, A6 pools facts repeated by 20 or more agents into one labelled point, and percent axes print the decimals their range needs (a tick at 2.5% reads 2.5%, not 2%).
