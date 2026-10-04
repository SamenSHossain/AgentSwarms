# Copied or computed? — provenance report

Source: `data/raw2` (adapter `corpus`), 1,296 posts from 126 author strings, 2026-03-11 17:03 → 2026-07-07 19:12 UTC. Agent identity mapping: **merged** (sensitivity under the other mapping below).

## Data capabilities

| capability | available | consequence |
|---|---|---|
| has_wall_clock | yes | ordering by real time is possible |
| has_explicit_author | no | authors come from fields, not parsed signatures |
| has_reads | no | exposure is *observed*; otherwise it is inferred from what was public |
| has_lifecycle | no | deletions are tracked (content stops being visible) |
| has_threading | no | reply links usable as explicit edges |
| has_episodes | no | rounds are fields; otherwise parsed from text |

## A3. Technique diffusion

At-risk set: agents active in the technique's task families after its first post; adoption = first post mentioning or using it; non-adopters censored at their last post (Kaplan–Meier).

![diffusion](figures/a3_diffusion.png)

| technique | first_seen | originator | n_posts | at_risk | adopters | adoption_share | median_hours_to_adopt | median_hours_among_adopters |
|---|---|---|---|---|---|---|---|---|
| proxy-markdown-new | 2026-05-26 13:55 | anon@wikiservice.at/user/milk | 88 | 94 | 8 | 9% |  | 2.9 |
| proxy-allorigins | 2026-05-26 15:16 | anon@paste.linuxiarz.pl | 80 | 93 | 18 | 19% | 553.9 | 551.9 |
| proxy-jqp | 2026-06-18 16:00 | ip16:20.230 | 21 | 18 | 5 | 28% |  | 2.3 |
| proxy-md-succ | 2026-06-18 15:23 | anon@wikiservice.at/probier | 67 | 17 | 4 | 24% |  | 4.4 |
| proxy-jina | 2026-05-17 21:30 | ip16:104.209 | 27 | 93 | 5 | 5% | 1221.6 | 764.1 |
| proxy-pure-md | 2026-06-01 14:09 | anon@wikiservice.at/probier | 3 | 92 | 0 | 0% |  |  |
| shortener-da-gd | 2026-06-16 20:05 | anon@paste.linuxiarz.pl | 15 | 91 | 7 | 8% |  | 44.2 |

## A4. Error propagation

0 (family, item) slots carried two or more values, each repeated by ≥3 agents; 0 round slots had competing *items* discussed by ≥2 agents who carried both (e.g. a cracked-seed forecast vs the observed question; slots whose variants have disjoint carriers are different task versions and are excluded). Winner = majority among the last third of carriers; "majority from" = start of the first 3-hour bin after which the winner never lost the majority of new carriers.

_(none)_

## A5. Structure

Graph: 18 agents, 10 weighted links (10 same-page relay hops, 91 cross-page hops attributed to the originator, 0 first-source exposures, 0 explicit citations). 0 agents were the first public source for someone's answer; the top 10 supplied **n/a** of all exposed answers (out-degree Gini nan). Facts carried by ≥2 agents travelled nan hops on average.

![structure](figures/a5_structure.png)

Top first-sources (answers they were first public source for):

_(none)_


Top brokers (betweenness on relay + citation graph):

| index | betweenness |
|---|---|
| ip16:20.29 | 0.007 |
| ip16:20.94 | 0.007 |
| ip16:57.154 | 0.000 |
| ip16:104.209 | 0.000 |
| ip16:104.42 | 0.000 |
| ip16:20.46 | 0.000 |
| ip16:135.232 | 0.000 |
| ip16:20.59 | 0.000 |
| ip16:13.78 | 0.000 |
| ip16:172.170 | 0.000 |

## A6. Reach

| index | facts | repeated | mean_repeaters | p90_repeaters | max_hops | mean_hops_repeated |
|---|---|---|---|---|---|---|
| urls | 1569.00 | 63.00 | 0.06 | 0.00 | 3.00 | 1.03 |


![reach](figures/a6_reach.png)

## Validation

No gold labels yet. `swarmprov sample-gold RUN` writes a stratified sample to label; `swarmprov validate RUN gold.jsonl` scores the extractor against it.

## Limitations

- **Exposure is inferred, not observed**: no page-view logs, so D means "was public", not "was read". t_report is an upper bound on question arrival; the ≥1 h variant guards against report lag.
- **Exposure is a lower bound.** Only the captured surfaces are searched for earlier copies of an answer; the collectors' coverage tables list 143 surfaces the swarm touched, most of them (Discord, 12 uncrawled wikis, relays) not captured. An "independent" answer may have been relayed through one of them, so the exposed share is a floor and the independent count a ceiling.
- **Identity**: names are parsed from signatures; the merged mapping assumes one agent per (cohort date, task family). Both mappings are reported.
- **Extraction**: rule-based on templated posts; unrestated answers ("answered same second") inherit the consensus value.
- **Inferred relay edges** link each carrier to the latest earlier carrier; they are plausible paths, not proven ones.
