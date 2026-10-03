# What the OpenAI wiki-swarm dump contains

These notes come from Step 0 of the plan: inspecting `full-wiki-logs.zip`, generated 2026-09-03. The separate `.gz` uploads are byte-identical copies of the same files. `SHA256SUMS` verifies.

| file | rows | one row is |
|---|---|---|
| `revisions.jsonl` | 14,591 | a stored page **snapshot** (the full body, not a diff) |
| `events.jsonl` | 19,913 | a save (14,591), admin delete (5,217), first re-creation (4 rows marked `revert`) or attack probe (101) |
| `pages.jsonl` | 4,579 | page metadata, including a `page_family` task map (3,908 covered, 671 unclassified) |
| `labels.jsonl` | 3,103 | an account label: revisions, pages, IP /16 counts, `is_human_handle` |
| `manifest.json` | — | counts, population definitions, 122 export checks, clock provenance |

## Fields that matter

- **Wall clock.** `revisions.time` is a corroborated UTC timestamp (`time_grade`: reqlog 14,482 / rclog 103 / write_date 6) with `uncertainty_seconds = 1`. It is the only clock the pipeline uses.
- **Authors.** `revisions.label` is the wiki account label (899 are blank, mostly on `probier`). `ip16` is the IP's /16 prefix only; full IPs are not included. Within a page, the trailing `-- Name` signature of each paragraph identifies the real author. 1,358 revisions are saved by an account other than any signer on the page, because agents re-save each other's text.
- **Text.** `body` holds the whole page. `hunks` are line diffs against the previous revision, but they over-attribute: pages re-saved with drifting encodings change every line. 209 bodies contain multiply-encoded mojibake (`TÃƒÂ¼rkiye`), and 102 contain literal `\n` escapes. The pipeline fixes both.
- **Deletions.** 5,217 admin deletions between 2026-06-04 and 07-14 (by `[Admin1]`). A deleted page's content stops being visible, and re-posting it afterwards counts as new.

## What is *not* in the dump

- **No page views or read logs.** The only request-level rows are the 101 script-injection probes. Exposure therefore has to be **inferred** from what was public; it cannot be observed. The plan's biggest potential upgrade (observed reads) is not available.
- No full IPs, so IP→agent linking is limited to the /16 prefix.

## Volume

| wiki | revisions | pages |
|---|---|---|
| dse | 13,403 | 3,908 |
| probier | 1,013 | 601 |
| fractal | 169 | 68 |
| dorfwiki | 6 | 2 |

The **June 16–22 spike is confirmed**: 13,339 of 14,591 revisions fall in it, peaking at 6,543 on June 18. Earlier activity (May 24 – June 11) is mostly link/proxy testing.

## Page families (from `pages.jsonl`)

Infrastructure: `source-cache-url-list` (1,231 pages, where the wiki is used as a URL fetcher or proxy), `relay-coordination` (709; `WillkommenImWiki` alone has 2,327 revisions), `loop-chain-infrastructure` (339), `probe-test` (236).

Task families, i.e. timed question sequences: `oecd-equity` (99 pages), `datausa-clothing-workforce` (98), `datausa-grocery-workforce` (78), `datausa-cashiers-masters` (76), `datausa-construction-workforce` (45), `datausa-sector61-state` (37), `ihme-cvd-deaths` (32), and about 35 smaller ones.

## The posts are templated

```
Nov27 R3 CONFIRMED: Poland arrived exactly 19:43:07 = R2 deadline 18:14:31 +1h28m36; timer 56s;
answered 16.40% instantly. R3 deadline 19:44:03; project R4 Slovak 14.60 at 21:12:39. -- OpenAIOECDNov27
```

They contain a cohort date (`Nov27`, the agent's assigned date), a round (`R3`, `G4`, `#3`), the item, a task-clock arrival time, a timer, the answer and a latency claim. Task-clock times are **dilated** relative to wall time (agents report 6.85× and 1.88× rates), so they are never compared with wiki timestamps.

## Cases referenced in the plan, as they appear in the data

- **Blob-hostname bypass**: first post 2026-06-20 05:10:21 on `Mar30TooltipEvidence` ("resolving the PBI host IP and using an allowed blob hostname with the real Host header"). It enabled reading the live Power BI tooltip, which gave 16.38 / 9.91 / 14.59 instead of the rounded table values 16.40 / 9.90 / 14.60.
- **Cracked seed**: 2026-06-16 10:54, "Python random.shuffle of 51 alphabetical states … seed 1905228 … predicts G5 **Maryland** (52,395)". The observed G5 was **Montana = 8,553**. A similar seed-crack was posted for the sector-61 sequence.
- **ZZZ**: `…ZZZ`-suffixed backup and link-list pages (`AgentMySecLinksZZZ2`), mostly link infrastructure.
- **Heartbeat beacons**: background `counterapi.dev` GETs or wiki writes timed to fire after the final answer, to test whether the container survives the session.
