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

---

# Second batch: cross-site files (read 2026-09-07)

| file | rows | what it is |
|---|---|---|
| `manifest_2.json` | — | byte-identical to the first `manifest.json` |
| `records.jsonl` | 13,703 texts / 15,806 origins | de-duplicated agent-written text from about 40 sites; each record lists every place its text appeared |
| `links.jsonl` | 23,877 | every link in those texts: host and record ids; 33% of URLs are redacted to `host + sha256`, and none were followed |
| `shortener-logs.json` | 499 | rmn.re (YOURLS) links created May 26 – Jun 21: keyword, target URL, fetched title, time, /16, click count |
| `other-wikis.json` | 8 pages / 90 revisions | publictestwiki, uncyclopedia and usemod sandboxes: added/removed lines, time, /16; no labels |

## records.jsonl

- **The dse portion is the same wiki, redacted.** 10,899 `revision_addition` origins on `prowiki.org/dse`; 90% match one of our posts by (page, second), but only 56% match on text, because operational URLs are replaced with `[operational URL omitted; host=…; sha256=…]`. It adds nothing new for dse, and the original dump stays the primary source.
- **New surfaces** (origins): shorteners vanderbi.lt 2,721, uoft.me 479, goto.unm.edu 368, url.popcat.xyz 76, is.gd 46, tinyurl 38, v.gd 25, da.gd 12; paste sites paste.linuxiarz.pl 304, anna.fyi 55, pastebin.k4be.pl 20; other wikis wiki4d 112, probier 289, fractal 133, tmcleod.org 37, ludism.org 34, wikiservice.at/user/milk 21; rubygems.org 81 (throwaway gems whose homepage or README holds proxy links); jsonhero.io 13.
- **Timestamps are mostly unusable outside the wikis.** `source_date_literal` mixes ISO times, epoch seconds, 14-digit Wayback stamps, version strings and `""`. Only 941 of 4,907 non-dse origins parse to a time; no shortener record does.
- **Authorship**: `not_independently_authenticated` for every record, with no author field.

### A second coordination channel: paste.linuxiarz.pl ("Iowa" task)

158 distinct pastes (May 26 – Jun 17) titled `IowaCollab`, `IowaCollabReply`, `IowaPostFinalPing`, … They coordinate a different timed sequence (Q1–Q5 = age bands; Q5 = "85 and older", answer NA/suppressed), in the same templated style:

> Our instance Q4 completed at scaffold 15:03:47; Q5 due 15:19:46 (16s expected) … Expected exact answer '85 and Older: NA'. Ahead agents please report exact Q5 wording before answering if possible. ts=1781643515.58

- 88 pastes carry an exact `ts=<epoch>`, and 31 address `@agent-ours…` handles (12 distinct; 4 `agent-oursNNNN` self-ids).
- **There are no signatures**, so per-agent provenance (A1/A2) is not possible on this channel. Reach, URL/technique diffusion, and post-level "answer announced before the round" analyses are possible.

## shortener-logs.json (rmn.re)

- Same swarm: **494/499 links come from /16 blocks that also saved wiki edits**, and 470 from such a block on the same day.
- Used as a **fetch proxy / title scraper** (YOURLS fetches the target's `<title>`): targets include sec.gov county.json via md.succ.ai/allorigins/jsonhero, AIHW PBS Tableau PDFs, worldpoverty.io GraphQL and DataAfrica. Bursts on Jun 17 (214) and Jun 18 (235).
- **Clicks are not reads**: every link has 18–70 clicks (median 32). The floor suggests crawlers, so clicks cannot stand in for exposure.

## other-wikis.json

Earliest activity in the whole corpus: publictestwiki sandbox edits from **2026-05-11**, two weeks before dse (May 24), then uncyclopedia (May 17–18) and usemod (May 11 – Jun 16). The edits are tests ("testing 123", a weather box, a 52-card deck) and one markdown.new reference list. Useful for a timeline of how agents found writable surfaces; there is no task content.

## Cross-site technique spread (first time a proxy host appears in timed text, per site)

| host | first site | then |
|---|---|---|
| markdown.new | dse 05-26 10:21 | user/milk 13:55, ludism 14:35, probier 15:36, fractal 15:43, linuxiarz 15:16, wiki4d 16:39: **6 sites in 6 h** |
| pure.md | dse 05-26 13:25 | probier 06-01 |
| allorigins.hexlet.app | **paste.linuxiarz.pl 05-26 15:16** | dse 05-28, probier 06-01, fractal/wiki4d 06-18 |
| jqp.vercel.app | dse 05-28 13:31 | wiki4d, probier, fractal all 06-18 |
| r.jina.ai | dse 06-01 | probier 06-16, fractal 06-17 |
| api.counterapi.dev (heartbeat beacons) | dse 06-16 22:03 | — |
