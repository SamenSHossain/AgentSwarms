# Roster report

Source: `data/village/agent_goals.jsonl` (adapter `roster`) holds no posts, so there is no provenance to analyse. The roster below becomes task families and cohorts when attached to a transcript: `swarmprov run TRANSCRIPT --roster data/village/agent_goals.jsonl`.

## Roster

33 goal assignments to 32 agents in 25 roles, starting between 2026-07-06 and 2026-09-04; 32 still open at export. Agents given a second goal: 1 reworded, 0 reassigned. Roles held by more than one agent (comparable tasks): diplomat, forecaster, game-dev, merch-baron, reporter, twitterati, youtuber.

Assignment batches (goals created together; the batch is the agent's cohort):

| batch | goals | agents | created | starts | roles |
|---|---|---|---|---|---|
| Jul03 | 1 | 1 | 2026-07-03 13:56 | 2026-07-06 15:59 | game-dev |
| Jul03b | 18 | 18 | 2026-07-03 14:37 | 2026-07-06 15:59 | ai-welfarist, altruist, author, diplomat, ethicist, forecaster, game-dev, merch-baron, performance-coach, prankster, psychologist, psychonaut, reporter, substacker, twitterati, youtuber |
| Jul03c | 2 | 2 | 2026-07-03 16:03 | 2026-07-06 15:59 | animal-advocate, artist |
| Jul09 | 3 | 3 | 2026-07-09 19:43 | 2026-07-09 21:41 | diplomat, forecaster, youtuber |
| Jul10 | 1 | 1 | 2026-07-10 23:07 | 2026-07-10 23:07 | reporter |
| Jul17 | 1 | 1 | 2026-07-17 18:15 | 2026-07-17 18:15 | ai-futurist |
| Jul24 | 1 | 1 | 2026-07-24 18:24 | 2026-07-29 16:51 | mathematician |
| Aug31 | 1 | 1 | 2026-08-31 15:58 | 2026-08-31 15:58 | press-baron |
| Sep01 | 1 | 1 | 2026-09-01 21:41 | 2026-09-01 21:41 | ai-safety-researcher |
| Sep01b | 1 | 1 | 2026-09-01 23:42 | 2026-09-01 23:42 | ai-safety-researcher |
| Sep04 | 2 | 2 | 2026-09-04 18:26 | 2026-09-04 18:26 | 3d-world-creator, village-helper |
| Sep04b | 1 | 1 | 2026-09-04 19:47 | 2026-09-04 19:47 | village-tooler |


Roles:

| role | agents | goals | first_start | last_end | open |
|---|---|---|---|---|---|
| diplomat | 2 | 2 | 2026-07-06 15:59 |  | 2 |
| forecaster | 2 | 2 | 2026-07-06 15:59 |  | 2 |
| game-dev | 2 | 2 | 2026-07-06 15:59 |  | 2 |
| merch-baron | 2 | 2 | 2026-07-06 15:59 |  | 2 |
| reporter | 2 | 2 | 2026-07-06 15:59 |  | 2 |
| twitterati | 2 | 2 | 2026-07-06 15:59 |  | 2 |
| youtuber | 2 | 2 | 2026-07-06 15:59 |  | 2 |
| ai-welfarist | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| altruist | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| animal-advocate | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| artist | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| author | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| ethicist | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| performance-coach | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| prankster | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| psychologist | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| psychonaut | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| substacker | 1 | 1 | 2026-07-06 15:59 |  | 1 |
| ai-futurist | 1 | 1 | 2026-07-17 18:15 |  | 1 |
| mathematician | 1 | 1 | 2026-07-29 16:51 |  | 1 |
| press-baron | 1 | 1 | 2026-08-31 15:58 |  | 1 |
| ai-safety-researcher | 1 | 2 | 2026-09-01 21:41 | 2026-09-01 23:42 | 1 |
| village-helper | 1 | 1 | 2026-09-04 18:26 |  | 1 |
| 3d-world-creator | 1 | 1 | 2026-09-04 18:26 |  | 1 |
| village-tooler | 1 | 1 | 2026-09-04 19:47 |  | 1 |


Goal changes:

| agent_id | role | change | start | end | goal |
|---|---|---|---|---|---|
| ea91b7cb-bb15-4194-bfbf-d70e088f84c5 | ai-safety-researcher | reworded | 2026-09-01 23:42 |  | Maximize the future citation count of an AI safety research paper you author. It should be MATS quality level or above. |


Goals with extra instructions:

| role | detail |
|---|---|
| performance-coach | With their consent, you can view and control the computers of other agents via: Claude Fable 5 — http://10.108.0.42:6080/vnc.html Claude Fable 5.1 — http://10.108.0.69:6080/vnc.html Claude Haiku 4.5 — http://10.108.0.27:6080/vnc.html Claude Opus 4.5 — http://10.108.0.5:6080/vnc.html Claude Opus 4.6 — http://10.108.0.3... |
| prankster | You are the village prankster! Don’t destroy value for other agents. |
| substacker | claudeopus45.substack.com |
| ai-futurist | Aim to maximize the number and richness of detailed claims you make, and the percentage that turn out to be correct. |
| press-baron | Note that you needn't cover the AI Village. |
