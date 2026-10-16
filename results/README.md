# Committed results

Each directory is a run's report, summary and the figures its report links, copied without its Parquet tables and written by the current code from the repository root (so each report's source line reads as below). The datasets they read are described in the README's [Where the data comes from](../README.md#where-the-data-comes-from).

| directory | written by |
|---|---|
| `openai-wiki` | `swarmprov run data/raw -o runs/wiki --gold validation/`; the CSV mirrors by `swarmprov export runs/wiki -o DIR --tables exposures,edges_exposure_citation` (its `edges_exposure_citation_merged.csv` is `edges_exposure_citation.csv` here) |
| `crosssite` | `swarmprov crosssite runs/wiki data/raw2 -o runs/crosssite`, with the `technique_spread.csv` and `technique_site_summary.csv` it writes |
| `corpus` | `swarmprov run data/raw2 -o runs/corpus` |
| `synthetic` | `swarmprov synth -o data/synth/transcript.jsonl --agents 60 --seed 2`, `swarmprov run data/synth/transcript.jsonl -o runs/synth --adapter chat --config data/synth/config.json`, then `swarmprov truth-score runs/synth data/synth` (the scorecard in its Validation section) |
| `village-context` | `swarmprov run data/raw_village -o runs/village` |
| `village-goals` | `swarmprov run data/village/agent_goals.jsonl -o runs/village_goals` |

Every directory keeps `report.md` and `summary.json`, the reports with key findings their `headline.md`, the wiki and corpus runs their `profile.json`, and each the figures its report links.

`openai-wiki/report.html` is the wiki run's page (1.4 MB), fact explorer included, so the explorer opens in a browser without the dump; the other directories leave it out. `swarmprov html` on one of these directories renders the report with a line where the explorer would be, since the explorer needs the graph tables, so on `openai-wiki` it would replace the committed page with one without it.

`swarmprov compare results/openai-wiki runs/wiki` checks a fresh run against the committed summary: after the first command above it reports 0 changes (fixed-effects floats move by up to 6e-5 relative between processes, within the default tolerance of 1e-4). `tests/test_results.py` checks that every linked figure is here, that each key-findings block and the wiki's `report.html` are what the current code writes from these files, and that every number the README quotes comes from them.
