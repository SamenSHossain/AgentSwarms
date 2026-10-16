# Tests

```bash
pip install -e '.[dev]'     # pytest
python -m pytest -q         # from the repository root
```

The suite takes about 6 min with the wiki dump present (`data/raw`; the tests that need it are
skipped without it). Shared fixtures live in `conftest.py`, built once per session: `synth_run` (the seed-2 synthetic
swarm, run end to end), `synth_first` (copiers take the first post), `synth_strict`, `synth_null` (no copying) and
`wiki_run`, which ingests and extracts the wiki dump once, in 25–40 s, for the label and dump-count checks. Figures
are drawn but saved as a 1-pixel PNG, except in the 15 tests marked `figures`, which draw real PNGs.

| file | what it checks |
|---|---|
| `test_rules.py`, `test_segment.py` | the rule extractor (answer reports, the `=` form, values, latency, round sequences) and wiki segmentation (signatures, novelty since the last deletion, mojibake, a linear signature regex) |
| `test_chat_and_llm.py` | field guessing, LLM extraction (its cache keyed on content, failed calls counted, fatal errors, the post block it cannot close), family-less chats and the skip notes |
| `test_chat_formats.py`, `test_chat_ingest.py`, `test_remote.py` | epoch strings and the sequence clock; CSV, Slack and Discord exports, directories, field mapping, ingest warnings, config-key validation, adapter errors and `inspect`; `hf://` inputs |
| `test_framework_roles.py` | task turns and tool results in OpenAI, LangChain and AutoGen traces, and own lookups against the synthetic AutoGen swarm's truth |
| `test_corpus_adapter.py`, `test_crosssite.py`, `test_swarmtraces.py`, `test_village.py`, `test_roster.py`, `test_wiki_events.py` | the other adapters: the cross-site corpus and its technique spread (tables written as CSV), the Swarm traces reconstruction corpus, AI Village side tables and audience-aware exposure, rosters, the wiki's deletions, recreations, probes and clock grades |
| `test_identity.py` | cohorts, case variants resolved together, the alias audit, and a run whose two mappings group the authors alike (computed once and relabelled) |
| `test_synthetic_recovery.py`, `test_trace.py` | the pipeline recovers known provenance from the synthetic swarm; `trace` and the posts behind each answer, against synthetic truth |
| `test_self_originated.py` | answers whose agent posted the value first credit no source (a planner that summarises its researchers' findings is credited with none of them), the integrity check against an added edge, and older exposure tables |
| `test_attribution.py`, `test_structure.py`, `test_graph.py` | A5's attribution rules and origins against synthetic swarms that copy the latest post, the first post or nothing; exact betweenness; reply edges, agent profiles, the GraphML/JSON export and the supply figure |
| `test_errors.py`, `test_detail.py` | A4 status and outcomes, variant provenance and `y_winner`; the adoption and dispute rows, the contagion permutation test, `export` and the committed exposure mirror's columns |
| `test_dose.py`, `test_uncertainty.py`, `test_sweeps.py` | A2's dose response against a planted copy ramp on three seeds and its gates; the bootstrap, Greenwood and leave-one-item-out intervals (with a coverage simulation); the head-start, relay-window and instant-threshold sweeps against synthetic truth |
| `test_truthscore.py`, `test_checks.py`, `test_validate.py` | the scenario options and the truth scorecard over three scenarios (deletions, a technique, several boards, aliases, a confounded null); the integrity checks against corrupted copies; extractor scoring, label coverage and the wiki's labelled splits |
| `test_compare.py`, `test_suggest.py` | `compare` on synthetic runs (a run against itself, two configs on a moved transcript, two seeds, merged against strict, an edited config); `suggest` and `auto_techniques` on a chat fixture, with the summaries against `tests/golden` |
| `test_headline.py`, `test_report_integrity.py`, `test_figures.py` | the key findings (every number parsed back from `summary.json` and traced to its field, A2's reading rule, A5's ranges, the skip lines, the strict headline); report text that holds for its own data; what each figure draws |
| `test_html.py` | `report.html`: the Markdown subset with text escaped before formatting, a self-contained page, the explorer against `fact_lineage`, the byte cap and palette figures, and in headless Chromium (where node has playwright) the detail box, the phone layout and the sequence-clock axis |
| `test_cli.py`, `test_hostile_input.py`, `test_helpers.py` | one-line CLI errors that create nothing, README examples that write only where git ignores; control characters, surrogates and escape sequences in agent-written strings; strict JSON and the range validators |
| `test_results.py` | the committed `results/`: every linked figure there and linked, the 143-surface inventory cited only where it applies, each key-findings block and the wiki's `report.html` what the current code writes (the page byte for byte, its explorer data with the fields the explorer writes now), every number the README and `docs/DATA.md` quote from a run (the validation table and the dump's and roster's sizes included; the held-out split's scores from before the fixes are history), and every link in the docs |
