## Key findings

- **A1. Provenance split**: skipped: needs agent-rounds (an agent's answer to an item in a task family); 15 answer claims extracted, 0 agent-rounds: none names an item (add `extra_items`, or use `--llm`).
- **A2. Does availability cause copying?** skipped: needs more than 20 agent-rounds; this run has 0.
- **A3. Technique diffusion**: adoption share = the share of the agents at risk (active in its task families after its first post) that went on to use it: `proxy-jqp` 28% (5 of 18), `proxy-md-succ` 24% (4 of 17), `proxy-allorigins` 19% (18 of 93), `proxy-markdown-new` 9% (8 of 94), `shortener-da-gd` 8% (7 of 91); lower for the rest.
- **A4. Error propagation**: no slot carried competing values.
- **A5. Structure**: 18 agents linked by 10 same-page relay hops; no answer had a first public source, so there is no hub concentration to report.
- **A6. Reach**: 63 of 1,569 URLs were repeated by other agents, travelling 1.0 hops on average and at most 3 (a hop links a carrier to an inferred source: plausible paths, not proven ones).
