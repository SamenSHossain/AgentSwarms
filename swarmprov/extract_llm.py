"""Optional LLM extraction (Claude, structured outputs).

One call per candidate post returns the same fields the rule extractor
produces; results are cached on disk by event_id so re-runs are free.  Rule
and LLM rows are then merged: the LLM wins on semantic fields (item,
event type, wrong/correct), the rules fill anything the LLM left empty.

Requires ``pip install anthropic pydantic`` and an API key.  Default model is
the small, cheap ``claude-haiku-4-5``; pass ``--llm <model-id>`` to change it.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from .rules import norm_value
from .textutil import stable_id

DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM = """You extract structured facts from posts written by AI agents coordinating on a shared wiki.
Agents answer timed question sequences (rounds R1..R6, also written G3, #4) and report each round, e.g.
"R3 CONFIRMED: Poland arrived 19:43:07, timer 56s; answered 16.40% instantly -- AgentNov27".
For every round the AUTHOR reports having answered, emit one claim with event_type "answer".
For every value the author says they expect for a future round, emit event_type "prediction".
Do not emit claims for values the author merely lists in a table, quotes from someone else, or asks about.
item = the question's subject (state, country, field of study, ...). value = the answer exactly as written
(digits only, keep decimals, drop % and thousands separators; several numbers -> join with "/").
latency_class: "instant" if the author says instantly / same second / +1s / within 2 s, "delayed" if they
give a longer time, null if unstated. wrong = the author says that answer was wrong or a guess;
correct = the author says it was correct. If the post contains no such claims return an empty list."""


class Claim(BaseModel):
    event_type: Literal["answer", "prediction"]
    episode: Optional[int] = Field(None, description="round number, e.g. 3 for R3/G3/#3")
    item: Optional[str] = None
    value: Optional[str] = None
    latency_class: Optional[Literal["instant", "delayed"]] = None
    timer_s: Optional[float] = None
    wrong: bool = False
    correct: bool = False


class PostExtraction(BaseModel):
    claims: list[Claim]
    independent_claim: bool = Field(False, description="author says they computed/verified it themselves")
    disputes_value: bool = Field(False, description="author says some posted value is wrong")


def _candidates(events: pd.DataFrame, families: pd.Series) -> pd.DataFrame:
    kw = events["text"].str.contains(r"answer|submitt|expect|predict|CONFIRMED|arrived|\bR\d\b|\bG\d\b", case=False)
    return events[(families != "") & kw]


def _call(client, model: str, ev, family: str) -> dict:
    msg = (f"Task family: {family}\nAuthor (signature or account): {ev.author_raw}\n"
           f"Channel: {ev.channel}\nPost:\n<post>\n{ev.text[:6000]}\n</post>")
    resp = client.messages.parse(
        model=model, max_tokens=2000, system=SYSTEM,
        messages=[{"role": "user", "content": msg}], output_format=PostExtraction,
    )
    return resp.parsed_output.model_dump()


def run(events: pd.DataFrame, families: pd.Series, agents: pd.DataFrame, model: str = DEFAULT_MODEL,
        cache_dir: Path | None = None, workers: int = 8, client=None, limit: int | None = None) -> pd.DataFrame:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    cache_dir = Path(cache_dir or ".swarmprov/llm_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cand = _candidates(events, families)
    if limit:
        cand = cand.head(limit)
    fam = families.loc[cand.index]
    amap = agents.set_index("author_raw")

    def work(args):
        ev, f = args
        p = cache_dir / f"{model}__{ev.event_id}.json"
        if p.exists():
            return ev, f, json.loads(p.read_text())
        try:
            out = _call(client, model, ev, f)
        except Exception as e:  # keep going; failed posts fall back to rule output
            return ev, f, {"error": str(e)[:200]}
        p.write_text(json.dumps(out))
        return ev, f, out

    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for ev, f, out in pool.map(work, zip(cand.itertuples(index=False), fam)):
            if "error" in out:
                continue
            a = amap.loc[ev.author_raw] if ev.author_raw in amap.index else None
            for k, c in enumerate(out["claims"]):
                rows.append({
                    "claim_id": stable_id(ev.event_id, "llm", k), "event_id": ev.event_id, "ts": ev.ts,
                    "agent_strict": a["agent_strict"] if a is not None else ev.author_raw,
                    "agent_merged": a["agent_merged"] if a is not None else ev.author_raw,
                    "family": f, "episode": c["episode"], "item": c["item"], "value_raw": c["value"],
                    "value_norm": norm_value(c["value"]), "event_type": c["event_type"],
                    "latency_class": c["latency_class"], "latency_s": None, "timer_s": c["timer_s"],
                    "wrong_flag": c["wrong"], "correct_flag": c["correct"], "extractor": f"llm:{model}",
                })
    return pd.DataFrame(rows)


def merge(rule: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """Per post the LLM decides which claims exist; rule values fill its gaps."""
    if llm.empty:
        return rule
    from .gazetteer import canonical_item
    llm = llm.copy()
    llm["item"] = llm["item"].map(lambda s: canonical_item(s) if isinstance(s, str) else s)
    covered = set(llm["event_id"])
    keep_rule = rule[~rule["event_id"].isin(covered)]
    r_idx = {(r.event_id, r.event_type, r.item): r for r in rule[rule["event_id"].isin(covered)].itertuples(index=False)}
    filled = []
    for r in llm.to_dict("records"):
        rr = r_idx.get((r["event_id"], r["event_type"], r["item"]))
        if rr is not None:
            for col in ("value_raw", "value_norm", "episode", "latency_class", "latency_s", "timer_s"):
                if r.get(col) is None or (isinstance(r.get(col), float) and pd.isna(r[col])):
                    r[col] = getattr(rr, col)
            r["extractor"] = r["extractor"] + "+rule"
        filled.append(r)
    return pd.concat([keep_rule, pd.DataFrame(filled)], ignore_index=True)
