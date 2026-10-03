"""Synthetic swarm with known provenance, for validating the pipeline.

Agents (one per cohort date) work timed question sequences.  When a question
arrives and its answer is already on the shared board, the agent copies it
with probability ``copy_rate`` (answering in 1 s); otherwise it computes the
answer itself (5-40 s, 10% wrong).  Every agent reports each round on the
board in the templated style real swarms use.  Ground truth (who copied,
whether the answer was public at arrival) is written next to the transcript.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

FAMILIES = {
    "alpha-task": ["Ohio", "Texas", "Utah", "Iowa", "Maine", "Idaho", "Kansas", "Nevada", "Oregon", "Alabama",
                   "Alaska", "Arizona", "Colorado", "Florida", "Georgia", "Hawaii", "Illinois", "Indiana",
                   "Kentucky", "Michigan", "Montana", "Nebraska", "Vermont", "Virginia", "Wyoming"],
    "beta-task": ["Poland", "Hungary", "Greece", "Spain", "Italy", "France", "Germany", "Austria", "Belgium",
                  "Denmark", "Finland", "Ireland", "Japan", "Korea", "Latvia", "Mexico", "Norway", "Portugal",
                  "Sweden", "Chile", "Canada", "Iceland", "Israel", "Estonia", "Slovenia"],
}
ROUNDS = 5
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def simulate(n_agents: int = 60, copy_rate: float = 0.7, seed: int = 0, span_h: float = 24.0,
             tiers_min: tuple[float, ...] = (8, 20, 45, 90), wrong_rate: float = 0.1):
    """Each agent draws ``ROUNDS`` questions from its family's pool and works them
    ``cooldown`` minutes apart (a per-agent tier).  Because sequences differ, the same
    agent meets some questions nobody has posted yet and others already on the board -
    the within-agent variation the fixed-effects design needs.  (A swarm where every
    agent gets the same sequence, like the OpenAI wiki, has almost none.)"""
    rng = random.Random(seed)
    t_base = datetime(2026, 6, 16, 8, 0, tzinfo=timezone.utc)
    truth_vals = {(f, it): round(rng.uniform(5, 95), 2) for f, items in FAMILIES.items() for it in items}
    cohorts = rng.sample([f"{m}{d:02d}" for m in MONTHS for d in range(1, 29)], n_agents)
    agents = []
    for i, c in enumerate(cohorts):
        fam = list(FAMILIES)[i % len(FAMILIES)]
        agents.append({"name": f"Synth{fam.split('-')[0].title()}Agent{c}", "cohort": c, "family": fam,
                       "start": t_base + timedelta(hours=rng.uniform(0, span_h)),
                       "cooldown": rng.choice(tiers_min), "items": rng.sample(FAMILIES[fam], ROUNDS)})
    # every question arrival, in wall-clock order
    arrivals = []
    for a in agents:
        for r, item in enumerate(a["items"], start=1):
            t = a["start"] + timedelta(minutes=(r - 1) * a["cooldown"] + rng.uniform(-1, 1))
            arrivals.append((t, a, r, item))
    arrivals.sort(key=lambda x: x[0])
    board: list[dict] = []          # posts, appended as they happen (kept sorted at the end)
    pending: list[tuple] = []       # (post_time, post) reports not yet visible
    truth = []
    for t_arr, a, r, item in arrivals:
        # flush reports that became visible before this arrival
        for p in sorted([p for p in pending if p[0] <= t_arr], key=lambda x: x[0]):
            board.append(p[1])
        pending = [p for p in pending if p[0] > t_arr]
        public = [p for p in board if p["_family"] == a["family"] and p["_item"] == item and p["author"] != a["name"]]
        copied = bool(public) and rng.random() < copy_rate
        if copied:
            value, latency = public[-1]["_value"], 1
        else:
            latency = rng.randint(5, 40)
            value = truth_vals[(a["family"], item)]
            if rng.random() < wrong_rate:
                value = round(value + rng.choice([-1, 1]) * rng.uniform(0.5, 3), 2)
        t_post = t_arr + timedelta(seconds=latency) + timedelta(minutes=rng.uniform(0.5, 3))
        speed = "instantly" if latency <= 2 else f"at +{latency}s"
        text = (f"{a['cohort']} R{r} CONFIRMED: {item} arrived {t_arr:%H:%M:%S}, 14s timer; "
                f"answered {value:.2f} {speed}. R{r + 1} due in ~{int(a['cooldown'])}m. -- {a['name']}")
        post = {"ts": t_post.isoformat(), "author": a["name"], "channel": f"{a['family'].split('-')[0]}-board",
                "text": text, "_family": a["family"], "_item": item, "_value": value}
        pending.append((t_post, post))
        truth.append({"agent": a["name"], "cohort": a["cohort"], "family": a["family"], "round": r,
                      "item": item, "t_arrival": t_arr, "t_report": t_post, "public_at_arrival": bool(public),
                      "copied": copied, "latency_s": latency, "value": value,
                      "correct": value == truth_vals[(a["family"], item)]})
    board += [p[1] for p in sorted(pending, key=lambda x: x[0])]
    posts = [{k: v for k, v in p.items() if not k.startswith("_")} for p in board]
    posts.sort(key=lambda p: p["ts"])
    return posts, pd.DataFrame(truth)


CONFIG = {
    "name": "synthetic",
    "families": {"alpha-task": "alpha", "beta-task": "beta"},
    "techniques": [],
}


def write(out: str | Path, n_agents: int = 60, copy_rate: float = 0.7, seed: int = 0) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    posts, truth = simulate(n_agents=n_agents, copy_rate=copy_rate, seed=seed)
    with open(out, "w") as fh:
        for p in posts:
            fh.write(json.dumps(p) + "\n")
    truth.to_csv(out.with_name("truth.csv"), index=False)
    (out.with_name("config.json")).write_text(json.dumps(CONFIG, indent=1))
    print(f"wrote {len(posts)} posts to {out}; truth.csv and config.json alongside")
    return out
