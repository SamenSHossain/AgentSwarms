"""Synthetic swarm with known provenance, for validating the pipeline.

Agents (one per cohort date) work timed question sequences.  When a question
arrives and its answer is already on the shared board, the agent copies it
with probability ``copy_rate`` (answering in 1 s); otherwise it computes the
answer itself (5-40 s, 10% wrong).  With ``copy_ramp_h`` (``synth --copy-ramp-h``)
the copy probability grows with how long the answer has been on the board,
``copy_rate * min(1, head_h / copy_ramp_h)`` with ``head_h`` the hours since its
first post by another agent: the dose response A2's exploratory block looks for.
Every agent reports each round on the board in the templated style real swarms
use.  Ground truth (who copied, from whom, whether the answer was public at
arrival) is written next to the transcript.

``fmt="autogen"`` (``synth --format autogen``) writes the same swarm as an AutoGen message stream with no
timestamps (so it runs on message order): a ``user`` task message first, and
before each *computed* answer the agent's own tool call (a ``ToolCallRequestEvent``
and the ``ToolCallExecutionEvent`` carrying the value it then answers with).  A
copied answer has no tool call.

Five scenario options exercise the other analyses (``truthscore`` scores a run against them); each is off by default
and draws on its own random stream, so without them the transcript and truth are those of earlier versions:
``delete_rate`` (posts get a ``deleted_at``; a deleted post stops being on the board), ``technique`` (a token one agent
uses from its first post and every agent of its family uses once it has read a post carrying it), ``n_boards`` (each
family posts on several boards, every agent on its own, and reads them all), ``alias_rate`` (posts signed with the
agent's variant handle; truth ``agent`` is the handle signed, ``true_agent`` the agent) and ``confound`` (that share of
agents, the latest to start, never copy and answer each question instantly with probability ``FAST_P``, whatever is
on the board: late starters find more answers public, so between agents instant answers go with exposure, while
within an agent they do not, and at ``copy_rate=0`` the instant outcome varies under a true effect of zero).
"""

from __future__ import annotations

import json
import random
import re
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


def _truth_values(rng: random.Random) -> dict[tuple[str, str], float]:
    return {(f, it): round(rng.uniform(5, 95), 2) for f, items in FAMILIES.items() for it in items}


COPY_FROM = ("latest", "first", "random")
FAST_P = 0.5   # a fast agent's chance of answering a question instantly (``confound``)
FORMATS = ("chat", "autogen")
TOKEN = r"[A-Za-z][\w-]*"   # a technique token
TASK = "Work through the questions you are given in order and post each answer on the board."


def _check(copy_from, copy_ramp_h, delete_rate, technique, n_boards, alias_rate, confound) -> None:
    if copy_from not in COPY_FROM:
        raise ValueError(f"copy_from must be one of {COPY_FROM}")
    if copy_ramp_h is not None and not 0 < copy_ramp_h < float("inf"):
        raise ValueError("copy_ramp_h must be a positive, finite number of hours")
    for name, v in (("delete_rate", delete_rate), ("alias_rate", alias_rate), ("confound", confound)):
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must be a share between 0 and 1")
    if isinstance(n_boards, bool) or not isinstance(n_boards, int) or n_boards < 1:
        raise ValueError("n_boards must be a whole number of at least 1")
    if technique is not None and not re.fullmatch(TOKEN, technique):
        raise ValueError("technique must be a token of letters, digits, '_' and '-', starting with a letter")


def simulate(n_agents: int = 60, copy_rate: float = 0.7, seed: int = 0, span_h: float = 24.0,
             tiers_min: tuple[float, ...] = (8, 20, 45, 90), wrong_rate: float = 0.1, copy_from: str = "latest",
             trace: bool = False, copy_ramp_h: float | None = None, delete_rate: float = 0.0,
             technique: str | None = None, n_boards: int = 1, alias_rate: float = 0.0, confound: float = 0.0):
    """Each agent draws ``ROUNDS`` questions from its family's pool and works them
    ``cooldown`` minutes apart (a per-agent tier).  Because sequences differ, the same
    agent meets some questions nobody has posted yet and others already on the board -
    the within-agent variation the fixed-effects design needs.  (A swarm where every
    agent gets the same sequence, like the OpenAI wiki, has almost none.)

    A copying agent takes the value of the ``latest``, the ``first`` or a ``random`` post
    on the board for its question; truth records that post's author as ``src_agent``.

    ``trace`` adds framework rows (they carry a ``type``): the task message, and a tool call and its result
    before each computed answer.  They draw on their own random stream, so the board posts stay the same.

    The scenario options (module docstring) add truth columns only when on: ``deleted_at`` (the round's post),
    ``uses_technique``, ``true_agent`` and ``fast``."""
    _check(copy_from, copy_ramp_h, delete_rate, technique, n_boards, alias_rate, confound)
    rng = random.Random(seed)
    pick_rng = random.Random(seed + 1000)   # own stream, so 'latest' keeps the transcript of earlier versions
    tool_rng = random.Random(seed + 2000)   # and so does trace mode
    del_rng = random.Random(seed + 3000)    # and the deletions, aliases and fast answers
    alias_rng = random.Random(seed + 4000)
    fast_rng = random.Random(seed + 5000)
    t_base = datetime(2026, 6, 16, 8, 0, tzinfo=timezone.utc)
    truth_vals = _truth_values(rng)
    cohorts = rng.sample([f"{m}{d:02d}" for m in MONTHS for d in range(1, 29)], n_agents)
    agents = []
    for i, c in enumerate(cohorts):
        fam = list(FAMILIES)[i % len(FAMILIES)]
        short, board = fam.split("-")[0], (i // len(FAMILIES)) % n_boards
        agents.append({"name": f"Synth{short.title()}Agent{c}", "cohort": c, "family": fam,
                       "start": t_base + timedelta(hours=rng.uniform(0, span_h)),
                       "cooldown": rng.choice(tiers_min), "items": rng.sample(FAMILIES[fam], ROUNDS),
                       "board": f"{short}-board" + (f"-{board + 1}" if board else ""),
                       "alias": f"{short.title()}Runner-{c}", "fast": False, "uses": False})
    by_start = sorted(agents, key=lambda a: a["start"])
    for a in by_start[len(agents) - round(confound * len(agents)):] if confound else []:
        a["fast"] = True   # chosen before any question arrives, by start time alone
    if technique is not None:   # the first agent of the first family to start uses it from its first post
        next(a for a in by_start if a["family"] == list(FAMILIES)[0])["uses"] = True
    # every question arrival, in wall-clock order
    arrivals = []
    for a in agents:
        for r, item in enumerate(a["items"], start=1):
            t = a["start"] + timedelta(minutes=(r - 1) * a["cooldown"] + rng.uniform(-1, 1))
            arrivals.append((t, a, r, item))
    arrivals.sort(key=lambda x: x[0])
    board: list[dict] = []          # posts, appended as they happen (kept sorted at the end)
    pending: list[tuple] = []       # (post_time, post) reports not yet visible
    tools: list[dict] = []          # trace mode: the task message and each computed answer's own tool call
    if trace:
        tools.append({"ts": (t_base - timedelta(minutes=1)).isoformat(), "author": "user", "type": "TextMessage",
                      "text": TASK})
    truth = []
    for t_arr, a, r, item in arrivals:
        # flush reports that became visible before this arrival
        for p in sorted([p for p in pending if p[0] <= t_arr], key=lambda x: x[0]):
            board.append(p[1])
        pending = [p for p in pending if p[0] > t_arr]
        # what the agent reads for its question, on every board: other agents' posts on the item, not yet deleted
        public = [p for p in board if p["_family"] == a["family"] and p["_item"] == item and p["_agent"] != a["name"]
                  and (p["_deleted"] is None or p["_deleted"] > t_arr)]
        a["uses"] = a["uses"] or technique is not None and any(technique in p["text"] for p in public)
        p_copy = copy_rate
        if public and copy_ramp_h is not None:   # the same single draw, so no new random stream is needed
            p_copy *= min(1.0, (t_arr - min(p["_t"] for p in public)).total_seconds() / 3600 / copy_ramp_h)
        copied = bool(public) and rng.random() < p_copy and not a["fast"]
        src = None
        if copied:
            k = pick_rng.randrange(len(public)) if copy_from == "random" else (0 if copy_from == "first" else -1)
            src = public[k]
            value, latency = src["_value"], 1
        else:
            latency = rng.randint(5, 40)
            if a["fast"] and fast_rng.random() < FAST_P:   # drawn anyway, so the other agents' draws stay in step
                latency = 1
            value = truth_vals[(a["family"], item)]
            if rng.random() < wrong_rate:
                value = round(value + rng.choice([-1, 1]) * rng.uniform(0.5, 3), 2)
            if trace:   # the lookup is made after the question arrives and returns before the answer
                call = f"call_{len(truth)}"   # the truth row of this answer
                tools += [{"ts": (t_arr + timedelta(seconds=tool_rng.uniform(0, 1))).isoformat(), "author": a["name"],
                           "type": "ToolCallRequestEvent",
                           "content": [{"id": call, "name": "lookup", "arguments": json.dumps({"item": item})}]},
                          {"ts": (t_arr + timedelta(seconds=tool_rng.uniform(1, latency))).isoformat(), "author": a["name"],
                           "type": "ToolCallExecutionEvent",
                           "content": [{"content": f"{item}: {value:.2f}", "name": "lookup", "call_id": call,
                                        "is_error": False}]}]
        t_post = t_arr + timedelta(seconds=latency) + timedelta(minutes=rng.uniform(0.5, 3))
        speed = "instantly" if latency <= 2 else f"at +{latency}s"
        handle = a["alias"] if alias_rate and alias_rng.random() < alias_rate else a["name"]
        deleted = t_post + timedelta(hours=del_rng.uniform(0.25, 6)) \
            if delete_rate and del_rng.random() < delete_rate else None
        text = (f"{a['cohort']} R{r} CONFIRMED: {item} arrived {t_arr:%H:%M:%S}, 14s timer; "
                f"answered {value:.2f} {speed}. R{r + 1} due in ~{int(a['cooldown'])}m."
                + (f" Tip: {technique} saves a step." if a["uses"] else "") + f" -- {handle}")
        post = {"ts": t_post.isoformat(), "author": handle, "channel": a["board"], "text": text}
        if deleted is not None:
            post["deleted_at"] = deleted.isoformat()
        pending.append((t_post, post | {"_family": a["family"], "_item": item, "_value": value, "_t": t_post,
                                        "_agent": a["name"], "_deleted": deleted}))
        truth.append({"agent": handle, "cohort": a["cohort"], "family": a["family"], "round": r,
                      "item": item, "t_arrival": t_arr, "t_report": t_post, "public_at_arrival": bool(public),
                      "copied": copied, "latency_s": latency, "value": value,
                      "correct": value == truth_vals[(a["family"], item)],
                      "src_agent": src["author"] if src else None}
                     | ({"deleted_at": deleted} if delete_rate else {})
                     | ({"uses_technique": a["uses"]} if technique else {})
                     | ({"true_agent": a["name"]} if alias_rate else {}) | ({"fast": a["fast"]} if confound else {}))
    board += [p[1] for p in sorted(pending, key=lambda x: x[0])]
    posts = [{k: v for k, v in p.items() if not k.startswith("_")} for p in board] + tools
    posts.sort(key=lambda p: p["ts"])
    return posts, pd.DataFrame(truth)


def technique_truth(truth: pd.DataFrame, technique: str) -> pd.DataFrame:
    """One row per agent of the technique's family (the originator's): its first read of a post carrying the
    technique (the arrival of its first round using it; none for the originator), its first use and its last post."""
    agent = truth["true_agent"] if "true_agent" in truth else truth["agent"]
    t = truth.assign(_agent=agent)
    use = t[t["uses_technique"]]
    origin = use.loc[use["t_report"].idxmin()]
    fam = t[t["family"] == origin["family"]]
    first = use.sort_values("t_report").groupby("_agent").first()
    rows = [{"technique": technique, "agent": a, "family": origin["family"], "is_originator": a == origin["_agent"],
             "t_first_read": None if a == origin["_agent"] or a not in first.index else first.loc[a, "t_arrival"],
             "t_first_use": first.loc[a, "t_report"] if a in first.index else None, "t_last_post": g["t_report"].max(),
             "adopted": a in first.index} for a, g in fam.groupby("_agent")]
    return pd.DataFrame(rows).sort_values(["t_first_use", "agent"], na_position="last", kind="stable")


def _autogen(p: dict) -> dict:
    """A post as an AutoGen message: the speaker is ``source``; no timestamp and no channel."""
    return {"source": p["author"], "models_usage": None, "metadata": {}, "content": p.get("content", p.get("text")),
            "type": p.get("type", "TextMessage")}


CONFIG = {
    "name": "synthetic",
    "families": {"alpha-task": "alpha", "beta-task": "beta"},
    "techniques": [],
}


def write(out: str | Path, n_agents: int = 60, copy_rate: float = 0.7, seed: int = 0, copy_from: str = "latest",
          fmt: str = "chat", copy_ramp_h: float | None = None, **scenario) -> Path:
    """``scenario``: ``simulate``'s delete_rate, technique, n_boards, alias_rate and confound.  A technique adds its
    pattern to config.json and its adoptions as ``truth_techniques.csv`` (``technique_truth``).  ValueError, and
    nothing written, on an invalid option, or deletions or several boards in the autogen format (it has neither
    timestamps nor channels to carry them)."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    if fmt == "autogen" and (scenario.get("delete_rate") or scenario.get("n_boards", 1) > 1):
        raise ValueError("the autogen format writes no timestamps and no channels, so it cannot carry deletions "
                         "(delete_rate) or several boards (n_boards); write the chat format for those")
    posts, truth = simulate(n_agents=n_agents, copy_rate=copy_rate, seed=seed, copy_from=copy_from,
                            trace=fmt == "autogen", copy_ramp_h=copy_ramp_h, **scenario)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for p in posts:
            fh.write(json.dumps(_autogen(p) if fmt == "autogen" else p) + "\n")
    truth.to_csv(out.with_name("truth.csv"), index=False)
    # the true values are the seeded generator's first draws, so they come back without touching the posts
    refs = {f"{f}|{it}": v for (f, it), v in _truth_values(random.Random(seed)).items()}
    cfg, tech = CONFIG | {"reference_values": refs}, scenario.get("technique")
    also = ""
    if tech:
        cfg["techniques"] = [{"name": tech, "pattern": re.escape(tech), "families": [list(FAMILIES)[0]],
                              "description": "synthetic: one agent uses it from its first post, the others of its "
                                             "family once they have read a post carrying it"}]
        technique_truth(truth, tech).to_csv(out.with_name("truth_techniques.csv"), index=False)
        also = ", truth_techniques.csv"
    (out.with_name("config.json")).write_text(json.dumps(cfg, indent=1))
    n_calls = sum(p.get("type") == "ToolCallExecutionEvent" for p in posts)
    what = f"messages ({n_calls} tool calls, each with its result)" if fmt == "autogen" else "posts"
    print(f"wrote {len(posts)} {what} to {out}; truth.csv{also} and config.json (true values as reference_values) "
          "alongside")
    return out
