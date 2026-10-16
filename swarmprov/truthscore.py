"""``swarmprov truth-score RUN TRUTH_DIR``: score a run of the synthetic swarm against what ``synth`` planted.

Reads ``truth.csv`` (and ``truth_techniques.csv`` and ``config.json``) beside the transcript and writes
``truth_validation.json``:

* per answer round, ``D`` against the answer being on the board at arrival (a confusion matrix) and against an
  earlier post of the same value, and ``D_visible`` against such a post still undeleted at the report;
* A2's estimate against the true effect, the share of rounds public at arrival that copied (a round answers
  instantly by copying, which needs the answer on the board, or as a fast agent, whatever is on the board);
* for each A5 attribution rule (and, with deletions, the first copy still visible) the share of copied answers it
  credits to the agent whose post was copied, and its top-10 share on instant answers against the true one;
* A3's adoption share and Kaplan-Meier median against the planted adoptions;
* A4's status and outcome per disputed slot against the same rule run on the true answers;
* each identity mapping's purity (the share of posts whose agent's main true agent wrote them) and how many true
  agents it splits.

Every number holds under the synthetic generative model only: it says the analyses recover what this simulation
planted, not that they are right on a real swarm.  ``truth-score`` then rebuilds the run's report, whose Validation
section shows the scorecard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analysis import causal, diffusion, errors, structure
from .analysis.agents import AGENT_COLS
from .analysis.uncertainty import km_greenwood, km_median
from .claims import public, value_key
from .report import _json_safe
from .rules import norm_value
from .schema import RunDir

LABEL = "under the synthetic generative model"
RULES = {"first source": "src_agent", "latest source": "src_latest", "same-page relay parent": "relay_parent"}
VISIBLE = "first copy still visible"


def _rate(hits: pd.Series) -> float | None:
    return float(hits.mean()) if len(hits) else None


def _earlier_copy(truth: pd.DataFrame, visible: bool = False) -> pd.Series:
    """Whether another true agent posted the round's value on the item before its report (and, with ``visible``,
    had not deleted it by then): the target ``D`` (``D_visible``) estimates."""
    out = []
    for r in truth.itertuples(index=False):
        o = truth[(truth["family"] == r.family) & (truth["item"] == r.item) & (truth["key"] == r.key)
                  & (truth["who"] != r.who) & (truth["t_report"] < r.t_report)]
        if visible and "deleted_at" in truth:
            o = o[o["deleted_at"].isna() | (o["deleted_at"] > r.t_report)]
        out.append(len(o) > 0)
    return pd.Series(out, index=truth.index)


def _true_agents(agents: pd.DataFrame, handle: dict, col: str) -> pd.Series:
    """Each label of a mapping -> the true agent most of its posts come from."""
    n = agents.assign(true=agents["author_raw"].map(handle)).groupby([col, "true"])["n_events"].sum()
    return n.groupby(level=0).idxmax().map(lambda k: k[1])


def _identity(agents: pd.DataFrame, handle: dict) -> dict:
    """Over the author strings the truth knows (a framework trace's ``task`` is nobody's)."""
    a = agents.assign(true=agents["author_raw"].map(handle)).dropna(subset=["true"])
    out = {}
    for name, col in AGENT_COLS.items():
        top = a.groupby([col, "true"])["n_events"].sum().groupby(level=0).max()
        out[name] = {"agents": int(a[col].nunique()), "true_agents": int(a["true"].nunique()),
                     "purity": float(top.sum() / a["n_events"].sum()),
                     "true_agents_split": int((a.groupby("true")[col].nunique() > 1).sum())}
    return out


def _a2(ex: pd.DataFrame, truth: pd.DataFrame) -> dict:
    est = causal.estimate(ex).set_index(["outcome", "treatment"])
    r = est.loc[("y_instant", "D_cons")]
    exposed = truth[truth["public_at_arrival"]]
    true = float(exposed["copied"].mean()) if len(exposed) else float("nan")
    ok = pd.notna(r["ci_lo"])
    return {"outcome": "y_instant", "treatment": "D_cons", "beta": r["beta"], "se": r["se"], "p": r["p"],
            "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "note": r["note"] if isinstance(r["note"], str) else None,
            "true_effect": true, "covers": bool(ok and r["ci_lo"] <= true <= r["ci_hi"]) if ok else None,
            "y_instant_sd": float(ex["y_instant"].std()), "mean_y_D0": r["mean_y_D0"], "mean_y_D1": r["mean_y_D1"]}


def _sources(ex: pd.DataFrame, j: pd.DataFrame, edges: pd.DataFrame, events: pd.DataFrame, label: pd.Series,
             handle: dict, deleted: bool) -> list[dict]:
    """Per A5 rule: the share of copied answers it credits to the agent whose post was copied, and its top-10 share on
    instant answers (A5's *instant only* row: in the simulation an answer is instant when copied, or from a fast agent)
    against the true one on the copied answers.  With deletions, also the author of the first copy still visible at
    the report (``VISIBLE``, not an A5 rule): a copier can only take a post that is still there."""
    att = structure.attribution(ex, edges, events).set_index(["rule", "answers"])
    shares = {rule: att.loc[(rule, "instant only"), "top10_share"] for rule in RULES}
    rules = dict(RULES)
    if deleted:
        rules[VISIBLE] = "src_visible_agent"
        shares[VISIBLE] = structure.concentration(
            ex.loc[(ex["D"] == 1) & (ex["y_instant"] == 1), "src_visible_agent"])["top10_share"]
    cp = j[j["copied"]]
    if cp.empty:   # nobody copied: no source to name
        return []
    src = cp["true_src"].map(handle)
    vc = src.value_counts()
    true_top = float(vc.head(10).sum() / vc.sum()) if vc.sum() else None
    return [{"rule": rule, "copied_answers": int(len(cp)), "names_true_source": _rate(cp[c].map(label) == src),
             "credits_nobody": int(cp[c].isna().sum()), "top10_share": shares[rule], "true_top10_share": true_top}
            for rule, c in rules.items()]


def _a3(tags: pd.DataFrame, techniques, truth_dir: Path, col: str, wall: bool) -> list[dict]:
    """Under a sequence clock the run's times are message positions, so no median is compared (both are None)."""
    path = truth_dir / "truth_techniques.csv"
    if not path.exists():
        return []
    tt = pd.read_csv(path, parse_dates=["t_first_read", "t_first_use", "t_last_post"])
    est = diffusion.analyze(tags, techniques, col) if len(tags) and techniques else {}
    rows = []
    for name, g in tt.groupby("technique"):
        t0 = g.loc[g["is_originator"], "t_first_use"].min()
        risk = g[~g["is_originator"] & (g["t_last_post"] >= t0)]
        hours = (risk["t_first_use"].where(risk["adopted"], risk["t_last_post"]) - t0).dt.total_seconds() / 3600
        median = km_median(km_greenwood(hours.to_numpy(float), risk["adopted"].to_numpy(int)))[0]
        e = est.get(name, {})
        rows.append({"technique": name, "at_risk": e.get("at_risk"), "true_at_risk": int(len(risk)),
                     "adoption_share": e.get("adoption_share"), "true_adoption_share": _rate(risk["adopted"]),
                     "median_hours_to_adopt": e.get("median_hours_to_adopt") if wall else None,
                     "true_median_hours_to_adopt": median if wall else None})
    return rows


def _a4(mentions: pd.DataFrame, truth: pd.DataFrame, refs: dict, col: str, wall: bool) -> dict:
    """The run's value slots and those of A4's rule on the true answers, under the run's rule (one majority bin on a
    sequence clock, as the report has it)."""
    rule = {} if wall else {"bin_h": None}
    run_slots = {s["slot"]: s for s in errors.value_slots(public(mentions), col, **rule)}
    tm = truth.rename(columns={"t_report": "ts", "key": "value_key"}).assign(**{col: truth["who"]})
    true_slots = {s["slot"]: s for s in errors.value_slots(tm, col, **rule)}
    keys = errors.reference_keys(refs)
    rows = []
    for slot in sorted(set(run_slots) | set(true_slots)):
        s, t = run_slots.get(slot), true_slots.get(slot)
        ref = keys.get(((s or t)["family"], (s or t)["item"]))
        rows.append({"slot": slot, "status": s and s["status"], "true_status": t and t["status"],
                     "winner": s and s["winner"], "true_winner": t and t["winner"],
                     "outcome": s and ref is not None and errors.outcome(s, ref) or None,
                     "true_outcome": t and ref is not None and errors.outcome(t, ref) or None})
    both = [r for r in rows if r["status"] and r["true_status"]]
    return {"slots": len(run_slots), "true_slots": len(true_slots), "in_both": len(both),
            "status_agrees": sum(r["status"] == r["true_status"] for r in both),
            "outcome_agrees": sum(r["outcome"] == r["true_outcome"] for r in both), "rows": rows}


def score(run: RunDir, truth_dir: str | Path, mapping: str = "merged") -> dict:
    """The scorecard as a dict, also written to ``truth_validation.json`` in the run directory.  Rounds are matched
    to the truth by the report post's author string and the item (each agent answers an item once); ValueError, and
    nothing written, when none matches."""
    from .pipeline import load_config
    truth_dir = Path(truth_dir)
    truth = pd.read_csv(truth_dir / "truth.csv")
    for c in ("t_arrival", "t_report", "deleted_at"):
        if c in truth:
            truth[c] = pd.to_datetime(truth[c], utc=True)
    handle = dict(zip(truth["agent"], truth["true_agent"] if "true_agent" in truth else truth["agent"]))
    truth = truth.assign(who=truth["agent"].map(handle), key=[value_key(norm_value(f"{v:.2f}")) for v in truth["value"]])
    truth = truth.assign(target=_earlier_copy(truth), visible=_earlier_copy(truth, visible=True))
    col = AGENT_COLS[mapping]
    wall = json.loads((run.path / "profile.json").read_text()).get("clock") != "sequence"
    ex, edges, events, agents = (run.read(t) for t in (f"exposures_{mapping}", f"edges_{mapping}", "events", "agents"))
    refs = json.loads((truth_dir / "config.json").read_text()).get("reference_values", {})
    author = events.drop_duplicates("event_id").set_index("event_id")["author_raw"]
    ex = ex.assign(relay_parent=structure.relay_parents(ex, edges, events),
                   src_visible_agent=ex["src_visible_event_id"].map(author).map(agents.set_index("author_raw")[col]))
    cols = {"agent": "author_raw", "family": "family", "item": "item", "public_at_arrival": "public_at_arrival",
            "copied": "copied", "src_agent": "true_src", "target": "target", "visible": "visible"}
    j = (ex.merge(events[["event_id", "author_raw"]].drop_duplicates("event_id"), left_on="report_event_id",
                  right_on="event_id", how="left")
         .merge(truth[list(cols)].rename(columns=cols), on=["author_raw", "family", "item"], how="inner"))
    if j.empty:
        raise ValueError(f"no answer round of {run.path} matches {truth_dir / 'truth.csv'}: is it the run of that "
                         "transcript?")
    d, pub, vis = j["D"] == 1, j["public_at_arrival"].astype(bool), j["D_visible"] == 1
    out = {
        "label": LABEL, "truth_dir": str(truth_dir), "mapping": mapping,
        "rounds": {"truth": int(len(truth)), "run": int(len(ex)), "matched": int(len(j))},
        "D": {"confusion": {"D1_public": int((d & pub).sum()), "D1_not_public": int((d & ~pub).sum()),
                            "D0_public": int((~d & pub).sum()), "D0_not_public": int((~d & ~pub).sum())},
              "share": _rate(d), "true_public_at_arrival": _rate(pub), "accuracy_vs_public_at_arrival": _rate(d == pub),
              "true_earlier_copy": _rate(j["target"]), "accuracy_vs_earlier_copy": _rate(d == j["target"])},
        "D_visible": {"share": _rate(vis), "true_visible_copy": _rate(j["visible"]), "accuracy": _rate(vis == j["visible"]),
                      "deleted_posts": int(truth["deleted_at"].notna().sum()) if "deleted_at" in truth else 0},
        "A2": _a2(ex, truth) if len(ex) > 20 and wall else None,   # A2's latency outcome needs a wall clock
        "sources": _sources(ex, j, edges, events, _true_agents(agents, handle, col), handle, "deleted_at" in truth),
        "A3": _a3(run.read("tags"), load_config(run).techniques, truth_dir, col, wall),
        "A4": _a4(run.read("mentions"), truth, refs, col, wall),
        "identity": _identity(agents, handle),
    }
    out = _json_safe(out)
    (run.path / "truth_validation.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    return out


def main(run: RunDir, truth_dir: str | Path, mapping: str = "merged") -> int:
    """``swarmprov truth-score``: write the scorecard, print its headline, and rebuild the report when there is one."""
    if not (Path(truth_dir) / "truth.csv").exists():
        print(f"{truth_dir}: no truth.csv (it is written beside the transcript by `swarmprov synth`)")
        return 1
    try:
        r = score(run, truth_dir, mapping)
    except ValueError as e:
        print(e)
        return 1
    d, a2 = r["D"], r["A2"]
    print(f"truth score {LABEL} ({mapping} mapping), {r['rounds']['matched']} of {r['rounds']['truth']} true rounds "
          f"matched: D agrees with on-the-board-at-arrival for {d['accuracy_vs_public_at_arrival']:.0%} of rounds, "
          f"with an earlier post of the value for {d['accuracy_vs_earlier_copy']:.0%}"
          + (f"; A2 beta {a2['beta']:.3f} against a true {a2['true_effect']:.3f} ("
             + ("not estimated" if a2["covers"] is None else "inside" if a2["covers"] else "outside") + " its 95% CI)"
             if a2 and a2["beta"] is not None else ""))
    print(f"wrote {run.path / 'truth_validation.json'}")
    if (run.path / "summary.json").exists():   # rebuilt as it was built: the same mapping and bootstrap resamples
        from . import report
        built = json.loads((run.path / "summary.json").read_text())
        report.build(run, mapping=built.get("mapping", "merged"),
                     boot=((built.get("A1") or {}).get("bootstrap") or {}).get("reps", 0))
    return 0
