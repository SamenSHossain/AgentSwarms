"""Hand-label validation of the extractor.

``sample`` writes ``gold_template.jsonl``: a stratified sample of posts, each
with the extractor's output pre-filled under ``pred`` and an editable ``gold``
block.  Annotators correct ``gold`` (set ``is_answer`` and the answer fields;
for posts with several answers use the ``answers`` list) and save the file as
gold.jsonl.  ``score`` compares predictions with labels and writes
``validation.json``, which the report picks up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .claims import value_key
from .schema import RunDir

FIELDS = ("item", "value", "episode", "latency_class")


def _answers_for(cl: pd.DataFrame, eid: str) -> list[dict]:
    a = cl[(cl["event_id"] == eid) & (cl["event_type"] == "answer")]
    def clean(v):
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v
    return [{"item": clean(r["item"]), "value": clean(r["value_norm"]),
             "episode": None if pd.isna(r["episode"]) else int(r["episode"]),
             "latency_class": clean(r["latency_class"])} for _, r in a.iterrows()]


def sample(run: RunDir, n: int = 120, seed: int = 0, exclude: list[str | Path] = (),
           name: str = "gold_template.jsonl") -> Path:
    """Stratified sample of task posts; posts already in any ``exclude`` file are skipped."""
    ev, tags, cl = run.read("events"), run.read("tags"), run.read("claims")
    t = tags.set_index("event_id")
    task = ev[ev["family"].fillna("") != ""].copy()
    seen = {json.loads(l)["event_id"] for f in exclude for l in open(f) if l.strip()}
    task = task[~task["event_id"].isin(seen)]
    task["is_answer"] = task["event_id"].map(t["is_answer"]).fillna(False)
    task["arrival_kw"] = task["event_id"].map(t["is_arrival"]).fillna(False)
    # half predicted answers, a quarter posts that mention arrivals but no answer (recall), a quarter other
    parts = [task[task["is_answer"]], task[~task["is_answer"] & task["arrival_kw"]],
             task[~task["is_answer"] & ~task["arrival_kw"]]]
    quota = [n // 2, n // 4, n - n // 2 - n // 4]
    pick = pd.concat([p.sample(min(q, len(p)), random_state=seed) for p, q in zip(parts, quota)])
    out = run.path / name
    with open(out, "w") as fh:
        for _, r in pick.iterrows():
            pred = _answers_for(cl, r["event_id"])
            fh.write(json.dumps({
                "event_id": r["event_id"], "family": r["family"], "author": r["author_raw"],
                "text": r["text"], "pred": {"is_answer": bool(pred), "answers": pred},
                "gold": {"is_answer": bool(pred), "answers": pred, "labelled": False},
            }, default=str) + "\n")
    print(f"wrote {len(pick)} posts to {out}; edit each 'gold' block, set labelled=true, save as gold.jsonl")
    return out


def _norm(field: str, v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if field == "value":
        return value_key(str(v).replace(",", ""))
    if field == "episode":
        return int(v)
    return str(v).strip().lower()


def score(run: RunDir, gold_path: str | Path, out_name: str = "validation.json") -> dict:
    """Score the run's *current* claims against the labels (not the template's stale ``pred``)."""
    rows = [json.loads(l) for l in open(gold_path) if l.strip()]
    rows = [r for r in rows if r["gold"].get("labelled", True)]
    cl = run.read("claims")
    for r in rows:
        ans = _answers_for(cl, r["event_id"])
        r["pred"] = {"is_answer": bool(ans), "answers": ans}
    tp = fp = fn = 0
    field_hits = {f: [0, 0] for f in FIELDS}  # correct, compared
    for r in rows:
        g, p = r["gold"], r["pred"]
        if g["is_answer"] and p["is_answer"]:
            tp += 1
            # match answers by item (fallback: by position)
            gmap = {_norm("item", a.get("item")): a for a in g["answers"]}
            for k, pa in enumerate(p["answers"]):
                ga = gmap.get(_norm("item", pa.get("item"))) or (g["answers"][k] if k < len(g["answers"]) else None)
                if ga is None:
                    continue
                for f in FIELDS:
                    if ga.get(f) is None and pa.get(f) is None:
                        continue
                    field_hits[f][1] += 1
                    field_hits[f][0] += int(_norm(f, ga.get(f)) == _norm(f, pa.get(f)))
        elif p["is_answer"]:
            fp += 1
        elif g["is_answer"]:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    n_gold_answers = sum(len(r["gold"]["answers"]) for r in rows if r["gold"]["is_answer"])
    n_pred_answers = sum(len(r["pred"]["answers"]) for r in rows if r["pred"]["is_answer"])
    res = {"n": len(rows), "labeller": rows[0].get("labeller", "unknown") if rows else None,
           "gold_answers": n_gold_answers, "predicted_answers": n_pred_answers, "fields": {
        "answer_post": {"precision": prec, "recall": rec, "support": tp + fn},
        **{f: {"accuracy": (c / n) if n else float("nan"), "support": n} for f, (c, n) in field_hits.items()},
    }}
    res["split"] = rows[0].get("split") if rows else None
    res["labels_file"] = str(gold_path)
    (run.path / out_name).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res
