"""Extractor validation against hand labels (``validate.score``, ``run --gold``): a toy claims table with known
precision and recall, label coverage, and the wiki's labelled splits when the dump is present."""

import json
from pathlib import Path

import pandas as pd
import pytest

from swarmprov import pipeline, validate
from swarmprov.schema import RunDir, conform

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
LABELS = Path(__file__).resolve().parents[1] / "validation"


def _label(eid, answers, **extra):
    return {"event_id": eid, "gold": {"is_answer": bool(answers), "answers": answers, "labelled": True},
            "labeller": "toy", "split": "toy"} | extra


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """Three posts: e1 a right answer, e2 a false answer (an arrival notice), e3 a missed answer."""
    run = RunDir(tmp_path_factory.mktemp("validate"))
    run.write("events", conform(pd.DataFrame({"event_id": ["e1", "e2", "e3"], "text": ["a", "b", "c"]}), "events"))
    claims = pd.DataFrame([{"claim_id": "c1", "event_id": "e1", "event_type": "answer", "item": "Ohio",
                            "value_norm": "5.5", "episode": 1, "latency_class": "instant"},
                           {"claim_id": "c2", "event_id": "e2", "event_type": "answer", "item": "Texas",
                            "value_norm": "7", "episode": 2, "latency_class": None}])
    run.write("claims", conform(claims, "claims"))
    labels = [_label("e1", [{"item": "ohio", "value": "5.5", "episode": 1, "latency_class": "instant"}]),
              _label("e2", []),
              _label("e3", [{"item": "Utah", "value": "3", "episode": 1, "latency_class": None}]),
              _label("elsewhere", [{"item": "Iowa", "value": "1", "episode": 1, "latency_class": None}])]
    gold = run.path / "toy_labels.jsonl"
    gold.write_text("".join(json.dumps(r) + "\n" for r in labels))
    (run.path / "other_labels.jsonl").write_text(json.dumps(_label("nowhere", [])) + "\n")
    return run, gold


def test_score_gives_known_precision_recall_and_coverage(toy):
    run, gold = toy
    res = validate.score(run, gold, "validation_toy.json")
    ap = res["fields"]["answer_post"]
    assert (ap["precision"], ap["recall"], ap["support"]) == (0.5, 0.5, 2)
    assert res["fields"]["item"] == {"accuracy": 1.0, "support": 1}           # case is not a difference
    assert res["fields"]["value"]["accuracy"] == 1.0 and res["fields"]["episode"]["accuracy"] == 1.0
    assert res["fields"]["latency_class"]["accuracy"] == 1.0
    assert res["n"] == 3 and res["coverage"] == {"labelled": 4, "in_run": 3, "share": 0.75}
    assert json.loads((run.path / "validation_toy.json").read_text()) == res


def test_labels_of_another_source_are_skipped(toy):
    run, _ = toy
    assert validate.score(run, run.path / "other_labels.jsonl", "validation_other.json") is None
    assert not (run.path / "validation_other.json").exists()
    assert validate.split_name("validation/test_labels.jsonl") == "test" and validate.split_name("gold.jsonl") == "gold"


def test_score_gold_writes_one_file_per_split(toy, capsys):
    run, _ = toy
    pipeline.score_gold(run, [run.path])
    out = capsys.readouterr().out
    assert "3 of 4 labelled posts in the run" in out and "skipped" in out
    assert (run.path / "validation_toy.json").exists() and not (run.path / "validation_other.json").exists()


@pytest.mark.skipif(not (RAW / "events.jsonl").exists(), reason="full wiki dump not present")
def test_wiki_labels_hold(tmp_path):
    """The extractor on the wiki still meets its labelled accuracy (ingest and extract, about 25 s)."""
    run = RunDir(tmp_path / "wiki")
    pipeline.ingest(RAW, run)
    pipeline.extract(run)
    pipeline.score_gold(run, [LABELS])
    res = {s: json.loads((run.path / f"validation_{s}.json").read_text()) for s in ("dev", "heldout", "test")}
    assert all(r["coverage"]["share"] == 1.0 for r in res.values())
    f = res["test"]["fields"]
    assert f["answer_post"]["precision"] >= 0.95 and f["answer_post"]["recall"] >= 0.9   # 1.00 / 0.91
    assert f["item"]["accuracy"] >= 0.95 and f["value"]["accuracy"] >= 0.95              # 0.96, 1.00
