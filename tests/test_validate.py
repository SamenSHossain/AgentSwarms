"""Extractor validation against hand labels (``validate.score``, ``run --gold``): a toy claims table with known
precision and recall, label coverage, and the wiki's labelled splits when the dump is present."""

import json
from pathlib import Path

import pandas as pd
import pytest

from swarmprov import cli, pipeline, validate
from swarmprov.schema import RunDir, conform

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
    (run.path / "notes.jsonl").write_text(json.dumps({"event_id": "e1", "text": "a"}) + "\n")   # no gold block
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
    assert "3 of 4 labelled posts in the run" in out and "none of its labelled posts is in this run" in out
    assert "notes.jsonl: not a label file" in out and not (run.path / "validation_notes.json").exists()
    assert (run.path / "validation_toy.json").exists() and not (run.path / "validation_other.json").exists()
    with pytest.raises(ValueError, match="not a label file"):
        validate.score(run, run.path / "notes.jsonl")
    assert validate.read_labels(run.path / "notes.jsonl") is None and len(validate.read_labels(run.path / "toy_labels.jsonl")) == 4


def test_run_gold_scores_each_label_file_before_the_report(synth_null):
    run, _ = synth_null
    toy, held = (json.loads((run.path / f"validation_{s}.json").read_text()) for s in ("toy", "heldout"))
    assert toy["coverage"] == {"labelled": 8, "in_run": 4, "share": 0.5} and held["coverage"]["share"] == 1.0
    assert not (run.path / "validation_notes.json").exists()   # the transcript beside the labels is skipped
    text = (run.path / "report.md").read_text()
    assert "**validation_toy** — 4 of 8 (the others are not posts of this run) labelled posts (toy; labeller: toy)" in text
    assert "**validation_heldout** — 4 labelled posts (heldout; labeller: toy)" in text


def test_run_gold_refuses_a_missing_or_non_label_file_before_any_stage(synth_null, tmp_path, capsys):
    run, d = synth_null
    for gold, why in ((tmp_path / "nope_labels.jsonl", "does not exist"), (d / "transcript.jsonl", "is not a label file")):
        with pytest.raises(SystemExit) as e:
            cli.main(["run", str(d / "transcript.jsonl"), "-o", str(tmp_path / "run"), "--gold", str(gold)])
        assert e.value.code == 2 and why in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
    assert cli.main(["validate", str(run.path), str(d / "transcript.jsonl")]) == 1
    assert "not a label file" in capsys.readouterr().err


def test_wiki_labels_hold(wiki_run):
    """The extractor on the wiki still meets its labelled accuracy."""
    run = wiki_run
    pipeline.score_gold(run, [LABELS])
    res = {s: json.loads((run.path / f"validation_{s}.json").read_text()) for s in ("dev", "heldout", "test")}
    assert all(r["coverage"]["share"] == 1.0 for r in res.values())
    f = res["test"]["fields"]
    assert f["answer_post"]["precision"] >= 0.95 and f["answer_post"]["recall"] >= 0.9   # 1.00 / 0.91
    assert f["item"]["accuracy"] >= 0.95 and f["value"]["accuracy"] >= 0.95              # 0.96, 1.00
