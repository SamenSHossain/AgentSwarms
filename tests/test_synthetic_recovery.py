"""End-to-end: the pipeline must recover known provenance from a synthetic swarm."""

import pandas as pd
import pytest

from swarmprov import pipeline, synth


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    path = synth.write(d / "transcript.jsonl", n_agents=60, copy_rate=0.7, seed=2)
    run = pipeline.run_all(path, d / "run", adapter="chat", config=str(d / "config.json"))
    return run, pd.read_csv(d / "truth.csv")


def test_every_round_is_extracted(synth_run):
    run, truth = synth_run
    ex = run.read("exposures_merged")
    assert len(ex) == len(truth)                      # one agent-round per true round
    assert ex["agent"].nunique() == truth["agent"].nunique()


def test_exposure_matches_truth(synth_run):
    run, truth = synth_run
    ex = run.read("exposures_merged")
    # exposure is value-specific, so it can only be below truth by the wrong-answer rate (~10%)
    assert truth["public_at_arrival"].mean() - 0.12 <= ex["D"].mean() <= truth["public_at_arrival"].mean() + 0.02


def test_copy_effect_is_recovered(synth_run):
    run, truth = synth_run
    from swarmprov.analysis import causal
    est = causal.estimate(run.read("exposures_merged")).set_index(["outcome", "treatment"])
    beta = est.loc[("y_instant", "D_cons"), "beta"]
    assert est.loc[("y_instant", "D_cons"), "p"] < 0.05
    assert abs(beta - 0.7) < 0.25                     # true P(instant | public) - P(instant | not public) = 0.7
    # consensus matching: copied and computed answers are both (mostly) right, so ~no effect
    assert abs(est.loc[("y_consensus", "D_cons"), "beta"]) < 0.15


def test_report_is_written(synth_run):
    run, _ = synth_run
    text = (run.path / "report.md").read_text()
    assert "## A1. Provenance split" in text and "## A2." in text
