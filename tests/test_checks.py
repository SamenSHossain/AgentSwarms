"""Run integrity checks (``checks.check_run``, ``swarmprov check``): every check passes on the synthetic run, and a
copy with one table corrupted fails exactly the checks that table feeds."""

import json
import shutil

import pandas as pd
import pytest

from swarmprov import checks, cli
from swarmprov.schema import RunDir


@pytest.fixture(scope="module")
def clean(synth_run, tmp_path_factory):
    """The shared seed-2 run, copied without its summary.json (one test puts a doctored one back)."""
    d = tmp_path_factory.mktemp("checks") / "run"
    shutil.copytree(synth_run[0].path, d)
    (d / "summary.json").unlink()
    return d


def _failing(run: RunDir) -> set[str]:
    out = checks.check_run(run)
    return set(out.loc[out["n_violations"] > 0, "check"])


def _copy(clean, tmp_path) -> RunDir:
    shutil.copytree(clean, tmp_path / "run")
    return RunDir(tmp_path / "run")


def test_every_check_passes_on_the_synthetic_run(synth_run, clean):
    out = checks.check_run(RunDir(clean))
    assert list(out.columns) == checks.COLS and set(out["check"]) == set(checks.SEVERITY) - {"summary_matches_tables"}
    assert out["n_violations"].sum() == 0 and set(out["table"]) >= {"agents", "exposures_merged", "edges_strict"}
    saved = json.loads((clean / "checks.json").read_text())
    assert saved["n_checks"] == saved["n_pass"] == len(out) and saved["n_errors"] == 0
    # the report runs them with its own summary: 18 checks, the summary one included
    text = (synth_run[0].path / "report.md").read_text()
    assert "**Integrity: 18/18 checks pass**" in text
    assert json.loads((synth_run[0].path / "summary.json").read_text())["integrity"]["n_pass"] == 18


def test_a_flipped_exposure_flag_fails_the_exposure_checks(clean, tmp_path):
    run = _copy(clean, tmp_path)
    ex = run.read("exposures_merged")
    i = ex.index[ex["D"] == 0][0]
    ex.loc[i, "D"] = 1
    run.write("exposures_merged", ex)
    assert _failing(run) == {"exposure_matches_mentions", "exposure_edges"}
    out = checks.check_run(run)
    assert out.loc[out["check"] == "exposure_edges", "example_ids"].iloc[0] == [ex.loc[i, "claim_id"]]


def test_a_deleted_relay_edge_fails_the_chain_check(clean, tmp_path):
    run = _copy(clean, tmp_path)
    e = run.read("edges_strict")
    run.write("edges_strict", e.drop(index=e.index[e["kind"] == "relay"][0]))
    out = checks.check_run(run)
    bad = out[out["n_violations"] > 0]
    assert bad[["check", "table"]].values.tolist() == [["relay_hops_per_chain", "edges_strict"]]


def test_a_shifted_public_time_fails_the_rederivation(clean, tmp_path):
    run = _copy(clean, tmp_path)
    ex = run.read("exposures_merged")
    i = ex.index[ex["D"] == 1][0]
    ex.loc[i, "t_public"] = ex.loc[i, "t_public"] - pd.Timedelta(hours=1)
    run.write("exposures_merged", ex)
    assert _failing(run) == {"exposure_matches_mentions"}


def test_other_corruptions_fail_their_own_check(clean, tmp_path):
    run = _copy(clean, tmp_path)
    ag = run.read("agents")
    twin = ag.iloc[[0]].assign(author_raw=ag["author_raw"].iloc[0].upper(), agent_merged="Elsewhere|alpha-task")
    run.write("agents", pd.concat([ag, twin], ignore_index=True))   # one strict handle, two merged agents
    ex = run.read("exposures_strict")
    i = ex.index[ex["D"] == 1][0]
    ex.loc[i, "t_self_prepared"] = ex.loc[i, "t_public"] - pd.Timedelta(minutes=5)
    run.write("exposures_strict", ex)
    e = run.read("edges_merged")
    e.loc[e.index[e["kind"] == "exposure"][0], "src_event_id"] = "no-such-post"
    run.write("edges_merged", e)
    out = checks.check_run(run)
    bad = out[out["n_violations"] > 0].set_index("check")
    assert set(bad.index) == {"strict_refines_merged", "self_prepared_before_public", "edge_posts_resolve"}
    assert bad.loc["self_prepared_before_public", "severity"] == "warning"
    assert bad.loc["strict_refines_merged", "example_ids"] == [ag["agent_strict"].iloc[0]]


def test_summary_numbers_are_checked_against_the_tables(synth_run, clean, tmp_path):
    run = _copy(clean, tmp_path)
    summary = json.loads((synth_run[0].path / "summary.json").read_text())
    assert _failing(RunDir(run.path)) == set() and not checks.check_run(run, summary)["n_violations"].any()
    summary["A1"]["share_exposed"] += 0.01
    summary["A5"]["n_relay_edges"] += 1
    out = checks.check_run(run, summary).set_index("check")
    assert out.loc["summary_matches_tables", "example_ids"] == ["A1.share_exposed", "A5.n_relay_edges"]
    assert out.loc["summary_matches_tables", "severity"] == "warning"


def test_check_command_exits_1_on_an_error(clean, tmp_path, capsys):
    assert cli.main(["check", str(clean)]) == 0
    assert "17 of 17 checks pass: 0 errors, 0 warnings" in capsys.readouterr().out   # no summary.json here
    run = _copy(clean, tmp_path)
    e = run.read("edges_merged")
    run.write("edges_merged", e[e["kind"] != "relay"])
    assert cli.main(["check", str(run.path)]) == 1
    out = capsys.readouterr().out
    assert "ERROR relay_hops_per_chain" in out and "1 error, 0 warnings" in out
    assert cli.main(["check", str(tmp_path / "nothing")]) == 2
