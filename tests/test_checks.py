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
    # the report runs them with its own summary: 19 checks, the summary one included
    text = (synth_run[0].path / "report.md").read_text()
    assert "**Integrity: 19/19 checks pass**" in text and "None fails." in text and "Not run" not in text
    assert json.loads((synth_run[0].path / "summary.json").read_text())["integrity"]["n_pass"] == 19


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
    # the doctored own-post time follows from no mention either
    assert set(bad.index) == {"strict_refines_merged", "self_prepared_before_public", "edge_posts_resolve",
                              "exposure_matches_mentions"}
    assert bad.loc["exposure_matches_mentions", "example_ids"] == [f"{ex.loc[i, 'claim_id']} (t_self_prepared)"]
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
    assert "18 of 18 checks pass: 0 errors, 0 warnings" in capsys.readouterr().out   # no summary.json here
    run = _copy(clean, tmp_path)
    e = run.read("edges_merged")
    run.write("edges_merged", e[e["kind"] != "relay"])
    assert cli.main(["check", str(run.path)]) == 1
    out = capsys.readouterr().out
    assert "ERROR relay_hops_per_chain" in out and "1 error, 0 warnings" in out
    assert cli.main(["check", str(tmp_path / "nothing")]) == 2


def test_each_exposure_column_is_rederived(clean, tmp_path):
    """One round per column, that column alone corrupted: the loop names each round and the column that is off."""
    run = _copy(clean, tmp_path)
    ex = run.read("exposures_merged")
    hour, used = pd.Timedelta(hours=1), set()

    def row(mask):
        i = next(i for i in ex.index[mask] if i not in used)
        used.add(i)
        return i
    d1, dc, d0 = ex["D"] == 1, ex["D_cons"] == 1, ex["D"] == 0
    edits = [("D", d0, 1), ("D_cons", dc, 0), ("t_cons", dc, lambda v: v - hour), ("gap_cons_s", dc, lambda v: v * 3),
             ("D_cons_w60m", dc & (ex["gap_cons_s"] < 3600), 1), ("D_visible", d1, 0),
             ("t_public_visible", d1, pd.Timestamp("2000-01-01", tz="UTC")), ("src_visible_event_id", d1, "elsewhere"),
             ("src_deleted_before_report", d1, 1), ("src_latest", d1, "Nobody"), ("t_latest", d1, lambda v: v - hour),
             ("latest_event_id", d1, "elsewhere"), ("src_channel", d1, "another-board"),
             ("self_prepared", d0, 1), ("t_self_prepared", d0, pd.Timestamp("2000-01-01", tz="UTC")),
             ("gap_s", d1, lambda v: v + 1), ("D_w10m", d1 & (ex["gap_s"] < 600), 1)]
    want = set()
    for col, mask, v in edits:
        i = row(mask)
        ex.loc[i, col] = v(ex.loc[i, col]) if callable(v) else v
        want.add(f"{ex.loc[i, 'claim_id']} ({col})")
    run.write("exposures_merged", ex)
    got = checks._exposure(run.read("exposures_merged"), checks._facts(run.read("mentions"), "agent_merged"), {}, True)
    assert set(got) == want
    out = checks.check_run(run).set_index(["check", "table"])
    assert out.loc[("exposure_matches_mentions", "exposures_merged"), "n_violations"] == len(edits)
    assert out.loc[("exposure_matches_mentions", "exposures_strict"), "n_violations"] == 0


def test_handles_kept_apart_by_cohort_or_roster_only_warn(clean, tmp_path, capsys):
    ag = pd.DataFrame({"author_raw": ["Bot", "bot", "Ann", "ann", "Cy", "cy"],
                       "agent_strict": ["bot", "bot", "ann", "ann", "cy", "cy"],
                       "agent_merged": ["Mar16|a", "Apr02|a", "r1", "r2", "Mar16|a", "Mar16|b"],
                       "cohort": ["Mar16", "Apr02", None, None, "Mar16", "Mar16"],
                       "roster_agent": [None, None, "r1", "r2", None, None]})
    assert checks._split(ag) == (["cy"], ["ann", "bot"])   # cy: one cohort, no roster, yet two merged agents
    run = _copy(clean, tmp_path)
    ag = run.read("agents")
    twin = ag.iloc[[0]].assign(author_raw=ag["author_raw"].iloc[0].upper(), cohort="Dec31",
                               agent_merged="Dec31|alpha-task")   # what identity.resolve does with a second cohort
    run.write("agents", pd.concat([ag, twin], ignore_index=True))
    assert _failing(run) == {"strict_handles_kept_apart"}
    assert cli.main(["check", str(run.path)]) == 0
    assert "warn  strict_handles_kept_apart" in capsys.readouterr().out


def test_check_needs_stage_tables_and_a_summary_to_compare(clean, tmp_path, capsys):
    (tmp_path / "ingested").mkdir()
    shutil.copy(clean / "profile.json", tmp_path / "ingested")
    assert cli.main(["check", str(tmp_path / "ingested")]) == 2
    assert "no exposure or edge tables to check" in capsys.readouterr().out
    assert checks._summary({"mapping": "merged"}, RunDir(clean)) is None   # a summary stating no A1 or A5 number


def test_integrity_line_names_what_ran_what_had_nothing_to_compare_and_what_fails():
    from swarmprov import report
    chk = pd.DataFrame([["strict_refines_merged", "agents", 0, "error", []],
                        ["relay_hops_per_chain", "edges_merged", 2, "error", ["f|a|1", "f|b|2"]]], columns=checks.COLS)
    line = report._integrity(chk, summary := {}, "merged")[0]
    assert line.startswith("**Integrity: 1/2 checks pass** (`checks.json`; `swarmprov check RUN` reruns them): ")
    assert "reruns them): relay hops and chain carriers against the mentions and the strict mapping inside the " \
           "merged one. Not run, as this run has nothing for them to compare: exposure re-derived from the mentions, " in line
    assert ", rounds held before the first public copy, and summary.json against the tables. Not passing" in line
    assert line.rstrip().endswith("Not passing: `relay_hops_per_chain` (error; 2 in edges_merged: facts without one "
                                  "relay hop per carrier after the first).")
    assert summary["integrity"]["n_checks"] == 2 and summary["integrity"]["failing"][0]["n_violations"] == 2
