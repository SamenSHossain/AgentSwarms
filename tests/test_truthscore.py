"""The synthetic scenarios against their truth (``synth`` options, ``truthscore``): exposure, deletion-aware exposure,
A2 (with a confounded null), source attribution, A3, A4 and identity.  Thresholds come from seeds 0-7 (0-9 for the
null), each with a margin; the ranges seen are in the comments."""

import hashlib
import json
import shutil

import pandas as pd
import pytest

from swarmprov import checks, cli, pipeline, report, synth, truthscore
from swarmprov.adapters.chat import ChatAdapter
from swarmprov.schema import RunDir

SCENARIOS = {
    "copy latest": dict(copy_rate=0.7),
    "copy first, boards, aliases, deletions, technique": dict(copy_rate=0.3, copy_from="first", n_boards=3,
                                                              alias_rate=0.2, delete_rate=0.1, technique="TRICK-A"),
    "confounded null": dict(copy_rate=0.0, confound=0.3),
}
# sha256 of the seed-2 chat swarm and the seed-3 AutoGen swarm as written before the scenario options existed
DEFAULT_SHA = {"transcript.jsonl": "52a3220d7225fe6d0760096e3c3da7e5cbf718f06e6b4aeb4bf3cccec9f24722",
               "truth.csv": "b7bc52a857d7ecc7a347933235f158d1132d76d6e40a14f102001ec6c324e8c5",
               "config.json": "3859aa93ea8fb11bd46feb0e4724a0eba578af626e1adb98b9ae72281e3261c8"}
AUTOGEN_SHA = "6be147cdf1c68e23c2899c08b9378a69da406bc9f24ece179f7f7faeb938a18a"


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_scenario_options_leave_the_default_swarm_byte_identical(tmp_path):
    synth.write(tmp_path / "chat" / "transcript.jsonl", seed=2)
    assert {f: _sha(tmp_path / "chat" / f) for f in DEFAULT_SHA} == DEFAULT_SHA
    synth.write(tmp_path / "ag" / "autogen.jsonl", seed=3, fmt="autogen")
    assert _sha(tmp_path / "ag" / "autogen.jsonl") == AUTOGEN_SHA
    off = dict(delete_rate=0.0, technique=None, n_boards=1, alias_rate=0.0, confound=0.0)
    assert synth.simulate(n_agents=20, seed=4, **off)[0] == synth.simulate(n_agents=20, seed=4)[0]


def test_scenario_options_change_only_what_they_name():
    base, truth = synth.simulate(n_agents=30, seed=1)
    for opt in ({"delete_rate": 0.2}, {"alias_rate": 0.3}, {"n_boards": 2}):
        posts, t = synth.simulate(n_agents=30, seed=1, **opt)
        # the same questions arrive; deletions only take posts off the board, the others change no draw at all
        assert (t["t_arrival"] == truth["t_arrival"]).all() and (t["item"] == truth["item"]).all()
        if "delete_rate" not in opt:
            assert t["copied"].equals(truth["copied"]) and t["value"].equals(truth["value"])
    posts, t = synth.simulate(n_agents=30, seed=1, alias_rate=0.3)
    assert (t["agent"] != t["true_agent"]).mean() == pytest.approx(0.3, abs=0.1)
    assert {p["author"] for p in posts} == set(t["agent"]) and t.groupby("agent")["true_agent"].nunique().max() == 1
    posts, t = synth.simulate(n_agents=30, seed=1, n_boards=2)
    assert {p["channel"] for p in posts} == {"alpha-board", "alpha-board-2", "beta-board", "beta-board-2"}
    posts, t = synth.simulate(n_agents=30, seed=1, delete_rate=0.2)
    gone = [p for p in posts if "deleted_at" in p]
    assert len(gone) == t["deleted_at"].notna().sum() and all(p["deleted_at"] > p["ts"] for p in gone)


def test_confound_is_fast_late_starters_who_never_copy():
    _, t = synth.simulate(n_agents=40, seed=3, copy_rate=0.7, confound=0.25)
    fast = t[t["fast"]]
    assert fast["agent"].nunique() == 10 and not fast["copied"].any()
    assert fast["latency_s"].eq(1).mean() == pytest.approx(synth.FAST_P, abs=0.2)
    start = t.groupby("agent")["t_arrival"].min()   # the start, give or take the minute of arrival jitter
    late = fast["agent"].unique()
    assert start[late].min() > start.drop(late).max() - pd.Timedelta(minutes=2)   # the latest to start


def test_technique_spreads_by_reading_and_is_written_as_truth(tmp_path):
    cli.main(["synth", "-o", str(tmp_path / "transcript.jsonl"), "--seed", "2", "--technique", "TRICK-A",
              "--boards", "2", "--alias-rate", "0.1", "--delete-rate", "0.1", "--confound", "0.1"])
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["techniques"][0] | {"description": ""} == {"name": "TRICK-A", "pattern": "TRICK\\-A",
                                                          "families": ["alpha-task"], "description": ""}
    tt = pd.read_csv(tmp_path / "truth_techniques.csv", parse_dates=["t_first_read", "t_first_use", "t_last_post"])
    truth = pd.read_csv(tmp_path / "truth.csv")
    assert {"deleted_at", "uses_technique", "true_agent", "fast"} <= set(truth.columns)
    assert tt["is_originator"].sum() == 1 and set(tt["family"]) == {"alpha-task"}
    adopted = tt[tt["adopted"] & ~tt["is_originator"]]
    assert (adopted["t_first_read"] < adopted["t_first_use"]).all()   # read on arrival, used in that round's post
    assert tt.loc[tt["is_originator"], "t_first_use"].iloc[0] == tt["t_first_use"].min()
    lines = [json.loads(l) for l in open(tmp_path / "transcript.jsonl")]
    assert sum("TRICK-A" in p["text"] for p in lines) == truth["uses_technique"].sum()


@pytest.mark.parametrize("bad", [{"delete_rate": 1.5}, {"alias_rate": -0.1}, {"confound": 2}, {"n_boards": 0},
                                 {"n_boards": 1.5}, {"technique": "two words"}, {"technique": ""}])
def test_scenario_options_are_checked(bad):
    with pytest.raises(ValueError):
        synth.simulate(n_agents=4, **bad)


def test_bad_scenario_options_are_usage_errors_and_autogen_takes_no_deletions_or_boards(tmp_path, capsys):
    for opt in ({"delete_rate": 0.1}, {"n_boards": 2}):   # no timestamps or channels to carry them
        with pytest.raises(ValueError, match="autogen format"):
            synth.write(tmp_path / "ag" / "autogen.jsonl", n_agents=4, fmt="autogen", **opt)
    for argv in (["--format", "autogen", "--boards", "2"], ["--delete-rate", "2"], ["--alias-rate", "-0.1"],
                 ["--confound", "x"], ["--technique", "two words"]):
        with pytest.raises(SystemExit) as e:
            cli.main(["synth", "-o", str(tmp_path / "x" / "t.jsonl"), "--agents", "4", *argv])
        assert e.value.code == 2 and "error:" in capsys.readouterr().err
    assert not (tmp_path / "ag").exists() and not (tmp_path / "x").exists()   # nothing written


def test_chat_reads_deletion_times(tmp_path, capsys):
    msgs = [{"ts": f"2026-06-16T08:0{i}:00Z", "author": a, "text": f"post {i}", "deleted_at": d}
            for i, (a, d) in enumerate([("A", "2026-06-16T09:00:00Z"), ("B", None), ("C", None)])]
    p = tmp_path / "chat.json"
    p.write_text(json.dumps(msgs))
    assert cli.main(["inspect", str(p)]) == 0 and "  visible_until <- deleted_at (1)" in capsys.readouterr().out
    b = ChatAdapter().load(p)
    assert b.capabilities.has_lifecycle and b.notes["n_deleted"] == 1
    assert b.events["visible_until"].iloc[0] == pd.Timestamp("2026-06-16T09:00:00Z") and b.events["visible_until"].iloc[1:].isna().all()
    p.write_text(json.dumps([{k: v for k, v in m.items() if k != "deleted_at"} | {"visible_until": m["deleted_at"]}
                             for m in msgs]))
    assert ChatAdapter().load(p).notes["n_deleted"] == 1
    p.write_text(json.dumps([{k: v for k, v in m.items() if k != "ts"} for m in msgs]))   # message order only
    b = ChatAdapter().load(p)
    # message order only: the deletion times are set aside, so nothing is deletion-aware, but the count is kept
    assert b.notes["clock"] == "sequence" and b.events["visible_until"].isna().all()
    assert not b.capabilities.has_lifecycle and b.notes["n_deleted"] == 1
    assert "the deletion times of 1 message were set aside too" in report._when({"clock": "sequence"}, b.notes)
    p.write_text(json.dumps([{k: v for k, v in m.items() if k != "deleted_at"} for m in msgs]))
    assert not ChatAdapter().load(p).capabilities.has_lifecycle


def _scored(tmp_path_factory, name: str):
    d = tmp_path_factory.mktemp(name.split(",")[0].replace(" ", "_"))
    path = synth.write(d / "transcript.jsonl", n_agents=60, seed=2, **SCENARIOS[name])
    run = RunDir(d / "run")
    pipeline.ingest(path, run, "chat", str(d / "config.json"))
    pipeline.extract(run)
    pipeline.expose(run)
    pipeline.build_graph(run)
    return run, truthscore.score(run, d)


@pytest.fixture(scope="module")
def scored(synth_run, tmp_path_factory):
    """Each scenario's run and scorecard; the plain swarm is the shared seed-2 run, copied so its report stays as is."""
    d = tmp_path_factory.mktemp("latest")
    for f in ("transcript.jsonl", "truth.csv", "config.json"):
        shutil.copy(synth_run[0].path.parent / f, d / f)
    shutil.copytree(synth_run[0].path, d / "run")
    run = RunDir(d / "run")
    out = {"copy latest": (run, truthscore.score(run, d))}
    return out | {n: _scored(tmp_path_factory, n) for n in SCENARIOS if n != "copy latest"}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_exposure_and_identity_match_the_truth(scored, name):
    run, r = scored[name]
    assert r["label"] == truthscore.LABEL and r["rounds"]["matched"] == r["rounds"]["truth"] == 300
    assert r["D"]["accuracy_vs_public_at_arrival"] >= 0.8      # 0.86-0.97
    assert r["D"]["accuracy_vs_earlier_copy"] >= 0.99          # 1.0: D is what it says
    assert r["D_visible"]["accuracy"] >= 0.95                  # 1.0
    assert r["identity"]["merged"]["purity"] >= 0.95 and r["identity"]["merged"]["true_agents_split"] == 0
    a4 = r["A4"]
    assert a4["slots"] == a4["true_slots"] == a4["in_both"] == a4["status_agrees"] == a4["outcome_agrees"]
    n_x = int((run.read("edges_merged")["kind"] == "relay_xchannel").sum())
    assert (n_x > 0) == (SCENARIOS[name].get("n_boards", 1) > 1)   # 80-85 hops on 3 boards, none on one
    # the integrity checks re-derive exposure under deletions, boards and aliases too
    assert not checks.check_run(run)["n_violations"].any()


@pytest.mark.parametrize("name", ["copy latest", "copy first, boards, aliases, deletions, technique"])
def test_copy_effect_and_sources_are_recovered(scored, name):
    _, r = scored[name]
    a2 = r["A2"]
    assert a2["covers"]                     # on 16 of 16 runs
    assert a2["p"] < 0.05 or name != "copy latest"   # copy rate 0.7: p < 0.001; 0.3: 7 of 8 under 0.05
    src = {s["rule"]: s for s in r["sources"]}
    match = src["latest source" if name == "copy latest" else "first source"]
    # the rule that matches how the agents copy: 0.98-1.0; under deletions the first post may be gone (0.87-1.0),
    # and the first copy still visible names the source every time
    assert match["names_true_source"] >= (0.95 if name == "copy latest" else 0.8)
    # within 0.005 of the truth on one board, 0.06 on three with deletions (the other rule misses by 0.33-0.40)
    assert abs(match["top10_share"] - match["true_top10_share"]) < (0.05 if name == "copy latest" else 0.1)
    if name != "copy latest":
        assert src[truthscore.VISIBLE]["names_true_source"] >= 0.95
        assert src["latest source"]["names_true_source"] < 0.5   # 0.19-0.35
        assert match["true_top10_share"] - src["latest source"]["top10_share"] > 0.2


def test_aliases_merge_and_the_technique_is_recovered(scored):
    _, r = scored["copy first, boards, aliases, deletions, technique"]
    strict = r["identity"]["strict"]
    assert strict["agents"] > strict["true_agents"] == 60 and strict["true_agents_split"] > 20
    (a3,) = r["A3"]
    assert a3["at_risk"] == a3["true_at_risk"] and abs(a3["adoption_share"] - a3["true_adoption_share"]) <= 0.1
    assert abs(a3["median_hours_to_adopt"] - a3["true_median_hours_to_adopt"]) <= 1   # equal on seeds 0-7
    assert r["D_visible"]["deleted_posts"] > 0 and r["D_visible"]["share"] < r["D"]["share"]


def test_the_confounded_null_is_not_significant(scored):
    run, r = scored["confounded null"]
    a2 = r["A2"]
    assert a2["true_effect"] == 0 and a2["covers"] and a2["p"] > 0.05   # p 0.06-0.89 on seeds 0-9
    assert abs(a2["beta"]) < 0.15 and a2["y_instant_sd"] > 0.2          # |beta| <= 0.10; sd 0.34-0.39
    assert a2["mean_y_D1"] - a2["mean_y_D0"] > 0.1                      # 0.14-0.22 without agent FE: the confound


def test_truth_score_command_rebuilds_the_report(scored, capsys):
    run, _ = scored["copy latest"]   # a copy of the shared run, which has a report
    assert cli.main(["truth-score", str(run.path), str(run.path.parent)]) == 0
    assert "truth score under the synthetic generative model" in capsys.readouterr().out
    text = (run.path / "report.md").read_text()
    block = text[text.index("**Synthetic truth**"):text.index("## Limitations")]
    assert "under the synthetic generative model" in block and "| A2 `y_instant ~ D_cons` |" in block
    assert "| identity, strict mapping | 60 agents | 60 agents | purity 100%, 0 true agents split |" in block
    assert "D_visible" not in block and "TRICK" not in block   # nothing deleted, no technique planted
    summary = json.loads((run.path / "summary.json").read_text())
    assert summary["truth_validation"]["mapping"] == "merged" and summary["A1"]["bootstrap"]["reps"] == 1000   # as built
    assert cli.main(["truth-score", str(run.path), str(run.path)]) == 1   # no truth.csv there
    other = run.path.parent / "other"
    other.mkdir()
    shutil.copy(run.path.parent / "config.json", other)
    pd.read_csv(run.path.parent / "truth.csv").assign(agent="Nobody", src_agent=None).to_csv(other / "truth.csv")
    before = (run.path / "truth_validation.json").read_text()
    assert cli.main(["truth-score", str(run.path), str(other)]) == 1   # another swarm's truth: no round matches
    assert "no answer round" in capsys.readouterr().out and (run.path / "truth_validation.json").read_text() == before


def test_scorecard_rows_for_deletions_techniques_and_aliases(scored):
    _, r = scored["copy first, boards, aliases, deletions, technique"]
    table = report._truth_block(r, summary := {})[1]
    assert summary["truth_validation"] is r
    rows = {line.split(" | ")[0].lstrip("| "): line for line in table.splitlines()[2:]}
    assert rows["`D_visible` (%d posts deleted)" % r["D_visible"]["deleted_posts"]].endswith("| 100% of rounds |")
    assert "A3 `TRICK-A` adoption share" in rows and "A3 `TRICK-A` median time to adopt" in rows
    assert "A5 top-10 share of instant answers, first copy still visible (not an A5 rule)" in rows
    strict = r["identity"]["strict"]
    assert rows["identity, strict mapping"].endswith(f"| purity 100%, {strict['true_agents_split']} true agents split |")


def test_truth_score_rebuilds_each_report_the_run_has(tmp_path, monkeypatch):
    """A run with a merged and a strict report gets both rebuilt, each under its own mapping and resamples."""
    run = RunDir(tmp_path / "run")
    (tmp_path / "truth.csv").write_text("")
    (run.path / "summary.json").write_text(json.dumps({"mapping": "merged", "A1": {"bootstrap": {"reps": 7}}}))
    (run.path / "summary_strict.json").write_text(json.dumps({"mapping": "strict"}))
    built = []
    monkeypatch.setattr(truthscore, "score", lambda *a: {"D": {"accuracy_vs_public_at_arrival": 1,
                                                               "accuracy_vs_earlier_copy": 1},
                                                         "A2": None, "rounds": {"matched": 1, "truth": 1}})
    monkeypatch.setattr(report, "build", lambda r, mapping, boot: built.append((mapping, boot)))
    monkeypatch.setattr(truthscore, "_missing", lambda *a: None)
    assert truthscore.main(run, tmp_path) == 0
    assert built == [("merged", 7), ("strict", 0)]


def test_truth_score_on_a_swarm_that_never_copies(synth_null, tmp_path, capsys):
    """copy_rate 0 without the confound: no answer is instant, so A2's fit is a point at 0 and claims no coverage;
    truth-score and the report it rebuilds run through.  The copy has its exposure edges dropped, so the rebuilt
    report also names that failing check."""
    run = RunDir(shutil.copytree(synth_null[0].path, tmp_path / "run"))
    e = run.read("edges_merged")
    run.write("edges_merged", e[e["kind"] != "exposure"])
    assert cli.main(["truth-score", str(run.path), str(synth_null[1])]) == 0
    assert "; A2 not estimated: every round has the same y_instant\n" in capsys.readouterr().out
    tv = json.loads((run.path / "truth_validation.json").read_text())
    a2 = tv["A2"]
    assert a2["covers"] is None and a2["se"] == 0 and a2["y_instant_sd"] == 0 and a2["true_effect"] == 0
    row = "| A2 `y_instant ~ D_cons` | not estimated: every round has the same y_instant | 0.000 (the share of rounds " \
          "public at arrival that copied) | – |"
    text = (run.path / "report.md").read_text()
    assert row in text and "Not passing: `exposure_edges` (error; " in text
    # a scorecard written before a point interval claimed no coverage renders without a p value too
    old = report._truth_block(tv | {"A2": a2 | {"covers": True, "note": None}}, {})[1]
    assert "| A2 `y_instant ~ D_cons` | not estimated | 0.000 (the share" in old and old.count("| – |") >= 1


def test_truth_score_names_what_is_missing(synth_null, tmp_path, capsys):
    run, d = synth_null
    part = tmp_path / "part"
    part.mkdir()
    for f in ("profile.json", "events.parquet"):   # an ingest, nothing after it
        shutil.copy(run.path / f, part / f)
    assert cli.main(["truth-score", str(part), str(d)]) == 1
    assert f"has no agents, tags: run `swarmprov extract {part}`" in capsys.readouterr().out
    assert not (part / "truth_validation.json").exists()
    bare = tmp_path / "truth"
    bare.mkdir()
    shutil.copy(d / "truth.csv", bare)
    assert cli.main(["truth-score", str(run.path), str(bare)]) == 1 and "no config.json" in capsys.readouterr().out
