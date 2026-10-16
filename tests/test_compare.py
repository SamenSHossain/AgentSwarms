"""``swarmprov compare``: the setup, the summary keyed by meaning within a tolerance, and the rows on their stable
ids, on synthetic runs: a run against itself, two configs on a moved transcript, two seeds, merged against strict."""

import json
import re
import shutil

import pandas as pd
import pytest

from swarmprov import cli, compare, pipeline, synth
from swarmprov.schema import RunDir


def _stages(transcript, d, config) -> RunDir:
    run = RunDir(d)
    pipeline.ingest(transcript, run, "chat", str(config))
    pipeline.extract(run)
    pipeline.expose(run)
    pipeline.build_graph(run)
    return run


@pytest.fixture(scope="module")
def slow(synth_run, tmp_path_factory):
    """The seed-2 transcript moved to another path, so every chat event id changes, and run with an instant
    threshold of 60 s instead of 2 s (no report)."""
    src, d = synth_run[0].path.parent, tmp_path_factory.mktemp("moved")
    shutil.copy(src / "transcript.jsonl", d / "transcript.jsonl")
    (d / "config.json").write_text(json.dumps(json.loads((src / "config.json").read_text()) | {"instant_threshold_s": 60}))
    return _stages(d / "transcript.jsonl", d / "run", d / "config.json")


@pytest.fixture(scope="module")
def seed3(tmp_path_factory):
    d = tmp_path_factory.mktemp("seed3")
    return _stages(synth.write(d / "transcript.jsonl", n_agents=30, seed=3), d / "run", d / "config.json")


def _tables_line_up(md: str) -> bool:
    """Every row of every markdown table has its header's number of cells (a stray pipe would add one)."""
    for block in re.findall(r"(?:^\|.*\|$\n?)+", md, flags=re.M):
        rows = block.strip().splitlines()
        if len({len(re.findall(r"(?<!\\)\|", r)) for r in rows}) != 1:
            return False
    return True


def test_a_run_against_itself_has_no_changes(synth_run, tmp_path, capsys):
    p = str(synth_run[0].path)
    assert cli.main(["compare", p, p, "-o", str(tmp_path)]) == 0
    assert f"compare {p} (merged) vs {p} (merged): 0 changes" in capsys.readouterr().out
    res = json.loads((tmp_path / "compare.json").read_text())
    assert res["n_changes"] == 0 and res["summary"]["n_compared"] > 500 and res["summary"]["largest_within_tol"] is None
    assert not res["alignment"]["fallback"] and res["alignment"]["event_id_overlap"] == 1
    assert "**0 changes**" in (tmp_path / "compare.md").read_text()
    assert cli.main(["compare", p, str(tmp_path), "-o", str(tmp_path)]) == 2   # not a run directory


def test_summary_rows_are_keyed_by_meaning_within_a_tolerance():
    sa = {"mapping": "merged", "A3": [{"technique": "x", "adopters": 3, "first_seen": "2026-06-20 05:10:21+00:00"},
                                      {"technique": "y", "adopters": 5}],
          "A4": {"rows": [{"slot": "s1", "status": "held"}, {"slot": "s2", "status": "overturned"}]},
          "A2": [{"outcome": "y_instant", "treatment": "D_cons", "beta": 0.1, "note": None}],
          "A5": {"attribution": [{"rule": "first source", "answers": "all exposed", "top10_share": 0.7}]},
          "agents": {"top_suppliers": [{"label": "Mar16|t", "answers_supplied": 4}]}, "validation": {"n": 3}}
    sb = {"mapping": "strict", "A3": [{"technique": "y", "adopters": 5},   # one time, spelt another way
                                      {"technique": "x", "adopters": 3, "first_seen": "2026-06-20T05:10:21+00:00"}],
          "A4": {"rows": [{"slot": "s2", "status": "overturned"}]},
          "A2": [{"outcome": "y_instant", "treatment": "D_cons", "beta": 0.1 * (1 + 1e-9), "note": float("nan")}],
          "A5": {"attribution": [{"rule": "first source", "answers": "all exposed", "top10_share": 0.6}]},
          "agents": {"top_suppliers": [{"label": "mar16scout", "answers_supplied": 4}]}}
    d = compare.summary_diff(sa, sb, labels_b={"mar16scout": "Mar16|t"})
    assert d["sections_only_a"] == ["validation"] and not d["sections_only_b"]
    assert d["rows_only_a"] == ["A4.rows[s1]"] and not d["rows_only_b"]   # reordered A3 rows match, NaN equals null
    assert [(c["path"], c["a"], c["b"]) for c in d["changed"]] == [
        ("A5.attribution[first source | all exposed].top10_share", 0.7, 0.6)]
    assert d["largest_within_tol"]["path"] == "A2[y_instant | D_cons].beta"
    assert d["largest_within_tol"]["rel_delta"] == pytest.approx(1e-9, rel=1e-3)
    assert compare.summary_diff(sa, sb, tol=0.2, labels_b={"mar16scout": "Mar16|t"})["changed"] == []
    later = {**sb, "A3": [{"technique": "x", "adopters": 3, "first_seen": "2026-06-20T05:10:22+00:00"}]}
    assert [c["path"] for c in compare.summary_diff(sa, later, tol=0.2)["changed"]] == ["A3[x].first_seen"]
    assert compare.summary_diff(sa, sb)["rows_only_b"] == ["agents.top_suppliers[mar16scout]"]   # untranslated


def test_the_default_tolerance_hides_solver_noise():
    """Fixed-effects estimates move by up to 6e-5 relative between processes on the same tables: by default that is
    no change, so a rerun of the committed wiki compares with 0 changes; --tol 0 shows it."""
    sa = {"A2": [{"outcome": "y_instant", "treatment": "D", "ci_hi": 0.0628}]}
    sb = {"A2": [{"outcome": "y_instant", "treatment": "D", "ci_hi": 0.0628 * (1 + 6e-5)}]}
    assert cli._parser().parse_args(["compare", "a", "b"]).tol == compare.TOL == 1e-4
    assert compare.summary_diff(sa, sb)["changed"] == [] and len(compare.summary_diff(sa, sb, tol=0)["changed"]) == 1


def test_counts_compare_exactly():
    """A count of 22,317 against 22,318 is within 1e-4 relative but is a change; a count JSON wrote as a float is one
    too, and a fraction next to it still gets the tolerance."""
    sa = {"n_edges": 22317, "A1": {"n_rounds": 15987.0, "share": 0.925, "n_same": 4}}
    sb = {"n_edges": 22318, "A1": {"n_rounds": 15988, "share": 0.92505, "n_same": 4.0}}
    d = compare.summary_diff(sa, sb)
    assert [c["path"] for c in d["changed"]] == ["A1.n_rounds", "n_edges"]
    assert d["largest_within_tol"]["path"] == "A1.share"


def test_two_configs_on_a_moved_transcript(synth_run, slow):
    """Moving the file changes every chat event id, so posts are aligned on (time, author, text hash); the only row
    change left is y_instant, and its flips are exactly those a join on the agent-round finds."""
    a = synth_run[0]
    res = compare.compare(a.path, slow.path)
    setup = {r["field"]: (r["a"], r["b"]) for r in res["setup"]}
    assert setup["config.instant_threshold_s"] == (2.0, 60) and {"source", "config_path"} <= set(setup)
    al = res["alignment"]
    assert al["fallback"] and al["event_id_overlap"] == 0 and al["aligned"] == len(a.read("events")) and al["collisions"] == 0
    r = res["rows"]
    assert all(r[t]["added"] == r[t]["dropped"] == 0 for t in ("events", "claims", "exposures", "chains", "agents"))
    assert not r["claims"]["changed"] and set(r["exposures"]["changed"]) == {"y_instant"}
    m = a.read("exposures_merged").merge(slow.read("exposures_merged"), on=["agent", "family", "item", "episode"],
                                          suffixes=("_a", "_b"))
    assert len(m) == len(a.read("exposures_merged"))
    flips = m[m["y_instant_a"].fillna(-1) != m["y_instant_b"].fillna(-1)]
    got = r["exposures"]["changed"]["y_instant"]
    assert got["n"] == len(flips) > 0
    assert got["by_change"] == (flips["y_instant_a"].map("{:g}".format) + "→" + flips["y_instant_b"].map("{:g}".format)
                                ).value_counts().to_dict()
    assert res["summary"] == {"missing": [f"{slow.path}/summary.json"]}   # the moved run has no report


def test_two_seeds_share_no_post(synth_run, seed3):
    a = synth_run[0]
    res = compare.compare(a.path, seed3.path)
    assert res["alignment"]["fallback"] and res["alignment"]["aligned"] == 0
    r = res["rows"]
    for t, name in (("events", "events"), ("exposures", "exposures_merged")):
        assert (r[t]["dropped"], r[t]["added"]) == (len(a.read(name)), len(seed3.read(name)))
    assert {"n_events", "source", "config_path"} <= {s["field"] for s in res["setup"]}


def test_merged_against_strict_translates_labels(synth_strict, tmp_path, capsys):
    """Each synthetic agent signs one handle, so the mappings group alike: no row changes once the strict labels
    are read as merged ones (without that, every source and originator would differ)."""
    p = str(synth_strict.path)
    assert cli.main(["compare", p, p, "--mapping-b", "strict", "-o", str(tmp_path), "--limit", "3"]) == 0
    assert "summary: " in capsys.readouterr().out
    res = json.loads((tmp_path / "compare.json").read_text())
    assert res["setup"] == [{"field": "mapping", "a": "merged", "b": "strict"}] and res["b"]["mapping"] == "strict"
    r = res["rows"]
    assert all(r[t]["added"] == r[t]["dropped"] == 0 and not r[t]["changed"] for t in r)
    assert r["agents"]["regrouped"] == 0 and r["agents"]["agents_a"] == r["agents"]["agents_b"] == 60
    s = res["summary"]   # summary.json against summary_strict.json, which was built without resamples: only they differ
    paths = [c["path"] for c in s["changed"]]
    assert s["n_compared"] > 500 and "A1.share_exposed_ci[0]" in paths and "A6.task values.mean_repeaters_lo" in paths
    assert all("_ci[" in x or x.startswith("A1.bootstrap.") or x.endswith(("_lo", "_hi")) for x in paths), paths
    # top-source rows keyed by merged labels on both sides: only ties at a top-10 cut, which sort by label, differ
    assert len(s["rows_only_a"]) == len(s["rows_only_b"]) < 5
    assert all(x.startswith("agents.top_no_prior_copy[") for x in s["rows_only_a"] + s["rows_only_b"])
    md = (tmp_path / "compare.md").read_text()
    assert "| agent-rounds | claim_id | 300 | 300 | 0 | 0 | 0 |" in md and _tables_line_up(md)
    assert "more in `compare.json`." in md   # --limit 3


def test_regrouped_author_strings_and_duplicate_posts():
    ag = pd.DataFrame({"author_raw": ["A1", "a1", "B"], "agent_strict": ["a1", "a1", "b"], "agent_merged": ["X", "X", "X"]})
    g = compare._agents(ag, ag, "merged", "strict", 10)
    assert (g["agents_a"], g["agents_b"], g["regrouped"]) == (1, 2, 3) and "label" not in g["changed"]
    t = pd.Timestamp("2026-06-16T08:00:00Z")
    ev = pd.DataFrame({"ts": [t, t, t], "author_raw": ["x", "y", "y"], "text": ["hi", "dup", "dup"]})
    pairs, n_coll = compare.align_events(ev.assign(event_id=["a1", "a2", "a3"]), ev.assign(event_id=["b1", "b2", "b3"]))
    assert pairs == {"b1": "a1"} and n_coll == 4   # the two identical posts on each side are flagged, not paired


def test_the_config_is_read_as_each_run_used_it(tmp_path, capsys):
    """A config file edited in place between two runs shows in the setup, since each run's stages recorded the
    settings they ran with; a stage rerun after an edit is noted; a run recorded before that reads the file as it
    is now, and says so, and two identical unreadable configs are no change."""
    posts = [{"ts": f"2026-05-04T10:0{i}:00Z", "author": a, "text": f"R1 Poland: 41.{i}"} for i, a in enumerate("abcab")]
    (tmp_path / "t.jsonl").write_text("\n".join(map(json.dumps, posts)))
    cfg = tmp_path / "iter.json"
    for name, k in (("A", 2), ("B", 600)):
        cfg.write_text(json.dumps({"instant_threshold_s": k}))
        pipeline.ingest(tmp_path / "t.jsonl", RunDir(tmp_path / name), "chat", str(cfg))
        pipeline.extract(RunDir(tmp_path / name))
    res = compare.compare(tmp_path / "A", tmp_path / "B")
    assert res["setup"] == [{"field": "config.instant_threshold_s", "a": 2, "b": 600}] and res["n_changes"] == 1
    assert res["config"]["a"] == {"recorded": True, "stages_differ": None}
    cfg.write_text(json.dumps({"instant_threshold_s": 30}))
    pipeline.extract(RunDir(tmp_path / "B"))
    assert cli.main(["compare", str(tmp_path / "A"), str(tmp_path / "B"), "-o", str(tmp_path / "cmp")]) == 0
    assert "note: B's stages ran with different settings (ingest ≠ extract)" in capsys.readouterr().out
    assert "1 change**" in (tmp_path / "cmp" / "compare.md").read_text()   # 2 → 30, the latest stage's
    prof = json.loads((tmp_path / "A" / "profile.json").read_text())
    for k in ("config_settings", "config_stages"):
        prof.pop(k)
    (tmp_path / "A" / "profile.json").write_text(json.dumps(prof))
    old = compare.compare(tmp_path / "A", tmp_path / "A")
    assert old["n_changes"] == 0 and old["config"]["a"]["recorded"] is False
    assert "A's settings are its config as it reads now" in compare.to_markdown(old)
    cfg.unlink()
    gone = compare.compare(tmp_path / "A", tmp_path / "A")
    assert gone["n_changes"] == 0 and not gone["setup"]
