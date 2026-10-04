"""The wiki dump's events table beyond delete rows: first-recreation edges, probes,
clock grades, the recreation check, and the visibility-aware exposure variant."""

import json
from pathlib import Path

import pandas as pd
import pytest

from swarmprov import adapters, exposure, graph, lifecycle, pipeline

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
T = lambda s: f"2026-06-16T{s}:00Z"


def _dump(d: Path) -> Path:
    revs = [
        {"rev_id": "dse~P@1", "page_id": "dse/P", "wiki": "dse", "label": "acctA", "ip16": "1.1", "time": T("10:00"),
         "time_grade": "reqlog", "uncertainty_seconds": 1,
         "body": "Jun01 R1 CONFIRMED: Utah arrived 09:58:00; answered 73.74 at +30s. -- AgentA"},
        {"rev_id": "dse~P@2", "page_id": "dse/P", "wiki": "dse", "label": "acctB", "ip16": "9.9", "time": T("13:00"),
         "time_grade": "reqlog", "uncertainty_seconds": 1, "relation_type": "first_recreation_of",
         "related_event_id": ["delete:1", "delete:4"], "round_id": ["dse~P#round-1", "dse~P#round-2"],  # two edges, parallel lists
         "body": "Jun01 R1 CONFIRMED: Utah arrived 09:58:00; answered 73.74 at +30s. -- AgentA\n\nQ2 answered 16.4 -- AgentB"},   # restored + fresh
        {"rev_id": "dse~Q@1", "page_id": "dse/Q", "wiki": "dse", "label": "acctE", "ip16": "1.1", "time": T("12:30"),
         "time_grade": "reqlog", "uncertainty_seconds": 1,
         "body": "Jun02 R1 CONFIRMED: Utah arrived 12:28:00; answered 73.74 at +2s. -- AgentE"},   # same answer, after P's deletion
        {"rev_id": "dse~Late@1", "page_id": "dse/Late", "wiki": "dse", "label": "acctD", "ip16": "1.1", "time": T("09:00"),
         "time_grade": "rclog", "uncertainty_seconds": 1, "body": "Q3 answered 1.0 -- AgentD"},
        {"rev_id": "dse~Late@2", "page_id": "dse/Late", "wiki": "dse", "label": "acctD", "ip16": "1.1", "time": T("15:00"),
         "time_grade": "reqlog", "uncertainty_seconds": 1, "body": "Q4 answered 2.0 -- AgentD"},  # re-save after the cutoff, unmarked
    ]
    events = [
        {"event_id": "delete:1", "event_type": "delete", "wiki": "dse", "page": "P", "time": T("12:00"), "request_time": T("12:00"),
         "actor_label": "[Admin1]", "page_held": True, "round_id": "dse~P#round-1", "ip16": "2.2"},
        {"event_id": "delete:2", "event_type": "delete", "wiki": "dse", "page": "Gone", "time": T("12:30"), "request_time": T("12:30"),
         "actor_label": "[Admin1]", "page_held": False, "round_id": None, "ip16": "2.2"},
        {"event_id": "delete:3", "event_type": "delete", "wiki": "dse", "page": "Late", "time": T("11:00"), "request_time": T("10:59"),
         "actor_label": "[Admin1]", "page_held": True, "round_id": None, "ip16": "2.2"},
        {"event_id": "delete:4", "event_type": "delete", "wiki": "dse", "page": "P", "time": T("12:10"), "request_time": T("12:10"),
         "actor_label": "[Admin1]", "page_held": True, "round_id": "dse~P#round-2", "ip16": "2.2"},
        {"event_id": "revert:delete:2", "event_type": "revert", "wiki": "dse", "page": "Gone", "time": T("14:00"),
         "related_event_id": "delete:2", "relation_type": "first_recreation_of", "actor_label": "acctC", "page_held": False},
        {"event_id": "probe:attacklog_raw_dse_2606.jsonl:1", "event_type": "probe", "time": T("13:00"), "ip16": "9.9",
         "request_action": "form_editprefs", "param_family": "old_plist", "success_observed": False, "time_grade": "reqlog",
         "source_refs": ["corpus/live/attacklog_raw_dse_2606.jsonl:1"]},
        {"event_id": "probe:attacklog_raw_dse_2606.jsonl:2", "event_type": "probe", "time": T("13:05"), "ip16": "20.1",
         "request_action": "<script>alert('XSS')</script>", "param_family": "action", "success_observed": False,
         "time_grade": "reqlog", "source_refs": ["corpus/live/attacklog_raw_dse_2606.jsonl:2"]},
        {"event_id": "probe:attacklog_raw_dse_2606.jsonl:3", "event_type": "probe", "time": T("10:00"), "ip16": "1.1",
         "request_action": "browse", "param_family": "id", "success_observed": False, "time_grade": "reqlog",
         "source_refs": ["corpus/live/attacklog_raw_dse_2606.jsonl:3"]},        # same second as acctA's save, but 1.1 holds 3 labels
        {"event_id": "probe:attacklog_raw_dse_2606.jsonl:4", "event_type": "probe", "time": "2026-06-16T13:00:01Z", "ip16": "9.9",
         "request_action": "browse", "param_family": "id", "success_observed": False, "time_grade": "reqlog",
         "source_refs": ["corpus/live/attacklog_raw_dse_2606.jsonl:4"]},        # 1 s after a single-label save -> co-timed
        {"event_id": "probe:attacklog_raw_dse_2606.jsonl:5", "event_type": "probe", "time": "2026-06-16T13:00:02Z", "ip16": "9.9",
         "request_action": "browse", "param_family": "id", "success_observed": False, "time_grade": "reqlog",
         "source_refs": ["corpus/live/attacklog_raw_dse_2606.jsonl:5"]},        # 2 s -> not co-timed
    ] + [{"event_id": f"save:{r['rev_id']}", "event_type": "save", "wiki": "dse", "time": r["time"], "revision_ref": r["rev_id"]} for r in revs]
    pages = [{"page_id": "dse/P", "page_family": "oecd-equity"}, {"page_id": "dse/Late", "page_family": "oecd-equity"},
             {"page_id": "dse/Q", "page_family": "oecd-equity"}]
    labels = [{"label": "acctA", "is_human_handle": False}]
    (d / "revisions.jsonl").write_text("\n".join(json.dumps(r) for r in revs))
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (d / "pages.jsonl").write_text("\n".join(json.dumps(p) for p in pages))
    (d / "labels.jsonl").write_text("\n".join(json.dumps(l) for l in labels))
    cutoff = int(pd.Timestamp(T("14:30")).timestamp())
    (d / "manifest.json").write_text(json.dumps({"recreation_source": {"cutoff_unix_ts": cutoff}}))
    return d


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    d = _dump(tmp_path_factory.mktemp("dump"))
    return d, adapters.get("wiki").load(d)


def test_lifecycle_has_recreation_edges(mini):
    _, b = mini
    lc = b.lifecycle
    assert lc["action"].value_counts().to_dict() == {"delete": 4, "recreate": 3}
    r2 = lc[(lc["action"] == "recreate") & (lc["revision_ref"] == "dse~P@2")].sort_values("related_event_id")
    assert list(zip(r2["related_event_id"], r2["round_id"])) == [("delete:1", "dse~P#round-1"), ("delete:4", "dse~P#round-2")]
    assert set(r2["actor"]) == {"acctB"} and all(v is True or v == True for v in r2["page_held"])  # noqa: E712
    rv = lc[lc["event_id"] == "revert:delete:2"].iloc[0]
    assert rv["action"] == "recreate" and pd.isna(rv["revision_ref"]) and rv["actor"] == "acctC"
    d1 = lc[lc["event_id"] == "delete:1"].iloc[0]
    assert d1["round_id"] == "dse~P#round-1" and d1["actor"] == "[Admin1]"
    assert b.notes["n_recreations"] == 3 and b.notes["n_recreations_with_revision"] == 2


def test_recreation_check_against_pipeline_rule(mini):
    _, b = mini
    c = b.notes["recreation_check"]
    assert (c["dump_edges"], c["dump_edges_with_revision"], c["dump_edges_without_revision"]) == (3, 2, 1)
    # delete:1 -> P@2 and delete:4 -> P@2 (both at/after their second), delete:3 -> Late@2 (after the cutoff)
    assert (c["pipeline_edges"], c["overlap"], c["pipeline_only"], c["pipeline_only_after_cutoff"], c["dump_only"]) == (3, 2, 1, 1, 0)
    assert (c["posts_on_recreations"], c["restored_posts"], c["fresh_posts"]) == (2, 1, 1)


def test_recreation_rule_matches_the_segmenter_at_a_tie():
    t = pd.Timestamp(T("12:00"))
    revs = pd.DataFrame({"channel": ["dse/P", "dse/P"], "ts": [t - pd.Timedelta(hours=1), t], "rev_id": ["P@1", "P@2"],
                         "body": ["same text -- A", "same text -- A"]})
    lc = pd.DataFrame([{"channel": "dse/P", "ts": t, "action": "delete", "event_id": "d1"},
                       {"channel": "dse/P", "ts": t, "action": "recreate", "event_id": "r", "related_event_id": "d1", "revision_ref": "P@2"}])
    from swarmprov.segment import snapshots_to_posts
    posts = snapshots_to_posts(revs.assign(author_alt="a"), lc[lc["action"] == "delete"][["channel", "ts"]])
    assert list(posts["parent_id"]) == ["P@1", "P@2"]                  # the segmenter treats the same-second save as a recreation
    c = lifecycle.recreation_check(revs, lc, posts, None)
    assert (c["pipeline_edges"], c["overlap"], c["dump_only"], c["restored_posts"]) == (1, 1, 0, 1)   # and so does the check


def test_probes_table_and_cotiming(mini):
    _, b = mini
    p = b.probes.set_index("probe_id")
    assert len(p) == 5 and set(p["site"]) == {"dse"} and b.capabilities.has_request_log
    a = p.loc["probe:attacklog_raw_dse_2606.jsonl:1"]
    assert a["cotimed_label"] == "acctB" and a["nearest_save_dt_s"] == 0 and a["n_labels_on_prefix"] == 1
    x = p.loc["probe:attacklog_raw_dse_2606.jsonl:2"]
    assert pd.isna(x["cotimed_label"]) and x["n_labels_on_prefix"] == 0 and x["request_action"].startswith("<script")
    multi = p.loc["probe:attacklog_raw_dse_2606.jsonl:3"]            # same second as a save, but the prefix holds 3 labels
    assert pd.isna(multi["cotimed_label"]) and multi["n_labels_on_prefix"] == 3 and multi["nearest_save_dt_s"] == 0
    assert p.loc["probe:attacklog_raw_dse_2606.jsonl:4", "cotimed_label"] == "acctB"      # |dt| == 1 s counts
    assert pd.isna(p.loc["probe:attacklog_raw_dse_2606.jsonl:5", "cotimed_label"])        # |dt| == 2 s does not


def test_probe_prefix_join_survives_numeric_ip16():
    """A numeric ip16 (as a spreadsheet or pandas might deliver it) must still join saves on the same prefix as text."""
    from swarmprov.adapters.wiki import _probes
    t = pd.Timestamp(T("10:00"))
    events = pd.DataFrame([{"event_id": "probe:attacklog_raw_dse_2606.jsonl:9", "event_type": "probe", "time": t.isoformat(),
                            "ip16": 9.9, "request_action": "browse", "param_family": "id", "success_observed": False,
                            "time_grade": "reqlog", "source_refs": ["x:9"]}])
    revs = pd.DataFrame({"ts": [t], "label": ["acctB"], "ip16": ["9.9"]})
    out = _probes(events, revs)
    assert out["ip16"].iloc[0] == "9.9" and out["cotimed_label"].iloc[0] == "acctB" and out["n_labels_on_prefix"].iloc[0] == 1


def test_clock_notes(mini):
    _, b = mini
    n = b.notes
    assert n["clock_grades"] == {"reqlog": 4, "rclog": 1} and n["clock_uncertainty_s"] == [1.0]
    assert b.events["ts_grade"].value_counts().to_dict() == {"reqlog": 5, "rclog": 1}   # per post: P@2 yields two
    assert n["delete_request_lag_s"] == {"0": 3, "60": 1}


def test_full_run_reports_lifecycle_blocks(mini):
    d, _ = mini
    run = pipeline.run_all(d, d / "run")
    text = (run.path / "report.md").read_text()
    for needle in ("Clock quality:", "## Deletions and recreations", "## Probing", "Recreation edges",
                   "co-timed with", "1 of 2 posts on those revisions restore pre-deletion text"):
        assert needle in text, needle
    assert "4 deletions by [Admin1]" in text and "3 first-recreation edges" in text
    assert len(run.read("probes")) == 5
    ev = run.read("events")
    assert f"of {len(ev)} posts" in text                              # the clock note counts the analysed posts
    mn = run.read("mentions")
    assert "visible_until" in mn.columns and mn["visible_until"].notna().any()   # the pipeline-side merge happened
    ex = run.read("exposures_merged")
    e = ex[ex["agent"].str.startswith("Jun02")].iloc[0]                # AgentE (cohort Jun02): Utah 73.74 after AgentA's deleted copy
    assert e["D"] == 1 and e["src_deleted_before_report"] == 1 and e["D_visible"] == 0
    assert "1 of 1 exposed answers were preceded by a public copy that had been deleted" in text
    strict = run.read("exposures_strict")
    assert {"D_visible", "t_public_visible", "t_src_deleted"} <= set(strict.columns)
    lc = run.read("lifecycle")                                      # parquet round trip keeps the extended columns
    assert list(lc.columns)[:9] == ["channel", "ts", "action", "actor", "event_id", "related_event_id",
                                    "revision_ref", "page_held", "round_id"]


def test_exposure_visibility_variant():
    t = lambda s: pd.Timestamp(T(s))
    rounds = pd.DataFrame([{"claim_id": "c", "agent": "B", "family": "f", "episode": 1, "item": "Utah", "value_norm": "73.74",
                            "t_report": t("14:00"), "latency_class": None, "wrong_flag": False, "correct_flag": False}])
    base = {"family": "f", "item": "Utah", "value_norm": "73.74", "value_key": "73.74", "channel": "p"}
    gone = {"event_id": "a", "ts": t("10:00"), "agent_merged": "A", "visible_until": t("12:00"), **base}
    alive = {"event_id": "c", "ts": t("13:30"), "agent_merged": "C", "visible_until": pd.NaT, **base}
    cons = {("f", "Utah"): "73.74"}
    both = exposure.build(rounds, pd.DataFrame([gone, alive]), cons, "agent_merged").iloc[0]
    assert both["D"] == 1 and both["t_public"] == t("10:00") and both["src_deleted_before_report"] == 1
    assert both["D_visible"] == 1 and both["t_public_visible"] == t("13:30")
    only_gone = exposure.build(rounds, pd.DataFrame([gone]), cons, "agent_merged").iloc[0]
    assert only_gone["D"] == 1 and only_gone["D_visible"] == 0 and only_gone["src_deleted_before_report"] == 1
    no_col = exposure.build(rounds, pd.DataFrame([{k: v for k, v in gone.items() if k != "visible_until"}]), cons, "agent_merged").iloc[0]
    assert no_col["D_visible"] == 1 and no_col["src_deleted_before_report"] == 0   # sources without lifecycle: D_visible == D


def test_relay_edges_flag_clock_resolution():
    t0 = pd.Timestamp(T("10:00"))
    m = pd.DataFrame([{"event_id": f"e{i}", "ts": t0 + pd.Timedelta(seconds=s),
                       "agent_merged": a, "channel": "p", "family": "f", "item": "Utah", "value_norm": "1", "value_key": "1"}
                      for i, (a, s) in enumerate([("A", 0), ("B", 1), ("C", 30)])])
    edges, _ = graph.relay_edges(m, "agent_merged")
    e = edges.set_index("dst")
    assert bool(e.loc["B", "within_clock_res"]) and not bool(e.loc["C", "within_clock_res"])


def test_sweeps_cluster_by_gap():
    t0 = pd.Timestamp(T("10:00"))
    d = pd.DataFrame({"channel": list("abcde"), "action": "delete",
                      "ts": [t0, t0 + pd.Timedelta(minutes=5), t0 + pd.Timedelta(minutes=50), t0 + pd.Timedelta(minutes=55), t0 + pd.Timedelta(hours=5)]})
    assert list(lifecycle.sweeps(d)["n"]) == [2, 2, 1]
    assert list(lifecycle.sweeps(d, gap_min=60)["n"]) == [4, 1]


@pytest.mark.skipif(not (RAW / "events.jsonl").exists(), reason="full wiki dump not present")
def test_real_dump_counts():
    b = adapters.get("wiki").load(RAW)
    lc, p, n = b.lifecycle, b.probes, b.notes
    assert lc["action"].value_counts().to_dict() == {"delete": 5217, "recreate": 68}
    rec = lc[lc["action"] == "recreate"]
    assert rec["revision_ref"].map(lambda v: isinstance(v, str) and bool(v)).sum() == 64
    two = rec[rec["revision_ref"] == "dse~OECDEducationEquitySequence@14"]
    assert set(zip(two["related_event_id"], two["round_id"])) == {
        ("delete:dse:rclog:146261", "dse~OECDEducationEquitySequence#round-2"),
        ("delete:dse:rclog:146265", "dse~OECDEducationEquitySequence#round-3")}
    assert lc.loc[lc["action"] == "delete", "page_held"].value_counts().to_dict() == {True: 3969, False: 1248}
    assert len(p) == 101 and p["cotimed_label"].value_counts().to_dict() == {"AgentDataHelperX": 14}
    c = n["recreation_check"]
    assert (c["pipeline_edges"], c["overlap"], c["pipeline_only_after_cutoff"], c["dump_only"]) == (66, 64, 2, 0)
    assert (c["restored_posts"], c["fresh_posts"]) == (30, 77)
    assert n["clock_grades"] == {"reqlog": 14482, "rclog": 103, "write_date": 6}


def test_report_helpers_on_small_frames():
    from swarmprov import report
    t = lambda h: pd.Timestamp(T(h))
    ex = pd.DataFrame([
        {"D": 1, "D_visible": 1, "src_deleted_before_report": 1, "t_report": t("14:00"), "t_public": t("10:00"), "t_src_deleted": t("12:00"), "gap_s": 14400.0},
        {"D": 1, "D_visible": 0, "src_deleted_before_report": 1, "t_report": t("14:00"), "t_public": t("13:00"), "t_src_deleted": t("13:30"), "gap_s": 3600.0},
        {"D": 0, "D_visible": 0, "src_deleted_before_report": 0, "t_report": t("14:00"), "t_public": pd.NaT, "t_src_deleted": None, "gap_s": None},
    ])
    note = report._visibility_note(ex)
    assert "2 of 2 exposed answers were preceded by a public copy" in note
    assert "for 1 another copy was still visible, for 1 nothing visible" in note
    assert "would be 2 (67%) under D_visible instead of 1 (33%)" in note
    assert report._visibility_counts(ex) == {"n_independent_visible": 2, "n_src_deleted_before_report": 2, "n_only_deleted_copies": 1}
    assert report._visibility_note(ex.assign(src_deleted_before_report=0)).endswith("so D_visible equals D.")
    edges = pd.DataFrame([{"kind": "relay", "family": "f", "within_clock_res": True}, {"kind": "relay_xchannel", "family": "f", "within_clock_res": False},
                          {"kind": "relay", "family": "url", "within_clock_res": True}, {"kind": "citation", "family": "f", "within_clock_res": True}])
    assert report._relay_tie_note(edges) == " 1 of 2 answer relay hops (and 1 URL hops) fall within 2 s of their source and carry no reliable direction."
    events = pd.DataFrame({"ts_grade": ["reqlog", "reqlog", "rclog"]})
    cn = report._clock_note({"clock_grades": {"reqlog": 2, "rclog": 1}, "clock_uncertainty_s": [1.0], "delete_request_lag_s": {"0": 1, "1": 1}}, ex, events)
    assert "reqlog for 2, rclog for 1 of 3 posts (stated uncertainty 1 s)" in cn and "1 of 2 one second after the request" in cn
    assert "0 of 2 exposed answers within the 2 s summed uncertainty, 0 within 2 s of the 10 min threshold, 1 within 2 s of the 1 h threshold" in cn
