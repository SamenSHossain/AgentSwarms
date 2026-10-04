"""Roster (agent_goals) support: adapter detection, eras/batches, and the roster as
families + cohorts when attached to a transcript."""

import json
from pathlib import Path

import pandas as pd
import pytest

from swarmprov import adapters, pipeline, roster
from swarmprov.schema import RunDir

GOALS = Path(__file__).resolve().parents[1] / "data" / "village" / "agent_goals.jsonl"
TWITTERATI = "169ea37e-c664-4012-acba-cb583aaab1f3"      # Jul03b batch, open goal from 2026-07-06 15:59
TWITTERATI2 = "f69b132c-d4bd-49d5-b2a5-cef3f60f2246"
RESEARCHER = "ea91b7cb-bb15-4194-bfbf-d70e088f84c5"      # two goals: reworded at 2026-09-01 23:42


@pytest.fixture(scope="module")
def ros():
    return adapters.get("roster").load(GOALS).roster


def test_detects_agent_goals_as_roster():
    assert adapters.detect(GOALS).name == "roster"


def test_normalize_counts(ros):
    assert len(ros) == 33 and ros["agent_id"].nunique() == 32 and ros["role"].nunique() == 25
    assert ros["end"].isna().sum() == 32
    assert set(ros.loc[ros["agent_id"] == TWITTERATI, "batch"]) == {"Jul03b"}
    assert ros.loc[ros["created"].idxmin(), "batch"] == "Jul03"           # the lone 13:56 game-dev goal


def test_eras_marks_rewording(ros):
    er = roster.eras(ros)
    r = er[er["agent_id"] == RESEARCHER].sort_values("start")
    assert list(r["change"]) == ["first", "reworded"]
    assert (er["change"] == "reassigned").sum() == 0


def test_summary_shared_roles(ros):
    s = roster.summary(ros)
    assert "twitterati" in s["shared_roles"] and "altruist" not in s["shared_roles"]
    assert s["changes"] == {"first": 32, "reworded": 1}


def _events(rows):
    ev = pd.DataFrame(rows)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    ev["author_alt"] = ev.get("author_alt", "")
    return ev


def test_annotate_by_id_name_alias_and_window(ros):
    ev = _events([
        {"author_raw": "Claude Opus 4.5", "author_alt": TWITTERATI, "ts": "2026-07-07T10:00:00Z"},   # id in window
        {"author_raw": "Claude Opus 4.5", "author_alt": TWITTERATI, "ts": "2026-07-01T10:00:00Z"},   # before the goal
        {"author_raw": "Altruist", "author_alt": "", "ts": "2026-07-07T10:00:00Z"},                    # by role name
        {"author_raw": "Twitterati", "author_alt": "", "ts": "2026-07-07T10:00:00Z"},                  # ambiguous role
        {"author_raw": "Gemini", "author_alt": "", "ts": "2026-09-01T22:00:00Z"},                      # alias, first goal
        {"author_raw": "Gemini", "author_alt": "", "ts": "2026-09-02T00:00:00Z"},                      # alias, reworded goal
        {"author_raw": "Someone", "author_alt": "", "ts": "2026-07-07T10:00:00Z"},                     # unmatched
    ])
    ann = roster.annotate(ev, ros, {"gemini": RESEARCHER})
    assert list(ann["role"]) == ["twitterati", "twitterati", "altruist", None, "ai-safety-researcher",
                                 "ai-safety-researcher", None]
    assert list(ann["in_window"]) == [True, False, True, None, True, True, None]
    assert ann["goal_id"][4] != ann["goal_id"][5]                      # the window picks the goal


@pytest.fixture(scope="module")
def village_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("village")
    t0 = pd.Timestamp("2026-07-07T10:00:00Z")
    msgs = []
    for i in range(30):
        who, aid = ("Claude Opus 4.5", TWITTERATI) if i % 2 else ("GPT-5", TWITTERATI2)
        msgs.append({"id": f"m{i}", "created_at": (t0 + pd.Timedelta(minutes=i)).isoformat(),
                     "agent_name": who, "agent_id": aid, "content": f"R{1 + i % 3} posted tweet {i}, 120 followers"})
    msgs.append({"id": "x", "created_at": t0.isoformat(), "agent_name": "Observer", "agent_id": "",
                 "content": "watching the village"})
    p = d / "chat_messages.jsonl"
    p.write_text("\n".join(json.dumps(m) for m in msgs))
    run = pipeline.run_all(p, d / "run", adapter="chat", roster=str(GOALS))
    return run


def test_roster_sets_family_and_cohort(village_run):
    ev, agents = village_run.read("events"), village_run.read("agents")
    assert set(ev.loc[ev["author_raw"] != "Observer", "family"]) == {"twitterati"}
    assert ev.loc[ev["author_raw"] == "Observer", "family"].iloc[0] == ""
    a = agents.set_index("author_raw")
    # two agents were given the Twitterati goal in the same batch: the roster keeps them apart
    assert a.loc["Claude Opus 4.5", "agent_merged"] == f"Jul03b|twitterati|{TWITTERATI[:8]}"
    assert a.loc["GPT-5", "agent_merged"] == f"Jul03b|twitterati|{TWITTERATI2[:8]}"
    assert a.loc["GPT-5", "cohort"] == "Jul03b" and a.loc["GPT-5", "roster_agent"] == TWITTERATI2
    assert a.loc["Observer", "agent_merged"] == "observer"                # unmatched: strict handle
    assert pd.isna(a.loc["Observer", "roster_agent"])


def test_merged_ids_unique_per_roster_agent(ros):
    labels = roster.merged_id(ros)
    assert len(set(labels.values())) == len(labels) == 32
    assert labels[RESEARCHER] == "Sep01|ai-safety-researcher"            # alone in its role: plain label


def test_roster_report_coverage(village_run):
    text = (village_run.path / "report.md").read_text()
    assert "## Roster" in text and "30/31 posts by 2/3 authors matched" in text
    assert "`Observer` (1)" in text
    prof = json.loads((village_run.path / "profile.json").read_text())
    assert prof["roster"].endswith("agent_goals.jsonl")
    ann = village_run.read("roster_events")
    assert ann["in_window"].dropna().all()


def test_roster_only_run_writes_roster_report(tmp_path):
    run = pipeline.run_all(GOALS, tmp_path / "run")
    text = (run.path / "report.md").read_text()
    assert text.startswith("# Roster report") and "33 goal assignments to 32 agents" in text
    assert not run.has("claims")
    pipeline.extract(run)                                              # re-running a stage is a no-op, not a crash


def test_roster_from_csv_and_rejects_text_tables(tmp_path):
    csv = tmp_path / "goals.csv"
    pd.read_json(GOALS, lines=True).to_csv(csv, index=False)
    assert adapters.detect(csv).name == "roster"
    ros = adapters.get("roster").load(csv).roster
    assert len(ros) == 33 and (ros["detail"] == "").sum() == 28       # CSV NaN is "no detail", not "nan"
    chat = tmp_path / "chat.jsonl"
    chat.write_text(json.dumps({"agent_id": "a", "short_name": "x", "start_time": "2026-01-01", "content": "hi"}))
    assert adapters.detect(chat).name == "chat"


SHEET = GOALS.parent / "agent_goals_sheet.tsv"   # the same table after a spreadsheet round-trip


def test_sheet_export_is_detected_and_mangled_times_are_flagged():
    assert adapters.detect(SHEET).name == "roster"
    b = adapters.get("roster").load(SHEET)
    ros, issues = b.roster, b.notes["timestamp_issues"]
    assert len(ros) == 10 and ros["agent_id"].nunique() == 10
    assert ros["start"].notna().sum() == 9                      # "26:37.1" is not a date
    assert ros["created"].isna().all()                          # "03:57.5" is not 03:57 today either
    assert issues["created"]["n_unparsed"] == 10 and issues["created"]["examples"] == ["03:57.5", "07:59.8", "26:37.3"]
    assert issues["start"] == {"n_values": 10, "n_unparsed": 1, "examples": ["26:37.1"]}
    assert set(ros["batch"]) == {"Jul06", "Jul09", "Jul10", ""}  # batches fall back to start times


def test_sheet_report_warns(tmp_path):
    run = pipeline.run_all(SHEET, tmp_path / "run")
    text = (run.path / "report.md").read_text()
    assert "Timestamp problems in the roster export" in text and "`26:37.1`" in text
    assert "Oct" not in text                                    # no phantom batch dated today


@pytest.mark.parametrize("raw,ok", [("2026-07-03 14:37:49.366903", True), ("7/6/26 15:59", True),
                                    ("2026-07-06T15:59:00Z", True), ("37:49.4", False), ("15:59", False),
                                    (1783000000, True), (None, False)])
def test_ts_needs_a_date(raw, ok):
    assert pd.notna(roster._ts(raw)) is ok
