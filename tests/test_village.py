"""AI Village side tables (agents, chat_rooms, agent_goals): detection, assembly, and
audience-aware exposure when a message table sits next to them."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from swarmprov import adapters, exposure, pipeline, village
from swarmprov.adapters.village import discover

RAW = Path(__file__).resolve().parents[1] / "data" / "raw_village"
BEST = "d45ec7c6-6adb-49cb-8c40-dc5d18c37d84"       # room "best": whitelisted agents only
GENERAL = "18a3b2fb-9d2e-4ce7-b9b1-52e09c5408a8"
GPT55 = "6365764a-b6e2-4dfa-94cd-2d1aef5b54f7"      # game-dev, in "best"
OPUS47 = "78f39924-1ced-4be5-94a6-e7bbf0c90d66"     # game-dev, not in "best"
SONNET45 = "169ea37e-c664-4012-acba-cb583aaab1f3"   # twitterati
GEMINI31 = "f69b132c-d4bd-49d5-b2a5-cef3f60f2246"   # twitterati


def test_tables_are_recognised_by_columns():
    found = discover(RAW)
    assert {k: v.name for k, v in found.items()} == {
        "goals": "agent_goals.jsonl.gz", "agents": "agents.jsonl.gz", "rooms": "chat_rooms.jsonl.gz",
        "sessions": "claude_code_sessions.jsonl.gz", "eras": "village_goals.jsonl.gz"}
    assert adapters.detect(RAW / "village_goals.jsonl.gz").name == "village"
    assert adapters.detect(RAW).name == "village"
    assert adapters.detect(RAW / "agents.jsonl.gz").name == "village"
    assert adapters.detect(RAW / "chat_rooms.jsonl.gz").name == "village"
    assert adapters.detect(RAW / "agent_goals.jsonl.gz").name == "roster"    # a lone roster stays a roster


@pytest.fixture(scope="module")
def bundle():
    return adapters.get("village").load(RAW)


def test_directory_roster_and_channels(bundle):
    d, r, c = bundle.directory, bundle.roster, bundle.channels
    assert len(d) == 46 and d["participating"].sum() == 32
    assert d.set_index("agent_id").loc[SONNET45, "name"] == "Claude Sonnet 4.5"
    assert d["vendor"].value_counts()["Anthropic"] == 16
    assert len(r) == 33 and r["agent_name"].notna().all()                   # every goal's agent is in the directory
    assert r.set_index("agent_id").loc[OPUS47, "agent_name"] == "Claude Opus 4.7"
    assert len(c) == 16 and c["deleted"].notna().sum() == 12 and len(village.restricted(c)) == 11
    best = c.set_index("channel").loc["best"]
    assert "Kimi K2.6" in best["allow"] and "GPT-5.5" in best["allow"] and "Claude Opus 4.7" not in best["allow"]
    assert set(c.set_index("channel").loc["rest", "deny"]) == set(best["allow"])
    assert len(bundle.lifecycle) == 16 + 12 and bundle.capabilities.has_lifecycle
    assert len(bundle.events) == 0


def test_room_lists_name_known_agents(bundle):
    names = set(bundle.directory["name"])
    listed = {n for lst in list(bundle.channels["allow"]) + list(bundle.channels["deny"]) for n in lst}
    assert listed <= names


def _post(i, agent_id, room, t, text):
    return {"id": f"m{i}", "agent_id": agent_id, "chat_room_id": room, "created_at": t.isoformat(), "content": text}


@pytest.fixture(scope="module")
def village_run(tmp_path_factory):
    """Real side tables + a synthetic message table shaped like chat_messages
    (agent_id and chat_room_id, no names): answers for a known round."""
    d = tmp_path_factory.mktemp("village")
    for f in RAW.glob("*.jsonl.gz"):
        shutil.copy(f, d / f.name)
    t0 = pd.Timestamp("2026-07-07T10:00:00Z")
    m = pd.Timedelta(minutes=1)
    msgs = [
        # game-dev: GPT-5.5 answers first, but in "best", which Claude Opus 4.7 cannot read
        _post(0, GPT55, BEST, t0, "R1 CONFIRMED: Utah arrived 09:58:00, 14s timer; answered 73.74 at +30s."),
        _post(1, OPUS47, GENERAL, t0 + 30 * m, "R1 CONFIRMED: Utah arrived 10:28:00, 14s timer; answered 73.74 at +2s."),
        # twitterati: same pattern in the public room -> exposed
        _post(2, SONNET45, GENERAL, t0, "R1 CONFIRMED: Utah arrived 09:58:00, 14s timer; answered 73.74 at +30s."),
        _post(3, GEMINI31, GENERAL, t0 + 30 * m, "R1 CONFIRMED: Utah arrived 10:28:00, 14s timer; answered 73.74 at +2s."),
    ]
    # second round so items/rounds are learned from more than one post per agent
    for k, (aid, room) in enumerate([(GPT55, BEST), (OPUS47, GENERAL), (SONNET45, GENERAL), (GEMINI31, GENERAL)]):
        msgs.append(_post(10 + k, aid, room, t0 + (60 + 5 * k) * m,
                          "R2 CONFIRMED: Idaho arrived 11:00:00, 14s timer; answered 41.2 at +35s."))
    (d / "chat_messages.jsonl").write_text("\n".join(json.dumps(x) for x in msgs))
    return pipeline.run_all(d, d / "run")


def test_messages_get_names_rooms_and_families(village_run):
    ev = village_run.read("events")
    assert set(ev["author_raw"]) == {"GPT-5.5", "Claude Opus 4.7", "Claude Sonnet 4.5", "Gemini 3.1 Pro"}
    assert set(ev["channel"]) == {"best", "general"}
    fam = ev.set_index("author_raw")["family"]
    assert set(fam["GPT-5.5"]) == {"game-dev"} and set(fam["Gemini 3.1 Pro"]) == {"twitterati"}
    agents = village_run.read("agents").set_index("author_raw")
    assert agents.loc["Claude Opus 4.7", "agent_merged"] == f"Jul03|game-dev"
    assert agents.loc["GPT-5.5", "agent_merged"] == "Jul03b|game-dev"
    assert agents.loc["Claude Sonnet 4.5", "agent_merged"] == f"Jul03b|twitterati|{SONNET45[:8]}"


def test_exposure_respects_room_audience(village_run):
    ex = village_run.read("exposures_merged").set_index(["agent", "item"])
    # public room: Gemini's Utah answer was already posted by Sonnet -> exposed
    assert ex.loc[(f"Jul03b|twitterati|{GEMINI31[:8]}", "Utah"), "D"] == 1
    # restricted room: GPT-5.5's answer sat in "best", unreadable to Claude Opus 4.7 -> independent
    assert ex.loc[("Jul03|game-dev", "Utah"), "D"] == 0
    assert ex.loc[("Jul03b|game-dev", "Utah"), "D"] == 0                   # first poster, nothing before it


def test_report_has_village_block(village_run):
    text = (village_run.path / "report.md").read_text()
    assert "## Village" in text and "## Roster" in text
    assert "2 transcript posts are in restricted rooms" in text
    assert "only Kimi K2.6, GPT-5.5" in text


def test_audience_unit():
    rounds = pd.DataFrame([{"claim_id": "c", "agent": "B", "family": "f", "episode": 1, "item": "Utah",
                            "value_norm": "73.74", "t_report": pd.Timestamp("2026-07-07T11:00Z"),
                            "latency_class": None, "wrong_flag": False, "correct_flag": False}])
    mentions = pd.DataFrame([{"event_id": "e", "ts": pd.Timestamp("2026-07-07T10:00Z"), "agent_merged": "A",
                              "channel": "best", "family": "f", "item": "Utah", "value_norm": "73.74", "value_key": "73.74"}])
    open_ = exposure.build(rounds, mentions, {("f", "Utah"): "73.74"}, "agent_merged")
    closed = exposure.build(rounds, mentions, {("f", "Utah"): "73.74"}, "agent_merged", audience={"best": {"A"}})
    assert open_["D"].iloc[0] == 1 and open_["D_cons"].iloc[0] == 1
    assert closed["D"].iloc[0] == 0 and closed["D_cons"].iloc[0] == 0


# --- presence log (claude_code_sessions) -------------------------------------------------

def test_sessions_table_is_an_activity_log(bundle):
    act = bundle.activity
    assert len(act) == 303 and act["agent_id"].nunique() == 1
    assert set(act["agent_name"]) == {"Opus 4.5 (Claude Code)"} and set(act["kind"]) == {"session"}
    assert act["ref"].nunique() == 42                                        # sdk session ids, one resumed 260 times
    assert bundle.capabilities.has_activity
    assert adapters.detect(RAW / "claude_code_sessions.jsonl.gz").name == "village"


def test_activity_overlap_is_reported_honestly(bundle):
    s = village.activity_summary(bundle.activity, bundle.roster, None)
    assert s["agents_in_roster"] == 0 and s["rows_in_goal_window"] == 0
    assert s["span"] == "2026-01-26 → 2026-03-31" and s["weekend_share"] == 0
    text = "\n".join(village.section(village.summary(bundle.directory, bundle.channels, None, bundle.activity, bundle.roster),
                                     lambda df, **k: ""))
    assert "covers none of the agents under study" in text


def test_unrecognised_files_are_listed(tmp_path):
    for f in RAW.glob("*.jsonl.gz"):
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "mystery.jsonl").write_text(json.dumps({"foo": 1, "bar": "x"}) + "\n")
    run = pipeline.run_all(tmp_path, tmp_path / "run")
    prof = json.loads((run.path / "profile.json").read_text())
    assert prof["notes"]["unrecognised"] == ["mystery.jsonl"]
    assert "`mystery.jsonl`" in (run.path / "report.md").read_text()
    assert len(run.read("activity")) == 303


# --- shared goals (village_goals) ------------------------------------------------------

def test_shared_goals_become_eras(bundle):
    e = bundle.eras
    assert len(e) == 51 and e["start"].notna().all() and e["end"].isna().sum() == 1       # no NaT from mixed formats
    e = e.sort_values("start").reset_index(drop=True)
    assert ((e["start"].shift(-1) - e["end"]).dt.total_seconds().abs().dropna() == 0).all()  # contiguous windows
    assert e["label"].is_unique and e["label"].iloc[0].startswith("e01-")
    assert e.iloc[-1]["goal"].startswith("Each agent: Maximize your assigned goal")
    assert e.iloc[-1]["start"] == bundle.roster["start"].min()                            # handover to per-agent goals


def test_era_label_is_short_and_stable():
    assert village.era_label(17, "Form two teams and debate each other, while one agent judges. Choose your teammates wisely!") == "e17-form-two-teams"
    assert village.era_label(3, "Holiday: do whatever you like! Next goal will begin soon") == "e03-holiday-goal-begin"
    assert village.era_label(5, "???") == "e05-goal"


def test_annotate_eras_picks_the_window(bundle):
    e = bundle.eras
    ev = pd.DataFrame({"ts": pd.to_datetime(["2026-06-16T12:00:00Z", "2025-01-01T00:00:00Z", "2026-09-01T00:00:00Z"], utc=True)})
    a = village.annotate_eras(ev, e)
    assert a["era"].iloc[0] == e.set_index("goal").loc["Reduce global suffering as much as you can!", "label"]
    assert a["era"].iloc[1] is None                                                       # before the first window
    assert a["era"].iloc[2] == e.sort_values("start")["label"].iloc[-1]                   # the open window runs on


def test_family_precedence_agent_goal_then_era(tmp_path):
    """A post inside the author's own goal window takes the role; a post by the same author
    before that window, or by an unknown author, takes the shared goal of its time."""
    for f in RAW.glob("*.jsonl.gz"):
        shutil.copy(f, tmp_path / f.name)
    msgs = [
        _post(0, SONNET45, GENERAL, pd.Timestamp("2026-07-07T10:00:00Z"), "R1 CONFIRMED: Utah arrived 09:58:00; answered 73.74"),
        _post(1, SONNET45, GENERAL, pd.Timestamp("2026-06-16T12:00:00Z"), "Working on suffering reduction, 12 ideas"),
        _post(2, "", GENERAL, pd.Timestamp("2026-06-16T13:00:00Z"), "Observer note 1"),
    ]
    (tmp_path / "chat_messages.jsonl").write_text("\n".join(json.dumps(m) for m in msgs))
    run = pipeline.run_all(tmp_path, tmp_path / "run")
    ev = run.read("events").set_index("event_id")
    eras = run.read("eras").set_index("goal")
    suffering = eras.loc["Reduce global suffering as much as you can!", "label"]
    assert ev.loc["m0", "family"] == "twitterati"
    assert ev.loc["m1", "family"] == suffering                       # not the nearest agent goal: it was not in force yet
    assert ev.loc["m2", "family"] == suffering
    re_ = run.read("roster_events")
    assert {"roster_agent", "role", "in_window", "era_id", "era"} <= set(re_.columns)
    text = (run.path / "report.md").read_text()
    assert "Shared goals: 51 windows" in text and "switched from shared to individual goals on 2026-07-06 15:59" in text
