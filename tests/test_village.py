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
        "goals": "agent_goals.jsonl.gz", "agents": "agents.jsonl.gz", "rooms": "chat_rooms.jsonl.gz"}
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
