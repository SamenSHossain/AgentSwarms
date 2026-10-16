"""Transcripts without a usable wall clock: epoch strings, framework message lists with no timestamps
(a sequence clock), partly timed files and concatenated rooms (no order is invented)."""

import json

import pandas as pd
import pytest

from swarmprov import pipeline
from swarmprov.adapters.chat import SEQUENCE_EPOCH, ChatAdapter, _to_ts
from swarmprov.schema import RunDir

ROUNDS = [("Poland", "0.2"), ("Hungary", "-0.9"), ("Chile", "0.2")]
AGENTS = ["researcher-1", "researcher-2", "researcher-3"]


def _msgs(author_key: str = "name") -> list[dict]:
    return [{author_key: a, "content": f"R{r} CONFIRMED: {item} arrived; answered {val} at +{3 + k}s."}
            for r, (item, val) in enumerate(ROUNDS, start=1) for k, a in enumerate(AGENTS)]


def _timed(i: int) -> dict:
    return {"timestamp": f"2026-05-04T10:{i:02d}:00Z"} if i % 3 else {}   # 6 of 9 timed


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("formats")
    files = {
        "openai": [{"role": "system", "content": "You are a research swarm."}]
                  + [{"role": "assistant", **m} for m in _msgs()],
        "rooms": {"room-a": [m | _timed(i) for i, m in enumerate(_msgs()[:5])],
                  "room-b": [m | _timed(i + 5) for i, m in enumerate(_msgs()[5:])]},
        "rooms_untimed": {"room-a": _msgs()[:5], "room-b": _msgs()[5:]},
        # 4 agents post 16.40 first, then 11 post 16.38; agents 3, 6 and 9 use a fetch proxy; then all answer R2
        "dispute": [{"name": f"agent-{i}", "content": f"R1 CONFIRMED: Poland arrived; answered {'16.40' if i < 4 else '16.38'} "
                                                       f"at +3s." + (" via r.jina.ai" if i in (3, 6, 9) else "")}
                    for i in range(15)]
                   + [{"name": f"agent-{i}", "content": "R2 CONFIRMED: Hungary arrived; answered 9.90 instantly."}
                      for i in range(15)],
    }
    (d / "tech.json").write_text(json.dumps({"techniques": [{"name": "jina-proxy", "pattern": r"r\.jina\.ai"}]}))
    out = {}
    for name, doc in files.items():
        (d / f"{name}.json").write_text(json.dumps(doc))
        out[name] = pipeline.run_all(d / f"{name}.json", d / name, adapter="chat",
                                     config=str(d / "tech.json") if name == "dispute" else None)
    return out


def _report(run) -> str:
    return (run.path / "report.md").read_text()


def _profile(run) -> dict:
    return json.loads((run.path / "profile.json").read_text())


def test_epoch_strings_parse_by_magnitude_and_short_digit_runs_do_not():
    assert _to_ts("1777885200.000100") == pd.Timestamp("2026-05-04 09:00:00.000100", tz="UTC")
    assert _to_ts("1777885200000") == pd.Timestamp("2026-05-04 09:00:00", tz="UTC")       # milliseconds
    assert pd.isna(_to_ts("12345678")) and pd.isna(_to_ts("12"))                          # ids, not times
    assert _to_ts("2026-05-04T10:00:00Z") == pd.Timestamp("2026-05-04 10:00", tz="UTC")


def test_slack_epoch_strings_give_a_wall_clock_and_ids_stay_ids(tmp_path):
    p = tmp_path / "2026-05-04.json"
    p.write_text(json.dumps([{"type": "message", "user": a, "text": f"note {i}", "ts": f"{1777885200 + 60 * i}.000100",
                              "client_msg_id": f"{1777885200 + i}"} for i, a in enumerate(AGENTS * 2)]))
    b = ChatAdapter(mapping={"id": "client_msg_id"}).load(p)
    assert b.events["ts"].notna().all() and b.capabilities.has_wall_clock is True
    assert b.events["ts"].iloc[1] - b.events["ts"].iloc[0] == pd.Timedelta(minutes=1)
    assert b.events["event_id"].iloc[0] == "1777885200" and b.notes["clock"] == "wall"


def test_untimed_message_list_runs_on_a_sequence_clock(runs):
    run = runs["openai"]
    prof = _profile(run)
    assert prof["clock"] == "sequence" and prof["first_ts"] is None and prof["events_per_day"] == {}
    caps = prof["capabilities"]
    assert all(isinstance(v, bool) for v in caps.values())
    assert caps["has_wall_clock"] is False and caps["has_order"] is True
    ev = run.read("events")
    pos = ev["source_ref"].str.rsplit(":", n=1).str[1].astype(int)
    assert (ev["ts"] == SEQUENCE_EPOCH + pd.to_timedelta(pos, unit="s")).all()
    text = _report(run)
    assert "ordered by message position (no wall clock)" in text and "| has_wall_clock | yes" not in text
    assert "## A1. Provenance split" in text and len(run.read("exposures_merged")) == 9


def test_autogen_style_result_is_one_message_list(tmp_path):
    """A wrapper object around one list (AutoGen TaskResult, LangGraph state) is ordered by position."""
    p = tmp_path / "task_result.json"
    p.write_text(json.dumps({"messages": [m | {"type": "TextMessage"} for m in _msgs()], "stop_reason": "max turns"}))
    b = ChatAdapter().load(p)
    assert b.notes["clock"] == "sequence" and b.notes["n_no_author"] == 0 and b.capabilities.has_order
    assert list(b.events["ts"]) == list(SEQUENCE_EPOCH + pd.to_timedelta(range(9), unit="s"))


def test_sequence_clock_suppresses_every_hour_based_number(runs):
    run = runs["openai"]
    text = _report(run)
    for gone in ("at least an hour before", "Median head start", "exposed_60m", "exposed ≥1 h", "Majority from",
                 "| hours to overtake", "winner first seen", "carry no reliable direction", "UTC"):
        assert gone not in text, gone
    assert text.count("need a wall clock and are not shown") == 2          # A1 shares, A4 bins
    assert "**Message order is not time.**" in text
    a1 = json.loads((run.path / "summary.json").read_text())["A1"]
    assert a1["share_exposed_60m"] is None and a1["median_gap_h"] is None and a1["share_exposed"] == pytest.approx(6 / 9)
    assert run.read("edges_merged").query("kind == 'relay'")["within_clock_res"].isna().all()


def test_sequence_clock_keeps_a4_status_and_a3_counts_without_hours(runs):
    run = runs["dispute"]
    text, summary = _report(run), json.loads((run.path / "summary.json").read_text())
    a4 = summary["A4"]
    assert a4["rule"]["bin_hours"] is None
    row = a4["featured"][0]
    assert row["slot"] == "all | Poland" and row["first variant"] == "16.4" and row["winner"] == "16.38"
    assert row["status"] == "overturned"                    # 11 of 15 carriers: the majority of all of them
    assert not {"winner first seen", "winner majority from", "hours to overtake"} & set(row)
    assert "a later variant won and holds the majority of all the slot's carriers" in text
    a3 = summary["A3"][0]
    assert a3["at_risk"] == 14 and a3["adopters"] == 2      # every other agent posts again after agent-3's first use
    assert not {"first_seen", "median_hours_to_adopt", "median_hours_among_adopters"} & set(a3)
    assert "(every post is in the catch-all family `all`, so for a technique without families of its own that is every " \
           "agent active after its first post" in text
    assert "Times to adoption (hours, the Kaplan–Meier median) and the first-seen time need a wall clock" in text
    assert "a3_diffusion.png" not in text and "a4_errors.png" not in text
    assert {(r["outcome"], r["treatment"]) for r in summary["A2"]} == {("y_consensus", "D_cons")}
    assert "The latency outcome (an instant answer) and the ≥1 h treatment need a wall clock" in text


def test_partly_timed_list_discards_its_timestamps_instead_of_mixing(tmp_path):
    from swarmprov import report
    (tmp_path / "partial.jsonl").write_text("\n".join(json.dumps(m | _timed(i)) for i, m in enumerate(_msgs())))
    run = RunDir(tmp_path / "run")
    prof = pipeline.ingest(tmp_path / "partial.jsonl", run, adapter="chat")
    assert prof["clock"] == "sequence" and prof["notes"]["timestamps_discarded"] == 6
    assert (run.read("events")["ts"] < pd.Timestamp("2001-01-01", tz="UTC")).all()   # no real time survives
    assert "the 6 of 9 messages that carried a timestamp were too few to order by" in report._when(prof, prof["notes"])


def test_concatenated_rooms_keep_missing_timestamps_and_invent_no_order(runs):
    run = runs["rooms"]
    prof = _profile(run)
    assert prof["clock"] == "none" and prof["n_untimed"] == 3
    assert prof["capabilities"]["has_order"] is False and prof["capabilities"]["has_wall_clock"] is False
    ex = run.read("exposures_merged")
    assert len(ex) == 6 and ex["t_report"].notna().all()                  # the 3 untimed rounds are dropped
    text = _report(run)
    assert "3 posts have no timestamp and take part in no time comparison" in text
    assert "3 agent-rounds have no timestamp and are left out" in text
    assert json.loads((run.path / "summary.json").read_text())["funnel"]["agent_rounds_untimed"] == 3


def test_untimed_rooms_skip_the_time_based_analyses(runs):
    run = runs["rooms_untimed"]
    assert len(run.read("exposures_merged")) == 0 and len(run.read("edges_merged")) == 0
    text = _report(run)
    assert "no timestamps." in text
    assert "_Skipped: needs agent-rounds (an agent's answer to an item in a task family); none of the 9 agent-rounds " \
           "has a timestamp, so nothing can be ordered against them._" in text
    assert "## A5. Structure\n\n_Skipped: needs provenance edges" in text


def test_missing_author_field_is_flagged(runs, tmp_path):
    p = tmp_path / "autogen.json"
    p.write_text(json.dumps({"messages": [m | {"type": "TextMessage"} for m in _msgs("source")]}))
    assert ChatAdapter().load(p).notes["n_no_author"] == 9          # AutoGen's "source" is not an author alias
    text = _report(runs["openai"])
    assert "**Warning:** 1 post names no author" in text and "`unknown`" in text   # the system message
    assert "no author" not in _report(runs["rooms"])


def test_order_alone_recovers_the_wall_clock_exposure_share(synth_run, tmp_path):
    """The synthetic swarm with its timestamps stripped: message order gives the same exposure as wall time."""
    from swarmprov.analysis import causal
    run, _ = synth_run
    src = run.read("events").sort_values("ts", kind="stable")
    (tmp_path / "untimed.jsonl").write_text("\n".join(json.dumps({"author": a, "channel": c, "text": t})
                                                      for a, c, t in zip(src["author_raw"], src["channel"], src["text"])))
    seq = RunDir(tmp_path / "run")
    assert pipeline.ingest(tmp_path / "untimed.jsonl", seq, "chat", str(run.path.parent / "config.json"))["clock"] == "sequence"
    pipeline.extract(seq)
    pipeline.expose(seq)
    wall, order = run.read("exposures_merged"), seq.read("exposures_merged")
    assert len(order) == len(wall) == 300
    assert abs(order["D"].mean() - wall["D"].mean()) < 0.01
    est = causal.estimate(order, wall_clock=False)
    assert set(zip(est["outcome"], est["treatment"])) == {("y_consensus", "D_cons")}   # no latency, no >= 1 h

