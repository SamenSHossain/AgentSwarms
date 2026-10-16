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


def _dispute() -> list[dict]:
    """4 agents post 16.40 first, then 11 post 16.38; agents 3, 6 and 9 use a fetch proxy; then all answer R2."""
    return ([{"name": f"agent-{i}", "content": f"R1 CONFIRMED: Poland arrived; answered {'16.40' if i < 4 else '16.38'} "
                                                f"at +3s." + (" via r.jina.ai" if i in (3, 6, 9) else "")} for i in range(15)]
            + [{"name": f"agent-{i}", "content": "R2 CONFIRMED: Hungary arrived; answered 9.90 instantly."} for i in range(15)])


def _timed(i: int) -> dict:
    return {"timestamp": f"2026-05-04T10:{i:02d}:00Z"} if i % 3 else {}   # 6 of 9 timed


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("formats")
    dispute = _dispute()
    files = {
        "openai": [{"role": "system", "content": "You are a research swarm."}]
                  + [{"role": "assistant", **m} for m in _msgs()],
        "rooms": {"room-a": [m | _timed(i) for i, m in enumerate(_msgs()[:5])],
                  "room-b": [m | _timed(i + 5) for i, m in enumerate(_msgs()[5:])]},
        "rooms_untimed": {"room-a": _msgs()[:5], "room-b": _msgs()[5:]},
        "dispute": dispute,
        "rooms_dispute": {"room-a": dispute[:15], "room-b": dispute[15:]},   # the same dispute, untimed, in two rooms
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
    # mis-scaled or placeholder values are no time: 11-digit seconds (5138), 13-digit ms past 2100 (2286), 1973, 1970
    for v in ("99999999999", "9999999999999", "100000000", 99999999999, 9999999999999, 1e20, "1970-01-01T00:00:00Z"):
        assert pd.isna(_to_ts(v)), v
    assert _to_ts("631152000") == pd.Timestamp("1990-01-01", tz="UTC")                     # the window's ends
    assert _to_ts("4102444799999") == pd.Timestamp("2099-12-31 23:59:59.999", tz="UTC")


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


def test_a_single_room_is_one_message_list(tmp_path):
    """{channel: [...]} with one channel has a file order; two channels do not."""
    for doc, clock in (({"general": _msgs()}, "sequence"), ({"general": _msgs(), "meta": {"exported": "x"}}, "sequence"),
                       ({"a": _msgs()[:4], "b": _msgs()[4:]}, "none")):
        p = tmp_path / "rooms.json"
        p.write_text(json.dumps(doc))
        b = ChatAdapter().load(p)
        assert b.notes["clock"] == clock and b.capabilities.has_order is (clock == "sequence"), doc.keys()
        assert b.events["ts"].notna().all() if clock == "sequence" else b.events["ts"].isna().all()


def test_sequence_clock_suppresses_every_hour_based_number(runs):
    run = runs["openai"]
    text = _report(run)
    for gone in ("at least an hour before", "Median head start", "exposed_60m", "exposed ≥1 h", "Majority from",
                 "| hours to overtake", "winner first seen", "carry no reliable direction", "UTC",
                 "Exposure vs required head start", "a1_threshold.png", "Sensitivity to the relay window", "within 1 h"):
        assert gone not in text, gone
    assert text.count("need a wall clock and are not shown") == 3          # A1 shares, A4 bins, A6 relay window
    assert "Relay-window sweeps (a window W in minutes and hours) need a wall clock" in text
    assert "**Message order is not time.**" in text
    a1 = json.loads((run.path / "summary.json").read_text())["A1"]
    assert a1["share_exposed_60m"] is None and a1["median_gap_h"] is None and a1["share_exposed"] == pytest.approx(6 / 9)
    assert a1["share_exposed_60m_ci"] is None and a1["median_gap_h_ci"] is None and "threshold_curve" not in a1
    assert "window_sensitivity" not in json.loads((run.path / "summary.json").read_text()).get("A6", {})
    assert run.read("edges_merged").query("kind == 'relay'")["within_clock_res"].isna().all()
    ex = run.read("exposures_merged")                       # positions are not seconds: no gap, no threshold flag
    assert ex[["gap_s", "D_w10m", "D_w60m", "D_cons_w60m"]].isna().all().all() and ex["D"].notna().all()


def test_sequence_clock_keeps_a4_status_and_a3_counts_without_hours(runs):
    run = runs["dispute"]
    text, summary = _report(run), json.loads((run.path / "summary.json").read_text())
    a4 = summary["A4"]
    assert a4["rule"]["bin_hours"] is None
    row = a4["rows"][0]
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
    assert "The contagion test compares the pages agents adopted on, and every post here is on one page or channel" in text
    assert "contagion" not in a3 and run.read("adoptions_merged")["in_at_risk"].sum() == 14
    assert "a3_diffusion.png" not in text and "a4_errors.png" not in text
    assert {(r["outcome"], r["treatment"]) for r in summary["A2"]} == {("y_consensus", "D_cons")}
    assert "On a message-order run only the consensus outcome under `D_cons` is estimated: the ≥1 h treatment needs a " \
           "wall clock, and the latency outcome (an instant answer, as the post states it) is left out with it" in text
    assert "latency outcome (an instant answer) and the ≥1 h treatment need a wall clock" not in text


def test_trace_counts_messages_under_a_sequence_clock(runs, capsys):
    from swarmprov import query
    assert query.trace(str(runs["dispute"].path), agent="agent-9", item="Poland") == 0
    out = capsys.readouterr().out
    assert "clock: message order, no wall clock" in out
    assert "at least 60 min before: needs a wall clock (D_w60m)" in out
    assert "head start needs a wall clock (5 messages earlier in the file)" in out
    assert "1 message before this agent's first post of the value" in out and " s before" not in out
    assert query.trace(str(runs["dispute"].path), fact="all|Poland|16.38", limit=1) == 0
    out = capsys.readouterr().out
    assert "+1 message" in out and "span 10 messages" in out and "clock resolution" not in out


def test_partly_timed_list_discards_its_timestamps_instead_of_mixing(tmp_path):
    from swarmprov import report
    (tmp_path / "partial.jsonl").write_text("\n".join(json.dumps(m | _timed(i)) for i, m in enumerate(_msgs())))
    run = RunDir(tmp_path / "run")
    prof = pipeline.ingest(tmp_path / "partial.jsonl", run, adapter="chat")
    assert prof["clock"] == "sequence" and prof["notes"]["timestamps_discarded"] == 6
    assert (run.read("events")["ts"] < pd.Timestamp("2001-01-01", tz="UTC")).all()   # no real time survives
    assert "the 6 of 9 messages that carried a timestamp were too few to order by" in report._when(prof, prof["notes"])


def test_header_counts_one_untimed_post_in_the_singular():
    from swarmprov import report
    prof = {"first_ts": "2026-05-04 10:00:00+00:00", "last_ts": "2026-05-04 11:00:00+00:00", "n_untimed": 1}
    assert report._when(prof, {}).endswith("UTC. 1 post has no timestamp and takes part in no time comparison. ")


def test_sequence_clock_hides_clock_quality_and_deletion_notes(runs, tmp_path):
    """Only wiki and village runs carry clock grades and deletions, so the sequence-clock run is given them by hand."""
    import shutil
    from swarmprov import report
    run = RunDir(shutil.copytree(runs["openai"].path, tmp_path / "run"))
    prof = _profile(run)
    prof["notes"]["clock_grades"] = {"reqlog": 10}
    prof["capabilities"]["has_lifecycle"] = True
    (run.path / "profile.json").write_text(json.dumps(prof))
    ev = run.read("events")
    run.write("lifecycle", pd.DataFrame({"action": ["delete"], "event_id": [ev["event_id"].iloc[1]], "ts": [ev["ts"].iloc[5]]}))
    text = report.build(run)
    for what in ("Clock quality notes", "Deletions and recreations (deletion times are real times)",
                 "Deletion-aware shares (D_visible)"):
        assert f"{what} need a wall clock and are not shown" in text, what
    for gone in ("Clock quality:", "Honouring deletions", "Deletion ends visibility"):
        assert gone not in text, gone


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
    assert "| has_wall_clock | no | only the 6 of 9 posts with a timestamp are ordered by real time |" in text
    assert "_Skipped: needs more than 20 timed agent-rounds; this run has 6 (3 have no timestamp)._" in text
    assert "3 of 9 item–value and URL mentions have no timestamp and are left out" in text
    summary = json.loads((run.path / "summary.json").read_text())
    assert summary["funnel"]["agent_rounds_untimed"] == 3 and summary["A4"]["n_mentions_untimed"] == 3


def test_untimed_rooms_skip_the_time_based_analyses(runs):
    run = runs["rooms_untimed"]
    assert len(run.read("exposures_merged")) == 0 and len(run.read("edges_merged")) == 0
    text = _report(run)
    assert "no timestamps." in text
    assert "_Skipped: needs agent-rounds (an agent's answer to an item in a task family); none of the 9 agent-rounds " \
           "has a timestamp, so nothing can be ordered against them._" in text
    assert "_Skipped: needs more than 20 timed agent-rounds; this run has 0 (9 have no timestamp)._" in text
    assert "## A4. Error propagation\n\n_Skipped: needs posts with a timestamp; none of the 9 posts with an item–value " \
           "pair or URL has one, and the file joins several message lists, so no order is invented for them._" in text
    assert "## A5. Structure\n\n_Skipped: needs provenance edges; 3 facts were carried by two or more agents, but not in " \
           "posts with a timestamp (relay edges need an order; the file joins several message lists, so none is " \
           "invented), none of the 9 agent-rounds has a timestamp to be exposed against and no post cites another " \
           "agent._" in text
    assert "no fact was carried" not in text and "no answer was exposed" not in text


def test_a_dispute_in_untimed_rooms_is_skipped_for_want_of_order_not_reported_absent(runs):
    run = runs["rooms_dispute"]
    mn = run.read("mentions").query("item == 'Poland'")
    assert mn.groupby("value_key")["agent_merged"].nunique().to_dict() == {"16.38": 11, "16.4": 4}
    text = _report(run)
    assert "## A4. Error propagation\n\n_Skipped: needs posts with a timestamp; none of the 30 posts" in text
    assert "slots carried two or more values" not in text and "3-hour bin" not in text
    assert "_Skipped: needs provenance edges; 3 facts were carried by two or more agents, but not in posts with a " \
           "timestamp" in text


def test_missing_author_field_is_flagged(runs, tmp_path):
    p = tmp_path / "autogen.json"
    p.write_text(json.dumps({"messages": [m | {"type": "TextMessage"} for m in _msgs("handle")]}))
    b = ChatAdapter().load(p)                                       # "handle" is not an author alias
    assert b.notes["n_no_author"] == 9
    assert "The first record's keys are `handle`, `content`, `type`; name the author's key with `--field author=<key>`." \
        in b.notes["warnings"][0]
    ev = runs["openai"].read("events")          # its role-only system message is the task's, not an unnamed agent's
    assert ev.loc[ev["source_kind"] == "task", "author_raw"].tolist() == ["task"] and "unknown" not in set(ev["author_raw"])
    assert "no author" not in _report(runs["openai"]) and "no author" not in _report(runs["rooms"])


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

