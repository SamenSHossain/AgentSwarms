"""Chat ingest that works or fails loudly: the field mapping (config ``fields``, ``--field``), CSV, Slack exports,
Discord exports, directories of message files, AutoGen's ``source``, ingest warnings, config-key validation, adapter
errors, the detection order and ``swarmprov inspect``."""

import csv
import gzip
import json
import shutil
import warnings
from pathlib import Path

import pandas as pd
import pytest

from swarmprov import adapters, cli, inspect, pipeline
from swarmprov.adapters.base import AdapterConfig, config_snippet
from swarmprov.adapters.chat import ChatAdapter, _get
from swarmprov.schema import RunDir

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
T0 = pd.Timestamp("2026-05-04T10:00:00Z")


def _answer(i: int) -> str:
    return f"R1 CONFIRMED: Poland arrived; answered {0.2 if i % 3 else 0.3} at +{3 + i}s."


def _custom_keys(d: Path) -> Path:
    """Keys no alias knows: step_time, agent_handle, output; only `room` is guessed (as the channel)."""
    p = d / "custom_keys.jsonl"
    p.write_text("\n".join(json.dumps({"step_time": (T0 + pd.Timedelta(minutes=i)).isoformat(), "agent_handle": NAMES[i % 5],
                                       "output": _answer(i), "room": "lab" if i < 10 else "ops"}) for i in range(20)))
    return p


def _slack(d: Path) -> Path:
    """Six users and a bot in three channels; day files whose concatenation is out of time order; one thread each."""
    root = d / "slack_export"
    users = [{"id": f"U0{i}ABC", "name": n, "real_name": f"{n.title()} Agent"} for i, n in enumerate(NAMES[:6])]
    root.mkdir()
    (root / "users.json").write_text(json.dumps(users))
    (root / "channels.json").write_text(json.dumps([{"id": f"C{i}", "name": c} for i, c in enumerate(["general", "research", "review"])]))
    k = 0
    for c, (ch, offset) in enumerate((("general", 600), ("research", 0), ("review", 300))):   # research posts first
        (root / ch).mkdir()
        msgs = []
        for j in range(4):
            k += 1
            ts = f"{int(T0.timestamp()) + offset + 60 * j}.{k:06d}"
            msgs.append({"type": "message", "user": users[(c + j) % 6]["id"], "ts": ts,
                         "text": f"<@{users[(c + j + 1) % 6]['id']}> {_answer(k)} see <https://example.org/t.csv|the table>"})
        msgs[0]["thread_ts"] = msgs[0]["ts"]                       # a thread's first message names itself
        msgs[2]["thread_ts"] = msgs[0]["ts"]
        msgs.append({"type": "message", "subtype": "bot_message", "bot_id": "B1", "username": "ci-bot",
                     "text": "build ok", "ts": f"{int(T0.timestamp()) + offset + 300}.{k:06d}"})
        (root / ch / "2026-05-04.json").write_text(json.dumps(msgs))
    return root


def _discord(d: Path) -> Path:
    """A DiscordChatExporter file: guild and channel metadata around the messages, replies by reference.messageId."""
    msgs = [{"id": str(9000 + i), "type": "Reply" if i in (3, 5) else "Default", "timestamp": (T0 + pd.Timedelta(minutes=i)).isoformat(),
             "content": _answer(i), "author": {"id": str(100 + i % 4), "name": NAMES[i % 4], "nickname": NAMES[i % 4].title()},
             "attachments": [], "reactions": []} | ({"reference": {"messageId": str(8998 + i), "channelId": "55"}} if i in (3, 5) else {})
            for i in range(8)]
    p = d / "discord_research.json"
    p.write_text(json.dumps({"guild": {"id": "1", "name": "Swarm HQ"}, "dateRange": {"after": None},
                             "channel": {"id": "55", "type": "GuildTextChat", "category": "Ops", "name": "research"},
                             "messages": msgs, "messageCount": len(msgs)}))
    return p


def _csv(d: Path, name: str = "chat.csv", sep: str = ",") -> Path:
    p = d / name
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh, delimiter=sep)
        w.writerow(["timestamp", "author", "text", "channel"])
        for i in range(20):
            w.writerow([str(int((T0 + pd.Timedelta(minutes=i)).timestamp())), NAMES[i % 7], _answer(i) + " (via a, b)",
                        "lab" if i % 2 else "ops"])
    return p


@pytest.fixture(scope="module")
def d(tmp_path_factory):
    return tmp_path_factory.mktemp("ingest")


def test_dotted_keys_reach_nested_objects():
    m = {"author": {"name": "a", "nickname": "A"}, "reference": {"messageId": "7"}, "a.b": 1}
    assert _get(m, "author.nickname") == "A" and _get(m, "reference.messageId") == "7"
    assert _get(m, "a.b") == 1 and _get(m, "reference.channelId") is None and _get(m, "author.name.x") is None


def test_inspect_suggests_a_mapping_that_runs(d, capsys):
    p = _custom_keys(d)
    with pytest.raises(ValueError, match="no adapter recognises.*swarmprov inspect"):
        adapters.detect(p)
    assert cli.main(["inspect", str(p)]) == 0
    out = capsys.readouterr().out
    assert "  chat         no\n" in out and "No adapter recognises it" in out
    assert "first record's keys: step_time, agent_handle, output, room" in out
    assert "  channel    <- room (20)" in out and "  author     <- (none)" in out
    assert "coverage: timestamps read 0%, authors known (not `unknown`) 0%, text non-empty 0%, 2 channels" in out
    snippet = json.loads(out[out.index("{"):])
    assert snippet == {"fields": {"ts": "step_time", "author": "agent_handle", "text": "output"}}
    (d / "suggested.json").write_text(json.dumps(snippet))
    run = pipeline.run_all(p, d / "run_custom", config=str(d / "suggested.json"))
    prof = json.loads((run.path / "profile.json").read_text())
    assert prof["adapter"] == "chat" and prof["n_authors"] == 5 and prof["clock"] == "wall"
    assert prof["fields"] == snippet["fields"] and prof["warnings"] == []
    assert "2026-05-04 10:00 → 2026-05-04 10:19 UTC" in (run.path / "report.md").read_text()


def test_inspect_with_fields_reports_coverage_and_stops_at_another_adapters_input(d, capsys):
    p = _custom_keys(d)
    assert cli.main(["inspect", str(p), "--field", "text=output", "--field", "ts=step_time"]) == 0
    out = capsys.readouterr().out
    assert "fields given: text=output, ts=step_time\n" in out and "  chat         yes   <- auto-detection picks this one" in out
    assert "  text       <- output (20) [mapped]" in out and "timestamps read 100%" in out
    assert json.loads(out[out.index("{"):]) == {"fields": {"text": "output", "ts": "step_time", "author": "agent_handle"}}
    (d / "untimed.jsonl").write_text("\n".join(json.dumps({"source": a, "content": _answer(i)}) for i, a in enumerate(NAMES)))
    assert cli.main(["inspect", str(d / "untimed.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "  author     <- source (7)" in out and out.endswith("\nNo key looks like the ts.\n")
    assert cli.main(["inspect", str(ROOT / "data" / "raw_village")]) == 0
    out = capsys.readouterr().out
    assert "  village      yes   <- auto-detection picks this one" in out and "first record's keys" not in out
    (d / "bad.json").write_text("{not json")
    assert cli.main(["inspect", str(d / "bad.json")]) == 1
    out = capsys.readouterr().out
    assert "  chat         error: ValueError: bad.json cannot be read" in out and "The chat reader cannot read it" in out


def test_field_flags_and_config_fields_reach_the_adapter(d, capsys):
    p = _custom_keys(d)
    (d / "partial.json").write_text(json.dumps({"fields": {"text": "output", "author": "room"}}))
    cli.main(["ingest", str(p), "-o", str(d / "run_flags"), "--config", str(d / "partial.json"),
              "--field", "ts=step_time", "--field", "author=agent_handle,channel=room"])
    prof = json.loads((d / "run_flags" / "profile.json").read_text())
    assert prof["fields"] == {"text": "output", "author": "agent_handle", "ts": "step_time", "channel": "room"}
    assert prof["n_authors"] == 5 and prof["clock"] == "wall"                 # the flag overrides the config's author
    assert "with adapter 'chat', fields {" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["ingest", str(p), "--field", "autor=agent_handle"])
    assert "unknown field `autor` (did you mean `author`?)" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["ingest", str(p), "--field", "author"])
    assert "expected FIELD=KEY" in capsys.readouterr().err
    with pytest.raises(ValueError, match="unknown field `txt` \\(did you mean `text`\\?\\)"):
        AdapterConfig.from_dict({"fields": {"txt": "output"}})


def test_autogen_source_is_the_last_author_alias(synth_run, tmp_path):
    """The synthetic swarm written as an AutoGen stream: 300 posts from 60 speakers, not one `unknown`."""
    run, _ = synth_run
    ev = run.read("events")
    p = tmp_path / "autogen.jsonl"
    p.write_text("\n".join(json.dumps({"source": a, "models_usage": None, "content": t, "type": "TextMessage"})
                           for a, t in zip(ev["author_raw"], ev["text"])))
    b = ChatAdapter().load(p)
    assert len(b.events) == 300 and b.events["author_raw"].nunique() == 60 and b.notes["n_no_author"] == 0
    rows = [{"source": "https://data.example.org/oecd.csv", "text": "x"}, {"source": "www.example.org", "text": "x"},
            {"source": "data/oecd.csv", "text": "x"}, {"source": "planner", "name": "named", "text": "x"},
            {"source": "planner", "text": "x"}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert list(ChatAdapter().load(p).events["author_raw"]) == ["unknown"] * 3 + ["named", "planner"]


def test_csv_and_tsv_transcripts(d):
    p = _csv(d)
    assert adapters.detect(p).name == "chat"
    run = pipeline.run_all(p, d / "run_csv")
    prof = json.loads((run.path / "profile.json").read_text())
    assert prof["n_events"] == 20 and prof["n_authors"] == 7 and prof["clock"] == "wall"
    ev = run.read("events")
    assert ev["ts"].min() == T0 and set(ev["channel"]) == {"lab", "ops"} and ev["text"].str.endswith("(via a, b)").all()
    assert (run.path / "report.md").read_text().startswith("# Copied or computed?")
    tsv = _csv(d, "chat.tsv.gz", "\t")
    tsv.write_bytes(gzip.compress(tsv.read_bytes()))
    b = adapters.detect(tsv).load(tsv)
    assert len(b.events) == 20 and b.events["author_raw"].nunique() == 7


def test_slack_export_directory(d):
    root = _slack(d)
    assert adapters.detect(root).name == "chat"
    run = RunDir(d / "run_slack")
    prof = pipeline.ingest(root, run)
    ev = run.read("events")
    assert prof["clock"] == "wall" and prof["capabilities"]["has_threading"] is True and ev["ts"].notna().all()
    assert ev["author_raw"].nunique() == 7 and set(ev["channel"]) == {"general", "research", "review"}
    assert {"Alpha Agent", "ci-bot"} <= set(ev["author_raw"]) and "U00ABC" in set(ev["author_alt"])
    replies = ev.dropna(subset=["parent_id"]).merge(ev, left_on="parent_id", right_on="event_id", suffixes=("", "_parent"))
    assert len(replies) == 3 and (replies["channel"] == replies["channel_parent"]).all()   # thread_ts names a message's ts
    b = adapters.get("chat").load(root)                           # the adapter's own order: by ts, not by file
    assert b.events["ts"].is_monotonic_increasing and b.events["channel"].iloc[0] == "research"
    first = b.events["text"].iloc[0]
    assert first.startswith("@Charlie Agent R1 CONFIRMED") and first.endswith("see https://example.org/t.csv")
    assert "<" not in "".join(b.events["text"])


def test_discord_export_keeps_its_channel_and_replies(d):
    p = _discord(d)
    b = adapters.detect(p).load(p)
    ev = b.events
    assert len(ev) == 8 and set(ev["channel"]) == {"research"} and ev["author_raw"].nunique() == 4
    assert ev["ts"].notna().all() and b.capabilities.has_threading
    replies = ev.dropna(subset=["parent_id"])
    assert list(replies["parent_id"]) == ["9001", "9003"] and replies["parent_id"].isin(ev["event_id"]).all()
    nick = ChatAdapter({"author": "author.nickname"}).load(p).events
    assert set(nick["author_raw"]) == {"Alpha", "Bravo", "Charlie", "Delta"}


def test_directory_of_message_files_names_channels_after_files(d):
    root = d / "rooms"
    root.mkdir()
    for ch in ("planning", "results"):
        (root / f"{ch}.jsonl").write_text("\n".join(json.dumps({"timestamp": (T0 + pd.Timedelta(hours=k)).isoformat(),
                                                                "author": NAMES[k], "text": f"{ch} {k}"}) for k in range(5)))
    (root / "extra.json").write_text(json.dumps({"messages": [{"author": "zed", "text": "own room", "room": "side"}]}))
    (root / "notes.txt").write_text("not a message file")
    assert adapters.detect(root).name == "chat"
    b = ChatAdapter().load(root)
    assert b.events.groupby("channel").size().to_dict() == {"planning": 5, "results": 5, "side": 1}
    assert b.notes["clock"] == "wall" and b.events["source_ref"].iloc[0] == "extra.json:0"
    untimed = d / "rooms_untimed"
    untimed.mkdir()
    (untimed / "only.jsonl").write_text("\n".join(json.dumps({"author": a, "text": _answer(i)}) for i, a in enumerate(NAMES)))
    b = ChatAdapter().load(untimed)                              # a directory is never one message list
    assert b.notes["clock"] == "none" and not b.capabilities.has_order and b.events["ts"].isna().all()


def test_detection_order_keeps_table_directories(tmp_path):
    raw_village = ROOT / "data" / "raw_village"
    assert adapters.detect(raw_village).name == "village"
    tables = tmp_path / "village"                                 # Village tables beside a message table
    tables.mkdir()
    shutil.copy(raw_village / "agents.jsonl.gz", tables)
    (tables / "chat_messages.jsonl").write_text(json.dumps({"author": "a", "content": "hi", "timestamp": "2026-05-04T10:00Z"}))
    assert adapters.detect(tables).name == "village"
    st = tmp_path / "st" / "redacted.jsonl"
    st.parent.mkdir()
    st.write_text(json.dumps({"id": "R1", "cite": "R1:a", "kind": "payload", "parent_id": None, "time_utc": None,
                              "tags": "", "text": "x", "source": "https://example.org"}) + "\n")
    assert adapters.detect(st).name == "swarmtraces" and adapters.detect(st.parent).name == "swarmtraces"
    if (ROOT / "data" / "raw2" / "records.jsonl").exists():
        assert adapters.detect(ROOT / "data" / "raw2").name == "corpus"
    if (ROOT / "data" / "raw" / "revisions.jsonl").exists():
        assert adapters.detect(ROOT / "data" / "raw").name == "wiki"


def test_adapter_names_and_sniff_errors_are_explained(d, capsys):
    with pytest.raises(ValueError, match="unknown adapter `slack`; the adapters are auto, wiki, corpus, swarmtraces, "
                                         "village, roster, chat"):
        adapters.get("slack")
    with pytest.raises(ValueError, match="did you mean `chat`"):
        adapters.get("chta")
    with pytest.raises(SystemExit):
        cli.main(["run", str(_csv(d)), "--adapter", "slack"])
    assert "invalid choice: 'slack' (choose from 'auto', 'wiki', 'corpus'" in capsys.readouterr().err
    (d / "broken.json").write_text("{not json")
    with pytest.raises(ValueError) as e:
        adapters.detect(d / "broken.json")
    msg = str(e.value)
    assert "Sniff errors:" in msg and "\n  chat: ValueError: broken.json cannot be read (" in msg
    assert "the chat adapter reads a JSON array of messages" in msg and "CSV/TSV with a header row" in msg


def test_unknown_config_keys_warn_with_a_suggestion_and_strict_raises():
    with pytest.warns(UserWarning, match="config key `technques` is not a setting \\(did you mean `techniques`\\?\\) "
                                         "and is ignored"):
        cfg = AdapterConfig.from_dict({"technques": [{"name": "x", "pattern": "x"}], "families": {"a": "a"}})
    assert cfg.techniques == [] and cfg.families == {"a": "a"}
    with pytest.warns(UserWarning, match="`family_of_text` is not a setting"):
        cfg = AdapterConfig.from_dict({"family_of_text": "oops"})
    assert callable(cfg.family_of_text)                             # a method is not a setting to overwrite
    with pytest.raises(ValueError, match="`technques` is not a setting.*`familes` is not a setting \\(did you mean `families`"):
        AdapterConfig.from_dict({"strict_config": True, "technques": [], "familes": {}})
    with pytest.raises(ValueError, match="`familes`"):
        AdapterConfig.from_dict({"familes": {}}, base=AdapterConfig(name="x", strict_config=True))
    snippet = config_snippet(fields={"author": "source"}, families={"a": "a"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert AdapterConfig.from_dict(json.loads(snippet)).fields == {"author": "source"}
    with pytest.raises(ValueError, match="not config settings"):
        config_snippet(feilds={})


def test_ingest_warns_in_the_log_the_profile_and_the_report(d, capsys):
    """One author, one channel, and timestamps that are missing or unreadable; plus a mistyped config key."""
    p = d / "solo.jsonl"
    p.write_text("\n".join(json.dumps({"author": "solo", "text": _answer(i)}
                                      | ({"timestamp": (T0 + pd.Timedelta(minutes=i)).isoformat()} if i < 6 else
                                         {"timestamp": "soon"} if i < 9 else {})) for i in range(12)))
    (d / "typo.json").write_text(json.dumps({"technques": []}))
    run = pipeline.run_all(p, d / "run_solo", config=str(d / "typo.json"))
    out = capsys.readouterr().out
    prof = json.loads((run.path / "profile.json").read_text())
    w = prof["warnings"]
    assert len(w) == 4 and "warnings" not in prof["notes"]
    assert w[0].startswith("config key `technques` is not a setting (did you mean `techniques`?)")
    assert w[1].startswith("All 12 posts have one author, `solo`") and "The first record's keys are `author`, `text`, " \
        "`timestamp`; if another one names the speaker, use `--field author=<key>`." in w[1]
    assert w[2] == ("3 of 12 posts have no timestamp field (the first record's keys are `author`, `text`, `timestamp`); "
                    "if one holds the time, map it with `--field ts=<key>`. 3 of 12 timestamps could not be read as a time "
                    "(ISO text, or epoch seconds or milliseconds from 1990 to 2100), e.g. `soon`.")
    assert w[3].startswith("All 12 posts are in one channel, `main`")
    assert all(f"warning: {x}" in out for x in w)
    text = (run.path / "report.md").read_text()
    capabilities = text[text.index("## Data capabilities"):text.index("## Pipeline funnel")]
    assert all(f"**Warning:** {x}\n" in capabilities for x in w)
    assert json.loads((run.path / "summary.json").read_text())["warnings"] == w
    with warnings.catch_warnings():
        warnings.simplefilter("error")                              # later stages do not repeat ingest's warnings
        pipeline.load_config(run)
    half = d / "half.jsonl"                                          # most posts unsigned: the keys are named
    half.write_text("\n".join(json.dumps({"author": "a", "text": "x"} if i < 4 else {"who": "b", "text": "x"}) for i in range(12)))
    assert ChatAdapter().load(half).notes["warnings"][0].endswith(
        "8 posts name no author (no author field was recognised), so they are pooled as one agent, `unknown`, which can "
        "look like a hub or a relay source in A5. The first record's keys are `author`, `text`; name the author's key "
        "with `--field author=<key>`.")
    roster = RunDir(d / "run_roster")
    pipeline.ingest(ROOT / "data" / "village" / "agent_goals.jsonl", roster, fields={"author": "x"})
    ignored = "`fields` reach the chat adapter only; the `roster` adapter reads its own schema and ignores them."
    assert json.loads((roster.path / "profile.json").read_text())["warnings"] == [ignored]
    from swarmprov import report
    assert f"**Warning:** {ignored}\n" in report.build(roster)            # the context report lists them too
