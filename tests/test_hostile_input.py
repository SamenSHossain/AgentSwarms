"""Agent-written strings that once broke a run or reached the terminal raw: control characters and lone surrogates
(ingest cleans them, GraphML writing is guarded), and escape or bidi sequences in printed names and keys."""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from swarmprov import cli, graph, pipeline, synth
from swarmprov.fmt import console
from swarmprov.textutil import clean

ROOT = Path(__file__).resolve().parents[1]
ESC = "X\x1b]0;T\x07\x1b[31mY"   # an OSC title-set and a colour change inside the handle: strict_handle keeps them
BIDI = "‮RLO"
FORGE, FORGED = "\rFAKE\nanswer forged\x85Y", " FAKE answer forgedY"   # a name that would print lines of its own


@pytest.fixture(scope="module")
def hostile(tmp_path_factory):
    """A 12-agent swarm with both security-finding payloads, an escape sequence inside one author string and lone
    surrogates (a truncated emoji's JSON escape) in another's text and name, plus a bidi override and line breaks in a
    third name."""
    d = tmp_path_factory.mktemp("hostile")
    path = synth.write(d / "clean.jsonl", n_agents=12, seed=2)
    rows = [json.loads(line) for line in open(path)]
    a, b, c = sorted({r["author"] for r in rows})[:3]
    for r in rows:
        if r["author"] == a:
            r["author"] = a + ESC
        elif r["author"] == b:
            r |= {"author": b + "\ud800", "text": r["text"] + " \udc00"}
        elif r["author"] == c:
            r["author"] = c + BIDI + FORGE
    (d / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))   # ensure_ascii: "\ud800" escapes
    cfg = json.loads((d / "config.json").read_text()) | {"bootstrap_reps": 0}
    (d / "cfg.json").write_text(json.dumps(cfg))
    run = pipeline.run_all(d / "t.jsonl", d / "run", adapter="chat", config=str(d / "cfg.json"))
    return run, (a, b, c)


def test_a_run_with_control_characters_and_surrogates_finishes(hostile):
    run, (a, b, c) = hostile
    assert (run.path / "report.md").exists() and (run.path / "provenance_strict.graphml").exists()
    ev = run.read("events")
    assert set(ev["author_raw"]) >= {a + "X]0;T[31mY", b + "�", c + BIDI + FORGED}
    assert ev["text"].str.endswith(" �").sum() == (ev["author_raw"] == b + "�").sum() > 0
    n_a, n_b, n_c = ((ev["author_raw"] == x).sum() for x in (a + "X]0;T[31mY", b + "�", c + BIDI + FORGED))
    assert run.profile()["warnings"] == [
        f"{n_a + 2 * n_b + n_c} strings held lone surrogates or control characters and were cleaned at ingest "
        f"(author_raw {n_a + n_b + n_c}, text {n_b}): surrogates became U+FFFD, control characters were removed from "
        "names and ids (a tab or line break became a space), and those XML forbids became a space in text."]
    assert f"**Warning:** {n_a + 2 * n_b + n_c} strings held" in (run.path / "report.md").read_text()
    labels = set(nx.read_graphml(run.path / "provenance_strict.graphml"))
    assert any("x]0;t[31my" in n for n in labels)


def test_clean_keeps_text_mojibake_and_tab_but_not_name_controls():
    """C1 is how a mis-decoded dash reads in the wiki's text ("â\\x80\\x93"), and XML allows it: text keeps it."""
    df = pd.DataFrame({"author_raw": ["bob\x9b\x1b[2J", "ann"], "text": ["Masterâ\x80\x99s\x0c\tok", "fine"],
                       "source_ref": pd.Series(["f\ud800", "g"], dtype=object)})   # as _load reads a surrogate
    fixed = Counter()
    out = pipeline._clean(df, fixed)
    assert out["author_raw"].tolist() == ["bob[2J", "ann"] and out["text"].tolist() == ["Masterâ\x80\x99s \tok", "fine"]
    assert out["source_ref"].tolist() == ["f�", "g"] and fixed == {"author_raw": 1, "text": 1, "source_ref": 1}
    assert clean("x\r\ny\tz\x85") == "x y z" and clean("x\r\ny\tz", "text") == "x\r\ny\tz"
    tidy = df.iloc[1:]
    assert pipeline._clean(tidy, Counter()) is tidy   # nothing to fix: the frame itself, not a copy


def test_an_attached_roster_is_cleaned_like_a_main_input(tmp_path):
    """A lone surrogate in a goal, in a roster attached to a transcript: ingest cleans it as it does in the same file
    given as the input, and reads the roster before it clears the run directory."""
    rows = [json.loads(line) for line in (ROOT / "data" / "village" / "agent_goals.jsonl").open(encoding="utf-8")]
    rows[0]["name"] += " \ud83d"
    (tmp_path / "goals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    chat = synth.write(tmp_path / "t.jsonl", n_agents=4, seed=2)
    assert cli.main(["ingest", str(chat), "-o", str(tmp_path / "run"), "--roster", str(tmp_path / "goals.jsonl")]) == 0
    run = pipeline.RunDir.open(tmp_path / "run")
    assert run.read("roster")["goal"].str.endswith(" \ufffd").sum() == 1
    assert run.profile()["warnings"][0].startswith("1 string held lone surrogates or control characters and was "
                                                   "cleaned at ingest (goal 1)")
    (tmp_path / "bad.jsonl").write_text("not a roster\n")
    assert cli.main(["ingest", str(chat), "-o", str(tmp_path / "run"), "--roster", str(tmp_path / "bad.jsonl")]) == 2
    assert run.has("roster") and run.has("events")


def test_graphml_backstop_fixes_ids_and_attributes(tmp_path):
    """A run ingested before the cleaning (or a configured label) still writes its graph."""
    G = nx.DiGraph()
    G.add_edge("a\x1bb", "c", note="x\x0cy\ud800")
    G.nodes["c"]["family"] = "f\x07"
    nx.write_graphml(graph.xml_safe(G), tmp_path / "g.graphml")
    H = nx.read_graphml(tmp_path / "g.graphml")
    assert set(H) == {"ab", "c"} and H.edges["ab", "c"]["note"] == "x y�" and H.nodes["c"]["family"] == "f "


def test_console_escapes_controls_and_bidi_overrides():
    assert console("a\x1b]0;T\x07\x9bb‮c⁦\td\ne") == "a\\x1b]0;T\\x07\\x9bb\\u202ec\\u2066\td\ne"


def test_trace_agent_and_inspect_print_no_raw_controls(hostile, tmp_path, capsys):
    run, (_, _, c) = hostile
    label = run.read("agents").set_index("author_raw").loc[c + BIDI + FORGED, "agent_strict"]
    assert cli.main(["agent", str(run.path), label, "--mapping", "strict"]) == 0
    assert cli.main(["trace", str(run.path), "--agent", label, "--mapping", "strict"]) == 0
    out = capsys.readouterr().out
    assert "\\u202e" in out and "‮" not in out
    assert FORGED in out and not any(line.startswith(("FAKE", "answer forged")) for line in out.splitlines())
    (tmp_path / "keys.jsonl").write_text(json.dumps({"who\x1b[31m": "a", "body\x1b[2J": "hi", "ts": 1}) + "\n")
    assert cli.main(["inspect", str(tmp_path / "keys.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "who\\x1b[31m, body\\x1b[2J" in out and "\x1b" not in out


def test_a_console_without_utf8_gets_escapes_not_a_crash(hostile, tmp_path):
    """Under the C locale (no coercion, no UTF-8 mode) a name's U+FFFD prints as an escape."""
    run, (_, b, _) = hostile
    label = run.read("agents").set_index("author_raw").loc[b + "\ufffd", "agent_strict"]
    env = os.environ | {"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-m", "swarmprov", "agent", str(run.path), label, "--mapping", "strict"],
                         env=env, cwd=tmp_path, capture_output=True, text=True, errors="replace")
    assert out.returncode == 0, out.stderr
    assert f"{b}\\ufffd" in out.stdout
