"""Agent-written strings that once broke a run or reached the terminal raw: control characters and lone surrogates
(ingest cleans them, GraphML writing is guarded), and escape or bidi sequences in printed names and keys."""

import json
from collections import Counter

import networkx as nx
import pandas as pd
import pytest

from swarmprov import cli, graph, pipeline, synth
from swarmprov.fmt import console

ESC = "X\x1b]0;T\x07\x1b[31mY"   # an OSC title-set and a colour change inside the handle: strict_handle keeps them
BIDI = "‮RLO"


@pytest.fixture(scope="module")
def hostile(tmp_path_factory):
    """A 12-agent swarm with both security-finding payloads, an escape sequence inside one author string and lone
    surrogates (a truncated emoji's JSON escape) in another's text and name, plus a bidi override in a third name."""
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
            r["author"] = c + BIDI
    (d / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))   # ensure_ascii: "\ud800" escapes
    cfg = json.loads((d / "config.json").read_text()) | {"bootstrap_reps": 0}
    (d / "cfg.json").write_text(json.dumps(cfg))
    run = pipeline.run_all(d / "t.jsonl", d / "run", adapter="chat", config=str(d / "cfg.json"))
    return run, (a, b, c)


def test_a_run_with_control_characters_and_surrogates_finishes(hostile):
    run, (a, b, c) = hostile
    assert (run.path / "report.md").exists() and (run.path / "provenance_strict.graphml").exists()
    ev = run.read("events")
    assert set(ev["author_raw"]) >= {a + "X]0;T[31mY", b + "�", c + BIDI}
    assert ev["text"].str.endswith(" �").sum() == (ev["author_raw"] == b + "�").sum() > 0
    n_a, n_b = (ev["author_raw"] == a + "X]0;T[31mY").sum(), (ev["author_raw"] == b + "�").sum()
    assert run.profile()["warnings"] == [
        f"{n_a + 2 * n_b} strings held lone surrogates or control characters and were cleaned at ingest (author_raw "
        f"{n_a + n_b}, text {n_b}): surrogates became U+FFFD, control characters were removed from names and ids, "
        "and those XML forbids became a space in text."]
    assert f"**Warning:** {n_a + 2 * n_b} strings held" in (run.path / "report.md").read_text()
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
    tidy = df.iloc[1:]
    assert pipeline._clean(tidy, Counter()) is tidy   # nothing to fix: the frame itself, not a copy


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
    label = run.read("agents").set_index("author_raw").loc[c + BIDI, "agent_strict"]
    assert cli.main(["agent", str(run.path), label, "--mapping", "strict"]) == 0
    assert cli.main(["trace", str(run.path), "--agent", label, "--mapping", "strict"]) == 0
    out = capsys.readouterr().out
    assert "\\u202e" in out and "‮" not in out
    (tmp_path / "keys.jsonl").write_text(json.dumps({"who\x1b[31m": "a", "body\x1b[2J": "hi", "ts": 1}) + "\n")
    assert cli.main(["inspect", str(tmp_path / "keys.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "who\\x1b[31m, body\\x1b[2J" in out and "\x1b" not in out
