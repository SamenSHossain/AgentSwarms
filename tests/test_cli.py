"""The command line on user errors: one `swarmprov: error:` line and exit status 2, nothing created or overwritten."""

import json

import pandas as pd
import pytest

from swarmprov import cli

CHAT = [{"author": "a", "text": "hi", "timestamp": "2026-01-01T00:00:00Z"}]


@pytest.fixture(scope="module")
def chat(tmp_path_factory):
    d = tmp_path_factory.mktemp("chat")
    (d / "chat.json").write_text(json.dumps(CHAT))
    (d / "bad.json").write_text("{bad")
    (d / "typo.json").write_text(json.dumps({"technques": [], "strict_config": True}))
    return d


def _error(capsys) -> str:
    err = capsys.readouterr().err
    assert err.startswith("swarmprov: error: ") and "Traceback" not in err
    return err


@pytest.mark.parametrize("argv, says", [
    (["run", "data/no-such-dump"], "no such file or directory (the datasets under data/ are not in git: README's "
                                   "'Where the data comes from'"),
    (["ingest", "{d}/missing.jsonl"], "missing.jsonl: no such file or directory\n"),
    (["run", "{d}/chat.json", "--config", "{d}/missing.json"], "--config: {d}/missing.json: no such file or directory"),
    (["run", "{d}/chat.json", "--roster", "{d}/missing.jsonl"], "--roster: {d}/missing.jsonl: no such file or directory"),
    (["inspect", "{d}/missing.json"], "no such file or directory"),
    (["run", "{d}/chat.json", "--config", "{d}/bad.json"], "bad.json: not a valid JSON config (Expecting property"),
    (["run", "{d}/chat.json", "--config", "{d}/typo.json"], "config key `technques` is not a setting"),
    (["run", "{d}/chat.json", "--adapter", "wiki"], "the `wiki` adapter needs a 'time' field that"),
])
def test_input_errors_end_in_one_line(chat, tmp_path, capsys, argv, says):
    out = tmp_path / "out"
    argv = [a.format(d=chat) for a in argv] + (["-o", str(out)] if argv[0] != "inspect" else [])
    assert cli.main(argv) == 2
    assert says.format(d=chat) in _error(capsys)
    if "no such file" in says:
        assert not out.exists()   # checked before any stage


@pytest.mark.parametrize("argv", [["extract", "R"], ["expose", "R"], ["graph", "R"], ["report", "R"],
                                  ["validate", "R", "g.jsonl"], ["truth-score", "R", "T"], ["check", "R"],
                                  ["trace", "R", "--list"], ["agent", "R", "x"], ["export", "R", "-o", "O"],
                                  ["html", "R"], ["suggest", "R"], ["compare", "R", "R"], ["sample-gold", "R"],
                                  ["crosssite", "R", "C", "-o", "O"]])
def test_commands_reading_a_run_refuse_a_missing_one(tmp_path, capsys, argv):
    """One message and exit status 2 from every command; the mistyped directory is not created."""
    r = tmp_path / "typo"
    assert cli.main([str(r) if a == "R" else str(tmp_path / a) if a in ("O", "T", "C") else a for a in argv]) == 2
    assert f"{r}: not a run directory (no profile.json); run `swarmprov run` first" in _error(capsys)
    assert not r.exists() and not (tmp_path / "O").exists()


def test_json_and_png_outputs_create_their_directory(synth_run, tmp_path):
    run = synth_run[0]
    r = run.read("exposures_merged").iloc[0]
    assert cli.main(["trace", str(run.path), "--fact", f"{r.family}|{r['item']}|{r.value_norm}",
                     "--json", str(tmp_path / "a" / "fact.json"), "--png", str(tmp_path / "b" / "fact.png")]) == 0
    assert json.loads((tmp_path / "a" / "fact.json").read_text())["mode"] == "fact"
    assert (tmp_path / "b" / "fact.png").exists()
    assert cli.main(["agent", str(run.path), r.agent, "--json", str(tmp_path / "c" / "agent.json")]) == 0
    assert json.loads((tmp_path / "c" / "agent.json").read_text())["result"]["label"] == r.agent


def test_export_and_crosssite_refuse_to_write_into_the_run_they_read(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "profile.json").write_text("{}")
    table = pd.DataFrame({"claim_id": ["c1"], "D": [1]})
    table.to_parquet(run / "exposures_merged.parquet")
    before = (run / "exposures_merged.parquet").read_bytes()
    assert cli.main(["export", str(run), "-o", str(tmp_path / "." / "run"), "--format", "parquet"]) == 2
    assert "is the run it reads, whose stage tables the files would overwrite" in _error(capsys)
    assert cli.main(["crosssite", str(run), str(tmp_path), "-o", str(run) + "/"]) == 2
    assert "is the wiki run it reads" in _error(capsys)
    assert (run / "exposures_merged.parquet").read_bytes() == before and sorted(p.name for p in run.iterdir()) == [
        "exposures_merged.parquet", "profile.json"]
