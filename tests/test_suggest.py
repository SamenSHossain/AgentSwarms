"""``swarmprov suggest`` and ``auto_techniques``: the precise URL-wrap test, the mined techniques and their evidence,
channel families with items found under them, the run's other settings carried over, A3 on mined techniques labelled
as such, and A6's url domains row, which is the only change a run without ``auto_techniques`` sees (against summaries
fixed before ``suggest`` existed)."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from swarmprov import adapters, cli, compare, pipeline, report, suggest
from swarmprov.adapters import ADAPTERS, AdapterConfig
from swarmprov.schema import RunDir

GOLDEN = Path(__file__).parent / "golden"

T0 = pd.Timestamp("2026-05-04T10:00:00Z")


def _messages() -> list[dict]:
    """Five agents on two boards: four fetch a World Bank page each through r.jina.ai (no exact URL repeats), three
    post the same census query, a shortener mostly carries plain links, and two quote a median income."""
    posts = [("r1", "research", "Start: https://r.jina.ai/https://data.worldbank.org/indicator/A1 Answer: Poland 41.2"),
             ("r2", "research", "via https://r.jina.ai/https://data.worldbank.org/indicator/A2 Answer: Poland 41.2"),
             ("r3", "review", "mirror https://r.jina.ai/https://data.worldbank.org/indicator/A3 Answer: Poland 41.3"),
             ("r4", "review", "reader https://r.jina.ai/https://data.worldbank.org/indicator/A4 Answer: Poland 41.2"),
             ("r1", "research", "census https://api.census.example/data/2020?get=POP&for=state:06"),
             ("r2", "review", "same https://api.census.example/data/2020?get=POP&for=state:06"),
             ("r5", "research", "also https://api.census.example/data/2020?get=POP&for=state:06"),
             ("r1", "research", "short https://tinyurl.example/abc1"), ("r2", "research", "short https://tinyurl.example/abc2"),
             ("r3", "review", "short https://tinyurl.example/abc3"),
             ("r4", "review", "short https://tinyurl.example/go?u=https://x.org/a"),
             ("r5", "review", "short https://tinyurl.example/go?u=https://x.org/b"),
             ("r3", "research", "Median Income was 54,000 last year"), ("r5", "research", "Median Income: 54,100 here")]
    posts += [(f"r{1 + i % 5}", ("research", "review")[i % 2], f"status note {i}") for i in range(10)]
    return [{"id": f"m{i}", "ts": (T0 + pd.Timedelta(minutes=7 * i)).isoformat(), "author": a, "channel": c, "text": t}
            for i, (a, c, t) in enumerate(posts)]


def _write(path: Path, messages: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(m) for m in messages))
    return path


def _extracted(transcript: Path, out: Path, config: Path | None = None) -> RunDir:
    """Ingest and extract only: what ``suggest`` reads."""
    run = RunDir(out)
    pipeline.ingest(transcript, run, "chat", str(config) if config else None)
    pipeline.extract(run)
    return run


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("suggest")
    _write(d / "chat.jsonl", _messages())
    (d / "auto.json").write_text(json.dumps({"auto_techniques": True}))
    plain = pipeline.run_all(d / "chat.jsonl", d / "plain", adapter="chat")
    auto = pipeline.run_all(d / "chat.jsonl", d / "auto", adapter="chat", config=str(d / "auto.json"))
    return plain, auto


@pytest.fixture(scope="module")
def mapped(tmp_path_factory):
    """The same posts under other keys, run with a config that maps them and sets a threshold, a reference value
    under the catch-all family, a technique and an item: (transcript, run, its config)."""
    d = tmp_path_factory.mktemp("mapped")
    rows = [{"step_time": m["ts"], "agent_handle": m["author"], "room": m["channel"], "output": m["text"]}
            for m in _messages()]
    cfg = {"fields": {"ts": "step_time", "author": "agent_handle", "channel": "room", "text": "output"},
           "instant_threshold_s": 30, "reference_values": {"all|Poland": 41.2},
           "techniques": [{"name": "reader", "pattern": "jina"}], "extra_items": {"all": ["Foo"]}}
    (d / "cfg.json").write_text(json.dumps(cfg))
    return _write(d / "fields.jsonl", rows), _extracted(d / "fields.jsonl", d / "run", d / "cfg.json"), cfg


def test_the_wrap_test_is_precise():
    for u in ("https://r.jina.ai/https://example.com/a", "https://x.dev/?url=https%3A%2F%2Fexample.com",
              "https://md.succ.ai/https%253A%252F%252Fexample.com", "https://markdown.new/example.com/page",
              "https://md.succ.ai//www.sec.gov/files/x.json", "https://p.io/raw?url=www.sec.gov%2Ffiles",
              "https://pure.md/web.archive.org", "https://markdown.new/www-sec-gov.translate.goog/files/county.json"):
        assert suggest.wraps(u), u
    for u in ("https://wikiservice.at/dse/wiki.cgi?action=browse&id=StartSeite",
              "https://api.datausa.io/tesseract/data.jsonrecords?cube=acs&drilldowns=Year",
              "https://example.com/index.html", "https://site.org/docs/example.com/x",   # a .tld/ further in is no wrap
              "https://example.com/example.com/x", "https://p.io/?q=foo.bar", "https://tinyurl.example/abc1",
              # a root-level file or a dotted name is not a host: its last label is no top-level domain, or a file's
              "https://example.com/report.xlsx", "https://example.com/README.md", "https://example.com/notes.docx",
              "https://example.com/image.webp", "https://example.com/data.parquet", "https://example.com/file.tar",
              "https://allorigins.hexlet.app/favicon.ico", "https://site.com/user.name/profile"):
        assert not suggest.wraps(u), u


def test_items_leave_trailing_punctuation_out():
    posts = pd.DataFrame({"event_id": list("abcd"), "family": "f", "author_raw": ["x", "y", "x", "y"], "channel": "c",
                          "text": ["Answer: Poland. 41.2", "Poland. 41.2", "Climate & Environment. 2020", "Climate & Environment, 2021"]})
    assert suggest.suggest_items(posts) == {"f": ["Climate & Environment"]}   # Poland is a built-in country


def test_suggest_writes_a_reviewable_config(runs, tmp_path, capsys):
    """Proposed families come with items found under them, so a run with the file matches them."""
    plain, _ = runs
    out = tmp_path / "new" / "s.json"   # its directory is made
    assert cli.main(["suggest", str(plain.path), "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    cfg = json.loads(out.read_text())
    assert [t["name"] for t in cfg["techniques"]] == ["proxy:r.jina.ai"]
    assert cfg["techniques"][0]["description"].startswith(suggest.MINED)
    assert cfg["families"] == {"research": "^research$", "review": "^review$"}
    assert cfg["extra_items"] == {"research": ["Median Income"]} and set(cfg) == set(suggest.SUGGESTED)
    lines = {ln.split()[0]: ln for ln in printed.splitlines() if ln.startswith("  ") and "." in ln.split()[0]}
    assert lines["r.jina.ai"].split()[1:5] == ["4", "4", "4", "100%"] and "technique proxy:r.jina.ai" in lines["r.jina.ai"]
    assert "2026-05-04 10:00" in lines["r.jina.ai"]
    assert "40%" in lines["tinyurl.example"] and "near the threshold" in lines["tinyurl.example"]
    assert "data source" in lines["api.census.example"] and "not applied" in printed
    assert f"swarmprov run {plain.path.parent / 'chat.jsonl'} --adapter chat --config {out}" in printed
    loaded = AdapterConfig.from_file(out, base=adapters.get("chat").config())
    assert loaded.techniques[0].name == "proxy:r.jina.ai" and loaded.techniques[0].families == ()
    rerun = _extracted(plain.path.parent / "chat.jsonl", tmp_path / "rerun", out)
    assert json.loads((rerun.path / "items.json").read_text()) == {"research": ["Median Income"]}
    ev = rerun.read("events")
    quoted = ev[ev["text"].str.startswith("Median Income")]
    im = pipeline._matcher({"research": ["Median Income"]})
    assert len(quoted) == 2 and all(im.find(f, t) for f, t in zip(quoted["family"], quoted["text"]))
    assert cli.main(["suggest", str(tmp_path / "none")]) == 2 and not (tmp_path / "none").exists()


def test_suggest_carries_the_run_settings_over(mapped, tmp_path, capsys):
    """The file replaces the run's config: its field mapping, threshold and reference value carry over, its items
    under the catch-all family go to each proposed family, and a run with it reads the same posts."""
    transcript, run, cfg = mapped
    out = tmp_path / "s.json"
    assert suggest.main(str(run.path), str(out)) == 0
    printed = capsys.readouterr().out
    got = json.loads(out.read_text())
    assert {k: got[k] for k in ("fields", "instant_threshold_s", "reference_values")} == {
        k: cfg[k] for k in ("fields", "instant_threshold_s", "reference_values")}
    assert got["techniques"][0] == {"name": "reader", "pattern": "jina"} and len(got["techniques"]) == 1
    assert got["extra_items"] == {"research": ["Foo", "Median Income"], "review": ["Foo"]}
    assert "already configured (reader)" in printed
    assert "kept from the run's config: fields, instant_threshold_s, reference_values" in printed
    assert "note: reference_values name the catch-all family `all`" in printed
    again = _extracted(transcript, tmp_path / "again", out)
    assert len(again.read("events")) == len(run.read("events")) == 24
    assert set(again.read("events")["family"]) == {"research", "review"}


def test_suggest_keeps_what_the_config_sets(mapped, tmp_path, capsys):
    """Families the run's config sets are kept, and its items come before the new ones; the settings are the ones
    the run recorded, so editing its config file afterwards changes nothing."""
    transcript, _, cfg = mapped
    (tmp_path / "cfg.json").write_text(json.dumps(cfg | {"families": {"all": "."}}))
    run = _extracted(transcript, tmp_path / "run", tmp_path / "cfg.json")
    (tmp_path / "cfg.json").write_text("{}")
    assert suggest.main(str(run.path), str(tmp_path / "s.json")) == 0
    assert "families: the config's 1 kept" in capsys.readouterr().out
    got = json.loads((tmp_path / "s.json").read_text())
    assert (got["families"], got["extra_items"], got["fields"]) == (
        {"all": "."}, {"all": ["Foo", "Median Income"]}, cfg["fields"])


def test_the_evidence_table_says_what_it_leaves_out(tmp_path, capsys):
    """A URL on a domain one agent posted is no row, and the table says how many there were; under message order
    the first post is a position; a domain two agents posted that wraps is listed, short of a technique."""
    one = _extracted(_write(tmp_path / "one.jsonl", [{"author": "a", "text": "see https://r.jina.ai/https://x.org/a"}]),
                     tmp_path / "one")
    assert suggest.main(str(one.path), str(tmp_path / "s1.json")) == 0
    assert "(no domain that two or more agents posted: 1 distinct URL on 1 domain)" in capsys.readouterr().out
    two = [{"author": a, "text": f"via https://r.jina.ai/https://x.org/{a}"} for a in ("x", "y", "z", "y")]
    run = _extracted(_write(tmp_path / "two.jsonl", [{"author": "w", "text": "hello"}, *two[:2]]), tmp_path / "two")
    assert json.loads((run.path / "profile.json").read_text())["clock"] == "sequence"
    assert suggest.main(str(run.path), str(tmp_path / "s2.json")) == 0
    row = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("  r.jina.ai"))
    assert "message 1" in row and "2000-01-01" not in row and "a technique needs 3+ agents" in row


def test_auto_techniques_is_off_by_default_everywhere():
    assert not any(adapters.get(n).config().auto_techniques for n in ADAPTERS)


def test_auto_techniques_runs_a3_on_mined_rows(runs, tmp_path):
    plain, auto = runs
    assert not (plain.path / "mined_techniques.json").exists()
    assert "`auto_techniques` mines them at extract" in (plain.path / "report.md").read_text()
    mined = json.loads((auto.path / "mined_techniques.json").read_text())
    assert [t["name"] for t in mined] == ["proxy:r.jina.ai"]
    tags = auto.read("tags")
    assert (tags["techniques"] == "proxy:r.jina.ai").sum() == 4
    a3 = json.loads((auto.path / "summary.json").read_text())["A3"]
    assert [(r["technique"], r["mined"], r["adopters"]) for r in a3] == [("proxy:r.jina.ai", True, 3)]
    text = (auto.path / "report.md").read_text()
    assert "| proxy:r.jina.ai (mined) |" in text and "Rows marked *(mined)* were not configured" in text
    # extract again without auto_techniques: the mined file goes, and so do the techniques
    shutil.copytree(auto.path, tmp_path / "run")
    run = RunDir(tmp_path / "run")
    prof = json.loads((run.path / "profile.json").read_text())
    (tmp_path / "off.json").write_text(json.dumps({"auto_techniques": False}))
    (run.path / "profile.json").write_text(json.dumps(prof | {"config_path": str(tmp_path / "off.json")}))
    pipeline.extract(run)
    assert not (run.path / "mined_techniques.json").exists() and not pipeline.load_config(run).techniques


def test_a6_gains_only_the_url_domains_row(runs):
    """Distinct agents per domain beside the exact URLs, naming the most shared; no interval note without one."""
    plain, _ = runs
    a6 = json.loads((plain.path / "summary.json").read_text())["A6"]
    assert a6["urls"]["repeated"] == 1   # only the census query repeats exactly
    dom = a6["url domains"]
    assert (dom["facts"], dom["repeated"], dom["max_hops"]) == (3, 3, None)   # r.jina.ai 4, census 3, tinyurl 5 agents
    assert dom["mean_repeaters"] == pytest.approx((3 + 2 + 4) / 3) and "mean_repeaters_lo" not in dom
    assert dom["top"] == [{"domain": "tinyurl.example", "agents": 5}, {"domain": "r.jina.ai", "agents": 4},
                          {"domain": "api.census.example", "agents": 3}]
    text = (plain.path / "report.md").read_text()
    assert "| url domains | 3.00 | 3.00 | 3.00 | 3.80 |  |  |" in text
    assert ("(3 domains, 3 posted by two or more agents; most shared: `tinyurl.example` 5 agents, `r.jina.ai` 4 agents, "
            "`api.census.example` 3 agents)") in text
    assert "it has no relay hops." in text and "resamples domains" not in text


@pytest.mark.parametrize("name", ["suggest_plain", "synth_seed2"])
def test_a_default_run_is_as_it_was_before_suggest(name, runs, synth_run):
    """Every summary value but A6's url domains row equals the summary written before ``suggest`` and
    ``auto_techniques`` existed (``tests/golden``, with the sections later changes added: the skip lines and the
    integrity count), within a relative 1e-4 (fixed-effects estimates move in the sixth digit between processes).
    After an intended change, write the run's ``summary.json`` without ``url domains`` over the golden file."""
    run = {"suggest_plain": runs[0], "synth_seed2": synth_run[0]}[name]
    now = json.loads((run.path / "summary.json").read_text())
    now["A6"].pop("url domains", None)
    d = compare.summary_diff(json.loads((GOLDEN / f"{name}_summary.json").read_text()), now, tol=1e-4)
    assert d["n_compared"] > 40 and not any(d[k] for k in ("sections_only_a", "sections_only_b", "rows_only_a",
                                                           "rows_only_b", "changed")), d
