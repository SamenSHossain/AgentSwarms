"""``swarmprov suggest`` and ``auto_techniques``: the precise URL-wrap test, the mined techniques and their evidence,
channel families and items, A3 on mined techniques labelled as such, and A6's url domains row, which is the only
change a run without ``auto_techniques`` sees."""

import json
import shutil

import pandas as pd
import pytest

from swarmprov import adapters, cli, pipeline, report, suggest
from swarmprov.adapters import ADAPTERS, AdapterConfig
from swarmprov.analysis import structure
from swarmprov.schema import RunDir

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
             ("r3", "research", "Median Income was 54,000 last year"), ("r5", "review", "Median Income: 54,100 here")]
    posts += [(f"r{1 + i % 5}", ("research", "review")[i % 2], f"status note {i}") for i in range(10)]
    return [{"id": f"m{i}", "ts": (T0 + pd.Timedelta(minutes=7 * i)).isoformat(), "author": a, "channel": c, "text": t}
            for i, (a, c, t) in enumerate(posts)]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("suggest")
    (d / "chat.jsonl").write_text("\n".join(json.dumps(m) for m in _messages()))
    (d / "auto.json").write_text(json.dumps({"auto_techniques": True}))
    plain = pipeline.run_all(d / "chat.jsonl", d / "plain", adapter="chat")
    auto = pipeline.run_all(d / "chat.jsonl", d / "auto", adapter="chat", config=str(d / "auto.json"))
    return plain, auto


def test_the_wrap_test_is_precise():
    for u in ("https://r.jina.ai/https://example.com/a", "https://x.dev/?url=https%3A%2F%2Fexample.com",
              "https://md.succ.ai/https%253A%252F%252Fexample.com", "https://markdown.new/example.com/page",
              "https://md.succ.ai//www.sec.gov/files/x.json", "https://p.io/raw?url=www.sec.gov%2Ffiles"):
        assert suggest.wraps(u), u
    for u in ("https://wikiservice.at/dse/wiki.cgi?action=browse&id=StartSeite",
              "https://api.datausa.io/tesseract/data.jsonrecords?cube=acs&drilldowns=Year",
              "https://example.com/index.html", "https://site.org/docs/example.com/x",   # a .tld/ further in is no wrap
              "https://example.com/example.com/x", "https://p.io/?q=foo.bar", "https://tinyurl.example/abc1"):
        assert not suggest.wraps(u), u


def test_suggest_writes_a_reviewable_config(runs, tmp_path, capsys):
    plain, _ = runs
    out = tmp_path / "s.json"
    assert cli.main(["suggest", str(plain.path), "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    cfg = json.loads(out.read_text())
    assert [t["name"] for t in cfg["techniques"]] == ["proxy:r.jina.ai"]
    assert cfg["techniques"][0]["description"].startswith(suggest.MINED)
    assert cfg["families"] == {"research": "^research$", "review": "^review$"}
    assert cfg["extra_items"] == {"all": ["Median Income"]}
    lines = {ln.split()[0]: ln for ln in printed.splitlines() if ln.startswith("  ") and "." in ln.split()[0]}
    assert lines["r.jina.ai"].split()[1:5] == ["4", "4", "4", "100%"] and "technique proxy:r.jina.ai" in lines["r.jina.ai"]
    assert "40%" in lines["tinyurl.example"] and "near the threshold" in lines["tinyurl.example"]
    assert "data source" in lines["api.census.example"] and "not applied" in printed
    loaded = AdapterConfig.from_file(out, base=adapters.get("chat").config())
    assert loaded.techniques[0].name == "proxy:r.jina.ai" and loaded.techniques[0].families == ()
    assert cli.main(["suggest", str(tmp_path / "none")]) == 2 and not (tmp_path / "none").exists()


def test_suggest_keeps_what_the_config_sets(runs, tmp_path, capsys):
    shutil.copytree(runs[0].path, tmp_path / "run")
    cfg = {"techniques": [{"name": "reader", "pattern": "jina"}], "families": {"all": "."}, "extra_items": {"all": ["Foo"]}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    prof = json.loads((tmp_path / "run" / "profile.json").read_text())
    (tmp_path / "run" / "profile.json").write_text(json.dumps(prof | {"config_path": str(tmp_path / "cfg.json")}))
    assert suggest.main(str(tmp_path / "run"), str(tmp_path / "s.json")) == 0
    assert "already configured (reader)" in capsys.readouterr().out
    assert json.loads((tmp_path / "s.json").read_text()) == {
        "families": {"all": "."}, "techniques": [{"name": "reader", "pattern": "jina"}],
        "extra_items": {"all": ["Foo", "Median Income"]}}


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


def test_a6_gains_only_the_url_domains_row(runs, monkeypatch):
    """Distinct agents per domain beside the exact URLs; dropping the row leaves the report as it was before."""
    plain, _ = runs
    a6 = json.loads((plain.path / "summary.json").read_text())["A6"]
    assert a6["urls"]["repeated"] == 1   # only the census query repeats exactly
    dom = a6["url domains"]
    assert (dom["facts"], dom["repeated"], dom["max_hops"]) == (3, 3, None)   # r.jina.ai 4, census 3, tinyurl 5 agents
    assert dom["mean_repeaters"] == pytest.approx((3 + 2 + 4) / 3)
    reach = structure.reach
    monkeypatch.setattr(structure, "reach", lambda *a, **k: {n: r for n, r in reach(*a, **k).items() if n != "url domains"})
    without = report.build(plain, boot=0).splitlines()
    monkeypatch.undo()
    with_row = report.build(plain, boot=0).splitlines()
    added = [ln for ln in with_row if ln not in without]
    assert [ln for ln in without if ln not in with_row] == [] and len(added) == 2
    assert added[0].startswith("| url domains | 3.00 | 3.00 | 3.00 | 3.80 |  |  |")
    assert added[1].startswith("The `url domains` row counts the distinct agents that posted any URL on each domain "
                               "(3 domains, 3 posted by two or more agents)")
