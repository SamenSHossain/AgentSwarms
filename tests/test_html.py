"""report.html: the Markdown subset the reports write, escaped before any formatting, as one self-contained page
with headline tiles and a fact explorer built from query.fact_lineage and the agent profiles."""

import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from swarmprov import cli, html, query
from swarmprov.schema import RunDir

ROOT = Path(__file__).resolve().parents[1]
DATA = re.compile(r'<script type="application/json" id="fx-data">(.*?)</script>', re.S)
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def test_table_cells_keep_escaped_pipes_and_escape_html(tmp_path):
    out = html.md_to_html("| slot | note |\n|---|---|\n| a \\| b | `<script>alert('x')</script>` and <b>bold</b> |\n",
                          tmp_path)
    assert "<td>a | b</td>" in out
    assert "<code>&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;</code>" in out and "&lt;b&gt;bold&lt;/b&gt;" in out
    assert "<script" not in out and "<b>" not in out


def test_inline_formatting_after_escaping(tmp_path):
    out = html.md_to_html("**1,108 answers** and *instant only*, _Sensitivity_: D_visible and y_consensus, "
                          "`**not bold**` & 3 < 4\n", tmp_path)
    assert "<strong>1,108 answers</strong>" in out and "<em>instant only</em>" in out and "<em>Sensitivity</em>" in out
    assert "D_visible and y_consensus" in out   # an underscore inside a word is no italic
    assert "<code>**not bold**</code>" in out and "&amp; 3 &lt; 4" in out


def test_headings_sections_and_bullets(tmp_path):
    out = html.md_to_html("# Title\n\n## A1. Provenance split\n## A2. Does it?\n\n- one **x**\n- two\n\ntext\n", tmp_path)
    assert '<section id="a1-provenance-split">\n<h2>A1. Provenance split</h2>' in out
    assert '</section><section id="a2-does-it">\n<h2>A2. Does it?</h2>' in out   # a heading straight after a heading
    assert "<ul><li>one <strong>x</strong></li><li>two</li></ul>" in out and out.endswith("</section>")


def test_figures_become_data_uris_and_nothing_else_is_read(tmp_path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "a1.png").write_bytes(PNG)
    (tmp_path / "secret.png").write_bytes(PNG)
    out = html.md_to_html("![provenance](figures/a1.png)\n\n![gone](figures/none.png)\n\n![up](figures/../secret.png)\n",
                          tmp_path)
    assert f'<img alt="provenance" src="data:image/png;base64,{base64.b64encode(PNG).decode()}">' in out
    assert "Figure not found: figures/none.png" in out and "Figure not found: figures/../secret.png" in out
    assert out.count("<img") == 1


@pytest.mark.parametrize("path", sorted((ROOT / "results").glob("*/report.md")), ids=lambda p: p.parent.name)
def test_every_committed_report_converts(path):
    md = path.read_text()
    out = html.md_to_html(md, path.parent)
    assert out.count("<h2>") == sum(line.startswith("## ") for line in md.splitlines())
    assert out.count("<table>") == len(re.findall(r"^\|(?:---\|)+$", md, re.M))
    assert "<script" not in out and out.count("<img") == out.count('src="data:image/png;base64,')


def test_script_json_cannot_close_or_hide_its_element():
    s = html._script_json({"x": "</script><!--<script>alert(1)"})
    assert "</" not in s and "<!--" not in s and json.loads(s) == {"x": "</script><!--<script>alert(1)"}


@pytest.fixture(scope="module")
def page(synth_run):
    run = synth_run[0]
    text = (run.path / "report.html").read_text()
    return run, text, json.loads(DATA.search(text).group(1)), json.loads((run.path / "summary.json").read_text())


def test_report_builds_a_self_contained_page(page):
    run, text, _, s = page
    md = (run.path / "report.md").read_text()
    assert 'src="figures/' not in text and not re.findall(r'(?:src|href)="(?:https?:)?//', text)
    assert text.count("<img") == md.count("](figures/") and text.count("data:image/png;base64,") == text.count("<img")
    heads = [h.replace("&", "&amp;") for h in re.findall(r"^## (.*)$", md, re.M)]
    assert re.findall(r"<h2>(.*?)</h2>", text) == heads[:1] + ["Fact explorer"] + heads[1:]   # after the findings
    assert "Inferred paths, not proven ones" in text
    assert f'<div class="v">{s["A1"]["n_answers"]:,}</div><div class="l">answers</div>' in text
    assert '<div class="v word">identified</div>' in text   # the seed-2 swarm's A2 effect


def test_explorer_facts_are_fact_lineage(page):
    run, _, d, _ = page
    f = d["facts"][0]
    (lin,) = query.fact_lineage(run, f["family"], f["item"], f["value"])
    assert f["carriers"] == lin["carriers"] and len(f["h"]) == len(lin["hops"]) == f["carriers"] - 1
    assert [d["agents"][h[1]] for h in f["h"]] == [h["dst"] for h in lin["hops"]]
    assert [h[2] for h in f["h"]] == [int(h["kind"] == "relay") for h in lin["hops"]]
    assert d["posts"][f["h"][0][6]][0] == lin["hops"][0]["event_id"]
    ch = run.read("chains_merged")
    assert len(d["facts"]) == min(html.MAX_FACTS, int(((ch["n_agents"] > 1) & (ch["family"] != "url")).sum()))


def test_explorer_posts_once_per_item_and_short(page):
    _, _, d, _ = page
    assert all(len(p[1]) <= html.EXCERPT + 1 for p in d["posts"])
    cited: dict = {}   # post index -> the (event, item) pairs that cite it
    for f in d["facts"]:
        for k in [f["o"][2], *(h[6] for h in f["h"])]:
            if k >= 0:
                cited.setdefault(k, set()).add((d["posts"][k][0], f["item"]))
    assert set(cited) == set(range(len(d["posts"]))) and all(len(v) == 1 for v in cited.values())
    assert len(set().union(*cited.values())) == len(d["posts"])


@pytest.fixture(scope="module")
def digest(tmp_path_factory):
    """Three posts on one page: Ohio's and Texas's first reports, then a digest that carries both values with the
    items further apart than an excerpt is long."""
    from swarmprov import pipeline
    d = tmp_path_factory.mktemp("digest")
    filler = " ".join(["the board looks steady and nobody disputes this reading"] * 4)
    posts = [("08:00", "AgentAlpha", "R1 CONFIRMED: Ohio arrived 07:59:50, 14s timer; answered 91.04 at +9s."),
             ("08:05", "AgentBeta", "R1 CONFIRMED: Texas arrived 08:04:50, 14s timer; answered 90.30 at +9s."),
             ("08:10", "AgentGamma", f"R1 CONFIRMED: Ohio arrived 08:09:50, 14s timer; answered 91.04 at +9s. {filler}. "
                                     "Texas 90.30 is on the board too.")]
    (d / "t.jsonl").write_text("".join(json.dumps({"ts": f"2026-06-16T{t}:00Z", "author": a, "channel": "alpha-board",
                                                   "text": x}) + "\n" for t, a, x in posts))
    (d / "c.json").write_text(json.dumps({"name": "t", "families": {"alpha-task": "alpha"},
                                          "extra_items": {"alpha-task": ["Ohio", "Texas"]}}))
    run = RunDir(d / "run")
    pipeline.ingest(d / "t.jsonl", run, "chat", str(d / "c.json"))
    pipeline.extract(run)
    pipeline.expose(run)
    pipeline.build_graph(run)
    return html.explorer_data(run)


def test_a_post_carrying_two_facts_shows_each_its_own_item(digest):
    assert sorted(f["item"] for f in digest["facts"]) == ["Ohio", "Texas"]
    for f in digest["facts"]:
        (hop,) = f["h"]   # the digest relays each value from its first report
        assert digest["agents"][hop[1]] == "agentgamma" and f["item"] in digest["posts"][hop[6]][1]
    ids = [p[0] for p in digest["posts"]]
    assert len(ids) == 4 and len(set(ids)) == 3   # the digest once per item


def test_explorer_slots_are_the_featured_a4_slots(page):
    _, _, d, s = page
    feat = [r for r in s["A4"]["rows"] if r["featured"]]
    assert [x["slot"] for x in d["slots"]] == [r["slot"] for r in feat]
    for x, r in zip(d["slots"], feat):
        assert x["status"] == r["status"] and x["winner"] == r["winner"]
        assert {v["value"] for v in x["variants"]} == set(r["tail counts"]) | {r["first variant"]} - {None}
        for v in x["variants"]:   # a value slot's variant links its relay tree, whose carriers are the slot's
            f = d["facts"][v["fact"]]
            assert f["value"] == v["value"] and f["carriers"] == len(v["c"])


def test_agent_links_follow_the_edges(page):
    run, _, d, _ = page
    edges = run.read("edges_merged")
    edges = edges[edges["kind"] != "reply"]
    rows = {d["agents"][r[0]]: r for r in d["people"]["rows"]}
    n = len(d["people"]["cols"])
    prof = run.read("agent_profiles_merged").set_index("label")
    for label, r in list(rows.items())[:15]:
        into = edges[edges["dst"] == label]
        assert r[n + 1] == into["src"].nunique()
        top = r[n + 2][0] if r[n + 2] else None
        if top:   # the first listed source has the most edges into the agent, kinds counted as in the edges
            k = into[into["src"] == d["agents"][top[0]]]["kind"].value_counts()
            assert top[1:] == [int(k.get(x, 0)) for x in d["people"]["kinds"]]
            assert sum(top[1:]) == into.groupby("src").size().max()
        assert r[1 + d["people"]["cols"].index("n_answers")] == prof.loc[label, "n_answers"]


def test_html_command_renders_a_report_again(synth_run, tmp_path):
    d = tmp_path / "run"
    shutil.copytree(synth_run[0].path, d)
    (d / "report.html").unlink()
    assert cli.main(["html", str(d)]) == 0 and (d / "report.html").exists()
    assert cli.main(["html", str(d), "--mapping", "strict"]) == 1   # no strict report to render
    assert cli.main(["html", str(tmp_path / "missing")]) == 1 and not (tmp_path / "missing").exists()


def test_a_page_without_graph_tables(tmp_path):
    """A cross-site or context report: the Markdown and no tiles, no explorer, no data block."""
    run = RunDir(tmp_path / "x")
    (run.path / "report.md").write_text("# Cross-site view\n\n12 posts across 2 surfaces.\n\n## Posts per surface\n\n"
                                        "| site | posts |\n|---|---|\n| dse | 10 |\n| rmn.re | 2 |\n")
    (run.path / "summary.json").write_text(json.dumps({"n_events": 12}))
    text = html.build_html(run).read_text()
    assert '<p class="lede">12 posts across 2 surfaces.</p>' in text and "<td class=num>10</td>" in text
    assert '<div class="tiles">' not in text and 'id="fx-data">' not in text and 'id="fact-explorer"' not in text


def test_clear_removes_the_page_and_headline(synth_run, tmp_path):
    d = tmp_path / "run"
    shutil.copytree(synth_run[0].path, d)
    RunDir(d).clear()
    assert not (d / "report.html").exists() and not (d / "headline.md").exists()


def _node_env() -> dict | None:
    """The environment for node with the global playwright package, or None without node or playwright."""
    node = shutil.which("node")
    if not node:
        return None
    env = {**os.environ, "NODE_PATH": str(Path(node).resolve().parents[1] / "lib" / "node_modules")}
    ok = subprocess.run([node, "-e", "require('playwright')"], capture_output=True, env=env)
    return env if ok.returncode == 0 else None


@pytest.mark.skipif(_node_env() is None, reason="needs node with playwright (headless Chromium)")
def test_page_runs_in_a_browser(page, tmp_path):
    """The page's script runs without errors offline: the explorer draws a row per carrier, a row shows its post,
    a slot draws, the agent search finds an agent and a header click sorts its table."""
    run, _, d, _ = page
    f = d["facts"][0]
    second = d["agents"][min(f["h"], key=lambda h: h[3])[1]]   # the row after the originator's: the earliest hop
    js = tmp_path / "check.js"
    js.write_text("""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(), p = await (await b.newContext({offline: true})).newPage(), errs = [];
  p.on('pageerror', e => errs.push(e.message));
  await p.goto('file://' + process.argv[2]);
  const rows = await p.$$eval('#fx-chart svg circle[stroke-width]', c => c.length);
  await p.focus('#fx-chart svg'); await p.keyboard.press('ArrowDown'); await p.keyboard.press('ArrowDown');
  const detail = await p.textContent('#fx-detail');
  await p.click('#fx-list button'); const slot = await p.textContent('#fx-head');
  await p.fill('#ag-q', process.argv[3]); const card = await p.textContent('#ag-card');
  const th = await p.$('#data-capabilities th'); await th.click(); const sort = await th.getAttribute('aria-sort');
  console.log(JSON.stringify({errs, rows, detail, slot, card, sort})); await b.close();
})();""")
    out = subprocess.run([shutil.which("node"), str(js), str(run.path / "report.html"), d["agents"][f["h"][0][1]]],
                         capture_output=True, text=True, timeout=120, env=_node_env())
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    assert r["errs"] == [] and r["sort"] == "ascending"
    assert r["rows"] >= f["carriers"] and r["detail"].startswith(second) and "after the parent's post" in r["detail"]
    assert r["slot"].startswith(d["slots"][0]["slot"]) and "Learned from" in r["card"]
