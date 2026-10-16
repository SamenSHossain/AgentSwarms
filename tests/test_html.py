"""report.html: the Markdown subset the reports write, escaped before any formatting, as one self-contained page
with headline tiles and a fact explorer built from query.fact_lineage and the agent profiles."""

import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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


@pytest.mark.parametrize("path", sorted((ROOT / "results").glob("*/summary*.json")), ids=lambda p: p.parent.name)
def test_tiles_read_every_committed_summary(path):
    """``swarmprov html`` re-renders a run written by an earlier version, so the tiles read what its summary has."""
    s = json.loads(path.read_text())
    tiles = {lab: (v, note) for v, lab, note in html.tiles(s)}
    if a1 := s.get("A1"):
        assert tiles["independent lookups"][0] == f"≤ {a1['n_independent'] + (a1.get('n_self_originated') or 0):,}"


def test_the_independent_tile_counts_self_originated_answers_apart():
    a1 = {"n_answers": 5, "n_agents": 5, "share_exposed": 1.0, "n_independent": 0}
    assert ("≤ 0", "independent lookups", "nothing carrying the answer was public") in html.tiles({"A1": a1})   # older
    a1 |= {"n_self_originated": 3, "n_independent_at_most": 3}
    assert ("≤ 3", "independent lookups", "0 with nothing public, 3 first posted by the agent itself") \
        in html.tiles({"A1": a1})


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
    """Every fact, the slots' included, is what fact_lineage gives on the run's whole tables."""
    run, _, d, _ = page
    tables = query.lineage_tables(run)
    for f in d["facts"]:
        (lin,) = [x for x in query.fact_lineage(run, f["family"], f["item"], f["value"], tables=tables)
                  if x["value"] == f["value"]]
        assert f["carriers"] == lin["carriers"] and len(f["h"]) == len(lin["hops"]) == f["carriers"] - 1
        assert d["agents"][f["o"][0]] == lin["origin"]["agent"]
        assert [[d["agents"][h[0]], d["agents"][h[1]], h[2], d["posts"][h[6]][0], h[7]] for h in f["h"]] \
            == [[h["src"], h["dst"], int(h["kind"] == "relay"), h["event_id"], h["depth"]] for h in lin["hops"]]
    ch = run.read("chains_merged")
    assert sum(f["kind"] == "task" for f in d["facts"]) \
        == min(html.MAX_FACTS, int(((ch["n_agents"] > 1) & (ch["family"] != "url")).sum()))


def test_each_fact_is_traced_from_its_own_rows(synth_run, monkeypatch):
    """The explorer groups the tables by fact once and hands fact_lineage only that fact's rows (tracing each fact
    on the whole tables took 28 s on a 30,000-post chat)."""
    run = synth_run[0]
    calls, lineage = [], query.fact_lineage

    def spy(run, family, item, value, mapping, tables):
        calls.append({n: {tuple(k) for k in t[["family", "item", c]].itertuples(index=False)}
                      for n, t, c in [("chains", tables["chains"], "value_key"), ("edges", tables["edges"], "value_norm"),
                                      ("mentions", tables["mentions"], "value_key")]} | {"key": (family, item, value)})
        return lineage(run, family, item, value, mapping, tables)
    monkeypatch.setattr(query, "fact_lineage", spy)
    html.explorer_data(run, summary=json.loads((run.path / "summary.json").read_text()))
    assert len(calls) > 10 and all(c[n] <= {c["key"]} for c in calls for n in ("chains", "edges", "mentions"))
    assert all(c["chains"] == {c["key"]} for c in calls if c["edges"])


def test_the_explorer_stops_tracing_past_its_byte_cap(synth_run, tmp_path, monkeypatch):
    """Past ``MAX_BYTES`` of facts and posts no further fact is traced (most carried first); the featured slots stay
    whole, and the page says how many facts are left out."""
    d = tmp_path / "run"
    shutil.copytree(synth_run[0].path, d)
    run, s = RunDir(d), json.loads((d / "summary.json").read_text())
    full = html.explorer_data(run, summary=s)
    monkeypatch.setattr(html, "MAX_BYTES", 3_000)
    cut = html.explorer_data(run, summary=s)
    url = run.read("chains_merged").query("n_agents > 1")["family"] == "url"
    picked = min(html.MAX_FACTS, int((~url).sum())) + min(html.MAX_URLS, int(url.sum()))
    kept = picked - cut["left_out"]   # the facts traced before the slots' own
    assert full["left_out"] == 0 and 1 <= kept < picked and cut["facts"][:kept] == full["facts"][:kept]

    def used(facts):   # their JSON and that of the posts they cite, as the page holds them
        cited = {k for f in facts for k in [f["o"][2], *(h[6] for h in f["h"])] if k >= 0}
        return sum(len(html._script_json(x).encode()) for x in [*facts, *(cut["posts"][k] for k in cited)])
    assert used(cut["facts"][:kept - 1]) <= 3_000 < used(cut["facts"][:kept])
    variants = [[(v["value"], v["fact"] is None, len(v["c"])) for v in x["variants"]] for x in full["slots"]]
    assert [[(v["value"], v["fact"] is None, len(v["c"])) for v in x["variants"]] for x in cut["slots"]] == variants
    text = html.build_html(run).read_text()
    assert f"until their hops and posts take 0.003 MB of the page; {cut['left_out']:,} more are left out." in text


def test_figures_are_inlined_with_a_palette(tmp_path):
    """A figure goes into the page re-encoded with a 256-colour palette: well under the file's bytes, the same size
    and nearly the same pixels."""
    from swarmprov.plotting import plt
    (tmp_path / "figures").mkdir()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(30), [x * x for x in range(30)], label="carriers")
    ax.fill_between(range(30), 0, [x * 20 for x in range(30)], alpha=0.3)
    ax.set_title("Who supplied whom")
    ax.legend()
    fig.savefig(tmp_path / "figures" / "a.png", dpi=150)
    plt.close(fig)
    raw = (tmp_path / "figures" / "a.png").read_bytes()
    png = base64.b64decode(re.search(r'base64,([^"]+)"', html.md_to_html("![a](figures/a.png)\n", tmp_path)).group(1))
    with Image.open(io.BytesIO(png)) as q, Image.open(io.BytesIO(raw)) as o:
        assert q.mode == "P" and q.size == o.size and len(png) < 0.6 * len(raw)
        assert np.abs(np.asarray(q.convert("RGBA"), float) - np.asarray(o.convert("RGBA"), float)).mean() < 1
    assert html._palette(b"not a png") == b"not a png" and html._palette(PNG) == PNG   # nothing smaller to write


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
    assert cli.main(["html", str(tmp_path / "missing")]) == 2 and not (tmp_path / "missing").exists()


def _c_locale() -> dict:
    """A console with no UTF-8: the C locale, its coercion to UTF-8 and Python's UTF-8 mode both off."""
    return os.environ | {"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "PYTHONPATH": str(ROOT)}


def test_html_under_a_c_locale(synth_run, tmp_path):
    """Every text file is read and written as UTF-8 whatever the locale: the report's title holds an em dash."""
    d = tmp_path / "run"
    shutil.copytree(synth_run[0].path, d)
    (d / "report.html").unlink()
    out = subprocess.run([sys.executable, "-m", "swarmprov", "html", str(d)], env=_c_locale(), cwd=tmp_path,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "<title>Copied or computed? — provenance report</title>" in (d / "report.html").read_text(encoding="utf-8")


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


browser = pytest.mark.skipif(_node_env() is None, reason="needs node with playwright (headless Chromium)")


def _browse(tmp_path, script: str, *args) -> dict:
    """Run a Playwright script under node; it prints one JSON object."""
    js = tmp_path / "check.js"
    js.write_text("const { chromium } = require('playwright');\n" + script)
    out = subprocess.run([shutil.which("node"), str(js), *map(str, args)], capture_output=True, text=True, timeout=120,
                         env=_node_env())
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@browser
def test_page_runs_in_a_browser(page, tmp_path):
    """The page's script runs without errors offline: the explorer draws a row per carrier, a row shows its post,
    a slot draws, the agent search finds an agent and a header click sorts its table."""
    run, _, d, _ = page
    f = d["facts"][0]
    second = d["agents"][min(f["h"], key=lambda h: h[3])[1]]   # the row after the originator's: the earliest hop
    r = _browse(tmp_path, """
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
})();""", run.path / "report.html", d["agents"][f["h"][0][1]])
    assert r["errs"] == [] and r["sort"] == "ascending"
    assert r["rows"] >= f["carriers"] and r["detail"].startswith(second) and "after the parent's post" in r["detail"]
    assert r["slot"].startswith(d["slots"][0]["slot"]) and "Learned from" in r["card"]


LONG_URL = "https://allorigins.hexlet.app/raw?url=" + "https%3A%2F%2Fapi.example.org%2Fv2%2Fseries%2F" * 6


@pytest.fixture(scope="module")
def long_chain(tmp_path_factory):
    """One value relayed down a board by 90 agents, a minute apart (a 90-row timeline, taller than a screen), three
    of whose posts also cite one long URL with no break in it."""
    from swarmprov import pipeline
    d = tmp_path_factory.mktemp("long")
    posts = [{"ts": f"2026-06-16T{8 + i // 60:02d}:{i % 60:02d}:00Z", "author": f"Scout{i:02d}", "channel": "alpha-board",
              "text": f"R1 CONFIRMED: Ohio arrived {8 + i // 60:02d}:{i % 60:02d}:00, 14s timer; answered 91.04 at +9s."
                      + (f" Source: {LONG_URL}" if i % 30 == 0 else "")} for i in range(90)]
    (d / "t.jsonl").write_text("".join(json.dumps(p) + "\n" for p in posts))
    (d / "c.json").write_text(json.dumps({"name": "t", "families": {"alpha-task": "alpha"},
                                          "extra_items": {"alpha-task": ["Ohio"]}, "bootstrap_reps": 0}))
    return pipeline.run_all(d / "t.jsonl", d / "run", adapter="chat", config=str(d / "c.json"))


@browser
def test_the_detail_stays_in_view_beside_the_active_row(long_chain, tmp_path):
    """Hovering (desktop) or tapping (phone) the rows at 50% and 95% of a timeline taller than the screen, scrolled to
    the middle of the viewport: the detail box is inside the viewport and names the carrier under the pointer, whose
    band holds the pointer; at phone width, where rows have no labels, the band names the carrier too."""
    r = _browse(tmp_path, """
(async () => {
  const b = await chromium.launch(), out = {};
  for (const [name, opt] of [["desktop", {viewport: {width: 1440, height: 900}}],
                             ["phone", {viewport: {width: 390, height: 844}, hasTouch: true, isMobile: true}]]) {
    const p = await (await b.newContext({offline: true, ...opt})).newPage(), errs = [], rows = [];
    p.on('pageerror', e => errs.push(e.message));
    await p.goto('file://' + process.argv[2]);
    for (const f of [0.5, 0.95]) {
      const [x, y, top] = await p.evaluate(f => { const svg = () => document.querySelector('#fx-chart svg').getBoundingClientRect();
        window.scrollTo(0, scrollY + svg().top + f * svg().height - innerHeight / 2);
        const r = svg(); return [r.left + r.width / 2, r.top + f * r.height, r.top]; }, f);
      if (opt.hasTouch) await p.touchscreen.tap(x, y); else await p.mouse.move(x, y);
      rows.push(await p.evaluate(([y, top]) => { const box = e => { const r = e.getBoundingClientRect(); return [r.top, r.bottom]; };
        const d = document.getElementById('fx-detail'), tag = document.querySelector('#fx-chart svg > text[visibility]');
        return {y, top, vh: innerHeight, detail: box(d), band: box(document.querySelector('#fx-chart svg > rect[visibility]')),
                who: d.querySelector('button') && d.querySelector('button').textContent,
                tag: tag && tag.getAttribute('visibility') === 'visible' ? tag.textContent : null}; }, [y, top]));
    }
    out[name] = {errs, rows};
  }
  console.log(JSON.stringify(out)); await b.close();
})();""", long_chain.path / "report.html")
    for name, v in r.items():
        assert v["errs"] == []
        for row in v["rows"]:
            assert row["top"] < 0   # the chart starts above the screen
            assert 0 <= row["detail"][0] and row["detail"][1] <= row["vh"], (name, row)
            assert row["band"][0] - 1 <= row["y"] <= row["band"][1] + 1 and re.fullmatch(r"scout\d\d", row["who"])
            assert row["tag"] == (row["who"] if name == "phone" else None)
        assert v["rows"][0]["who"] < v["rows"][1]["who"]   # rows in order of first post: scout00, scout01, ...


@pytest.fixture(scope="module")
def corpus_run(tmp_path_factory):
    """The cross-site corpus run (its explorer opens on a URL fact)."""
    if not (ROOT / "data" / "raw2" / "records.jsonl").exists():
        pytest.skip("cross-site corpus not present")
    from swarmprov import pipeline
    return pipeline.run_all(ROOT / "data" / "raw2", tmp_path_factory.mktemp("corpus") / "run")


@browser
@pytest.mark.parametrize("which", ["long_chain", "corpus_run"])
def test_no_sideways_scroll_at_phone_width(which, request, tmp_path):
    """At 390 px the page is as wide as the screen on load and after picking each URL fact, whose heading is one long
    unbroken string."""
    run = request.getfixturevalue(which)
    r = _browse(tmp_path, """
(async () => {
  const b = await chromium.launch(), p = await (await b.newContext({offline: true, viewport: {width: 390, height: 844}})).newPage();
  await p.goto('file://' + process.argv[2]);
  const width = () => p.evaluate(() => document.documentElement.scrollWidth), widths = [await width()];
  const urls = () => p.$$('xpath=//ul[@id="fx-list"]//button[small[starts-with(., "URL")]]');
  for (let i = 0; i < (await urls()).length; i++) { await (await urls())[i].click(); widths.push(await width()); }
  console.log(JSON.stringify({vw: await p.evaluate(() => innerWidth), widths})); await b.close();
})();""", run.path / "report.html")
    assert len(r["widths"]) >= 2 and set(r["widths"]) == {r["vw"]} == {390}


@pytest.fixture(scope="module")
def seq_chain(tmp_path_factory):
    """A message list with no timestamps (a sequence clock): four agents post one value in turn."""
    from swarmprov import pipeline
    d = tmp_path_factory.mktemp("seq")
    msgs = [{"role": "user", "content": "Find the 2023 value for Ohio."}] + [
        {"role": "assistant", "name": f"Scout{i}", "content": "R1 CONFIRMED: Ohio arrived; answered 91.04."} for i in range(4)]
    (d / "t.json").write_text(json.dumps(msgs))
    return pipeline.run_all(d / "t.json", d / "run")


@browser
def test_a_sequence_clock_axis_counts_whole_messages(seq_chain, tmp_path):
    """Four positions across a wide chart: ticks at whole message positions (not 1.5), under the label 'message
    position'."""
    r = _browse(tmp_path, """
(async () => {
  const b = await chromium.launch(), p = await (await b.newContext({offline: true})).newPage();
  await p.goto('file://' + process.argv[2]);
  console.log(JSON.stringify({labels: await p.$$eval('#fx-chart svg > text:not([visibility])', ts => ts.map(t => t.textContent))}));
  await b.close();
})();""", seq_chain.path / "report.html")
    *ticks, axis = r["labels"]
    assert axis == "message position" and ticks == ["1", "2", "3", "4"]


def test_a_directory_without_graph_tables_says_why_it_has_no_explorer(synth_run, tmp_path, capsys):
    """A committed results/ directory keeps the report, summary and figures but no graph tables: the page puts one
    line where the explorer would be, after the key findings, and the command prints it."""
    d = tmp_path / "results"
    d.mkdir()
    for name in ("report.md", "summary.json", "figures"):
        (shutil.copytree if name == "figures" else shutil.copy)(synth_run[0].path / name, d / name)
    assert cli.main(["html", str(d)]) == 0
    assert f"no fact explorer ({html.NO_TABLES})" in capsys.readouterr().out
    text = (d / "report.html").read_text()
    assert f'<h2>Fact explorer</h2><p class="missing">No fact explorer: {html.NO_TABLES}.</p></section>' in text
    assert re.findall(r"<h2>(.*?)</h2>", text)[:2] == ["Key findings", "Fact explorer"] and 'id="fx-data"' not in text


def test_a_run_with_no_shared_fact_says_so(synth_run, tmp_path, monkeypatch):
    run = RunDir(tmp_path / "run")
    for name in ("report.md", "summary.json"):
        shutil.copy(synth_run[0].path / name, run.path / name)
    monkeypatch.setattr(html, "explorer_data", lambda *a: {"facts": [], "slots": []})
    text = html.build_html(run).read_text()
    assert "No fact explorer: no value was posted by two or more agents and A4 features no slot." in text
    assert 'id="fx-data"' not in text
