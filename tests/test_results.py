"""The committed results/ directories: every figure a report links is there (and none it does not link), the
collectors' 143-surface inventory is cited only where it describes the source, each key-findings block and the wiki's
report.html are what the current code writes from them, and every number the README and docs/DATA.md quote from a run
(the scale of the wiki dump, the headline table, the wiki results table, the validation table, the integrity line, the
cross-site and corpus text, the roster's size, DATA.md's cross-site table) is formatted from the committed results:
the summary.json and profile.json files, the cross-site report and its technique_spread.csv, and the roster.  Only
the held-out split's scores from before the fixes are history no run writes.  Every relative link between the docs
resolves, anchors included."""

import itertools
import json
import math
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from swarmprov import checks, html, report
from swarmprov.analysis import causal, origins
from swarmprov.schema import RunDir

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = (ROOT / "README.md").read_text(encoding="utf-8")
DATA = (ROOT / "docs" / "DATA.md").read_text(encoding="utf-8")
DOCS = [ROOT / p for p in ("README.md", "docs/USAGE.md", "docs/METHODS.md", "docs/DATA.md", "docs/PLAN.md",
                            "tests/README.md", "results/README.md")]
REPORTS = sorted(RESULTS.glob("*/report.md"))
LINK = re.compile(r"\]\(([^)]+\.png)\)")
FX_DATA = re.compile(r'<script type="application/json" id="fx-data">(.*?)</script>', re.S)
MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
NAME = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d)\w+")   # Mar30, sector61, G5, p90: digits inside a name
# digits that name a definition, not a finding
DEFINITIONS = ("95% CI", "top 10", "≥2", "score 0", "clears 0.05")
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
G5 = "datausa-grocery-workforce | G5 (cracked-seed forecast vs observed)"


def _summary(name: str) -> dict:
    return json.loads((RESULTS / name / "summary.json").read_text())


def test_results_hold_the_six_committed_reports():
    assert [p.parent.name for p in REPORTS] == ["corpus", "crosssite", "openai-wiki", "synthetic", "village-context",
                                                "village-goals"]


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.parent.name)
def test_every_figure_link_resolves_and_every_figure_is_linked(path):
    links = set(LINK.findall(path.read_text()))
    assert all((path.parent / f).is_file() for f in links), sorted(f for f in links if not (path.parent / f).is_file())
    figures = {f"figures/{f.name}" for f in (path.parent / "figures").glob("*.png")}
    assert figures == links


def test_the_committed_wiki_page_is_what_the_current_code_writes(synth_run, tmp_path, monkeypatch):
    """results/openai-wiki keeps its run's report.html, fact explorer included, for readers without the dump.  It is
    under 3 MB, and html.build_html given the committed report, summary and figures and the page's own explorer data
    writes it byte for byte: its style, script, tiles, explorer text and body are the current code's.  The explorer
    data needs the graph tables, which results/ does not hold, so it is checked to have the fields explorer_data writes
    now (on the synthetic run)."""
    d = RESULTS / "openai-wiki"
    page = (d / "report.html").read_text(encoding="utf-8")
    assert len(page.encode()) < 3_000_000
    data = json.loads(FX_DATA.search(page).group(1))
    run, _ = synth_run
    assert set(data) == set(html.explorer_data(run, summary=json.loads((run.path / "summary.json").read_text())))
    assert len(data["facts"]) >= 100 and data["slots"]
    shutil.copytree(d, tmp_path / "w", ignore=shutil.ignore_patterns("*.csv", "report.html"))
    monkeypatch.setattr(html, "explorer_data", lambda *a: data)
    assert html.build_html(RunDir(tmp_path / "w")).read_text(encoding="utf-8") == page


def test_the_coverage_inventory_is_cited_only_where_it_describes_the_source():
    cited = {p.parent.name for p in REPORTS if "143 surfaces" in p.read_text()}
    assert cited == {"openai-wiki", "corpus", "crosssite"}


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.parent.name)
def test_committed_key_findings_are_what_the_current_code_writes(path):
    """A wording change in report.key_findings fails here until results/ is regenerated."""
    s = json.loads((path.parent / "summary.json").read_text())
    kf = report.key_findings(s)
    assert (path.parent / "headline.md").exists() == bool(kf)
    if kf:
        assert (path.parent / "headline.md").read_text() == "\n".join(kf)
        assert s["headline"] == [b[2:] for b in kf if b.startswith("- ")]
        assert "\n".join(kf) in path.read_text()


def _table(marker: str, text: str = README) -> dict[str, str]:
    """The first table after ``marker``: {first cell: second cell} for each row below the header."""
    lines = text.split(marker, 1)[1].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    rows = list(itertools.takewhile(lambda line: line.startswith("|"), lines[start:]))[2:]
    return dict(tuple(c.strip() for c in line.strip().strip("|").split(" | ", 1)) for line in rows)


def _numbers(text: str) -> list[str]:
    for d in DEFINITIONS:
        text = text.replace(d, "")
    return NUMBER.findall(NAME.sub("", text))


def _check(text: str, expected: list, what: str):
    found = _numbers(text)
    assert found == expected, f"{what}: README {found}, expected {expected}"


def pct(x) -> str:
    return f"{x * 100:.0f}"


def carriers(row: dict) -> dict[str, str]:
    """'Maryland (RNG forecast) (30), Montana (observed) (11)' -> {'Maryland (RNG forecast)': '30', ...}"""
    return dict(re.findall(r"(.+?) \((\d+)\)(?:, |$)", row["variants (agents)"]))


def late(row: dict) -> list[str]:
    """The winner's late carriers and all late carriers."""
    return [f"{row['tail counts'][row['winner']]}", row["late carriers (winner/total)"].split("/")[1]]


def f2(value: str) -> str:
    return f"{float(value):.2f}"


def _a2_readings(s: dict) -> tuple[list[int], list[dict]]:
    """The switchers of the A2 headline rows read as not (or weakly) identified, and the rows with no effect
    detected (``report.a2_identified``)."""
    _, by = report.a2_identified(s["A2"])
    return [r["agents_with_D_variation"] for r in by[causal.WEAK] + by[causal.NOT_IDENTIFIED]], by[causal.NO_EFFECT]


def _flat_numbers(r: dict) -> list[str]:
    return [f"{r['agents_with_D_variation']}", f"{abs(r['ci_lo']):.2f}", f"{r['ci_hi']:.2f}"]


def _within_a_day(cross: dict, technique: str) -> pd.Series:
    """Hours after the origin at which the other sites first carried ``technique``, those within 24 h."""
    spread = pd.read_csv(RESULTS / "crosssite" / "technique_spread.csv")
    origin = {r["technique"]: r for r in cross["technique_site_summary"]}[technique]["origin_site"]
    hours = spread[(spread["technique"] == technique) & (spread["site"] != origin)]["hours_after_origin"]
    return hours[hours <= 24]


def _headline_expected(s: dict, synth: dict, cross: dict) -> dict[str, list[str]]:
    """Each headline row's numbers in the order the README states them, formatted from the summaries."""
    a1, a5 = s["A1"], s["A5"]
    rule = {r["rule"]: r for r in a5["attribution"] if r["answers"] == "all exposed"}
    span, n_credited = report.top10_range(a5)   # the range the key findings give
    a3 = {r["technique"]: r for r in s["A3"]}["blob-hostname-bypass"]
    a4 = {r["slot"]: r for r in s["A4"]["rows"]}
    g5, hungary = a4[G5], a4["oecd-equity | Hungary"]
    unread, (flat,) = _a2_readings(s)
    proxy = {r["technique"]: r for r in cross["technique_site_summary"]}["proxy-markdown-new"]
    day = _within_a_day(cross, "proxy-markdown-new")
    truth = synth["truth_validation"]["A2"]
    return {
        "How much was already public?": [
            pct(a1["share_exposed"]), *map(pct, a1["share_exposed_ci"]), f"{a1['n_answers']:,}", f"{a1['n_agents']:,}",
            pct(a1["share_exposed_60m"]), f"{a1['median_gap_h']:.1f}",
            *(f"{a1[k]:,}" for k in ("n_independent_at_most", "n_independent", "n_self_originated",
                                     "n_independent_at_most_visible"))],
        "Who supplied it?": [*NUMBER.findall(span), f"{n_credited:,}", pct(rule["first source"]["top10_share"]),
                             pct(rule["latest source"]["top10_share"]), pct(rule["same-page relay parent"]["top10_share"]),
                             f"{rule['same-page relay parent']['n']:,}", f"{a5['items_one_source']}",
                             f"{a5['items_exposed']}"],
        "How fast did tricks spread?": [pct(a3["adoption_share"]), f"{a3['adopters']}", f"{a3['at_risk']}",
                                        f"{a3['median_hours_among_adopters']:.1f}", f"{proxy['sites_within_24h']}",
                                        f"{math.ceil(day.max())}"],
        "Did errors win?": [carriers(g5)[g5["first variant"]], carriers(g5)[g5["winner"]], *late(g5),
                            f2(hungary["first variant"]), carriers(hungary)[hungary["first variant"]],
                            carriers(hungary)[hungary["winner"]], f2(hungary["winner"]),
                            f"{hungary['tail counts'][hungary['winner']]}",
                            f"{hungary['tail counts'][hungary['first variant']]}"],
        "Does availability *cause* copying?": [f"{min(unread)}", f"{max(unread)}", *_flat_numbers(flat),
                                               f"{truth['beta']:.2f}", f"{truth['true_effect']:.2f}"],
    }


def test_readme_headline_quotes_the_committed_summaries():
    s, synth, cross = (_summary(d) for d in ("openai-wiki", "synthetic", "crosssite"))
    rows, expected = _table("## What it found"), _headline_expected(s, synth, cross)
    assert list(rows) == list(expected)
    for question, numbers in expected.items():
        _check(rows[question], numbers, question)
    slots = {r["slot"]: r for r in s["A4"]["rows"]}
    assert slots[G5]["status"] == "overturned" and "overturned it as soon as it appeared" in rows["Did errors win?"]
    assert slots["oecd-equity | Hungary"]["status"] == "contested" and "Hungary stays **contested**" in \
        rows["Did errors win?"]
    proxy = {r["technique"]: r for r in cross["technique_site_summary"]}["proxy-markdown-new"]
    assert len(_within_a_day(cross, "proxy-markdown-new")) == proxy["sites_within_24h"]


def test_readme_a2_wording_follows_the_reading_rule():
    """The headline's A2 row says what report.a2_identified reads from the committed rows: "Not or only weakly
    identified for most rows" while most headline rows are one or the other (some of each) and none is identified,
    "no effect detected" for a row with enough switchers whose CI covers 0; the results table states
    causal.reading's thresholds and that the key findings, which say "Not identified for most rows", fold the
    weakly identified rows in."""
    s = _summary("openai-wiki")
    main, by = report.a2_identified(s["A2"])
    unread = by[causal.WEAK] + by[causal.NOT_IDENTIFIED]
    row = _table("## What it found")["Does availability *cause* copying?"]
    assert not by[causal.IDENTIFIED] and by[causal.WEAK] and by[causal.NOT_IDENTIFIED] and 2 * len(unread) > len(main)
    assert row.startswith("**Not or only weakly identified for most rows**")
    assert bool(by[causal.NO_EFFECT]) == ("no effect is detected" in row)
    rule = _table("## Results on the OpenAI wiki swarm")["**A2 Does availability cause copying?**"]
    assert "Not identified for most rows" in "\n".join(report.key_findings(s)) and \
        "The report's key findings count a weakly identified row as not identified." in rule
    k = causal.MIN_SWITCHERS
    assert [causal.reading(n, -1, 1) for n in (k - 1, k, 2 * k - 1, 2 * k)] == [
        causal.NOT_IDENTIFIED, causal.WEAK, causal.WEAK, causal.NO_EFFECT] and causal.reading(2 * k, 0.1, 1) == \
        causal.IDENTIFIED
    assert all(f"*{w}*" in rule for w in (causal.NOT_IDENTIFIED, causal.WEAK, causal.IDENTIFIED, causal.NO_EFFECT))


def _results_expected(s: dict) -> dict[str, list]:
    """Each row of the README's wiki results table, its numbers in order, formatted from the summary."""
    a1, a4, a5, a6 = s["A1"], s["A4"], s["A5"], s["A6"]
    tc = a1["threshold_curve"]
    curve, at = {c["answers"]: c["share"] for c in tc["curves"]}, tc["hours"].index
    a3 = {r["technique"]: r for r in s["A3"]}
    tip, beat = a3["live-tooltip-value"], a3["heartbeat-beacon"]
    seed, wait = (a3[t]["contagion"] for t in ("rng-seed-crack", "clock-wait-fastforward"))
    slots = {r["slot"]: r for r in a4["rows"]}
    poland, slovak = slots["oecd-equity | Poland"], slots["oecd-equity | Slovak Republic"]
    status = a4["status_counts"]
    tie = next(r for r in a4["rows"] if r["tie"])
    small = next(r for r in a4["rows"] if r["status"] == "contested (small n)")
    disagree = a4["consensus_disagrees"]
    rule = {r["rule"]: r for r in a5["attribution"] if r["answers"] == "all exposed"}
    instant, _ = report.top10_range({"attribution": [dict(r, answers="all exposed") for r in a5["attribution"]
                                                     if r["answers"] == "instant only"]})
    by = a5["origins"]["by_mapping"]

    def share(mapping, origin):
        counts = by[mapping]["first source"]
        return pct(counts[origin] / sum(counts.values()))

    tv, urls = a6["task values"], a6["urls"]
    window = {w["W_s"]: w for w in a6["window_sensitivity"]}
    hour, ever = window[3600.0], window[None]
    return {
        "**A1 Provenance**": [
            pct(curve["all answers"][at(24)]), "6", pct(curve["oecd-equity"][at(24)]), "24",
            f"{a1['n_src_deleted_before_report']:,}", f"{a1['n_src_deleted_before_report'] - a1['n_only_deleted_copies']:,}",
            pct(a1["share_self_prepared"])],
        "**A2 Does availability cause copying?**": [f"{causal.MIN_SWITCHERS}", *[f"{2 * causal.MIN_SWITCHERS}"] * 2],
        "**A3 Diffusion**": [
            f"{tip['adopters']}", f"{tip['at_risk']}", pct(tip["adoption_share"]), f"{tip['median_hours_to_adopt']:.1f}",
            f"{tip['median_hours_to_adopt_lo']:.1f}", f"{tip['median_hours_to_adopt_hi']:.1f}",
            f"{beat['adopters']}", f"{beat['at_risk']:,}", f"{beat['median_hours_among_adopters']:.0f}",
            pct(seed["share_on_carrying"]), pct(wait["share_on_carrying"]), pct(seed["baseline"])],
        "**A4 Errors**": [
            f"{sum(status.values())}", f"{status['overturned']}", f"{status['held']}",
            f"{status['contested'] + status['contested (small n)']}", *map(str, tie["tail counts"].values()),
            small["late carriers (winner/total)"].split("/")[1], f"{a4['rule']['bin_hours']:.0f}",
            f2(poland["winner"]), *late(poland), f2(slovak["winner"]), *late(slovak),
            f"{len(disagree['slots'])}", f"{disagree['n_value_slots']}"],
        "**A5 Structure**": [
            f"{rule['first source']['sources']}", f"{rule['first source']['gini']:.2f}", f"{a5['gini_item_exposures']:.2f}",
            f"{a5['items_exposed']}", f"{rule['latest source']['sources']}", *NUMBER.findall(instant),
            share("merged", "computed"), share("merged", "no answer round observed"),
            share("strict", "no answer round observed"), share("merged", "dissent"), share("merged", origins.POSTED_FIRST)],
        "**A6 Reach**": [
            f"{tv['facts']:,}", f"{tv['repeated']:,}", f"{tv['mean_repeaters']:.1f}", f"{tv['mean_repeaters_lo']:.1f}",
            f"{tv['mean_repeaters_hi']:.1f}", f"{tv['p90_repeaters']:.0f}", f"{urls['facts']:,}", f"{urls['repeated']:,}",
            f"{ever['mean_max_depth']:.1f}", f"{tv['max_hops']}", f"{hour['mean_max_depth']:.1f}",
            f"{hour['unattributed']:,}", f"{hour['relay'] + hour['relay_xchannel'] + hour['unattributed']:,}"],
    }


def test_readme_results_table_quotes_the_committed_summaries():
    s = _summary("openai-wiki")
    rows, expected = _table("## Results on the OpenAI wiki swarm"), _results_expected(s)
    assert list(rows) == list(expected)
    for analysis, numbers in expected.items():
        _check(rows[analysis], numbers, analysis)
    a1, a3, a4 = rows["**A1 Provenance**"], rows["**A3 Diffusion**"], rows["**A4 Errors**"]
    tc = s["A1"]["threshold_curve"]
    assert dict((c["answers"], c["share"]) for c in tc["curves"])["datausa-sector61-state"][tc["hours"].index(6)] == 0
    assert "none of datausa-sector61-state's at 6 h" in a1
    techniques = {r["technique"]: r for r in s["A3"]}
    holm = {t for t, r in techniques.items() if not r["contagion"].get("excluded") and r["contagion"]["p_holm"] < 0.05}
    assert holm == {"rng-seed-crack", "clock-wait-fastforward"} and "Seed-crack and `clock.wait` adoptions" in a3
    assert pct(techniques["rng-seed-crack"]["contagion"]["baseline"]) == \
        pct(techniques["clock-wait-fastforward"]["contagion"]["baseline"])
    statuses = [r["status"] for r in s["A4"]["rows"]]
    assert {k: statuses.count(k) for k in s["A4"]["status_counts"]} == s["A4"]["status_counts"]
    slots = {r["slot"]: r for r in s["A4"]["rows"]}
    contested = [r for r in s["A4"]["rows"] if r["status"].startswith("contested")]
    assert {slots["oecd-equity | Hungary"]["slot"]} | {r["slot"] for r in contested if r["tie"] or r["first seen tie"]
                                                       or r["status"] == "contested (small n)"} == \
        {r["slot"] for r in contested} and "contested: Hungary" in a4
    assert sum(r["first seen tie"] for r in s["A4"]["rows"]) == 1 and "first posted in the same second" in a4
    assert all(slots[f"oecd-equity | {k}"]["hours to overtake"] < s["A4"]["rule"]["bin_hours"]
               for k in ("Poland", "Slovak Republic"))


def test_readme_integrity_line_is_what_check_prints():
    integ = _summary("openai-wiki")["integrity"]
    line = checks.summary_line(integ["n_checks"], integ["n_pass"], sum(f["severity"] == "error" for f in integ["failing"]))
    para = next(p for p in README.split("\n\n") if p.startswith("That section opens with an integrity line"))
    assert f"*{line}*" in para


def _scores(v: dict) -> list[str]:
    """A validation split's row: posts, answer-post precision / recall, then item, value, round and latency accuracy."""
    f = v["fields"]
    return [f"{v['n']}", f"{f['answer_post']['precision']:.2f} / {f['answer_post']['recall']:.2f}",
            *(f"{f[k]['accuracy']:.2f}" for k in ("item", "value", "episode", "latency_class"))]


def test_readme_validation_table_quotes_the_committed_summary():
    """The dev and untouched-test rows are the Quick start run's scores; the held-out row's scores are its first,
    from before the fixes it prompted (history), and the text gives what it scores now."""
    v = _summary("openai-wiki")["validation"]
    rows = {k: [c.strip().replace("*", "") for c in cells.split(" | ")] for k, cells in _table("### Validation").items()}
    dev, held, test = (rows[k] for k in ("dev (used for tuning)", "held-out, scored **before** fixes, then used for tuning",
                                         "**untouched test** (labelled after the last extractor change)"))
    assert dev == _scores(v["validation_dev"])
    *scores, latency = _scores(v["validation_test"])
    assert test == [*scores, f"{latency} (n={v['validation_test']['fields']['latency_class']['support']})"]   # a small n
    assert held[0] == f"{v['validation_heldout']['n']}"
    now = v["validation_heldout"]["fields"]
    assert all(x == 1.0 for f in now.values() for k, x in f.items() if k != "support") and \
        "the held-out split, tuned on since, now scores 1.00 on every field" in README


def test_readme_scale_numbers_quote_their_sources():
    """The wiki dump's revisions and posts (the committed profile.json) and the roster's goal assignments (its rows)."""
    prof = json.loads((RESULTS / "openai-wiki" / "profile.json").read_text())
    assert f"(the German message board: {prof['notes']['n_revisions']:,} wiki revisions, which become " \
           f"{prof['n_events']:,} posts;" in README
    goals = sum(map(bool, (ROOT / "data" / "village" / "agent_goals.jsonl").read_text(encoding="utf-8").splitlines()))
    assert _table("## Repository layout")["`data/village/`"] == f"AI Village roster ({goals} goal assignments)"


def _crosssite_expected(s: dict, cross: dict, corpus: dict) -> dict[str, list]:
    """The README's cross-site paragraph, table rows and corpus paragraph, formatted from the cross-site and corpus
    summaries, the cross-site report and its technique_spread.csv."""
    spread = pd.read_csv(RESULTS / "crosssite" / "technique_spread.csv")
    summ = {r["technique"]: r for r in cross["technique_site_summary"]}
    md = summ["proxy-markdown-new"]
    hops = _within_a_day(cross, md["technique"])
    earliest = min(summ.values(), key=lambda r: r["first_seen"])
    dse_proxy = spread[(spread["site"] == "dse") & spread["technique"].str.startswith("proxy-")].sort_values("first_seen")
    dse_first = re.search(r"^\| dse \| \d+ \| \d+ \| (\S+)", (RESULTS / "crosssite" / "report.md").read_text(), re.M)
    cov, n_wiki = cross["coverage"], s["profile"]["n_events"]

    def day(ts) -> str:
        return f"{pd.Timestamp(ts).day}"

    return {
        "intro": [f"{cross['n_events']:,}", f"{cross['n_sites']}", f"{n_wiki:,}", f"{cross['n_events'] - n_wiki:,}",
                  f"{cross['corpus_notes']['n_events'] - (cross['n_events'] - n_wiki):,}"],
        "**Fetch-proxy tricks crossed platforms in hours; task tricks never did.**": [
            *pd.Timestamp(md["first_seen"]).strftime("%H %M %d").split(), f"{md['sites_within_24h']}",
            f"{math.ceil(hops.max())}", *(f"{h:.1f}" for h in sorted(hops))],
        "**The earliest proxy use predates the main wiki by a week.**": [
            day(earliest["first_seen"]), day(dse_first.group(1)), day(dse_proxy.iloc[0]["first_seen"])],
        "**Coverage bounds.**": [f"{cov['n_sites']}", f"{cov['n_unsearched']}", f"{cov['relays_unvisited']}",
                                 f"{cov['shorteners']}", f"{cov['discord_urls']}"],
        "corpus": [f"{corpus['profile']['n_events']:,}", f"{corpus['profile']['n_authors']:,}",
                   f"{corpus['funnel']['answer_claims_no_item']:,}", f"{corpus['A6']['urls']['facts']:,}"],
    }


def test_readme_crosssite_and_corpus_text_quotes_the_committed_results():
    s, cross, corpus = (_summary(d) for d in ("openai-wiki", "crosssite", "corpus"))
    marker = "### Cross-site results"
    rows = {"intro": README.split(marker, 1)[1].split("\n\n")[1], **_table(marker),
            "corpus": next(p for p in README.split("\n\n") if p.startswith("Running the standard pipeline on the corpus"))}
    expected = _crosssite_expected(s, cross, corpus)
    assert list(rows) == list(expected)
    for what, numbers in expected.items():
        _check(rows[what], numbers, what)
    summ = {r["technique"]: r for r in cross["technique_site_summary"]}
    fetch = rows["**Fetch-proxy tricks crossed platforms in hours; task tricks never did.**"]
    first = pd.Timestamp(summ["proxy-markdown-new"]["first_seen"])
    assert summ["proxy-markdown-new"]["origin_site"] == "dse" and f"on dse at {first:%H:%M} on {first:%B %d}" in fetch
    spread = pd.read_csv(RESULTS / "crosssite" / "technique_spread.csv").set_index(["technique", "site"])
    days = spread.loc[("proxy-allorigins", "dse"), "hours_after_origin"] / 24
    assert summ["proxy-allorigins"]["origin_site"] == "paste.linuxiarz.pl" and \
        f"**pastebin**, {WORDS[round(days)]} days before it reached dse" in fetch
    stayed = [r for r in summ.values() if r["n_sites_adopted"] == 1]
    assert {r["origin_site"] for r in stayed} == {"dse"} and f"All {WORDS[len(stayed)]} task-level techniques" in fetch
    earliest = min(summ.values(), key=lambda r: r["first_seen"])
    assert f"May {pd.Timestamp(earliest['first_seen']).day} {earliest['origin_site']}" in \
        rows["**The earliest proxy use predates the main wiki by a week.**"]
    assert set(corpus["skipped"]) >= {"A1", "A2"} and "A1 and A2 skip" in rows["corpus"]
    assert f"the corpus config's {WORDS[len(corpus['A3'])]} URL techniques" in rows["corpus"]


# docs/DATA.md's cross-site table: the host it names for each technique and the short names it gives the sites
HOSTS = {"markdown.new": "proxy-markdown-new", "pure.md": "proxy-pure-md", "allorigins.hexlet.app": "proxy-allorigins",
         "jqp.vercel.app": "proxy-jqp", "md.succ.ai": "proxy-md-succ", "r.jina.ai": "proxy-jina",
         "da.gd": "shortener-da-gd"}
SHORT = {"wikiservice.at/user/milk": "user/milk", "paste.linuxiarz.pl": "paste.linuxiarz", "ludism.org": "ludism",
         "prowiki.org/wiki4d": "wiki4d", "tmcleod.org/apchem": "tmcleod.org"}


def test_data_md_quotes_the_committed_crosssite_spread_and_a1():
    """Each row of DATA.md's cross-site table is the technique's sites in technique_spread.csv, in order: the origin
    with its date and time, then each other site with its time on the origin's day and its date after that; the
    note after a ';' counts the sites within a day.  Its consequences quote A1's bound."""
    cross, s = _summary("crosssite"), _summary("openai-wiki")
    spread = pd.read_csv(RESULTS / "crosssite" / "technique_spread.csv", parse_dates=["first_seen"])
    rows = _table("## Cross-site technique spread", DATA)
    assert set(rows) == set(HOSTS)
    section = DATA.split("## Cross-site technique spread", 1)[1].split("\n---\n", 1)[0]
    assert sum("|" in line for line in section.splitlines()) == len(rows) + 2   # one table, nothing left of an older one
    for host, cells in rows.items():
        first, then = (c.strip() for c in cells.split(" | "))
        t = spread[spread["technique"] == HOSTS[host]].sort_values(["first_seen", "site"])
        o = t.iloc[0]
        assert first.strip("*") == f"{SHORT.get(o['site'], o['site'])} {o['first_seen']:%m-%d %H:%M}", host
        later = [f"{SHORT.get(r.site, r.site)} {r.first_seen:%H:%M}" if r.first_seen.date() == o["first_seen"].date()
                 else f"{SHORT.get(r.site, r.site)} {r.first_seen:%m-%d}" for r in t.iloc[1:].itertuples()]
        listed, _, note = then.partition("; ")
        assert listed == ", ".join(later), host
        if note:
            hops = _within_a_day(cross, HOSTS[host])
            assert _numbers(note) == [f"{len(hops)}", f"{math.ceil(hops.max())}"], host
    a1 = s["A1"]
    consequence = next(p for p in DATA.split("\n") if p.startswith("1. **Exposure is a lower bound.**"))[3:]
    assert _numbers(consequence) == [f"{a1['n_independent_at_most']:,}", f"{a1['n_answers']:,}",
                                     f"{a1['n_independent']:,}", pct(a1["share_exposed"]),
                                     f"{a1['n_independent_at_most']:,}"]


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: link markup and punctuation dropped, spaces as hyphens."""
    text = MD_LINK.sub(r"\1", heading.lstrip("#").strip()).lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    return {_slug(line) for line in text.splitlines() if line.startswith("#")}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_in_the_docs_resolves(doc):
    """Files and #anchors alike: the README moved most of its text to docs/USAGE.md and docs/METHODS.md, and links
    are how a reader gets there."""
    text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)
    for _, target in MD_LINK.findall(text):
        if re.match(r"[a-z]+://", target):
            continue
        path, _, anchor = target.partition("#")
        dest = (doc.parent / path).resolve() if path else doc
        assert dest.exists(), f"{doc.name}: {target}"
        if anchor:
            assert anchor in _anchors(dest), f"{doc.name}: {target}"


def test_readme_links_the_usage_and_methods_docs_and_keeps_the_sections_code_names():
    targets = {t.partition("#")[0] for _, t in MD_LINK.findall(README)}
    assert {"docs/USAGE.md", "docs/METHODS.md", "tests/README.md", "results/README.md", "docs/DATA.md"} <= targets
    # the CLI's missing-data error and A1's skip note send the reader to these README sections
    assert {"where-the-data-comes-from", "what-the-rule-extractor-recognises"} <= _anchors(ROOT / "README.md")
    assert max(map(len, README.splitlines())) <= 600
