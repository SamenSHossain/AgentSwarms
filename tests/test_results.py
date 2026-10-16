"""The committed results/ directories: every figure a report links is there (and none it does not link), the
collectors' 143-surface inventory is cited only where it describes the source, each key-findings block is the one the
current code writes from its summary, and every number the README quotes from a run (the headline table, the wiki
results table, the cross-site and corpus paragraphs) is formatted from the committed summary.json files, the
cross-site report and its technique_spread.csv.  The few numbers read off tables results/ does not hold are HAND."""

import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from swarmprov import report
from swarmprov.analysis import causal

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = (ROOT / "README.md").read_text()
REPORTS = sorted(RESULTS.glob("*/report.md"))
LINK = re.compile(r"\]\(([^)]+\.png)\)")
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
NAME = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d)\w+")   # Mar30, sector61, G5, p90: digits inside a name
# digits that name a definition, not a finding
DEFINITIONS = ("95% CI", "top 10", "≥2", "score 0", "clears 0.05")
HAND = None   # read off the events or disputes tables, which results/ does not hold: not checked
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


def test_the_coverage_inventory_is_cited_only_where_it_describes_the_source():
    cited = {p.parent.name for p in REPORTS if "143 surfaces" in p.read_text()}
    assert cited == {"openai-wiki", "corpus", "crosssite"}


# committed reports whose key findings predate a wording change in report.key_findings (the A2 reading rule):
# expected to fail until results/ is regenerated, which must drop them from here (strict: a pass then fails)
STALE = {"openai-wiki", "synthetic"}


@pytest.mark.parametrize("path", [pytest.param(p, marks=pytest.mark.xfail(
    p.parent.name in STALE, reason="results/ predates the current key-findings wording", strict=True))
    for p in REPORTS], ids=lambda p: p.parent.name)
def test_committed_key_findings_are_what_the_current_code_writes(path):
    """A wording change in report.key_findings fails here until results/ is regenerated."""
    s = json.loads((path.parent / "summary.json").read_text())
    kf = report.key_findings(s)
    assert (path.parent / "headline.md").exists() == bool(kf)
    if kf:
        assert (path.parent / "headline.md").read_text() == "\n".join(kf)
        assert s["headline"] == [b[2:] for b in kf if b.startswith("- ")]
        assert "\n".join(kf) in path.read_text()


def _table(marker: str) -> dict[str, str]:
    """The first README table after ``marker``: {first cell: second cell} for each row below the header."""
    lines = README.split(marker, 1)[1].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    rows = list(itertools.takewhile(lambda line: line.startswith("|"), lines[start:]))[2:]
    return dict(tuple(c.strip() for c in line.strip().strip("|").split(" | ", 1)) for line in rows)


def _numbers(text: str) -> list[str]:
    for d in DEFINITIONS:
        text = text.replace(d, "")
    return NUMBER.findall(NAME.sub("", text))


def _check(text: str, expected: list, what: str):
    found = _numbers(text)
    assert len(found) == len(expected) and all(e is HAND or f == e for f, e in zip(found, expected)), \
        f"{what}: README {found}, expected {expected}"


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


def _headline_expected(s: dict, synth: dict, cross: dict) -> dict[str, list[str]]:
    """Each headline row's numbers in the order the README states them, formatted from the summaries."""
    a1, a5, reach = s["A1"], s["A5"], s["A6"]["task values"]
    rule = {r["rule"]: r for r in a5["attribution"] if r["answers"] == "all exposed"}
    span, n_credited = report.top10_range(a5)   # the range the key findings give
    origin = a5["origins"]["by_mapping"]["merged"]["first source"]
    a3 = {r["technique"]: r for r in s["A3"]}["blob-hostname-bypass"]
    a4 = {r["slot"]: r for r in s["A4"]["rows"]}
    g5, poland, hungary = (a4[k] for k in (G5, "oecd-equity | Poland", "oecd-equity | Hungary"))
    unread, (flat,) = _a2_readings(s)
    proxy = {r["technique"]: r for r in cross["technique_site_summary"]}["proxy-markdown-new"]
    truth = synth["truth_validation"]["A2"]
    return {
        "How much was copied?": [pct(a1["share_exposed"]), *map(pct, a1["share_exposed_ci"]), f"{a1['n_answers']:,}",
                                 f"{a1['n_agents']:,}", pct(a1["share_exposed_60m"]), f"{a1['median_gap_h']:.1f}",
                                 f"{a1['n_independent']:,}", f"{a1['n_independent_visible']:,}",
                                 pct(a1["share_self_prepared"])],
        "Who supplied it?": [*NUMBER.findall(span), f"{n_credited:,}",
                             pct(rule["first source"]["top10_share"]), f"{rule['first source']['gini']:.2f}",
                             pct(rule["latest source"]["top10_share"]), pct(rule["same-page relay parent"]["top10_share"]),
                             f"{rule['same-page relay parent']['n']:,}", f"{a5['items_one_source']}",
                             f"{a5['items_exposed']}", pct(origin["computed"] / a5["n_exposure_edges"]),
                             pct(origin["no answer round observed"] / a5["n_exposure_edges"]),
                             f"{reach['mean_hops_repeated']:.1f}", f"{reach['max_hops']}"],
        "How fast did tricks spread?": [pct(a3["adoption_share"]), f"{a3['adopters']}", f"{a3['at_risk']}",
                                        f"{a3['median_hours_among_adopters']:.1f}", f"{proxy['sites_within_24h']}",
                                        f"{proxy['median_hours_to_site_adoption']:.1f}"],
        "Did errors win?": [carriers(g5)[g5["first variant"]], carriers(g5)[g5["winner"]], *late(g5),
                            f2(poland["winner"]), f2(poland["first variant"]),
                            f"{s['A4']['rule']['bin_hours']:.0f}", f2(hungary["first variant"]),
                            carriers(hungary)[hungary["first variant"]], carriers(hungary)[hungary["winner"]],
                            f2(hungary["winner"]), f"{hungary['tail counts'][hungary['winner']]}",
                            f"{hungary['tail counts'][hungary['first variant']]}"],
        "Does availability *cause* copying?": [f"{min(unread)}", f"{max(unread)}", *_flat_numbers(flat),
                                               f"{truth['beta']:.2f}", f"{truth['true_effect']:.2f}"],
    }


def test_readme_headline_quotes_the_committed_summaries():
    s, synth, cross = (_summary(d) for d in ("openai-wiki", "synthetic", "crosssite"))
    rows, expected = _table("**What it found on the wiki swarm**"), _headline_expected(s, synth, cross)
    assert list(rows) == list(expected)
    for question, numbers in expected.items():
        _check(rows[question], numbers, question)
    assert s["A4"]["rows"][0]["status"] == "overturned" and "overturned it as soon as it appeared" in rows["Did errors win?"]
    assert {r["slot"]: r for r in s["A4"]["rows"]}["oecd-equity | Hungary"]["status"] == "contested"
    assert all(r["beta"] is None or r["ci_lo"] <= 0 <= r["ci_hi"] for r in s["A2"]) and "Not identified" in \
        rows["Does availability *cause* copying?"]


def _results_expected(s: dict, synth: dict) -> dict[str, list]:
    """Each row of the README's wiki results table, its numbers in order, formatted from the summaries."""
    a1, a4, a5, a6 = s["A1"], s["A4"], s["A5"], s["A6"]
    tc = a1["threshold_curve"]
    curve, at = {c["answers"]: c["share"] for c in tc["curves"]}, tc["hours"].index
    a3 = {r["technique"]: r for r in s["A3"]}
    blob, tip, beat = a3["blob-hostname-bypass"], a3["live-tooltip-value"], a3["heartbeat-beacon"]
    seed, wait, bcon, tcon = (a3[t]["contagion"] for t in ("rng-seed-crack", "clock-wait-fastforward",
                                                            "blob-hostname-bypass", "live-tooltip-value"))
    slots = {r["slot"]: r for r in a4["rows"]}
    g5, poland, hungary, slovak, czech = (slots[k] for k in (G5, "oecd-equity | Poland", "oecd-equity | Hungary",
                                                             "oecd-equity | Slovak Republic",
                                                             "oecd-equity | Czech Republic"))
    status = Counter(r["status"].split(" (")[0] for r in a4["rows"])
    tie = next(r for r in a4["rows"] if r["tie"])
    small = next(r for r in a4["rows"] if r["status"] == "contested (small n)")
    vp = czech["variant_provenance"]["by_variant"]
    won, cons = vp[czech["winner"]], vp[czech["consensus"]]
    disagree = a4["consensus_disagrees"]
    rule = {r["rule"]: r for r in a5["attribution"] if r["answers"] == "all exposed"}
    first, latest, relay = rule["first source"], rule["latest source"], rule["same-page relay parent"]
    instant, _ = report.top10_range({"attribution": [dict(r, answers="all exposed") for r in a5["attribution"]
                                                     if r["answers"] == "instant only"]})
    by = a5["origins"]["by_mapping"]

    def share(mapping, credit, origin):
        counts = by[mapping][credit]
        return pct(counts[origin] / sum(counts.values()))

    tv, urls = a6["task values"], a6["urls"]
    window = {w["W_s"]: w for w in a6["window_sensitivity"]}
    hour, ever = window[3600.0], window[None]
    unread, (flat,) = _a2_readings(s)
    truth = synth["truth_validation"]["A2"]
    return {
        "**A1 Provenance**": [
            f"{a1['n_answers']:,}", f"{a1['n_agents']:,}", f"{a1['n_independent']:,}", f"{a1['n_independent_visible']:,}",
            f"{a1['n_only_deleted_copies']:,}", f"{a1['n_src_deleted_before_report']:,}",
            f"{a1['n_src_deleted_before_report'] - a1['n_only_deleted_copies']:,}",
            pct(a1["share_exposed"]), *map(pct, a1["share_exposed_ci"]),
            pct(a1["share_exposed_60m"]), *map(pct, a1["share_exposed_60m_ci"]),
            f"{a1['median_gap_h']:.1f}", *(f"{h:.1f}" for h in a1["median_gap_h_ci"]),
            pct(curve["all answers"][at(24)]), "6", pct(curve["oecd-equity"][at(24)]), "24",
            pct(a1["share_self_prepared"])],
        "**A2 Does availability cause copying?**": [f"{min(unread)}", f"{max(unread)}", f"{causal.MIN_SWITCHERS}",
                                                    *[f"{2 * causal.MIN_SWITCHERS}"] * 2, *_flat_numbers(flat),
                                                    f"{truth['beta']:.2f}", f"{truth['true_effect']:.2f}"],
        "**A3 Diffusion**": [
            *pd.Timestamp(blob["first_seen"]).strftime("%Y %m %d %H %M").split(),
            f"{blob['adopters']}", f"{blob['at_risk']}", pct(blob["adoption_share"]),
            f"{blob['median_hours_among_adopters']:.1f}",
            f"{tip['adopters']}", f"{tip['at_risk']}", pct(tip["adoption_share"]), f"{tip['median_hours_to_adopt']:.1f}",
            f"{tip['median_hours_to_adopt_lo']:.1f}", f"{tip['median_hours_to_adopt_hi']:.1f}",
            f"{beat['adopters']}", f"{beat['at_risk']:,}", f"{beat['median_hours_among_adopters']:.0f}",
            pct(seed["share_on_carrying"]), pct(seed["baseline"]), pct(wait["share_on_carrying"]), pct(wait["baseline"]),
            f"{seed['p_holm']:.3f}", f"{wait['p_holm']:.3f}",
            pct(bcon["share_on_carrying"]), pct(bcon["baseline"]), f"{bcon['p']:.2f}", f"{bcon['p_holm']:.2f}",
            pct(tcon["share_on_carrying"]), pct(tcon["baseline"])],
        "**A4 Errors**": [
            f"{a4['rule']['min_tail']}", carriers(g5)[g5["first variant"]], carriers(g5)[g5["winner"]],
            HAND,   # Montana's first post 12 h after Maryland's: the disputes table's first-seen times
            *late(g5), f2(poland["first variant"]), f2(poland["winner"]), *late(poland),
            f2(slovak["first variant"]), f2(slovak["winner"]), *late(slovak), f"{a4['rule']['bin_hours']:.0f}",
            carriers(hungary)[hungary["first variant"]], f2(hungary["first variant"]),
            carriers(hungary)[hungary["winner"]], f2(hungary["winner"]),
            f"{hungary['tail counts'][hungary['winner']]}", f"{hungary['tail counts'][hungary['first variant']]}",
            f"{len(a4['rows'])}", f"{status['overturned']}", f"{status['held']}", f"{status['contested']}",
            *map(str, tie["tail counts"].values()), small["late carriers (winner/total)"].split("/")[1],
            czech["winner"], f"{won['derived_independently']}", f"{won['answer_rounds']}", czech["consensus"],
            f"{won['adopted_after_public']}", czech["consensus"], f"{cons['adopted_after_public']}",
            f"{len(disagree['slots'])}", f"{disagree['n_value_slots']}", f"{disagree['of_them_y_consensus_0']}"],
        "**A5 Structure**": [
            f"{first['sources']}", pct(first["top10_share"]), f"{first['gini']:.2f}", f"{a5['gini_item_exposures']:.2f}",
            f"{a5['items_exposed']}", pct(first["top10_share_loo_lo"]), pct(first["top10_share_loo_hi"]),
            f"{latest['sources']}", pct(latest["top10_share"]), pct(relay["top10_share"]), f"{relay['n']}",
            *NUMBER.findall(instant), f"{a5['items_one_source']}", f"{a5['items_exposed']}",
            share("merged", "first source", "computed"), share("merged", "first source", "no answer round observed"),
            share("strict", "first source", "no answer round observed"), share("merged", "first source", "dissent"),
            share("merged", "first source", "relayed"), share("merged", "latest source", "relayed"),
            f"{a5['origins']['roots']['latest source']['chain cut (its latest source posted after it)']:,}",
            f"{tv['mean_hops_repeated']:.1f}", f"{tv['max_hops']}"],
        "**A6 Reach**": [
            f"{tv['facts']:,}", f"{tv['repeated']:,}", f"{tv['mean_repeaters']:.1f}", f"{tv['mean_repeaters_lo']:.1f}",
            f"{tv['mean_repeaters_hi']:.1f}", f"{tv['p90_repeaters']:.0f}", f"{urls['facts']:,}", f"{urls['repeated']:,}",
            f"{ever['mean_max_depth']:.1f}", f"{hour['mean_max_depth']:.1f}", f"{hour['unattributed']:,}",
            f"{hour['relay'] + hour['relay_xchannel'] + hour['unattributed']:,}"],
    }


def test_readme_results_table_quotes_the_committed_summaries():
    s, synth = _summary("openai-wiki"), _summary("synthetic")
    rows, expected = _table("## Results on the OpenAI wiki swarm"), _results_expected(s, synth)
    assert list(rows) == list(expected)
    for analysis, numbers in expected.items():
        _check(rows[analysis], numbers, analysis)
    a1, a3, a4, a5 = rows["**A1 Provenance**"], rows["**A3 Diffusion**"], rows["**A4 Errors**"], rows["**A5 Structure**"]
    tc = s["A1"]["threshold_curve"]
    assert dict((c["answers"], c["share"]) for c in tc["curves"])["datausa-sector61-state"][tc["hours"].index(6)] == 0
    assert "none of datausa-sector61-state's at 6 h" in a1
    techniques = {r["technique"]: r for r in s["A3"]}
    assert f"by the {techniques['blob-hostname-bypass']['originator'].split('|')[0]} OECD agent" in a3
    tested = [t for t, r in techniques.items() if not r["contagion"].get("excluded")]
    holm = [t for t in tested if techniques[t]["contagion"]["p_holm"] < 0.05]
    assert f"For {WORDS[len(holm)]} techniques" in a3 and f"for the {WORDS[len(tested)]} techniques tested" in a3
    assert techniques["zzz-backup-pages"]["contagion"]["excluded"] and "ZZZ pages are not tested" in a3
    slots = {r["slot"]: r for r in s["A4"]["rows"]}
    assert [slots[k]["status"] for k in (G5, "oecd-equity | Hungary")] == ["overturned", "contested"]
    assert all(slots[f"oecd-equity | {k}"]["hours to overtake"] < s["A4"]["rule"]["bin_hours"]
               for k in ("Poland", "Slovak Republic"))
    assert sum(r["first seen tie"] for r in s["A4"]["rows"]) == 1 and "first posted in the same second" in a4
    czech = slots["oecd-equity | Czech Republic"]["variant_provenance"]["by_variant"]["9.69"]
    assert czech["independent_dissent"] == czech["derived_independently"] and "all while 9.7 was public" in a4
    assert f"within {WORDS[s['A5']['origins']['max depth']['first source']]} hops" in a5


def _crosssite_expected(s: dict, cross: dict, corpus: dict) -> dict[str, list]:
    """The README's cross-site paragraph, table rows and corpus paragraph, formatted from the cross-site and corpus
    summaries, the cross-site report and its technique_spread.csv."""
    spread = pd.read_csv(RESULTS / "crosssite" / "technique_spread.csv")
    summ = {r["technique"]: r for r in cross["technique_site_summary"]}
    md = summ["proxy-markdown-new"]
    hops = spread[(spread["technique"] == md["technique"]) & (spread["site"] != md["origin_site"])
                  & (spread["hours_after_origin"] <= 24)]["hours_after_origin"]
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
            f"{math.ceil(hops.max())}", *(f"{h:.1f}" for h in hops)],
        "**The earliest proxy use predates the main wiki by a week.**": [
            day(earliest["first_seen"]), day(dse_first.group(1)), day(dse_proxy.iloc[0]["first_seen"])],
        "**Three activity waves.**": [HAND] * 11,   # read off the timeline figure (the events table)
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
