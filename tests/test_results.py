"""The committed results/ directories: every figure a report links is there (and none it does not link), the
collectors' 143-surface inventory is cited only where it describes the source, and the README's headline table
quotes results/openai-wiki/summary.json (the synthetic and cross-site summaries for the numbers that come from them)."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = sorted(RESULTS.glob("*/report.md"))
LINK = re.compile(r"\]\(([^)]+\.png)\)")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# digits that name a definition, not a finding
DEFINITIONS = ("95% CI", "top 10")


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


def _headline_rows() -> dict[str, str]:
    text = (ROOT / "README.md").read_text()
    table = text.split("**What it found on the wiki swarm**", 1)[1].split("\n\n")[1]
    rows = [line.strip().strip("|").split(" | ", 1) for line in table.splitlines()[2:]]
    return {q.strip(): a.strip() for q, a in rows}


def _numbers(text: str) -> list[str]:
    for d in DEFINITIONS:
        text = text.replace(d, "")
    return NUMBER.findall(text)


def _expected(s: dict, synth: dict, cross: dict) -> dict[str, list[str]]:
    """Each headline row's numbers in the order the README states them, formatted from the summaries."""
    def pct(x):
        return f"{x * 100:.0f}"

    a1, a5, reach = s["A1"], s["A5"], s["A6"]["task values"]
    rule = {r["rule"]: r for r in a5["attribution"] if r["answers"] == "all exposed"}
    origin = a5["origins"]["by_mapping"]["merged"]["first source"]
    shares = [r["top10_share"] for r in rule.values()]
    a3 = {r["technique"]: r for r in s["A3"]}["blob-hostname-bypass"]
    a4 = {r["slot"]: r for r in s["A4"]["rows"]}
    g5, poland, hungary = (a4[k] for k in ("datausa-grocery-workforce | G5 (cracked-seed forecast vs observed)",
                                            "oecd-equity | Poland", "oecd-equity | Hungary"))
    switch = [r["agents_with_D_variation"] for r in s["A2"] if r["outcome"] != "y_winner"]
    proxy = {r["technique"]: r for r in cross["technique_site_summary"]}["proxy-markdown-new"]
    truth = synth["truth_validation"]["A2"]

    def carriers(row):   # 'Maryland (RNG forecast) (30), Montana (observed) (11)' -> {label: '30', ...}
        return dict(re.findall(r"(.+?) \((\d+)\)(?:, |$)", row["variants (agents)"]))

    def late(row):
        return [f"{row['tail counts'][row['winner']]}", row["late carriers (winner/total)"].split("/")[1]]

    return {
        "How much was copied?": [pct(a1["share_exposed"]), *map(pct, a1["share_exposed_ci"]), f"{a1['n_answers']:,}",
                                 f"{a1['n_agents']:,}", pct(a1["share_exposed_60m"]), f"{a1['median_gap_h']:.1f}",
                                 f"{a1['n_independent']:,}", f"{a1['n_independent_visible']:,}",
                                 pct(a1["share_self_prepared"])],
        "Who supplied it?": [pct(min(shares)), pct(max(shares)), f"{a5['n_exposure_edges']:,}",
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
                            f"{float(poland['winner']):.2f}", f"{float(poland['first variant']):.2f}",
                            f"{s['A4']['rule']['bin_hours']:.0f}", f"{float(hungary['first variant']):.2f}",
                            carriers(hungary)[hungary["first variant"]], carriers(hungary)[hungary["winner"]],
                            f"{float(hungary['winner']):.2f}", f"{hungary['tail counts'][hungary['winner']]}",
                            f"{hungary['tail counts'][hungary['first variant']]}"],
        "Does availability *cause* copying?": [f"{min(switch)}", f"{max(switch)}", f"{truth['beta']:.2f}",
                                               f"{truth['true_effect']:.2f}"],
    }


def test_readme_headline_quotes_the_committed_summaries():
    s, synth, cross = (json.loads((RESULTS / d / "summary.json").read_text())
                       for d in ("openai-wiki", "synthetic", "crosssite"))
    rows, expected = _headline_rows(), _expected(s, synth, cross)
    assert list(rows) == list(expected)
    for question, numbers in expected.items():
        assert _numbers(rows[question]) == numbers, question
    assert s["A4"]["rows"][0]["status"] == "overturned" and "overturned it as soon as it appeared" in rows["Did errors win?"]
    assert {r["slot"]: r for r in s["A4"]["rows"]}["oecd-equity | Hungary"]["status"] == "contested"
    assert all(r["beta"] is None or r["ci_lo"] <= 0 <= r["ci_hi"] for r in s["A2"]) and "Not identified" in \
        rows["Does availability *cause* copying?"]
