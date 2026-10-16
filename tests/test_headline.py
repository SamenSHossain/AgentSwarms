"""The key findings block: it opens the report, is written alone to headline.md and summary['headline'], and every
number in it is a summary.json value (formatted, never recomputed)."""

import copy
import json
import re

import pytest

from swarmprov import report
from swarmprov.analysis import causal
from swarmprov.schema import TABLES

NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
# definitions that carry a digit but are no finding: the interval level, A2's rule and the top-10 share's name
DEFINITIONS = ("95% CI", "excludes 0", "top 10", *(f"A{k}." for k in range(1, 7)))


def _leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v, path)
    else:
        yield path, obj


def _forms(v) -> set[str]:
    """Every way the key findings may print a summary number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return set()
    out = {f"{v:,.0f}", f"{v:.1f}", f"{v:.2f}", f"{v:+.2f}", f"{v:.0%}", f"{v * 100:.0f}"}
    return out | ({f"{v:,}"} if isinstance(v, int) else set())


def unexplained(summary: dict, bullets: list[str]) -> tuple[list[str], int]:
    """The numbers in ``bullets`` that no summary value formats to (outside code spans, whose text must itself be
    summary text or an exposure column, and the skip lines, which are copied from ``summary['skipped']``), and how
    many were checked."""
    leaves = [v for p, v in _leaves({k: v for k, v in summary.items() if k != "headline"})]
    forms = set().union(*map(_forms, leaves))
    text = {str(v) for v in leaves if isinstance(v, str)} | set(TABLES["exposures"])
    bad, n = [], 0
    for b in bullets:
        for reason in (summary.get("skipped") or {}).values():
            b = b.replace(reason, "")
        for code in re.findall(r"`([^`]+)`", b):
            if code not in text and not all(p.split(" | ")[0] in text or p in {"Y", "agent + item"}
                                            for p in code.split(" ~ ")):
                bad.append(f"`{code}`")
        b = re.sub(r"`[^`]+`", "", b)
        for d in DEFINITIONS:
            b = b.replace(d, "")
        for tok in NUMBER.findall(b):
            n += 1
            if tok.lstrip("+") not in forms and tok not in forms and tok.rstrip("%") not in forms:
                bad.append(tok)
    return bad, n


@pytest.fixture(scope="module")
def synth(synth_run):
    run = synth_run[0]
    return run, json.loads((run.path / "summary.json").read_text()), (run.path / "report.md").read_text()


def test_key_findings_open_the_report(synth):
    run, s, md = synth
    lines = md.splitlines()
    k = lines.index("## Key findings")
    assert k < 10 and lines[k - 2].startswith("Source:") and lines.index("## Data capabilities") > k
    block = (run.path / "headline.md").read_text()
    assert block.startswith("## Key findings\n\n- **A1. Provenance split**:") and block in md
    assert [b[2:] for b in block.splitlines() if b.startswith("- ")] == s["headline"]
    assert not (run.path / "headline_strict.md").exists()


def test_every_number_is_a_summary_value(synth):
    _, s, _ = synth
    bad, n = unexplained(s, s["headline"])
    assert n >= 25 and not bad, bad
    assert f"{s['A1']['n_answers']:,} answers" in s["headline"][0]


def test_the_round_trip_catches_a_recomputed_number(synth):
    """The check above is not vacuous: a number no summary value formats to is caught."""
    _, s, _ = synth
    bad, _ = unexplained(s, [s["headline"][0].replace(f"{s['A1']['n_answers']:,} answers", "2,345 answers")])
    assert bad == ["2,345"]


def _put(d, path: tuple, v) -> None:
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = v


def _row(rows: list[dict], **match) -> int:
    (i,) = [i for i, r in enumerate(rows) if all(r.get(k) == v for k, v in match.items())]
    return i


@pytest.fixture(scope="module")
def sentinels(synth):
    """The seed-2 summary with each number the key findings print replaced by a distinct sentinel that keeps its
    meaning (signs, the A2 switcher gate), plus an A3 row and a ``D_visible`` count it lacks: (summary, bullets,
    the sentinels by path)."""
    _, s, _ = synth
    t = copy.deepcopy(s)
    t["A1"]["n_independent_visible"] = None
    t["A3"] = [{"technique": "census", "adoption_share": None, "adopters": None, "at_risk": None, "contagion": None}]
    a2, a5 = t["A2"], t["A5"]["attribution"]
    put = {("A1", "n_answers"): 4321, ("A1", "n_agents"): 876, ("A1", "n_independent"): 1234,
           ("A1", "n_independent_visible"): 1111, ("A1", "share_exposed"): 0.611, ("A1", "share_exposed_ci", 0): 0.512,
           ("A1", "share_exposed_ci", 1): 0.733, ("A1", "share_exposed_60m"): 0.444,
           ("A1", "share_exposed_60m_ci", 0): 0.333, ("A1", "share_exposed_60m_ci", 1): 0.552,
           ("A1", "median_gap_h"): 37.3, ("A1", "median_gap_h_ci", 0): 31.2, ("A1", "median_gap_h_ci", 1): 44.4,
           ("A3", 0, "adoption_share"): 0.391, ("A3", 0, "adopters"): 1212, ("A3", 0, "at_risk"): 3232,
           ("A4", "rule", "min_tail"): 11,
           ("A5", "top10_share_exposure_loo", 0): 0.652, ("A5", "top10_share_exposure_loo", 1): 0.781,
           ("A6", "task values", "facts"): 345, ("A6", "task values", "repeated"): 97,
           ("A6", "task values", "mean_hops_repeated"): 6.7, ("A6", "task values", "max_hops"): 19}
    for k, (outcome, sw, beta, lo, hi) in enumerate((("y_instant", 1024, 0.812, 0.621, 0.934),
                                                     ("y_instant|known", 1025, 0.823, 0.632, 0.945))):
        i = _row(a2, outcome=outcome, treatment="D_cons")
        put |= {("A2", i, "agents_with_D_variation"): sw, ("A2", i, "beta"): beta, ("A2", i, "ci_lo"): lo,
                ("A2", i, "ci_hi"): hi}
    for k, treatment in enumerate(("D_cons", "D_cons_w60m")):
        put[("A2", _row(a2, outcome="y_consensus", treatment=treatment), "agents_with_D_variation")] = 1026 + k
    for rule, n, share in (("first source", 2468, 0.712), ("latest source", 2468, 0.234),
                           ("same-page relay parent", 1357, 0.123)):   # the relay parent's share is the lowest
        i = _row(a5, rule=rule, answers="all exposed")
        put |= {("A5", "attribution", i, "n"): n, ("A5", "attribution", i, "top10_share"): share}
    put[("A5", "attribution", _row(a5, rule="first source", answers="all exposed"), "gini")] = 0.471
    for k, i in enumerate(i for i, r in enumerate(t["A4"]["rows"]) if r["featured"]):
        for j, v in enumerate(t["A4"]["rows"][i]["tail counts"]):
            put[("A4", "rows", i, "tail counts", v)] = 30 + 10 * k - j
    for path, v in put.items():
        _put(t, path, v)
    return t, report.key_findings(t)[1:-1], put


def test_each_printed_number_is_the_field_it_names(sentinels):
    """Every number the key findings print is its own summary field, formatted, not a neighbour's or a recomputed
    one: with each field set to a distinct sentinel, the sentinel appears where the bullet names that field."""
    t, bullets, put = sentinels
    text = "\n".join(bullets)
    expect = ["61% (95% CI 51–73%) of 4,321 answers by 876 agents", ", 44% (33–55%) at least an hour before",
              "median head start of 37.3 h (31.2–44.4 h)",
              "at most 1,234 were independent lookups (1,111 under `D_visible`",
              "`y_instant` under `D_cons` +0.81 (95% CI +0.62 to +0.93, 1,024 agents switch",
              "`y_instant|known` under `D_cons` +0.82 (95% CI +0.63 to +0.94, 1,025 agents switch",
              "Not identified for `y_consensus` (1,026–1,027 agents switch)", "`census` 39% (1,212 of 3,232)",
              "the top 10 sources supplied 23–71% of 2,468 exposed answers",
              "71% credited to the first agent to post the value (out-degree Gini 0.47; leaving out any one item moves "
              "it between 65% and 78%)", "23% to the latest earlier carrier", "12% of the 1,357 with a same-page relay parent",
              "97 of 345 task values were repeated by other agents, travelling 6.7 hops on average and at most 19"]
    for r in (r for r in t["A4"]["rows"] if r["featured"]):
        n = sorted(r["tail counts"].values(), reverse=True)
        expect.append(f"{r['slot']}: `{r['first variant']}` first, " + (
            f"`{r['winner']}` leads the late carriers {' to '.join(map(str, n))}" if len(n) > 1 else
            f"all {n[0]} late carriers carry `{r['winner']}`"))
        if r["status"] == "contested (small n)":
            expect.append(f"`{r['winner']}`, fewer than 11: contested (small n)")
    assert [e for e in expect if e not in text] == []
    # and nothing else: every number printed is one of the sentinels
    forms = set()
    for v in put.values():
        forms |= {f"{v:,}"} if isinstance(v, int) else {f"{v:.1f}", f"{v:.2f}", f"{v:+.2f}", f"{v:.0%}", f"{v * 100:.0f}"}
    for b in bullets:
        b = re.sub(r"`[^`]+`", "", b)
        for d in DEFINITIONS:
            b = b.replace(d, "")
        assert [x for x in NUMBER.findall(b) if x not in forms and x.rstrip("%") not in forms] == [], b


def test_the_tile_takes_the_range_the_headline_states(sentinels):
    """A5's range runs over the rules that credit every exposed answer; the relay parent's lower share is of fewer."""
    from swarmprov import html
    t, _, _ = sentinels
    assert ("23–71%", "top-10 source share", "of 2,468 exposed answers, across attribution rules") in html.tiles(t)


def test_a3_in_a_catch_all_run_counts_every_active_agent(sentinels):
    t = copy.deepcopy(sentinels[0])
    (a3,) = [b for b in report.key_findings(t) if "A3." in b]
    assert "(active in its task families after its first post)" in a3
    t["funnel"]["catch_all_family"] = "all"
    (a3,) = [b for b in report.key_findings(t) if "A3." in b]
    assert "(active after its first post)" in a3


def test_skipped_sections_repeat_their_skip_line(synth):
    _, s, md = synth
    reason = s["skipped"]["A3"]
    assert f"_Skipped: {reason}._" in md
    assert f"**A3. Technique diffusion**: skipped: {reason}." in s["headline"]
    assert [b[:5] for b in s["headline"]] == [f"**A{k}." for k in range(1, 7)]


def _a2(switch, lo, hi, outcome="y_instant", beta=0.5):
    return {"outcome": outcome, "treatment": "D_cons", "beta": beta, "ci_lo": lo, "ci_hi": hi,
            "agents_with_D_variation": switch}


@pytest.mark.parametrize("rows,identified", [
    ([_a2(2 * causal.MIN_SWITCHERS, 0.1, 0.9)], True),
    ([_a2(2 * causal.MIN_SWITCHERS - 1, 0.1, 0.9)], False),   # too few switchers, whatever the interval
    ([_a2(40, -0.1, 0.9)], False),                             # the interval covers 0
    ([_a2(40, -0.9, -0.2, beta=-0.5)], True),                  # a negative effect counts too
    ([_a2(3, None, None, beta=None), _a2(40, 0.1, 0.9, outcome="y_winner")], False),   # the extra row is no headline
])
def test_a2_is_identified_only_with_switchers_and_an_interval_off_zero(rows, identified):
    s = {"profile": {"n_events": 10}, "A2": rows}
    (bullet,) = [b for b in report.key_findings(s) if "A2." in b]
    assert ("Identified" in bullet) is identified and ("not identified" in bullet) is not identified
    if not identified:
        main = [r["agents_with_D_variation"] for r in rows if r["outcome"] != "y_winner"]
        k = f"{min(main)}" + (f"–{max(main)}" if min(main) != max(main) else "")
        assert f"not identified ({k} agents switch exposure status)" in bullet


def test_a2_names_the_headline_rows_it_could_not_identify():
    rows = [_a2(3, None, None, beta=None), {**_a2(5, -0.1, 0.4, outcome="y_consensus"), "treatment": "D_cons_w60m"},
            _a2(40, 0.1, 0.9, outcome="y_winner")]
    (bullet,) = [b for b in report.key_findings({"profile": {"n_events": 10}, "A2": rows}) if "A2." in b]
    assert ("not identified (3–5 agents switch exposure status): no estimate of `y_instant` or `y_consensus` under "
            "`D_cons` or `D_cons_w60m` from two-way fixed effects") in bullet and "y_winner" not in bullet


def test_a5_without_a_first_source_counts_in_the_singular():
    s = {"profile": {"n_events": 10}, "A5": {"n_nodes": 2, "n_relay_edges": 1, "attribution": []}}
    assert "2 agents linked by 1 same-page relay hop; no answer" in report.key_findings(s)[1]


def _slot(status, first, winner, tail, **kw):
    return {"slot": "f | X", "featured": True, "status": status, "first variant": first, "winner": winner,
            "tail counts": tail, "tie": kw.get("tie", False), "first seen tie": kw.get("ftie", False),
            "hours to overtake": kw.get("h"), "outcome": kw.get("outcome")}


@pytest.mark.parametrize("row,seq,says", [
    (_slot("overturned", "a", "b", {"b": 9, "a": 1}, h=0.0), False, "`a` → `b`, overturned as soon as it appeared"),
    (_slot("overturned", "a", "b", {"b": 9, "a": 1}, h=2.5), False, "overturned 2.5 h after it appeared"),
    (_slot("held", "a", "a", {"a": 9}), False, "`a` held"),
    (_slot("contested", "9.9", "9.91", {"9.91": 16, "9.9": 13}, outcome="precision variant: unresolved"), False,
     "`9.91` leads the late carriers 16 to 13 but never kept the majority of new carriers: contested (against the "
     "analyst's reference value: precision variant: unresolved)"),
    (_slot("contested", "9.9", "9.91", {"9.91": 16, "9.9": 13}), True,
     "but does not hold the majority of all the slot's carriers: contested"),
    (_slot("contested", "a", None, {"a": 2, "b": 2}, tie=True), False, "contested: the late carriers tie 2 to 2"),
    (_slot("contested", None, "b", {"b": 5, "a": 1}, ftie=True), False, "two variants were first seen at the same moment"),
    (_slot("contested (small n)", "a", "b", {"b": 4}), False, "all 4 late carriers carry `b`, fewer than 5: contested (small n)"),
    (_slot("contested", "a", "b", {"b": 3}), True,
     "`a` first, all 3 late carriers carry `b`, which does not hold the majority of all the slot's carriers: contested"),
])
def test_a4_states_each_featured_slot(row, seq, says):
    s = {"profile": {"n_events": 10, "clock": "sequence" if seq else "wall"},
         "A4": {"rule": {"bin_hours": None if seq else 3.0, "tail_share": 1 / 3, "min_tail": 5}, "rows": [row]}}
    (bullet,) = report.key_findings(s)[1:-1]
    assert says in bullet and "the last third of a slot's carriers" in bullet


def test_no_findings_without_posts():
    assert report.key_findings({}) == []
    assert report.key_findings({"profile": {"adapter": "village", "source": "x", "n_events": 0},
                                "funnel": {"posts": 0}}) == []


def test_a_strict_report_writes_its_own_headline(synth_strict):
    s = json.loads((synth_strict.path / "summary_strict.json").read_text())
    block = (synth_strict.path / "headline_strict.md").read_text()
    assert [b[2:] for b in block.splitlines() if b.startswith("- ")] == s["headline"]
    assert block in (synth_strict.path / "report_strict.md").read_text()
    assert (synth_strict.path / "headline.md").read_text() != block   # the merged one is left as it was
    assert "Key findings" in (synth_strict.path / "report_strict.html").read_text()


def test_compare_leaves_the_key_findings_out():
    """The block restates summary values that compare already diffs one by one."""
    from swarmprov import compare
    d = compare.summary_diff({"A1": {"n_answers": 5}, "headline": ["5 answers"]},
                             {"A1": {"n_answers": 6}, "headline": ["6 answers"]})
    assert [c["path"] for c in d["changed"]] == ["A1.n_answers"] and not d["sections_only_a"]


def test_a1_under_a_sequence_clock_states_no_hours():
    a1 = {"n_answers": 12, "n_agents": 3, "n_independent": 4, "share_exposed": 2 / 3, "share_exposed_60m": None,
          "median_gap_h": None}
    (bullet,) = report.key_findings({"profile": {"n_events": 30, "clock": "sequence"}, "A1": a1})[1:-1]
    assert "67% of 12 answers by 3 agents" in bullet and "before means earlier in the file" in bullet
    assert "hour" not in bullet and "head start" not in bullet and "at most 4 were independent lookups" in bullet
