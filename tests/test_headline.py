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
# definitions that carry a digit but are no finding: the interval level and the top-10 share's name
DEFINITIONS = ("95% CI", "top 10", *(f"A{k}." for k in range(1, 7)))


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


def test_key_findings_read_the_summary_they_are_given(synth):
    _, s, _ = synth
    t = copy.deepcopy(s)
    t["A1"]["n_answers"], t["A6"]["task values"]["max_hops"] = 4321, 77
    text = "\n".join(report.key_findings(t))
    assert "4,321 answers" in text and "at most 77" in text


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
