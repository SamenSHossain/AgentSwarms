"""A4 slot status (held / overturned / contested) and the analyst-reference outcome."""

import json

import pandas as pd
import pytest

from swarmprov.adapters.base import AdapterConfig
from swarmprov.adapters.wiki import WikiAdapter
from swarmprov.analysis import errors

T0 = pd.Timestamp("2026-06-20", tz="UTC")


def _stats(spec, **rule):
    """spec: (hour, variant) per carrier, one agent each."""
    car = pd.DataFrame({"ts": [T0 + pd.Timedelta(hours=h) for h, _ in spec],
                        "agent": [f"a{i}" for i in range(len(spec))], "variant": [v for _, v in spec]})
    return errors._slot_stats(car, "fam | item", **rule)


def _spec(*groups):
    """groups: (variant, n, start hour); carriers 10 minutes apart, off the 3-hour bin edges."""
    return [(h0 + i / 6, v) for v, n, h0 in groups for i in range(n)]


HELD = _spec(("A", 10, 0), ("B", 3, 3.5), ("A", 6, 6.5))
OVERTURNED = _spec(("A", 6, 0), ("B", 12, 3.5))                      # B holds every bin from 3 h on
HUNGARY = _spec(("A", 14, 0), ("B", 4, 3.5), ("A", 2, 6.5), ("B", 1, 7))   # B leads the tail, loses the last bin
SMALL = _spec(("A", 4, 0), ("B", 5, 3.5))                            # 3 late carriers, all B
TIE = _spec(("A", 8, 0), ("B", 1, 3.5), ("A", 1, 4), ("B", 1, 4.5), ("A", 1, 5))


def test_held():
    s = _stats(HELD)
    assert (s["status"], s["winner"], s["first_variant"]) == ("held", "A", "A")


def test_overturned_needs_a_lasting_majority():
    s = _stats(OVERTURNED)
    assert (s["status"], s["winner"]) == ("overturned", "B")
    assert s["tail_counts"] == {"B": 6} and s["n_tail"] == 6
    assert s["t_winner_majority"] == T0 + pd.Timedelta(hours=3.5)     # the bin opens at 3 h, B first at 3.5 h
    assert s["hours_to_overtake"] == 0


def test_late_lead_without_a_lasting_majority_is_contested():
    s = _stats(HUNGARY)
    assert s["winner"] == "B" and s["tail_counts"] == {"B": 5, "A": 2}
    assert s["t_winner_majority"] is None and s["status"] == "contested"


@pytest.mark.parametrize("order", [1, -1])
def test_tied_tail_is_contested_whatever_the_order(order):
    head, tail = TIE[:8], TIE[8:]
    tail = [(h, v) for (h, _), (_, v) in zip(tail, tail[::order])]
    s = _stats(head + tail)
    assert s["tie"] and s["winner"] is None and s["status"] == "contested"
    assert s["tail_counts"] == {"A": 2, "B": 2}
    assert s["t_winner_first"] is None and s["t_winner_majority"] is None


def test_small_tail_cannot_overturn():
    s = _stats(SMALL)
    assert s["n_tail"] == 3 and s["t_winner_majority"] is not None
    assert s["status"] == "contested (small n)"
    assert _stats(SMALL, min_tail=3)["status"] == "overturned"


def test_tail_share_is_a_parameter():
    assert _stats(HUNGARY, tail_share=1)["status"] == "held"      # A leads all 21 carriers 16 to 5


def _relabel(spec, a, b):
    return [(h, a if v == "A" else b) for h, v in spec]


@pytest.mark.parametrize("spec, ref, label", [
    (HELD, "A", "truth held"),
    (HELD, "B", "error persisted"),
    (HELD, "C", "error persisted"),
    (OVERTURNED, "B", "error corrected"),
    (OVERTURNED, "A", "truth displaced by error"),
    (OVERTURNED, "C", "error persisted"),
    (HUNGARY, "B", "error persisted"),           # contested: the first variant still stands
    (HUNGARY, "A", "truth held"),
    (SMALL, "B", "error persisted"),
])
def test_outcome_labels(spec, ref, label):
    s = _stats(spec)
    status = s["status"]
    assert errors.outcome(s, ref) == label
    assert s["status"] == status


@pytest.mark.parametrize("spec, first, later, ref, label", [
    (OVERTURNED, "16.4", "16.38", "16.38", "precision variant: reference prevailed"),
    (HUNGARY, "9.9", "9.91", "9.91", "precision variant: reference did not prevail"),
    (HELD, "14.59", "14.6", "14.59", "precision variant: reference prevailed"),
    (OVERTURNED, "14.59", "14.6", "14.59", "precision variant: reference did not prevail"),
    (OVERTURNED, "12.4", "11.7", "11.7", "error corrected"),
    (HUNGARY, "9.9", "9.97", "9.97", "error persisted"),
])
def test_precision_variants(spec, first, later, ref, label):
    assert errors.outcome(_stats(_relabel(spec, first, later)), ref) == label


@pytest.mark.parametrize("a, b, ok", [
    ("9.9", "9.91", True), ("9.91", "9.9", True), ("16.4", "16.38", True), ("14.6", "14.59", True),
    ("10", "9.96", True), ("25276", "25276.4", True), ("9.9", "9.96", False), ("9.91", "9.91", False),
    ("9.69", "9.7", True), ("14751", "25276", False), ("Montana", "Maryland", False), ("nan", "9.9", False),
])
def test_rounds_to(a, b, ok):
    assert errors._rounds_to(a, b) is ok


def test_reference_keys_match_variant_keys():
    refs = errors.reference_keys({"oecd-equity | Poland": 16.38, "fam|Item": "16.40%", "x|G5": "Montana (observed)"})
    assert refs == {("oecd-equity", "Poland"): "16.38", ("fam", "Item"): "16.4", ("x", "G5"): "Montana (observed)"}
    assert AdapterConfig.from_dict({"reference_values": {"a|b": 1.5}}).reference_values == {"a|b": 1.5}


def test_wiki_reference_values_are_the_live_tooltip_values():
    assert WikiAdapter().config().reference_values == {
        "oecd-equity|Poland": 16.38, "oecd-equity|Hungary": 9.91, "oecd-equity|Slovak Republic": 14.59}


def test_synthetic_a4_against_truth(synth_run):
    run, truth = synth_run
    cfg = json.loads((run.path.parent / "config.json").read_text())
    true_vals = truth[truth["correct"]].drop_duplicates(["family", "item"]).set_index(["family", "item"])["value"]
    assert all(cfg["reference_values"][f"{f}|{i}"] == v for (f, i), v in true_vals.items())
    rows = json.loads((run.path / "summary.json").read_text())["A4"]["featured"]
    assert rows
    for r in rows:
        fam, item = r["slot"].split(" | ")
        ref = r["reference (analyst input)"]
        assert float(ref) == true_vals[(fam, item)]
        assert r["status"] in ("held", "overturned", "contested", "contested (small n)")
        assert r["corrected"] == (r["status"] == "overturned")
        assert sum(r["tail counts"].values()) == int(r["late carriers (winner/total)"].split("/")[1].split()[0])
        if r["outcome"] == "error corrected":
            assert r["winner"] == ref and r["first variant"] != ref
        if r["first variant"] == ref and r["winner"] not in (None, ref):        # the old 'corrected' mislabel
            assert r["outcome"] in ("truth held", "truth displaced by error")
    assert any(r["first variant"] == r["reference (analyst input)"] != r["winner"] for r in rows)
    a4 = (run.path / "report.md").read_text().split("## A4")[1].split("## A5")[0]
    assert "NaT" not in a4 and "analyst input" in a4 and "| status |" in a4
    assert "last 1/3 of carriers" in a4 and "3-hour bin" in a4 and "at least 5 late carriers" in a4
