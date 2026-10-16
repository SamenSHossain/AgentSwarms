"""A4 slot status (held / overturned / contested) and the analyst-reference outcome."""

import json
import math
import re

import pandas as pd
import pytest

from swarmprov.adapters.base import AdapterConfig
from swarmprov.adapters.wiki import WikiAdapter
from swarmprov.analysis import errors
from swarmprov.rules import norm_value

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
FIRST_TIE = [(0, "A"), (0, "B")] + _spec(("A", 5, 0.5), ("B", 10, 3.5))   # both first seen at 0 h; B then takes over
MIXED = _spec(("A", 6, 0), ("B", 1, 3.1), ("A", 3, 3.3), ("B", 8, 4.2))    # B leads 3-6 h, but not 3-4 h


def test_held():
    s = _stats(HELD)
    assert (s["status"], s["winner"], s["first_variant"]) == ("held", "A", "A")
    assert s["t_winner_majority"] is not None and math.isnan(s["hours_to_overtake"])   # nothing was overtaken


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


def test_first_seen_tie_names_no_first_variant():
    s = _stats(FIRST_TIE)
    assert s["first_tie"] and s["first_variant"] is None and s["winner"] == "B"
    assert s["status"] == "contested" and math.isnan(s["hours_to_overtake"])
    assert errors.outcome(s, "B") == "unresolved"


def _relabel(spec, a, b):
    return [(h, a if v == "A" else b) for h, v in spec]


@pytest.mark.parametrize("spec", [HELD, OVERTURNED, HUNGARY, SMALL, TIE, FIRST_TIE, MIXED])
def test_status_does_not_depend_on_labels(spec):
    """'Maryland' < 'Montana' once decided the G5 status; swapping the alphabetical order must change nothing."""
    s, z = _stats(spec), _stats(_relabel(spec, "Z", "A"))
    assert z["status"] == s["status"] and z["first_tie"] == s["first_tie"]
    assert z["t_winner_majority"] == s["t_winner_majority"]


def test_bin_width_is_a_parameter():
    s3, s1 = _stats(MIXED), _stats(MIXED, bin_h=1)
    assert s3["t_winner_majority"] == T0 + pd.Timedelta(hours=3.1)     # B first at 3.1 h, inside the B-majority 3-6 h bin
    assert s1["t_winner_majority"] == T0 + pd.Timedelta(hours=4)       # A still leads 3-4 h
    assert s1["hours_to_overtake"] == pytest.approx(0.9) and s3["status"] == s1["status"] == "overturned"


def test_small_tail_cannot_overturn():
    s = _stats(SMALL)
    assert s["n_tail"] == 3 and s["t_winner_majority"] is not None
    assert s["status"] == "contested (small n)"
    assert _stats(SMALL, min_tail=3)["status"] == "overturned"


def test_tail_share_is_a_parameter():
    assert _stats(HUNGARY, tail_share=1)["status"] == "held"      # A leads all 21 carriers 16 to 5


@pytest.mark.parametrize("spec, ref, label", [
    (HELD, "A", "truth held"),
    (HELD, "B", "error persisted"),
    (HELD, "C", "reference not among variants"),
    (OVERTURNED, "B", "error corrected"),
    (OVERTURNED, "A", "truth displaced by error"),
    (OVERTURNED, "C", "reference not among variants"),
    (HUNGARY, "B", "unresolved (error first)"),   # contested: no verdict either way
    (HUNGARY, "A", "unresolved (truth first)"),
    (SMALL, "B", "unresolved (error first)"),
    (TIE, "A", "unresolved (truth first)"),
])
def test_outcome_labels(spec, ref, label):
    s = _stats(spec)
    status = s["status"]
    assert errors.outcome(s, ref) == label
    assert s["status"] == status


@pytest.mark.parametrize("spec, first, later, ref, label", [
    (OVERTURNED, "16.4", "16.38", "16.38", "precision variant: reference prevailed"),
    (HUNGARY, "9.9", "9.91", "9.91", "precision variant: unresolved"),
    (HELD, "9.9", "9.91", "9.91", "precision variant: reference did not prevail"),
    (HELD, "14.59", "14.6", "14.59", "precision variant: reference prevailed"),
    (OVERTURNED, "14.59", "14.6", "14.59", "precision variant: reference did not prevail"),
    (OVERTURNED, "12.4", "11.7", "11.7", "error corrected"),
    (HUNGARY, "9.9", "9.97", "9.97", "unresolved (error first)"),
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
    assert refs == {("oecd-equity", "Poland"): "16.38", ("fam", "Item"): "16.40%", ("x", "G5"): "Montana (observed)"}
    assert AdapterConfig.from_dict({"reference_values": {"a|b": 1.5}}).reference_values == {"a|b": 1.5}
    match = errors.match_reference
    assert match(["16.4", "16.38"], refs[("fam", "Item")]) == "16.4"
    assert match(["Maryland (RNG forecast)", "Montana (observed)"], "montana (observed)") == "Montana (observed)"
    assert match(["$1,418", "$1,481"], "1,481") == match(["$1,418", "$1,481"], 1481) == "$1,481"
    assert match(["9.9", "9.91"], "9.97") is None


@pytest.mark.parametrize("ref", ["$1,481", "1,481", 1481])
def test_declared_labels_match_numeric_references(ref):
    s = _stats(_relabel(OVERTURNED, "$1,418", "$1,481"))
    assert errors.outcome(s, ref) == "error corrected"
    assert errors.outcome(s, "1,500") == "reference not among variants"


def test_wiki_reference_values_are_the_live_tooltip_values():
    assert WikiAdapter().config().reference_values == {
        "oecd-equity|Poland": 16.38, "oecd-equity|Hungary": 9.91, "oecd-equity|Slovak Republic": 14.59}


def test_wiki_g5_patterns_skip_the_forecast_runner_up_list():
    d = WikiAdapter().config().disputes[0]
    forecast = "seed 1905228 predicts G5 **Maryland** (52,395), then HI, MT, IA, WV. Treat only as prep"
    observed = "CRITICAL UPDATE: G5 independently relayed by Apr20 cohort: **MONTANA = 8,553**"
    hits = {t: {v for v, rx in d["variants"].items() if re.search(rx, t)} for t in (forecast, observed)}
    assert hits == {forecast: {"Maryland (RNG forecast)"}, observed: {"Montana (observed)"}}


def test_unused_references():
    refs = errors.reference_keys({"f|A": 1, "f|B": 2, "f|Typo": 3, "no-bar": 4})
    items, disputed = {("f", "A")}, {("f", "A"), ("f", "B")}
    assert errors.unused_references(refs, items, disputed) == ["f|Typo", "no-bar|"]


@pytest.fixture(scope="module")
def fourteen_slots(tmp_path_factory):
    """A chat whose 14 items each carry two values, three agents apiece: 14 disputed value slots."""
    from swarmprov import pipeline
    d = tmp_path_factory.mktemp("fourteen")
    items = ["Poland", "Hungary", "Chile", "Spain", "Italy", "France", "Germany", "Austria", "Belgium", "Denmark",
             "Finland", "Ireland", "Japan", "Norway"]
    msgs = [{"id": f"m{6 * i + a}", "timestamp": (T0 + pd.Timedelta(minutes=6 * i + a)).isoformat(),
             "author": f"agent-{a}", "text": f"I answered {item} {1 + (a >= 3) + i}.5."}
            for i, item in enumerate(items) for a in range(6)]
    (d / "chat.jsonl").write_text("".join(json.dumps(m) + "\n" for m in msgs))
    (d / "cfg.json").write_text(json.dumps({"bootstrap_reps": 0}))
    return pipeline.run_all(d / "chat.jsonl", d / "run", adapter="chat", config=str(d / "cfg.json"))


def test_every_disputed_slot_is_in_the_table(fourteen_slots):
    a4 = json.loads((fourteen_slots.path / "summary.json").read_text())["A4"]
    assert a4["n_value_slots_disputed"] == len(a4["rows"]) == 14 and sum(r["featured"] for r in a4["rows"]) == 4
    assert a4["status_counts"] == {"held": 0, "overturned": 0, "contested": 0, "contested (small n)": 14}
    text = (fourteen_slots.path / "report.md").read_text().split("## A4")[1].split("## A5")[0]
    assert "All 14 disputed slots are in the table: 14 contested (small n)." in text
    assert sum(line.startswith("| all \\| ") for line in text.splitlines()) == 14


def _true_values(truth):
    return truth[truth["correct"]].drop_duplicates(["family", "item"]).set_index(["family", "item"])["value"]


def test_synthetic_a4_against_truth(synth_run):
    run, truth = synth_run
    cfg = json.loads((run.path.parent / "config.json").read_text())
    true_vals = _true_values(truth)
    assert all(cfg["reference_values"][f"{f}|{i}"] == v for (f, i), v in true_vals.items())
    summary = json.loads((run.path / "summary.json").read_text())["A4"]
    rows = summary["rows"]
    assert rows and [r["featured"] for r in rows] == [True] * min(4, len(rows)) + [False] * (len(rows) - 4)
    for r in rows:
        fam, item = r["slot"].split(" | ")
        ref = r["reference (analyst input)"]
        assert float(ref) == true_vals[(fam, item)]
        assert r["status"] in ("held", "overturned", "contested", "contested (small n)")
        assert r["corrected"] == (r["status"] == "overturned")
        assert sum(r["tail counts"].values()) == int(r["late carriers (winner/total)"].split("/")[1].split()[0])
        assert r["outcome"].startswith("unresolved") == r["status"].startswith("contested")
        if r["first variant"] == ref and r["winner"] not in (None, ref):        # the old 'corrected' mislabel
            assert r["outcome"] in ("truth held", "truth displaced by error", "unresolved (truth first)")
        # missing values are JSON null, never the text 'None' / 'NaT' or a NaN
        assert not any(v in ("None", "NaT") or (isinstance(v, float) and math.isnan(v)) for v in r.values())
    assert any(r["first variant"] == r["reference (analyst input)"] != r["winner"] for r in rows)
    assert any(r["winner majority from"] is None and r["hours to overtake"] is None for r in rows)
    # every true value the swarm never posted is listed, so a mistyped key would be too
    posted = set(zip(truth["family"], truth["item"]))
    assert summary["references_unknown"] == sorted(k for k in cfg["reference_values"] if tuple(k.split("|")) not in posted)
    assert summary["references_unknown"] and "references_outside_table" not in summary
    assert summary["status_counts"] == {k: sum(r["status"] == k for r in rows) for k in errors.STATUSES}
    a4 = (run.path / "report.md").read_text().split("## A4")[1].split("## A5")[0]
    assert "NaT" not in a4 and "analyst input" in a4 and "| status |" in a4
    assert "last 1/3 of carriers" in a4 and "3-hour bin" in a4 and "at least 5 late carriers in all" in a4
    assert f"`{summary['references_unknown'][0]}`" in a4


@pytest.fixture(scope="module")
def synth200(tmp_path_factory):
    """A 200-agent swarm through extraction and the mention index only (no exposure stage): enough
    carriers to overturn slots."""
    from swarmprov import claims, pipeline, synth
    from swarmprov.schema import RunDir
    d = tmp_path_factory.mktemp("synth200")
    path = synth.write(d / "transcript.jsonl", n_agents=200, copy_rate=0.7, seed=0)
    run = RunDir(d / "run")
    pipeline.ingest(path, run, adapter="chat", config=str(d / "config.json"))
    pipeline.extract(run)
    ev, cl = run.read("events"), run.read("claims")
    im = pipeline._matcher(json.loads((run.path / "items.json").read_text()))
    mn = claims.index_mentions(ev, ev["family"].fillna(""), run.read("agents"), im, set(zip(cl["family"], cl["item"])))
    return errors.value_slots(mn, "agent_merged"), _true_values(pd.read_csv(d / "truth.csv"))


def test_synthetic_outcomes_follow_truth(synth200):
    slots, true_vals = synth200
    seen = set()
    for s in slots:
        truth = norm_value(str(true_vals[(s["family"], s["item"])]))
        out, first, win, st = errors.outcome(s, truth), s["first_variant"], s["winner"], s["status"]
        seen |= {st, out}
        standing = win if st == "overturned" else first
        if truth not in s["variants"]:
            assert out == "reference not among variants"
        elif st.startswith("contested"):
            assert out == f"unresolved ({'truth' if first == truth else 'error'} first)"
        else:
            assert out == {(True, True): "truth held", (False, True): "error corrected",
                           (True, False): "truth displaced by error", (False, False): "error persisted"}[
                (first == truth, standing == truth)]
    assert {"held", "overturned", "truth held", "error corrected", "truth displaced by error"} <= seen
