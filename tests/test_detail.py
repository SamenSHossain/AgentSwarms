"""The detail behind A1/A3/A4 (adoption and dispute rows, every family, strict summary.json), A3's contagion test,
A4's variant provenance and y_winner, and `swarmprov export`."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swarmprov import cli, exposure, pipeline
from swarmprov import export as export_mod
from swarmprov.adapters.base import Technique
from swarmprov.analysis import diffusion, errors, provenance, uncertainty
from swarmprov.claims import value_key

T0 = pd.Timestamp("2026-06-20", tz="UTC")
BLOB = Technique("blob", "blobtrick", ("f",))
COMMITTED = Path(__file__).resolve().parents[1] / "results" / "openai-wiki"


def _posts(rows):
    """rows: (minute, agent, family, page, text) -> (tags, events); a post uses `blob` when its text says blobtrick."""
    ev = pd.DataFrame([{"event_id": f"e{i:03d}", "ts": T0 + pd.Timedelta(minutes=m), "agent_merged": a, "family": f,
                        "channel": c, "text": t} for i, (m, a, f, c, t) in enumerate(rows)])
    tags = ev.assign(techniques=np.where(ev["text"].str.contains("blobtrick"), "blob", ""))
    return tags.drop(columns=["channel", "text"]), ev


def test_adoption_rows_are_the_km_inputs():
    rows = [(0, "o", "f", "P", "blobtrick")]
    rows += [(60 * i, f"a{i}", "f", "Q", "blobtrick" if i <= 6 else "no") for i in range(1, 10)]   # a7-a9 never adopt
    rows += [(600, "x", "g", "Q", "blobtrick"), (-5, "y", "f", "Q", "before the first post")]
    tags, _ = _posts(rows)
    r = diffusion.analyze(tags, [BLOB], "agent_merged")["blob"]
    ad = r["adoptions"]
    assert list(ad.columns) == diffusion.ADOPTION_COLS
    risk = ad[ad["in_at_risk"]]
    assert (len(risk), int(risk["adopted"].sum())) == (r["at_risk"], r["adopters"]) == (9, 6)
    dur = np.where(risk["adopted"], risk["hours_from_first_seen"], risk["censored_at_h"])
    pd.testing.assert_frame_equal(uncertainty.km_greenwood(dur, risk["adopted"].astype(int)), r["km"])
    assert risk.loc[risk["adopted"], "hours_from_first_seen"].median() == r["median_hours_among_adopters"] == 3.5
    assert risk.loc[~risk["adopted"], "censored_at_h"].tolist() == [7.0, 8.0, 9.0]
    origin, outside = ad.set_index("agent").loc["o"], ad.set_index("agent").loc["x"]
    assert origin["is_originator"] and not origin["in_at_risk"] and origin["hours_from_first_seen"] == 0
    assert outside["adopted"] and not outside["in_at_risk"] and not outside["is_originator"]   # it posts in g only
    assert "y" not in set(ad["agent"])                    # posted only before the first appearance


def test_permutation_p_is_the_exact_within_agent_shuffle():
    eligible = [(1, 2), (1, 4), (0, 3)]                    # (posts on a carrying page, eligible posts) per adopter
    probs = [k / n for k, n in eligible]
    for k in range(4):
        hits = [sum(c) for c in itertools.product(*[[1] * a + [0] * (n - a) for a, n in eligible])]
        assert diffusion.permutation_p(probs, k) == pytest.approx(np.mean([h >= k for h in hits]))
    assert diffusion.permutation_p([0.5, 0.5], 2) == pytest.approx(0.25) and diffusion.permutation_p([0.3], 0) == 1


def _contagion(rows, tech=BLOB):
    tags, ev = _posts(rows)
    return diffusion.contagion(tags, ev, [tech], "agent_merged")[tech.name]


def test_adoption_after_exposure_reads_as_contagion():
    """Ten adopters post on four quiet pages, then adopt on the page the originator posted it on."""
    rows = [(0, "o", "f", "P", "blobtrick")]
    for i in range(1, 11):
        rows += [(10 * i + k, f"a{i}", "f", f"Q{k}", "no") for k in range(4)] + [(10 * i + 5, f"a{i}", "f", "P", "blobtrick")]
    rows += [(4, "a1", "f", "P", "reading the page first"), (3, "a1", "f", "P", "and before that")]   # a1 was there twice
    c = _contagion(rows)
    assert (c["n_adopters"], c["n_on_carrying"], c["share_on_carrying"]) == (10, 10, 1.0)
    agents = c["agents"].set_index("agent")
    assert agents.loc["a2", "baseline_share"] == pytest.approx(1 / 5) and agents.loc["a1", "baseline_share"] == 3 / 7
    assert c["baseline"] == pytest.approx((9 / 5 + 3 / 7) / 10) and c["p"] < 1e-6 and c["p_holm"] == c["p"]
    assert agents.loc["a1", "t_exposed"] == T0 + pd.Timedelta(minutes=3) and c["n_exposed_before"] == 1   # the first
    assert agents["adoption_channel"].eq("P").all() and agents.loc["a2", "n_eligible_posts"] == 5


def test_adoption_away_from_exposure_does_not():
    """Adopters post on the carrying page but adopt on a page of their own; posts in another family do not count."""
    rows = [(0, "o", "f", "P", "blobtrick")]
    for i in range(1, 11):
        rows += [(10 * i + k, f"a{i}", "f", "P" if k < 2 else "Q", "no") for k in range(4)]
        rows += [(10 * i + 5, f"a{i}", "f", f"own{i}", "blobtrick"), (10 * i + 6, f"a{i}", "g", "P", "elsewhere")]
    c = _contagion(rows)
    assert (c["n_on_carrying"], c["share_on_carrying"]) == (0, 0.0)
    assert c["agents"]["baseline_share"].eq(2 / 5).all() and c["baseline"] == pytest.approx(0.4) and c["p"] == 1.0


def test_a_page_carries_only_what_another_agent_posted_strictly_before():
    rows = [(0, "o", "f", "P", "blobtrick"), (5, "b", "f", "R", "blobtrick"), (5, "c", "f", "R", "blobtrick"),
            (7, "d", "f", "S", "no"), (8, "d", "f", "S", "blobtrick"), (9, "d", "f", "S", "no")]
    agents = _contagion(rows)["agents"].set_index("agent")
    assert not agents.loc["b", "adopted_on_carrying_page"] and not agents.loc["c", "adopted_on_carrying_page"]
    assert not agents.loc["d", "adopted_on_carrying_page"]          # its own use never makes its page carry it
    assert agents.loc["d", "baseline_share"] == 0 and pd.isna(agents.loc["d", "t_exposed"])


def test_a_page_carries_for_its_first_poster_once_another_agent_posts_there():
    """a is the first to post the technique on R; R carries it for a only after b has posted it there too."""
    rows = [(0, "o", "f", "P", "blobtrick"), (10, "a", "f", "R", "blobtrick"), (15, "a", "f", "R", "no"),
            (20, "b", "f", "R", "blobtrick"), (30, "a", "f", "R", "no")]
    tags, ev = _posts(rows)
    agents = diffusion.contagion(tags, ev, [BLOB], "agent_merged")["blob"]["agents"].set_index("agent")
    assert not agents.loc["a", "adopted_on_carrying_page"] and agents.loc["b", "adopted_on_carrying_page"]
    first = pd.DataFrame({"agent_merged": ["a"], "ts": [T0 + pd.Timedelta(minutes=10)],
                          "ts2": [T0 + pd.Timedelta(minutes=20)]}, index=pd.Index(["R"], name="channel"))
    mine = tags.assign(channel=ev["channel"])[lambda d: d["agent_merged"] == "a"]
    assert diffusion._carrying(mine, first, "agent_merged").tolist() == [False, False, True]   # at 10, 15 and 30
    assert agents.loc["a", "baseline_share"] == pytest.approx(1 / 3) and agents.loc["b", "baseline_share"] == 1


def test_holm_adjusts_for_the_techniques_tested():
    p = pd.Series({"rng": 0.0033, "clock": 0.0058, "blob": 0.0382, "beacon": 0.0868, "tooltip": 0.8146})
    assert diffusion.holm(p).round(4).to_dict() == {"rng": 0.0165, "clock": 0.0232, "blob": 0.1146, "beacon": 0.1736,
                                                    "tooltip": 0.8146}
    assert diffusion.holm(pd.Series({"a": 0.4, "b": 0.3})).tolist() == [0.6, 0.6]       # step-down keeps the order
    rows = [(0, "o", "f", "P", "blobtrick")]
    rows += [r for i in range(1, 4) for r in ((10 * i, f"a{i}", "f", f"Q{i}", "no"), (10 * i + 1, f"a{i}", "f", "P", "blobtrick"))]
    tags, ev = _posts(rows)
    tags["techniques"] = tags["techniques"].replace("blob", "blob,blob2")
    c = diffusion.contagion(tags, ev, [BLOB, Technique("blob2", "blobtrick", ("f",))], "agent_merged")
    assert c["blob"]["p"] == c["blob2"]["p"] == 0.5 ** 3 and c["blob"]["p_holm"] == c["blob2"]["p_holm"] == 0.25


def test_a_technique_nobody_adopted_has_no_test():
    c = _contagion([(0, "o", "f", "P", "blobtrick"), (1, "a", "f", "P", "no")])
    assert (c["n_adopters"], c["n_on_carrying"], c["n_exposed_before"]) == (0, 0, 0) and np.isnan(c["p"])
    assert c["agents"].empty and np.isnan(c["baseline"]) and "p_holm" not in c


def test_a_pattern_that_matches_page_names_is_not_tested():
    zzz = Technique("blob", "ZZZ", ("f",))
    rows = [(0, "o", "f", "dse/LinksZZZ", "the list"), (1, "a", "f", "dse/Home", "see LinksZZZ for the list"),
            (2, "b", "f", "dse/Home", "a ZZZ backup page")]
    tags, ev = _posts(rows)
    tags["techniques"] = np.where(ev["text"].str.contains("ZZZ"), "blob", "")
    c = diffusion.contagion(tags, ev, [zzz], "agent_merged")["blob"]
    assert c["n_posts_page_name_only"] == 1 and "1 of its 2 posts match its pattern only by naming a page" in c["excluded"]
    assert "agents" not in c
    assert diffusion._names_only(ev["text"], "ZZZ", ["dse/Other"]).sum() == 0   # no page name matches: nothing to strip


def test_a_pattern_that_also_names_pages_is_tested_and_flagged():
    """Every post matches on its own words, so the technique is tested, but the report says which adoptions landed on
    a page whose name matches the pattern."""
    from swarmprov.report import _contagion_note
    beacon = Technique("beacon", "beacon", ("f",))
    rows = [(0, "o", "f", "P", "beacon trick"), (1, "o", "f", "dse/BeaconAlert", "beacon trick"),
            (2, "a1", "f", "Q", "no"), (3, "a1", "f", "dse/BeaconAlert", "a beacon here"),
            (5, "a2", "f", "Q", "beacon too"), (6, "a3", "f", "dse/BeaconAlert", "no"), (7, "a3", "f", "P", "beacon")]
    tags, ev = _posts(rows)
    tags["techniques"] = np.where(ev["text"].str.contains("beacon"), "beacon", "")
    c = diffusion.contagion(tags, ev, [beacon], "agent_merged")
    b = c["beacon"]
    assert b["excluded"] is None and b["n_posts_page_name_only"] == 0 and b["n_page_names"] == 1
    assert (b["n_adopters"], b["n_on_carrying"], b["n_adopted_on_named_page"], b["n_carrying_on_named_page"]) == (3, 2, 1, 1)
    assert ("beacon's pattern also matches 1 page name, where a post can match because of the page: 1 of its 3 "
            "adoptions landed on one, 1 of the 2 on a carrying page.") in _contagion_note(c)


def _slot(spec, kind="value"):
    car = pd.DataFrame({"ts": [T0 + pd.Timedelta(hours=h) for h, _, _ in spec], "agent": [a for _, a, _ in spec],
                        "variant": [v for _, _, v in spec]})
    s = errors._slot_stats(car, "f | i")
    s.update(family="f", item="i", kind=kind, agents_per_variant=car.groupby("variant")["agent"].nunique().to_dict())
    return s


def test_dispute_rows_keep_every_carrier_and_the_status():
    s = _slot([(0, "a0", "9.7"), (1, "a1", "9.7"), (2, "a2", "9.69"), (3, "a0", "9.69"), (4, "a3", "9.69")])
    d = errors.dispute_rows([s])
    assert list(d.columns) == errors.DISPUTE_COLS and len(d) == 5
    assert d.groupby("variant")["agent"].nunique().to_dict() == s["agents_per_variant"] == {"9.69": 3, "9.7": 2}
    assert set(d.loc[d["carried_both"], "agent"]) == {"a0"} and s["overlap_agents"] == 1
    assert (d["is_first_variant"] == (d["variant"] == "9.7")).all() and (d["is_winner"] == (d["variant"] == s["winner"])).all()
    assert d["status"].eq(s["status"]).all() and d["t_first"].min() == T0
    assert list(errors.dispute_rows([]).columns) == errors.DISPUTE_COLS


def test_precision_variants_and_the_consensus_flag():
    assert errors.precision_variants(["9.69", "9.7"]) and errors.precision_variants(["9.9", "9.91"])
    assert not errors.precision_variants(["1.25", "1.32"]) and not errors.precision_variants(["5"])
    s = _slot([(0, "a0", "9.7"), (1, "a1", "9.69"), (2, "a2", "9.69")])
    assert s["winner"] == "9.69"
    assert errors.consensus_disagrees(s, "9.7") and not errors.consensus_disagrees(s, "9.69")
    assert not errors.consensus_disagrees(s, None) and not errors.consensus_disagrees(s | {"kind": "sequence"}, "9.7")


def test_variant_provenance_counts_answer_rounds_not_carriers():
    s = _slot([(0, "a0", "9.7"), (1, "a1", "9.69"), (2, "a2", "9.69")])
    ex = pd.DataFrame([("9.69", 0, 1, np.nan), ("9.69", 0, 0, np.nan), ("9.69", 1, 1, 7200), ("9.7", 1, 1, 3600),
                       ("9.7", 1, 1, 10800), (None, 1, 1, 60), ("12", 0, 1, np.nan)],
                      columns=["value_norm", "D", "D_cons", "gap_s"]).assign(family="f", item="i")
    p = errors.variant_provenance(s, pd.concat([ex, ex.assign(item="other")]), "9.7")
    assert p["by_variant"]["9.69"] == {"answer_rounds": 3, "derived_independently": 2, "independent_dissent": 1,
                                       "adopted_after_public": 1, "median_head_start_h": 2.0}
    assert p["by_variant"]["9.7"] == {"answer_rounds": 2, "derived_independently": 0, "independent_dissent": 0,
                                      "adopted_after_public": 2, "median_head_start_h": 2.0}
    assert (p["n_inherited"], p["n_other_value"]) == (1, 1)   # the round with no value would inflate the consensus


def test_y_winner_scores_overturned_slots_only_beside_y_consensus():
    t = T0 + pd.Timedelta(hours=1)
    rounds = pd.DataFrame([{"claim_id": f"c{k}", "report_event_id": f"r{k}", "agent": f"a{k}", "family": "f",
                            "episode": 1, "item": item, "value_norm": v, "t_report": t, "latency_class": None,
                            "wrong_flag": wrong, "correct_flag": right}
                           for k, (item, v, wrong, right) in enumerate([
                               ("i", "9.69", False, False), ("i", "9.7", False, False), ("i", None, False, False),
                               ("i", "9.69", True, False), ("j", "3", False, False), ("i", None, False, True),
                               ("i", "9.7", False, True)])])
    mentions = pd.DataFrame([{"event_id": "m0", "ts": T0, "agent_merged": "z", "channel": "P", "family": "f",
                              "item": "i", "value_norm": "9.7", "value_key": "9.7", "source_kind": None}])
    ex = exposure.build(rounds, mentions, {("f", "i"): "9.7", ("f", "j"): "3"}, "agent_merged",
                        winners={("f", "i"): "9.69"})
    y = ex["y_winner"]
    assert (y[0], y[1], y[3]) == (1.0, 0.0, 1.0) and np.isnan(y[2])   # the stated value, whatever the report's flag
    assert np.isnan(y[5]) and y[6] == 0.0                              # a correct flag grades the task, not the winner
    assert np.isnan(y[4]) and ex["y_consensus"][4] == 1.0             # j is not overturned
    assert ex["y_consensus"].iloc[[0, 1, 3, 5, 6]].tolist() == [0.0, 1.0, 0.0, 1.0, 1.0]   # scored as before
    no_win = exposure.build(rounds, mentions, {("f", "i"): "9.7", ("f", "j"): "3"}, "agent_merged")
    assert no_win["y_winner"].isna().all() and no_win["y_consensus"].equals(ex["y_consensus"])


def test_every_family_is_kept_beside_the_printed_cut():
    ex = pd.DataFrame({"family": ["big"] * 12 + ["small"] * 3, "item_fe": [f"big|{i % 6}" for i in range(12)] + ["s|0"] * 3,
                       "agent": [f"a{i}" for i in range(15)], "D": [1, 0] * 6 + [1, 1, 0], "D_w10m": 0, "D_w60m": 0,
                       "self_prepared": 0, "self_originated": 0, "y_instant": 0, "gap_s": 60.0, "episode": 1.0})
    r = provenance.summarize(ex, boot=50)
    assert list(r["by_family"].index) == ["big"] and list(r["by_family_all"].index) == ["big", "small"]
    pd.testing.assert_frame_equal(r["by_family_all"].loc[["big"]], r["by_family"])   # the same row, intervals included
    assert r["by_family_all"]["n"].sum() == len(ex) and np.isnan(r["by_family_all"].loc["small", "exposed_lo"])


def test_synthetic_report_persists_the_detail(synth_run):
    run, _ = synth_run
    text = (run.path / "report.md").read_text()
    raw = (run.path / "summary.json").read_text()
    summary = json.loads(raw, parse_constant=lambda c: pytest.fail(f"summary.json holds {c}"))
    assert "NaT" not in raw and '"nan"' not in raw
    a4 = summary["A4"]
    split = a4["consensus_disagrees"]
    assert len(split["slots"]) == 3 and split["n_value_slots"] == 3 and split["answer_rounds_stating_winner"] == 9
    assert "**3 of 3 slots: consensus differs from the later majority (9 rounds).**" in text
    assert "**Warning:** on 3 disputed slots the consensus value that `y_consensus` scores against is not A4's winner" in text
    assert "y_winner" not in {r["outcome"] for r in summary["A2"]}   # no overturned slot: no y_winner row
    header = next(line for line in text.split("## A4")[1].splitlines() if line.startswith("| slot |"))
    assert header.startswith("| slot | kind | variants (agents) | agents carrying both | first variant | winner | status | "
                             "late carriers (winner/total) | winner first seen | winner majority from | hours to overtake | "
                             "reference (analyst input) | outcome |")      # the existing columns, then the new ones
    assert header.endswith("| independent / after-public rounds | consensus ≠ winner | precision variants |")
    d = run.read("disputes_merged")
    for r in a4["rows"]:
        assert r["winner first seen"] is None or "T" in r["winner first seen"]       # ISO timestamps
        mine = d[d["slot"] == r["slot"]]
        assert ", ".join(f"{v} ({n})" for v, n in mine.groupby("variant")["agent"].nunique().items()) == r["variants (agents)"]
        assert mine.loc[mine["carried_both"], "agent"].nunique() == r["agents carrying both"]
        prov = r["variant_provenance"]["by_variant"]
        listed = [x.rsplit(" (", 1)[0] for x in r["variants (agents)"].split(", ")]
        assert r["independent / after-public rounds"] == "; ".join(
            f"{v}: {prov[v]['derived_independently']} / {prov[v]['adopted_after_public']}" for v in listed)
    fam = run.read("a1_by_family_merged")
    assert fam["n"].sum() == summary["A1"]["n_answers"] and set(fam["family"]) == {"alpha-task", "beta-task"}
    assert not run.has("adoptions_merged")                # no technique is configured


def test_synthetic_wrong_values_are_derived_once_then_copied(synth_run):
    """The swarm's wrong answers are computed by one agent (D = 0) and copied after that (D = 1)."""
    run, truth = synth_run
    ex = run.read("exposures_merged")
    true = truth[truth["correct"]].drop_duplicates(["family", "item"]).set_index(["family", "item"])["value"]
    ex = ex.assign(key=ex["value_norm"].map(value_key),
                   true=[f"{true[(f, i)]:g}" for f, i in zip(ex["family"], ex["item"])])
    wrong = ex[ex["key"].notna() & (ex["key"].astype(float) != ex["true"].astype(float))].sort_values("t_report")
    assert wrong.groupby(["family", "item", "key"]).ngroups >= 5
    first = ~wrong.duplicated(["family", "item", "key"])
    assert wrong.loc[first, "D"].eq(0).all() and wrong.loc[~first, "D"].eq(1).all() and (~first).sum() >= 5
    rows = json.loads((run.path / "summary.json").read_text())["A4"]["rows"]
    for r in rows:                                         # so a disputed wrong variant was derived independently once
        fam, item = r["slot"].split(" | ")
        for v, p in r["variant_provenance"]["by_variant"].items():
            if p["answer_rounds"] and float(v) != float(f"{true[(fam, item)]:g}"):
                assert p["derived_independently"] == 1 and p["adopted_after_public"] == p["answer_rounds"] - 1


def test_export_writes_the_mirrors_and_the_detail(synth_run, tmp_path, capsys):
    run, _ = synth_run
    out = tmp_path / "out"
    assert cli.main(["export", str(run.path), "-o", str(out)]) == 0
    assert "adoptions: no adoptions_merged.parquet" in capsys.readouterr().out
    ex = pd.read_csv(out / "exposures_merged.csv")
    assert list(ex.columns) == export_mod.EXPOSURE_COLUMNS and len(ex) == len(run.read("exposures_merged"))
    edges = pd.read_csv(out / "edges_exposure_citation_merged.csv")
    assert list(edges.columns) == export_mod.EDGE_COLUMNS and set(edges["kind"]) <= {"exposure", "citation"}
    if COMMITTED.is_dir():                                 # the committed results/ mirrors have these headers
        assert list(pd.read_csv(COMMITTED / "exposures_merged.csv", nrows=0).columns) == export_mod.EXPOSURE_COLUMNS
        assert list(pd.read_csv(COMMITTED / "edges_exposure_citation.csv", nrows=0).columns) == export_mod.EDGE_COLUMNS
    assert len(pd.read_csv(out / "disputes_merged.csv")) == len(run.read("disputes_merged"))
    assert cli.main(["export", str(run.path), "-o", str(out), "--tables", "exposures,claims", "--format", "parquet",
                     "--mapping", "strict", "--all-columns"]) == 0
    pd.testing.assert_frame_equal(pd.read_parquet(out / "exposures_strict.parquet"), run.read("exposures_strict"))
    assert (out / "claims.parquet").exists()
    with pytest.raises(SystemExit):
        cli.main(["export", str(run.path), "-o", str(out), "--tables", "exposures,nope"])
    assert cli.main(["export", str(tmp_path / "missing"), "-o", str(out)]) == 2
    assert not (tmp_path / "missing").exists()


@pytest.fixture(scope="module")
def tech_run(tmp_path_factory):
    """A timed chat on several pages: agents adopt a trick on the page where another agent had posted it."""
    d = tmp_path_factory.mktemp("contagion")
    msgs = [{"timestamp": "2026-05-04T10:00:00Z", "name": "origin", "channel": "lab", "content": "the blobtrick works"}]
    for i in range(10):
        msgs += [{"timestamp": f"2026-05-04T{11 + i}:00:00Z", "name": f"agent-{i}", "channel": f"desk-{i}",
                  "content": "working on it"},
                 {"timestamp": f"2026-05-04T{11 + i}:30:00Z", "name": f"agent-{i}", "channel": "lab",
                  "content": "trying the blobtrick here too"}]
    (d / "chat.json").write_text(json.dumps(msgs))
    (d / "cfg.json").write_text(json.dumps({"techniques": [{"name": "blob", "pattern": "blobtrick"}]}))
    return pipeline.run_all(d / "chat.json", d / "run", adapter="chat", config=str(d / "cfg.json"))


def test_report_a3_shows_contagion_and_persists_adoptions(tech_run):
    text = (tech_run.path / "report.md").read_text()
    a3 = text.split("## A3")[1].split("## A4")[0]
    assert "| adopted on carrying page | baseline | p (within-agent permutation) |" in a3
    assert "| 10/10 (100%) | 50% | <0.001 |" in a3                   # 0.5 ** 10
    assert "**consistent with contagion** (blob), not proof of it" in a3 and "too few to read" not in a3
    c = json.loads((tech_run.path / "summary.json").read_text())["A3"][0]["contagion"]
    assert (c["n_adopters"], c["n_on_carrying"], c["baseline"], c["excluded"]) == (10, 10, 0.5, None)
    assert c["p"] == pytest.approx(0.5 ** 10)
    ad = tech_run.read("adoptions_merged")
    assert ad["in_at_risk"].sum() == 10 and ad["adopted_on_carrying_page"].sum() == 10
    assert ad.loc[ad["is_originator"], "adopted_on_carrying_page"].isna().all()
    assert "`adoptions_merged.parquet` lists every at-risk agent" in a3


def test_contagion_note_warns_on_small_n_and_names_untested_techniques():
    from swarmprov.report import _contagion_note
    row = {"excluded": None, "n_adopters": 6, "n_on_carrying": 6, "share_on_carrying": 1.0, "baseline": 0.5, "p": 0.016,
           "p_holm": 0.048, "n_page_names": 0}
    note = _contagion_note({"few": row, "zzz": {"excluded": "3 of its 4 posts match its pattern only by naming a page"},
                            "many": row | {"n_adopters": 40, "p": 0.2, "p_holm": 0.4},
                            "lost": row | {"n_adopters": 40, "p": 0.03, "p_holm": 0.06}})
    assert "with p < 0.05 after Holm's correction for the 3 techniques tested is **consistent with contagion** " \
           "(none here)" in note                                    # few is too small, many and lost fail Holm
    assert "Below 0.05 only before the correction: lost (p = 0.030, 0.060 after it)." in note
    assert "zzz is not tested: 3 of its 4 posts match its pattern only by naming a page." in note
    assert note.endswith("Fewer than 10 adopters, too few to read: few (6).\n") and "page name" not in note
    one = _contagion_note({"many": row | {"n_adopters": 40, "p": 0.01, "p_holm": 0.01}})
    assert "with p < 0.05 is **consistent with contagion** (many)" in one and "Holm" not in one   # one test: no correction


def test_export_of_an_empty_exposure_table_keeps_the_mirror_header(tech_run, tmp_path):
    assert tech_run.read("exposures_merged").empty
    assert cli.main(["export", str(tech_run.path), "-o", str(tmp_path), "--tables", "exposures"]) == 0
    assert list(pd.read_csv(tmp_path / "exposures_merged.csv").columns) == export_mod.EXPOSURE_COLUMNS


@pytest.fixture(scope="module")
def win_run(tmp_path_factory):
    """A timed chat with three overturned value slots: Poland (answer rounds, the consensus wins), Chile (predictions
    only: no answer rounds) and Hungary (the consensus 9.90 loses to 9.91; one 9.91 report says it scored); and a
    family of three answers, too small for A1's printed table."""
    d = tmp_path_factory.mktemp("winners")

    def post(h, m, i, channel, text):
        return {"timestamp": f"2026-05-04T{h:02d}:{m:02d}:00Z", "name": f"agent-{i}", "channel": channel, "content": text}
    msgs = [post(0 if i < 4 else 6, i, i, "big", f"R1 CONFIRMED: Poland arrived; answered {'16.40' if i < 4 else '16.38'} "
                 "at +3s.") for i in range(15)]
    msgs += [post(1 if i < 4 else 7, i, i, "big", f"expected Chile {'3.10' if i < 4 else '3.15'} on the board")
             for i in range(13)]
    msgs += [post(9 if i < 10 else 14, i, i, "big", f"R2 CONFIRMED: Hungary arrived; answered {'9.90' if i < 10 else '9.91'} "
                  "instantly." + (" Scored." if i == 15 else "")) for i in range(16)]
    msgs += [post(10, i, i, "small", "R1 CONFIRMED: Norway arrived; answered 4.4 at +3s.") for i in range(3)]
    (d / "chat.json").write_text(json.dumps(msgs))
    (d / "cfg.json").write_text(json.dumps({"family_from_channel": True, "extra_items": {"big": ["Chile"]},
                                            "bootstrap_reps": 0}))
    return pipeline.run_all(d / "chat.json", d / "run", adapter="chat", config=str(d / "cfg.json"))


def test_a2_counts_every_overturned_slot_and_what_y_consensus_scores(win_run):
    text = (win_run.path / "report.md").read_text()
    summary = json.loads((win_run.path / "summary.json").read_text())
    over = [r["slot"] for r in summary["A4"]["rows"] if r["status"] == "overturned"]
    assert sorted(over) == ["big | Chile", "big | Hungary", "big | Poland"]
    ex = win_run.read("exposures_merged")
    assert ex.loc[ex["y_winner"].notna(), "item"].value_counts().to_dict() == {"Hungary": 16, "Poland": 15}
    assert "on the value slots A4 calls overturned only (3 here, 2 with answer rounds that state a value, 31 rounds in " \
           "all)" in text
    assert "and 5 of the 6 answer rounds there that state the winner score y_consensus = 0, the rest 1 by their report's " \
           "own correct flag (see A4)" in text
    assert "**1 of 3 slots: consensus differs from the later majority (6 rounds).**" in text
    assert summary["A4"]["consensus_disagrees"] | {"slots": None} == {
        "slots": None, "n_value_slots": 3, "answer_rounds_stating_winner": 6, "of_them_y_consensus_0": 5}
    hu = ex[(ex["item"] == "Hungary") & (ex["value_norm"] == "9.91")]
    assert (hu["y_consensus"] == 0).sum() == 5 and hu["y_winner"].eq(1).all()


def test_a1_names_the_families_its_table_leaves_out(win_run):
    text = (win_run.path / "report.md").read_text()
    a1 = text.split("## A1")[1].split("## A2")[0]
    assert "| big | 31 |" in a1 and "| small |" not in a1
    assert "1 smaller family (fewer than 10 answers, 3 answers in all) is not shown; `a1_by_family_merged.parquet` " \
           "lists all 2." in a1
    fam = win_run.read("a1_by_family_merged").set_index("family")
    assert fam["n"].to_dict() == {"big": 31, "small": 3}


def test_export_says_which_report_to_run_for_the_other_mapping(win_run, tmp_path, capsys):
    assert cli.main(["export", str(win_run.path), "-o", str(tmp_path), "--mapping", "strict",
                     "--tables", "disputes,exposures"]) == 0
    out = capsys.readouterr().out
    assert (f"disputes: no disputes_strict.parquet in {win_run.path} (the report was built under the merged mapping; run "
            f"`swarmprov report {win_run.path} --mapping strict` first, which writes report_strict.md and summary_strict.json under "
            "this one), skipped") in out
    assert "exposures_strict.csv" in out


def test_a4_header_without_a_split_slot_says_so():
    from swarmprov.report import _variant_note, _zero_clause
    note = _variant_note([{}, {}], [], 0, 0, {}, "merged")
    assert note.startswith("**0 of 2 slots: consensus differs from the later majority (0 rounds).**")
    assert "on no value slot here is it a value other than the winner (*consensus ≠ winner*)." in note
    assert "it is not the winner" not in note and "score y_consensus = 0" not in note
    assert _zero_clause(1, 1) == "the 1 answer round there that states the winner scores y_consensus = 0"
    assert _zero_clause(68, 68) == "the 68 answer rounds there that state the winner score y_consensus = 0"
