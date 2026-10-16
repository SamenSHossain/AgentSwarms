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
    rows.append((3, "a1", "f", "P", "reading the page first"))   # a1 was on the carrying page before adopting
    c = _contagion(rows)
    assert (c["n_adopters"], c["n_on_carrying"], c["share_on_carrying"]) == (10, 10, 1.0)
    agents = c["agents"].set_index("agent")
    assert agents.loc["a2", "baseline_share"] == pytest.approx(1 / 5) and agents.loc["a1", "baseline_share"] == 2 / 6
    assert c["baseline"] == pytest.approx((9 / 5 + 2 / 6) / 10) and c["p"] < 1e-6
    assert agents.loc["a1", "t_exposed"] == T0 + pd.Timedelta(minutes=3) and c["n_exposed_before"] == 1
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


def test_a_technique_nobody_adopted_has_no_test():
    c = _contagion([(0, "o", "f", "P", "blobtrick"), (1, "a", "f", "P", "no")])
    assert (c["n_adopters"], c["n_on_carrying"], c["n_exposed_before"]) == (0, 0, 0) and np.isnan(c["p"])
    assert c["agents"].empty and np.isnan(c["baseline"])


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
                            "wrong_flag": wrong, "correct_flag": False}
                           for k, (item, v, wrong) in enumerate([("i", "9.69", False), ("i", "9.7", False),
                                                                 ("i", None, False), ("i", "9.69", True),
                                                                 ("j", "3", False)])])
    mentions = pd.DataFrame([{"event_id": "m0", "ts": T0, "agent_merged": "z", "channel": "P", "family": "f",
                              "item": "i", "value_norm": "9.7", "value_key": "9.7", "source_kind": None}])
    ex = exposure.build(rounds, mentions, {("f", "i"): "9.7", ("f", "j"): "3"}, "agent_merged",
                        winners={("f", "i"): "9.69"})
    y = ex["y_winner"]
    assert (y[0], y[1], y[3]) == (1.0, 0.0, 0.0) and np.isnan(y[2])   # a report's own wrong flag comes first
    assert np.isnan(y[4]) and ex["y_consensus"][4] == 1.0             # j is not overturned
    assert ex["y_consensus"].iloc[[0, 1, 3]].tolist() == [0.0, 1.0, 0.0]          # unchanged: scored on the consensus
    no_win = exposure.build(rounds, mentions, {("f", "i"): "9.7", ("f", "j"): "3"}, "agent_merged")
    assert no_win["y_winner"].isna().all() and no_win["y_consensus"].equals(ex["y_consensus"])


def test_every_family_is_kept_beside_the_printed_cut():
    ex = pd.DataFrame({"family": ["big"] * 12 + ["small"] * 3, "item_fe": [f"big|{i % 6}" for i in range(12)] + ["s|0"] * 3,
                       "agent": [f"a{i}" for i in range(15)], "D": [1, 0] * 6 + [1, 1, 0], "D_w10m": 0, "D_w60m": 0,
                       "self_prepared": 0, "y_instant": 0, "gap_s": 60.0, "episode": 1.0})
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
    with pytest.raises(SystemExit):
        cli.main(["export", str(tmp_path / "missing"), "-o", str(out)])
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
    row = {"excluded": None, "n_adopters": 6, "n_on_carrying": 6, "share_on_carrying": 1.0, "baseline": 0.5, "p": 0.016}
    note = _contagion_note({"few": row, "zzz": {"excluded": "3 of its 4 posts match its pattern only by naming a page"},
                            "many": row | {"n_adopters": 40, "p": 0.2}})
    assert "**consistent with contagion** (none here)" in note       # few is too small, many is not significant
    assert "zzz is not tested: 3 of its 4 posts match its pattern only by naming a page." in note
    assert note.endswith("Fewer than 10 adopters, too few to read: few (6).\n")
