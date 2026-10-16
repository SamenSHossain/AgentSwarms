"""Identity resolution: text-lead cohorts only where authors are parsed, case variants resolved together,
and the alias audit."""

import json

import pandas as pd
import pytest

from swarmprov import claims, exposure, graph, identity, pipeline, report
from swarmprov.adapters import AdapterConfig
from swarmprov.analysis import errors

AGENTS = ["planner", "coder", "analyst", "critic"]


def _events(rows):
    """rows: (author, text, family)"""
    ev = pd.DataFrame([{"author_raw": a, "text": t, "channel": "c", "channel_family": f} for a, t, f in rows])
    return ev, ev["channel_family"]


def _datelead():
    return _events([(a, f"May 5 standup: R{r} CONFIRMED: Utah answered 7{r}.0", "gdp") for r in (1, 2, 3) for a in AGENTS])


@pytest.mark.parametrize("explicit, setting, n_merged", [
    (True, None, 4),      # explicit authors: a date the messages open with is not a cohort
    (False, None, 1),     # parsed signatures: the wiki habit of opening with the cohort ("Mar16 R3 ...") is used
    (True, True, 1),      # a chat source that signs cohorts can opt in
    (False, False, 4),    # ... and a parsed source can opt out
])
def test_text_lead_cohort_follows_capability_and_config(explicit, setting, n_merged):
    ev, fam = _datelead()
    ag = identity.resolve(ev, fam, AdapterConfig(name="t", cohort_from_text=setting), explicit_author=explicit)
    assert ag["agent_merged"].nunique() == n_merged
    if n_merged == 1:
        assert set(ag["agent_merged"]) == {"May05|gdp"}


def test_cohort_in_the_author_string_is_always_used():
    ev, fam = _events([("SynthAlphaAgentSep26", "R1 answered 1", "alpha"), ("SynthAlphaAgentSep26b", "R1 answered 2", "alpha")])
    ag = identity.resolve(ev, fam, AdapterConfig(name="t", cohort_from_text=False), explicit_author=True)
    assert set(ag["cohort"]) == {"Sep26"} and set(ag["agent_merged"]) == {"Sep26|alpha"}


def test_case_variants_share_one_merged_id():
    """The wiki's OpenAIResearcher (posts open with Nov22) and OpenAiResearcher (unsigned): one strict handle,
    so one merged id, with the family taken over both variants' posts."""
    ev, fam = _events([("OpenAIResearcher", "Nov22 R1 answered 1", "a")] * 2
                      + [("OpenAiResearcher", "R2 answered 2", "b")] * 3 + [("Other", "R1 answered 3", "a")])
    ag = identity.resolve(ev, fam, AdapterConfig(name="t"))
    assert ag.groupby("agent_strict")["agent_merged"].nunique().max() == 1
    a = ag.set_index("author_raw")
    assert a.loc["OpenAIResearcher", "agent_merged"] == a.loc["OpenAiResearcher", "agent_merged"] == "Nov22|b"
    assert a.loc["OpenAiResearcher", "cohort"] == "Nov22" and a.loc["OpenAiResearcher", "family"] == "b"
    assert a.loc["OpenAIResearcher", "n_events"] == 2 and a.loc["OpenAiResearcher", "n_events"] == 3


def test_conflicting_cohorts_stay_split_and_are_audited():
    ev, fam = _events([("AgentX", "Nov22 R1 answered 1", "f"), ("agentx", "Dec05 R1 answered 1", "f"),
                       ("Solo", "R1 answered 2", "f")])
    ag = identity.resolve(ev, fam, AdapterConfig(name="t"))
    a = ag.set_index("author_raw")
    assert (a.loc["AgentX", "agent_merged"], a.loc["agentx", "agent_merged"]) == ("Nov22|f", "Dec05|f")
    al = identity.alias_table(ag).set_index("author_raw")
    assert al.loc["AgentX", "n_strict_aliases"] == 2 and al.loc["AgentX", "n_merged_for_strict"] == 2
    assert al.loc["Solo", "n_merged_for_strict"] == 1


def test_alias_table_sorts_by_merged_group_size():
    ev, fam = _events([("AlphaNov27", "x", "f"), ("BetaNov27", "x", "f"), ("BetaNov27", "y", "f"), ("Gamma", "z", "f")])
    al = identity.alias_table(identity.resolve(ev, fam, AdapterConfig(name="t")))
    assert list(al["author_raw"]) == ["BetaNov27", "AlphaNov27", "Gamma"]
    assert list(al["n_merged_aliases"]) == [2, 2, 1] and list(al["n_strict_aliases"]) == [1, 1, 1]
    assert list(al.columns[:6]) == ["agent_merged", "agent_strict", "author_raw", "cohort", "family", "n_events"]


def _chat(d, cfg: dict):
    t0 = pd.Timestamp("2026-05-05T09:00:00Z")
    msgs = []
    for r, (item, val) in enumerate([("Utah", "73.74"), ("Idaho", "41.20"), ("Ohio", "12.50")], start=1):
        for k, a in enumerate(AGENTS):
            t = t0 + pd.Timedelta(minutes=60 * r + 10 * k)
            msgs.append({"id": f"m{r}{k}", "author": a, "timestamp": t.isoformat(), "channel": "standup",
                         "text": f"May 5 standup: R{r} CONFIRMED: {item} arrived {t:%H:%M:%S}, 14s timer; answered {val} at +{5 + k}s."})
    (d / "datelead.jsonl").write_text("\n".join(json.dumps(m) for m in msgs))
    (d / "cfg.json").write_text(json.dumps({"families": {"gdp": "standup"}} | cfg))
    return pipeline.run_all(d / "datelead.jsonl", d / "run", adapter="chat", config=str(d / "cfg.json"))


@pytest.fixture(scope="module")
def datelead_runs(tmp_path_factory):
    return {name: _chat(tmp_path_factory.mktemp(name), cfg)
            for name, cfg in (("default", {}), ("opt_in", {"cohort_from_text": True}))}


def test_explicit_authors_with_date_leads_stay_distinct(datelead_runs):
    run = datelead_runs["default"]
    assert run.read("agents")["agent_merged"].nunique() == 4
    text = (run.path / "report.md").read_text()
    assert "**12 answers given by 4 agents;" in text
    assert "Identity merges: none; each of the 4 merged ids is a single author string" in text
    assert "both identity mappings group the author strings identically" in text   # one line, no sensitivity numbers
    al = pd.read_csv(run.path / "agent_aliases.csv")
    assert len(al) == 4 and set(al["n_merged_aliases"]) == {1}


def test_merges_on_an_explicit_author_source_are_flagged(datelead_runs):
    run = datelead_runs["opt_in"]
    text = (run.path / "report.md").read_text()
    assert "Identity merges: 1 of 1 merged ids joins more than one author string (4 strings in all); largest: `May05|gdp` (4)." in text
    assert ("**Warning:** this source names its authors explicitly. 1 merge joins handles it keeps apart (a shared cohort "
            "date token and task family, or a roster); compare the strict-mapping sensitivity lines.") in text
    assert "_Sensitivity_: under the other identity mapping, 12 agent-rounds" in text and "identically" not in text
    al = pd.read_csv(run.path / "agent_aliases.csv")
    assert set(al["agent_merged"]) == {"May05|gdp"} and set(al["n_merged_aliases"]) == {4}


def _agents(rows):
    """rows: (author_raw, agent_strict, agent_merged)"""
    return pd.DataFrame(rows, columns=["author_raw", "agent_strict", "agent_merged"])


CASE = [("Planner", "planner", "planner"), ("planner", "planner", "planner"), ("Coder", "coder", "coder")]
COHORT = [("AlphaNov27", "alphanov27", "Nov27|f"), ("BetaNov27", "betanov27", "Nov27|f")]


def test_case_merges_are_counted_apart_from_cohort_merges():
    """Both mappings fold case, so the strict sensitivity line cannot show a case merge; only cohort and roster
    merges are pointed at it."""
    line = report._identity_merges(_agents(CASE), explicit=True)
    assert line.startswith("Identity merges: 1 of 2 merged ids joins more than one author string (2 strings in all)")
    assert "Of these, 1 only joins case variants of one handle" in line
    assert "differ only in case are one agent under both mappings" in line and "compare the strict-mapping" not in line
    line = report._identity_merges(_agents(CASE + COHORT), explicit=True)
    assert "2 of 3 merged ids join more than one" in line and "Of these, 1 only joins case variants" in line
    assert "1 merge joins handles it keeps apart" in line and "compare the strict-mapping sensitivity lines" in line
    line = report._identity_merges(_agents(COHORT), explicit=True)
    assert "case" not in line and "1 merge joins handles it keeps apart" in line
    assert "**Warning:**" not in report._identity_merges(_agents(CASE + COHORT), explicit=False)


def test_identity_merges_none_reads_in_the_singular():
    assert report._identity_merges(_agents(CASE[2:]), explicit=False) == (
        "Identity merges: none; the only merged id is a single author string (`agent_aliases.csv` lists both mappings).")


def test_one_to_one_only_where_both_mappings_group_the_authors_alike():
    assert identity.one_to_one(_agents(CASE)) == {"planner": "planner", "coder": "coder"}   # case folds in both
    assert identity.one_to_one(_agents(CASE + COHORT)) is None                                 # merged joins two handles
    assert identity.one_to_one(_agents([("A", "a", "M1"), ("a", "a", "M2")])) is None          # a handle kept apart


def test_a_one_to_one_run_relabels_its_strict_tables_and_reports_one_line(synth_run):
    """The seed-2 swarm's two mappings group its 60 authors alike, so expose and graph write the strict tables by
    relabelling the merged ones: they hold what computing under the strict labels gives.  The report says so in one
    line and repeats no estimate under the other mapping."""
    run, _ = synth_run
    labels = identity.one_to_one(run.read("agents"))
    assert len(labels) == 60 and all(k != v for k, v in labels.items())
    cl, mn = run.read("claims"), run.read("mentions")
    slots = errors.value_slots(claims.public(mn), "agent_strict")
    win = {(s["family"], s["item"]): s["winner"] for s in slots if s["status"] == "overturned"}
    ex = exposure.build(claims.agent_rounds(cl, "agent_strict"), mn, claims.consensus(cl, mn), "agent_strict", winners=win)
    pd.testing.assert_frame_equal(run.read("exposures_strict"), ex, check_dtype=False)
    rel, chains = graph.relay_edges(claims.public(mn), "agent_strict")
    stored = run.read("edges_strict")
    stored = stored[stored["kind"].isin(graph.RELAY_KINDS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(stored[rel.columns], rel, check_dtype=False)
    pd.testing.assert_frame_equal(run.read("chains_strict"), chains, check_dtype=False)
    text = (run.path / "report.md").read_text()
    assert ("_Sensitivity_: both identity mappings group the author strings identically (each merged agent is one "
            "strict agent), so every number in this report is the same under the strict mapping") in text
    assert "under the other identity mapping" not in text and "Sensitivity (other identity mapping)" not in text
    assert "credit every answer; answers in brackets" in text   # the origins table has one mapping's columns
    s = json.loads((run.path / "summary.json").read_text())
    assert "other_mapping" not in s["A2_dose"]
    by = s["A5"]["origins"]["by_mapping"]   # computed once, kept under both names as a two-mapping run keeps them
    assert list(by) == ["merged", "strict"] and by["strict"] == by["merged"]
