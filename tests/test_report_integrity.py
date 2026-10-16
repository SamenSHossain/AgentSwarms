"""Generated reports state only what holds for their own data: capability consequences match the
capability, no wiki-only coverage text, no empty headings, no dangling figure links."""

import dataclasses
import json
import re
import shutil

import pandas as pd
import pytest

from swarmprov import pipeline, report, village
from swarmprov.adapters import Capabilities
from swarmprov.schema import RunDir

ROW = re.compile(r"^\| (\w+) \| (yes|no) \| (.*) \|$", re.M)


def _chat(d, timed: bool):
    """timed: a threaded chat (replies point at the round's first message); untimed: no clock, no threads, no families."""
    t0 = pd.Timestamp("2026-07-07T10:00:00Z")
    msgs = [{"id": f"m{r}{k}", "author": a, "text": f"R{r} CONFIRMED: {item} arrived; answered {val} at +{3 + k}s."}
            | ({"timestamp": (t0 + pd.Timedelta(minutes=30 * r + 5 * k)).isoformat()} if timed else {})
            | ({"reply_to": f"m{r}0"} if timed and k else {})
            for r, (item, val) in enumerate([("Utah", "73.74"), ("Idaho", "41.20")], start=1)
            for k, a in enumerate(["Planner", "Coder", "Critic"])]
    (d / "chat.jsonl").write_text("\n".join(json.dumps(m) for m in msgs))
    # untimed: ordered by message position (a sequence clock) and carried to A1 by the chat's catch-all family
    (d / "cfg.json").write_text(json.dumps({"families": {"states": "CONFIRMED"}} if timed else {}))
    return pipeline.run_all(d / "chat.jsonl", d / "run", adapter="chat", config=str(d / "cfg.json"))


@pytest.fixture(scope="module")
def runs(synth_run, tmp_path_factory):
    return {"synthetic": synth_run[0], "chat": _chat(tmp_path_factory.mktemp("chat"), True),
            "chat_untimed": _chat(tmp_path_factory.mktemp("untimed"), False)}


@pytest.fixture(params=["synthetic", "chat", "chat_untimed"])
def run(request, runs):
    return runs[request.param]


def test_capability_rows_state_the_consequence_of_the_value(run):
    rows = ROW.findall((run.path / "report.md").read_text())
    assert {k for k, _, _ in rows} == {f.name for f in dataclasses.fields(Capabilities)}
    for k, avail, text in rows:
        yes, no = report.CONSEQUENCES[k]
        assert text == (yes if avail == "yes" else no), k
        assert avail == "yes" or yes not in text, k


def test_no_heading_without_content(run):
    lines = [x for x in (run.path / "report.md").read_text().splitlines() if x.strip()]
    empty = [a for a, b in zip(lines, lines[1:] + ["#"]) if a.startswith("#") and b.startswith("#")]
    assert not empty
    assert "village" not in json.loads((run.path / "summary.json").read_text())


def test_no_wiki_coverage_text(run):
    text = (run.path / "report.md").read_text()
    assert "143 surfaces" not in text and "Discord" not in text
    assert "Only the posts in this transcript are searched for earlier copies" in text
    assert "authors come from the source's author field, case-folded" in text and "parsed from signatures" not in text


def test_every_figure_link_resolves(run):
    links = re.findall(r"\]\((figures/[^)]+)\)", (run.path / "report.md").read_text())
    assert all((run.path / f).exists() for f in links), links


def test_synthetic_report_links_its_figures(runs):
    assert "](figures/a1_provenance.png)" in (runs["synthetic"].path / "report.md").read_text()


def test_threaded_chat_turns_reply_links_into_reply_edges(runs):
    """The threaded chat's replies (each round's two replies to its first message) become 'reply' edges from the
    parent's agent; A5 counts them for brokers, not for supply, and the report says so."""
    run = runs["chat"]
    assert json.loads((run.path / "profile.json").read_text())["capabilities"]["has_threading"] is True
    e = run.read("edges_merged")
    rep = e[e["kind"] == "reply"]
    assert sorted(zip(rep["src"], rep["dst"], rep["src_event_id"])) == [
        ("planner", "coder", "m10"), ("planner", "coder", "m20"), ("planner", "critic", "m10"), ("planner", "critic", "m20")]
    text = (run.path / "report.md").read_text()
    assert "0 explicit citations, and 4 replies; a reply is an interaction, not a knowledge transfer" in text
    assert "Top brokers (betweenness on relay + citation + reply graph" in text
    a5 = json.loads((run.path / "summary.json").read_text())["A5"]
    assert a5["n_reply_edges"] == 4 and a5["n_exposure_edges"] == 4 and a5["exposure_sources"] == 1


def test_untimed_transcript_reports_without_a_clock(runs):
    run = runs["chat_untimed"]
    caps = json.loads((run.path / "profile.json").read_text())["capabilities"]
    assert all(isinstance(v, bool) for v in caps.values()) and caps["has_wall_clock"] is False
    text = (run.path / "report.md").read_text()
    assert "6 posts from 3 author strings, ordered by message position (no wall clock)." in text
    assert "| has_wall_clock | no | no reliable wall clock" in text and "| has_order | yes |" in text


def test_report_reads_string_capabilities_of_older_runs(runs, tmp_path):
    """Runs written before as_dict cast to bool stored a numpy False as the string "False"."""
    shutil.copytree(runs["chat_untimed"].path, tmp_path / "run")
    prof_path = tmp_path / "run" / "profile.json"
    prof = json.loads(prof_path.read_text())
    prof["capabilities"] = {k: str(v) for k, v in prof["capabilities"].items()}
    prof_path.write_text(json.dumps(prof))
    text = report.build(RunDir(tmp_path / "run"))
    assert "| has_wall_clock | no | no reliable wall clock" in text
    assert "| has_explicit_author | yes |" in text


def test_capabilities_are_plain_bools():
    import numpy as np
    d = Capabilities(has_wall_clock=np.bool_(False), has_threading=np.bool_(True)).as_dict()
    assert d["has_wall_clock"] is False and d["has_threading"] is True
    assert json.loads(json.dumps(d))["has_wall_clock"] is False


def test_every_capability_has_two_distinct_consequences():
    for f in dataclasses.fields(Capabilities):
        yes, no = report.CONSEQUENCES[f.name]
        assert yes and no and yes != no and yes not in no


def test_village_section_needs_village_data():
    assert village.section(village.summary(None, None), report.md_table) == []
    assert village.section({"family_from_roster": True, "unrecognised": ["x.csv"]}, report.md_table)


def test_a_strict_report_writes_beside_the_merged_one(synth_run, synth_strict, capsys):
    base, d = synth_run[0].path, synth_strict.path
    for f in ["report.md", "summary.json", "checks.json", *(f"figures/{g.name}" for g in (base / "figures").glob("*.png"))]:
        assert (d / f).stat().st_mtime_ns == (base / f).stat().st_mtime_ns, f   # copied, never rewritten
    text = (d / "report_strict.md").read_text()
    assert "Agent identity mapping: **strict**" in text and "Agent identity mapping: **merged**" in (d / "report.md").read_text()
    links = re.findall(r"\]\((figures/[^)]+)\)", text)
    assert links and all(f.endswith("_strict.png") and (d / f).exists() for f in links), links
    assert json.loads((d / "summary_strict.json").read_text())["mapping"] == "strict"
    assert "(`checks_strict.json`; `swarmprov check RUN --mapping strict` reruns them)" in text
    assert "and summary_strict.json against the tables" in text
    chk = json.loads((d / "checks_strict.json").read_text())
    assert {c["table"] for c in chk["checks"] if c["check"] == "summary_matches_tables"} == {"summary_strict.json"}
    assert all(synth_strict.has(f"{t}_strict") for t in ("a1_by_family", "disputes"))
    from swarmprov import cli
    assert cli.main(["check", str(d), "--mapping", "strict"]) == 0
    assert "summary_matches_tables       summary_strict.json" in capsys.readouterr().out


def test_clear_removes_both_mappings_reports(tmp_path):
    run = RunDir(tmp_path / "run")
    names = ["report.md", "summary.json", "checks.json", "report_strict.md", "summary_strict.json", "checks_strict.json"]
    for n in names:
        (run.path / n).write_text("{}")
    run.clear()
    assert not [n for n in names if (run.path / n).exists()]
