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
    """timed: a threaded chat (replies point at the round's first message); untimed: no clock, no threads."""
    t0 = pd.Timestamp("2026-07-07T10:00:00Z")
    msgs = [{"id": f"m{r}{k}", "author": a, "text": f"R{r} CONFIRMED: {item} arrived; answered {val} at +{3 + k}s."}
            | ({"timestamp": (t0 + pd.Timedelta(minutes=30 * r + 5 * k)).isoformat()} if timed else {})
            | ({"reply_to": f"m{r}0"} if timed and k else {})
            for r, (item, val) in enumerate([("Utah", "73.74"), ("Idaho", "41.20")], start=1)
            for k, a in enumerate(["Planner", "Coder", "Critic"])]
    (d / "chat.jsonl").write_text("\n".join(json.dumps(m) for m in msgs))
    # untimed: no task family, so no rounds reach exposure (which needs a clock); the report must still build
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


def test_threading_row_does_not_claim_reply_edges(runs):
    """The threaded chat has reply links; the graph never reads them, and the row must say so."""
    run = runs["chat"]
    assert json.loads((run.path / "profile.json").read_text())["capabilities"]["has_threading"] is True
    assert "| has_threading | yes | reply links are recorded but not used; edges are still inferred from timing |" \
        in (run.path / "report.md").read_text()
    assert set(run.read("edges_merged")["kind"]) <= {"relay", "relay_xchannel", "exposure", "citation"}


def test_untimed_transcript_reports_without_a_clock(runs):
    run = runs["chat_untimed"]
    caps = json.loads((run.path / "profile.json").read_text())["capabilities"]
    assert all(isinstance(v, bool) for v in caps.values()) and caps["has_wall_clock"] is False
    text = (run.path / "report.md").read_text()
    assert "6 posts from 3 author strings, no timestamps." in text
    assert "| has_wall_clock | no | no reliable wall clock" in text


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
