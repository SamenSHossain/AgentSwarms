"""Framework traces: system/user turns are the task, tool results the calling agent's own reads.  Neither is a public
post, so no phantom `unknown` / `task` / `tool` agent becomes a source or a hub, and an agent's own tool output before
its answer is evidence of a lookup (``own_lookup_evidence``), never a copy.  Checked on a hand-written OpenAI trace and
on the synthetic swarm written as an untimed AutoGen stream, against its truth."""

import json

import pandas as pd
import pytest

from swarmprov import adapters, cli, pipeline, query, synth
from swarmprov.adapters import framework
from swarmprov.adapters.chat import ChatAdapter, _pick

PHANTOMS = {"unknown", "task", "tool", "user", "lookup"}
ROUNDS = [("Poland", "0.2"), ("Hungary", "-0.9"), ("Chile", "0.2")]
AGENTS = ["researcher-1", "researcher-2", "researcher-3"]
FAMILY_ROUNDS = [("alpha-task", "Poland", "0.2"), ("alpha-task", "Hungary", "-0.9"), ("beta-task", "Peru", "1.5")]


def _lookup(agent: str, call: str, item: str, result: str) -> list[dict]:
    """An OpenAI tool call that names no text, and its result."""
    return [{"role": "assistant", "name": agent, "content": None,
             "tool_calls": [{"id": call, "type": "function",
                             "function": {"name": "lookup", "arguments": json.dumps({"country": item})}}]},
            {"role": "tool", "tool_call_id": call, "name": "lookup", "content": result}]


def _openai() -> list[dict]:
    """researcher-1 looks each value up with a tool, then all three answer; the user prompt already gives Chile's, and
    researcher-3 names "the task's value" (no agent's name)."""
    msgs = [{"role": "system", "content": "You are a research swarm. Report 'R<n> CONFIRMED: <item> answered <value>'."},
            {"role": "user", "content": "Find the 2023 change for Poland, Hungary and Chile. Known so far: Chile 0.2."}]
    for r, (item, val) in enumerate(ROUNDS, start=1):
        msgs += _lookup("researcher-1", f"call_{r}", item, f"{item} {val} (2023)")
        msgs += [{"role": "assistant", "name": a, "content": f"R{r} CONFIRMED: {item} arrived; answered {val} at +{3 + k}s."
                  + (" It matches the task's value." if k == 2 else "")} for k, a in enumerate(AGENTS)]
    return msgs


@pytest.fixture(scope="module")
def oai(tmp_path_factory):
    d = tmp_path_factory.mktemp("openai")
    (d / "openai.json").write_text(json.dumps(_openai()))
    return pipeline.run_all(d / "openai.json", d / "run", adapter="chat")


@pytest.fixture(scope="module")
def configured(tmp_path_factory):
    """Configured families that neither the prompt nor the tool output names; researcher-1 works in both."""
    d = tmp_path_factory.mktemp("configured")
    msgs = [{"role": "user", "content": "Find the 2023 change for Poland, Hungary and Peru. Known so far: Peru 1.5."}]
    for r, (fam, item, val) in enumerate(FAMILY_ROUNDS, start=1):
        msgs += _lookup("researcher-1", f"call_{r}", item, f"{item} {val} (2023)")
        msgs += [{"role": "assistant", "name": a, "content": f"{fam} R{r} CONFIRMED: {item} arrived; answered {val}."}
                 for a in AGENTS]
    (d / "trace.json").write_text(json.dumps(msgs))
    (d / "config.json").write_text(json.dumps({"families": {"alpha-task": "alpha-task", "beta-task": "beta-task"}}))
    return pipeline.run_all(d / "trace.json", d / "run", adapter="chat", config=str(d / "config.json"))


@pytest.fixture(scope="module")
def autogen(tmp_path_factory):
    d = tmp_path_factory.mktemp("autogen")
    path = synth.write(d / "autogen.jsonl", n_agents=60, copy_rate=0.7, seed=2, fmt="autogen")
    run = pipeline.run_all(path, d / "run", adapter="chat", config=str(d / "config.json"))
    return run, pd.read_csv(d / "truth.csv"), path


def _role(m: dict):
    return framework.role(m, _pick(m, "author", {}))


def test_roles_of_openai_langchain_autogen_and_anthropic_messages():
    cases = [({"role": "system", "content": "x"}, "task"), ({"role": "user", "content": "x"}, "task"),
             ({"role": "developer", "content": "x"}, "task"), ({"role": "assistant", "name": "a", "content": "x"}, "agent"),
             ({"role": "tool", "name": "lookup", "content": "x"}, "tool"), ({"role": "function", "content": "x"}, "tool"),
             # a named user turn is an agent's message passed on (AutoGen 0.2 group chats, LangGraph supervisors)
             ({"role": "user", "name": "researcher-2", "content": "x"}, "agent"),
             ({"role": "user", "name": "user", "content": "x"}, "task"),
             ({"type": "human", "content": "x"}, "task"), ({"type": "system", "content": "x"}, "task"),
             ({"type": "ai", "name": "a", "content": "x"}, "agent"), ({"type": "tool", "content": "x"}, "tool"),
             ({"type": "TextMessage", "source": "user", "content": "x"}, "task"),
             ({"type": "TextMessage", "source": "planner", "content": "x"}, "agent"),
             ({"type": "TextMessage", "handle": "planner", "content": "x"}, "agent"),   # no speaker: still an agent turn
             ({"type": "ToolCallSummaryMessage", "source": "planner", "content": "x"}, "agent"),
             ({"type": "HandoffMessage", "source": "planner", "content": "x"}, "agent"),
             ({"type": "AssistantMessage", "source": "planner", "content": []}, "agent"),          # autogen-core
             ({"type": "SystemMessage", "content": "x"}, "task"),
             # an agent's inner events reach no other agent, whoever the source
             ({"type": "ToolCallRequestEvent", "source": "planner", "content": []}, "inner"),
             ({"type": "ThoughtEvent", "source": "planner", "content": "x"}, "inner"),
             ({"type": "ModelClientStreamingChunkEvent", "source": "planner", "content": "x"}, "inner"),
             ({"type": "UserInputRequestedEvent", "source": "user", "content": ""}, "inner"),
             ({"type": "ToolCallExecutionEvent", "source": "planner", "content": []}, "tool"),
             ({"type": "CodeExecutionEvent", "source": "planner", "result": {"output": "x"}}, "tool"),
             ({"type": "FunctionExecutionResultMessage", "content": []}, "tool"),
             ({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]}, "tool"),
             ({"type": "message", "user": "U1", "text": "x"}, None), ({"author": "a", "text": "x"}, None),
             ({"type": "Default", "author": {"name": "a"}, "content": "x"}, None)]   # Slack and Discord types are no roles
    assert [_role(m) for m, _ in cases] == [k for _, k in cases]


def test_tool_results_go_to_the_caller_by_call_id_else_the_one_pending_call():
    c = framework.Calls()
    c.open("a", ["1", "2"])
    c.open("b", ["3"])
    assert c.answer(["3"]) == "b" and c.answer(["2"]) == "a"
    assert c.answer([]) == "a"                      # only a's call 1 is left
    c.open("a", [None])
    c.open("b", [None])
    assert c.answer([]) is None                     # two calls pending, no id: nobody
    c.open("b", [])                                 # b speaks again: its unanswered call is closed
    assert c.answer(["x"]) == "a"


def test_a_result_naming_an_unseen_id_is_not_given_to_a_call_with_another_id():
    c = framework.Calls()
    c.open("b", ["y"])
    assert c.answer(["x1"]) is None and c.pending   # not b's: b's call has an id, and the result names another
    assert c.answer(["x1"], named="r1") == "r1"     # an AutoGen event names the agent whose tool ran
    assert c.answer(["y"], named="r1") == "b"       # a matching call id comes first
    c.open("a", ["1", "2"])
    assert c.answer([]) == "a" and c.answer([]) == "a" and not c.pending   # every pending call is a's


def test_autogen_inner_events_are_no_posts_and_tool_events_go_to_their_caller(tmp_path):
    p = tmp_path / "autogen.jsonl"
    msgs = [{"type": "TextMessage", "source": "user", "content": "Find Poland, Hungary and Chile."},
            {"type": "ThoughtEvent", "source": "r1", "content": "Poland is 0.2, I will report it"},
            {"type": "ModelClientStreamingChunkEvent", "source": "r1", "content": "Poland is"},
            {"type": "ToolCallRequestEvent", "source": "r2", "content": [{"id": "y", "name": "f", "arguments": "{}"}]},
            # no logged request: the event's source ran the tool (r2's pending call has another id)
            {"type": "ToolCallExecutionEvent", "source": "r1", "content": [{"call_id": "x1", "content": "Poland 0.2"}]},
            {"type": "CodeExecutionEvent", "source": "r3", "retry_attempt": 0,
             "result": {"exit_code": 0, "output": "Hungary -0.9"}},
            # autogen-core: the call ids are in the AssistantMessage's FunctionCalls
            {"type": "AssistantMessage", "source": "r4", "content": [{"id": "call_1", "name": "f", "arguments": "{}"}]},
            {"type": "FunctionExecutionResultMessage", "content": [{"call_id": "call_1", "content": "Chile 0.2"}]},
            {"type": "TextMessage", "source": "r1", "content": "R1 CONFIRMED: Poland arrived; answered 0.2"}]
    p.write_text("\n".join(json.dumps(m) for m in msgs))
    b = ChatAdapter().load(p)
    assert list(zip(b.events["author_raw"], b.events["source_kind"], b.events["text"])) == [
        ("task", "task", "Find Poland, Hungary and Chile."), ("r1", "tool_result", "Poland 0.2"),
        ("r3", "tool_result", "Hungary -0.9"), ("r4", "tool_result", "Chile 0.2"),
        ("r1", "message", "R1 CONFIRMED: Poland arrived; answered 0.2")]
    assert b.notes["n_inner_dropped"] == 3 and b.notes["n_tool_unmatched"] == 0 and list(b.reads["agent_raw"]) == \
        ["r1", "r3", "r4"]


def test_unattributable_tool_results_are_the_tool_s_with_a_warning(tmp_path):
    p = tmp_path / "parallel.json"
    p.write_text(json.dumps([{"role": "assistant", "name": "a", "content": None, "function_call": {"name": "f"}},
                             {"role": "assistant", "name": "b", "content": None, "function_call": {"name": "f"}},
                             {"role": "function", "name": "f", "content": "Poland 0.2"},
                             {"role": "function", "name": "f", "content": "Poland 0.3"}]))
    b = ChatAdapter().load(p)
    assert b.events["author_raw"].tolist() == ["tool", "tool"] and b.notes["n_tool_unmatched"] == 2
    assert b.reads is None and b.capabilities.has_reads is False   # a read needs a reader
    assert "2 tool results answer no call id seen in the file, and neither an AutoGen `source` nor a single agent with " \
           "calls pending names the caller, so they are attributed to `tool`" in b.notes["warnings"][-1]


def test_langchain_dicts_and_anthropic_blocks(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"messages": [
        {"type": "human", "data": {"content": "Find Poland.", "type": "human"}},
        {"type": "ai", "data": {"content": "", "name": "planner", "tool_calls": [{"name": "s", "args": {}, "id": "c1"}]}},
        {"type": "tool", "data": {"content": "Poland 0.2", "tool_call_id": "c1", "name": "s"}},
        {"role": "assistant", "name": "writer", "content": [{"type": "text", "text": "checking"},
                                                            {"type": "tool_use", "id": "t1", "name": "s", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text",
                                                                                               "text": "Hungary -0.9"}]}]}]}))
    ev = ChatAdapter().load(p).events
    assert list(zip(ev["author_raw"], ev["source_kind"], ev["text"])) == [
        ("task", "task", "Find Poland."), ("planner", "tool_result", "Poland 0.2"), ("writer", "message", "checking\n"),
        ("writer", "tool_result", "Hungary -0.9")]


def test_openai_trace_has_no_phantom_agent_and_tool_output_is_a_private_read(oai):
    ev = oai.read("events")
    assert set(ev["author_raw"]) == {"task", *AGENTS}                 # not `unknown` (role-only turns), not `lookup` (tool name)
    assert ev["source_kind"].value_counts().to_dict() == {"message": 9, "tool_result": 3, "task": 2}
    reads = oai.read("reads")
    assert set(reads["agent_raw"]) == {"researcher-1"} and set(reads["method"]) == {"tool_result"} and len(reads) == 3
    prof = json.loads((oai.path / "profile.json").read_text())
    assert prof["capabilities"]["has_reads"] is True and prof["clock"] == "sequence" and prof["notes"]["n_tool_unmatched"] == 0
    assert adapters.detect(prof["source"]).name == "chat"             # its tool-call turns have `content: null`
    private = set(ev.loc[ev["source_kind"] != "message", "event_id"])
    assert not private & set(oai.read("claims")["event_id"]) and not private & set(oai.read("tags")["event_id"])
    edges = oai.read("edges_merged")
    assert "task" in set(oai.read("tags")["cites"])                   # "the task's value" reads as a citation ...
    assert not PHANTOMS & (set(edges["src"]) | set(edges["dst"]))     # ... but `task` is no A5 node
    assert not private & set(edges["event_id"]) and not private & set(edges["src_event_id"].dropna())
    text = (oai.path / "report.md").read_text()
    assert "14 posts from 3 author strings plus `task`, ordered by message position" in text
    assert "Identity merges: none; each of the 3 merged ids is a single author string" in text


def test_own_lookup_preceded_the_report_and_the_task_gave_chile(oai):
    ex = oai.read("exposures_merged").set_index(["agent", "item"])
    assert not PHANTOMS & (set(ex["src_agent"].dropna()) | set(ex["src_latest"].dropna()))
    first = ex.loc[("researcher-1", "Poland")]
    assert first["D"] == 0 and first["own_lookup_evidence"] == 1 and first["self_prepared"] == 0   # a read, not a post
    assert pd.notna(first["t_own_lookup"]) and first["t_own_lookup"] < first["t_report"]
    copier = ex.loc[("researcher-2", "Poland")]
    assert copier["D"] == 1 and copier["src_agent"] == "researcher-1" and copier["own_lookup_evidence"] == 0
    ev = oai.read("events").set_index("event_id")
    assert ev.loc[copier["src_event_id"], "source_kind"] == "message"   # researcher-1's answer, not its tool output
    assert pd.isna(copier["t_own_lookup"])                           # researcher-1's tool output was not its to read
    assert ex["t_task"].notna().sum() == 3 and set(ex.index[ex["t_task"].notna()].get_level_values("item")) == {"Chile"}
    assert ex.loc[("researcher-1", "Chile"), "D"] == 0                # the task message is no public copy


def test_a1_states_own_lookups_and_values_given_in_the_task(oai):
    text = (oai.path / "report.md").read_text()
    assert "Of 3 with no earlier public copy, 3 were preceded by the agent's own tool output (*own lookup preceded " \
           "report*, `own_lookup_evidence`). Over all 9 answers: 6 (67%) had a public copy first and no own lookup, " \
           "3 (33%) an own lookup first, 0 (0%) neither." in text
    assert "A lookup is evidence that the agent fetched the value, not proof that it computed it: a tool can echo a " \
           "value the agent passed in." in text
    assert "3 answers had the value given in the task (a system or user message carried the item and value before the " \
           "report); 1 of them had no earlier public copy, so the independent count above includes them." in text
    assert "| posts | 14 | posts; 2 task messages (system or user) and 3 tool results among them are not agents' posts" in text
    assert "| has_reads | yes | tool results are recorded as reads by the agent that called the tool" in text
    a1 = json.loads((oai.path / "summary.json").read_text())["A1"]
    assert {k: a1[k] for k in ("n_own_lookup", "n_exposed_with_own_lookup", "n_exposed_no_lookup", "n_neither",
                               "n_given_in_task", "n_given_in_task_no_public_copy")} == {
        "n_own_lookup": 3, "n_exposed_with_own_lookup": 0, "n_exposed_no_lookup": 6, "n_neither": 0,
        "n_given_in_task": 3, "n_given_in_task_no_public_copy": 1}


def test_task_and_tool_rows_without_a_family_count_in_every_family(configured):
    ev = configured.read("events")
    assert set(ev.loc[ev["source_kind"] != "message", "family"]) == {""}   # they name no configured family
    ex = configured.read("exposures_merged").set_index(["agent", "item"])
    assert ex["t_task"].notna().sum() == 3 and set(ex.index[ex["t_task"].notna()].get_level_values("item")) == {"Peru"}
    own = ex.loc["researcher-1"]   # two alpha-task rounds and one beta-task round, each after its own lookup
    assert set(own["family"]) == {"alpha-task", "beta-task"} and (own["own_lookup_evidence"] == 1).all()
    text = (configured.path / "report.md").read_text()
    assert "3 answers had the value given in the task" in text and "No answer had its value given" not in text
    assert "Of 3 with no earlier public copy, 3 were preceded by the agent's own tool output" in text


def test_a4_declared_disputes_count_agents_posts_only(tmp_path):
    """The task and every agent's tool output say 1.5, but only r1 posts it: neither is a carrier."""
    msgs = [{"role": "user", "content": "Peru prep 1.5 is where to start."}]
    for k, v in enumerate(["1.5", "2.5", "2.5", "2.5"], start=1):
        msgs += _lookup(f"r{k}", f"call_{k}", "Peru", "Peru prep 1.5")
        msgs.append({"role": "assistant", "name": f"r{k}", "content": f"R1 CONFIRMED: Peru arrived; answered {v}. Peru prep {v}"})
    (tmp_path / "trace.json").write_text(json.dumps(msgs))
    (tmp_path / "config.json").write_text(json.dumps({"disputes": [
        {"family": "all", "slot": "Peru prep", "context": "Peru prep", "variants": {"low": r"prep 1\.5", "high": r"prep 2\.5"}}]}))
    run = pipeline.run_all(tmp_path / "trace.json", tmp_path / "run", adapter="chat", config=str(tmp_path / "config.json"))
    feat = json.loads((run.path / "summary.json").read_text())["A4"]["rows"][0]
    assert feat["kind"] == "declared" and feat["variants (agents)"] == "high (3), low (1)" and feat["agents carrying both"] == 0


def test_trace_shows_the_own_lookup_and_the_task_under_the_sequence_clock(oai, capsys):
    assert query.trace(str(oai.path), agent="researcher-1", item="Chile") == 0
    out = capsys.readouterr().out
    assert "clock: message order" in out and "first public copy: none before the report" in out
    assert "own tool output before the report: post " in out and "by researcher-1; own lookup preceded report" in out
    assert '"Chile 0.2 (2023)"' in out and "value given in the task: post " in out and "by task" in out
    ev = oai.read("events")
    tool_ev = ev.loc[ev["source_kind"] == "tool_result", "event_id"].iloc[0]
    assert query.trace(str(oai.path), event=tool_ev) == 0
    out = capsys.readouterr().out
    assert "by researcher-1 (tool result)" in out and "a tool result (the calling agent's private read) is not an agent's " \
           "post; exposure reads its mentions as the agent's own lookup" in out


def test_inspect_counts_the_roles(oai, capsys):
    from swarmprov import inspect
    assert inspect.inspect(json.loads((oai.path / "profile.json").read_text())["source"]) == 0
    out = capsys.readouterr().out
    assert "framework roles: 12 agent turns, 2 task messages (system or user: author `task`), 3 tool results" in out
    assert "authors known (not `unknown`) 100%" in out


def test_trace_mode_leaves_the_synthetic_board_unchanged():
    plain, truth = synth.simulate(n_agents=20, seed=5)
    traced, truth_t = synth.simulate(n_agents=20, seed=5, trace=True)
    assert [p for p in traced if "type" not in p] == plain and truth_t.equals(truth)
    calls = [p for p in traced if p.get("type") == "ToolCallExecutionEvent"]
    assert len(calls) == int((~truth["copied"]).sum())                 # one per computed answer, none for a copied one
    row = truth.iloc[int(calls[0]["content"][0]["call_id"].split("_")[1])]
    assert not row["copied"] and calls[0]["author"] == row["agent"] and calls[0]["content"][0]["content"] == \
        f"{row['item']}: {row['value']:.2f}"


def test_own_lookup_recovers_computed_rounds_on_the_autogen_swarm(autogen):
    run, truth, _ = autogen
    prof = json.loads((run.path / "profile.json").read_text())
    assert prof["clock"] == "sequence" and prof["notes"]["n_tool_results"] == 126 and prof["notes"]["n_tool_unmatched"] == 0
    ex = run.read("exposures_merged")
    names = run.read("agents").drop_duplicates("author_raw").set_index("author_raw")["agent_merged"]
    m = ex.merge(truth.assign(agent=truth["agent"].map(names)), on=["agent", "item"], how="inner")
    assert len(m) == len(ex) == 300
    look = m["t_own_lookup"].notna()
    # seeds 0-7 all agree on every round and give the true copied share exactly; the bounds leave a margin
    assert (look == ~m["copied"]).mean() >= 0.98
    assert abs(((m["D"] == 1) & ~look).mean() - truth["copied"].mean()) <= 0.02
    assert (~m.loc[m["own_lookup_evidence"] == 1, "copied"]).mean() >= 0.98 and m["own_lookup_evidence"].sum() == 58
    d0 = m[m["D"] == 0]                                               # with no earlier public copy, the flag is the truth
    assert (d0["own_lookup_evidence"] == (~d0["copied"]).astype(int)).mean() >= 0.98
    assert not PHANTOMS & (set(ex["src_agent"].dropna()) | set(ex["src_latest"].dropna()))
    edges = run.read("edges_merged")
    assert not PHANTOMS & (set(edges["src"]) | set(edges["dst"]))
    a1 = json.loads((run.path / "summary.json").read_text())["A1"]
    assert a1["n_exposed_no_lookup"] / a1["n_answers"] == pytest.approx(truth["copied"].mean(), abs=0.02)


def test_trace_says_whether_the_public_copy_came_before_or_after_the_lookup(autogen, capsys):
    run, _, _ = autogen
    ex = run.read("exposures_merged")
    both = ex[(ex["D"] == 1) & ex["t_own_lookup"].notna()]
    for when, row in (("after", both[both["t_own_lookup"] < both["t_public"]].iloc[0]),
                      ("before", both[both["t_own_lookup"] > both["t_public"]].iloc[0])):
        assert query.trace(str(run.path), agent=row["agent"], item=row["item"]) == 0
        assert f"a public copy was also there before the report ({when} this lookup), so no own_lookup_evidence" \
            in capsys.readouterr().out


def test_tool_output_changes_no_exposure_against_the_plain_swarm(autogen, synth_run):
    """The same posts, plus private tool rows, untimed: every round's D and first source match the wall-clock run."""
    run, _, path = autogen
    plain, _ = synth_run
    a, b = plain.read("exposures_merged"), run.read("exposures_merged")
    m = a.merge(b, on=["agent", "item"], suffixes=("_wall", "_trace"))
    assert len(m) == 300 and (m["D_wall"] == m["D_trace"]).all() and (m["self_prepared_wall"] == m["self_prepared_trace"]).all()
    assert (m["src_agent_wall"].fillna("") == m["src_agent_trace"].fillna("")).all()
    assert "own tool output" not in (plain.path / "report.md").read_text() and plain.read("events")["source_kind"].eq("message").all()
    assert not plain.has("reads") and plain.read("exposures_merged")["t_own_lookup"].isna().all()
    first = json.loads(path.read_text().splitlines()[0])
    assert first == {"source": "user", "models_usage": None, "metadata": {}, "content": synth.TASK, "type": "TextMessage"}


def test_synth_cli_writes_the_autogen_format(tmp_path, capsys):
    assert cli.main(["synth", "-o", str(tmp_path / "ag.jsonl"), "--agents", "4", "--format", "autogen"]) == 0
    assert "tool calls, each with its result" in capsys.readouterr().out
    types = {json.loads(line)["type"] for line in (tmp_path / "ag.jsonl").read_text().splitlines()}
    assert types == {"TextMessage", "ToolCallRequestEvent", "ToolCallExecutionEvent"}
    assert not any("ts" in json.loads(line) for line in (tmp_path / "ag.jsonl").read_text().splitlines())


def test_truth_score_on_message_order(autogen):
    """The scorecard on the untimed swarm: no A2 (its latency outcome needs a wall clock), A4 under the one-bin rule
    on both sides, and `task` left out of identity."""
    from swarmprov import truthscore
    run, _, path = autogen
    r = truthscore.score(run, path.parent)
    assert r["A2"] is None and r["D"]["accuracy_vs_earlier_copy"] == 1.0
    assert r["A4"]["slots"] == r["A4"]["status_agrees"] == r["A4"]["outcome_agrees"]
    assert r["identity"]["merged"] == {"agents": 60, "true_agents": 60, "purity": 1.0, "true_agents_split": 0}
    assert (run.path / "checks.json").exists() and json.loads((run.path / "summary.json").read_text())["integrity"][
        "n_pass"] == 18
