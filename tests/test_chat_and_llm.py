import json

import pandas as pd
import pytest

from swarmprov import extract_llm
from swarmprov.adapters.chat import ChatAdapter


def test_chat_adapter_guesses_fields(tmp_path):
    p = tmp_path / "village.json"
    p.write_text(json.dumps({"messages": [
        {"timestamp": 1781600000, "sender": {"name": "Claude 3.7"}, "content": "Found the donation link"},
        {"timestamp": 1781600060, "sender": {"name": "o3"}, "content": [{"type": "text", "text": "thanks"}]},
    ]}))
    a = ChatAdapter()
    assert a.sniff(p)
    b = a.load(p)
    assert list(b.events["author_raw"]) == ["Claude 3.7", "o3"]
    assert b.events["ts"].iloc[1] - b.events["ts"].iloc[0] == pd.Timedelta(seconds=60)
    assert b.capabilities.has_explicit_author


class _Parsed:
    def __init__(self, out):
        self.parsed_output = extract_llm.PostExtraction(**out)


class _FakeClient:
    """Stands in for anthropic.Anthropic(); records calls."""
    def __init__(self, out):
        self.out, self.calls = out, 0
        self.messages = self

    def parse(self, **kw):
        self.calls += 1
        assert kw["output_format"] is extract_llm.PostExtraction
        return _Parsed(self.out)


def test_llm_run_caches_and_merges(tmp_path):
    ev = pd.DataFrame({"event_id": ["e1"], "ts": pd.to_datetime(["2026-06-20"], utc=True), "channel": "c",
                       "author_raw": ["A"], "text": ["R3 CONFIRMED: Poland arrived; answered instantly"]})
    fam = pd.Series(["oecd-equity"])
    agents = pd.DataFrame({"author_raw": ["A"], "agent_strict": ["a"], "agent_merged": ["Nov27|oecd-equity"]})
    client = _FakeClient({"claims": [{"event_type": "answer", "episode": 3, "item": "poland", "value": None,
                                      "latency_class": "instant"}]})
    llm = extract_llm.run(ev, fam, agents, cache_dir=tmp_path, client=client, workers=1)
    llm2 = extract_llm.run(ev, fam, agents, cache_dir=tmp_path, client=client, workers=1)
    assert client.calls == 1 and len(llm) == len(llm2) == 1           # second run served from cache
    rule = pd.DataFrame([{"claim_id": "r", "event_id": "e1", "ts": ev.ts[0], "agent_strict": "a",
                          "agent_merged": "Nov27|oecd-equity", "family": "oecd-equity", "episode": 3,
                          "item": "Poland", "value_raw": "16.40%", "value_norm": "16.4", "event_type": "answer",
                          "latency_class": None, "latency_s": None, "timer_s": 56.0, "wrong_flag": False,
                          "correct_flag": False, "extractor": "rule"}])
    merged = extract_llm.merge(rule, llm)
    row = merged.iloc[0]
    assert len(merged) == 1 and row["value_norm"] == "16.4" and row["latency_class"] == "instant"
    assert row["extractor"].startswith("llm:") and row["extractor"].endswith("+rule")


# --- chats with no task family: the catch-all family, the A3 pool, the funnel and the skip notes ---

PLAIN = [("researcher-1", "R1 CONFIRMED: Poland arrived; answered 0.2 at +5s."),
         ("researcher-2", "R1 CONFIRMED: Hungary arrived; answered -0.9 at +4s, fetched via r.jina.ai."),
         ("researcher-3", "R1 CONFIRMED: Chile arrived; answered 0.2 at +6s with the r.jina.ai proxy."),
         ("researcher-4", "R1 CONFIRMED: Poland arrived; answered 0.2 instantly via r.jina.ai."),
         ("researcher-5", "R1 CONFIRMED: Poland arrived; answered 0.1 at +9s.")]
FREEFORM = [("a1", "Found the dataset at https://data.example.org/oecd.csv"),
            ("a2", "Using https://data.example.org/oecd.csv as the source"),
            ("a3", "Also see https://data.example.org/oecd.csv for the raw table"),
            ("a4", "Docs: https://docs.example.org/api and https://data.example.org/oecd.csv"),
            ("a5", "https://docs.example.org/api is the reference", "other-page")]   # a cross-page hop
TECH = [{"name": "jina-proxy", "pattern": r"r\.jina\.ai"}]


def _chat_run(d, name, posts, cfg):
    t0 = pd.Timestamp("2026-05-04T10:00:00Z")
    (d / f"{name}.jsonl").write_text("\n".join(json.dumps({"timestamp": (t0 + pd.Timedelta(minutes=i)).isoformat(),
                                                           "author": a, "text": t} | ({"channel": ch[0]} if ch else {}))
                                                for i, (a, t, *ch) in enumerate(posts)))
    (d / f"{name}.json").write_text(json.dumps(cfg))
    from swarmprov import pipeline
    return pipeline.run_all(d / f"{name}.jsonl", d / name, adapter="chat", config=str(d / f"{name}.json"))


@pytest.fixture(scope="module")
def familyless(tmp_path_factory):
    d = tmp_path_factory.mktemp("familyless")
    return {"catchall": _chat_run(d, "catchall", PLAIN, {"techniques": TECH}),
            "unmatched": _chat_run(d, "unmatched", PLAIN, {"families": {"gdp": "GDP"}, "techniques": TECH}),
            "freeform": _chat_run(d, "freeform", FREEFORM, {})}


def _text(run):
    return (run.path / "report.md").read_text()


def test_chat_without_families_reaches_a1_through_the_catch_all_family(familyless):
    run = familyless["catchall"]
    ex = run.read("exposures_merged")
    assert len(ex) == 5 and set(ex["family"]) == {"all"}
    text = _text(run)
    assert "## A1. Provenance split\n\n**5 answers given by 5 agents;" in text
    assert "| with a task family | 5 | posts; the catch-all family `all`, because no families are configured |" in text
    assert "**Catch-all family.**" in text
    assert json.loads((run.path / "summary.json").read_text())["funnel"]["catch_all_family"] == "all"


def test_a3_pools_every_agent_when_no_task_family_restricts_the_set(familyless):
    """The at-risk set is decided once (``at_risk_basis``) and both the A3 text and its key finding say it."""
    for name, basis, why in (
            ("catchall", "catch_all", "every agent active after the technique's first post, as every post is in the "
                                      "catch-all family `all` and the technique names no families of its own"),
            ("unmatched", "all_active", "every agent active after the technique's first post, because no post has a "
                                        "task family")):
        run = familyless[name]
        s = json.loads((run.path / "summary.json").read_text())
        a3 = s["A3"][0]
        assert (a3["originator"], a3["at_risk"], a3["adopters"], a3["at_risk_basis"]) == ("researcher-2", 3, 2, basis)
        assert why in _text(run), name
        assert "the agents at risk (active after its first post)" in s["headline"][2], name


def test_a3_with_only_the_task_message_in_a_family(tmp_path):
    """A framework trace whose task message names the family but no agent's post does: the posts set the pool, so
    the A3 text and the key finding both count every agent active after the first post."""
    from swarmprov import pipeline
    msgs = [{"role": "system", "content": "alpha-task: research the 2023 change for each country."}]
    for i in range(12):
        who = ("ann", "bob", "cy", "dee")[i % 4]
        msgs.append({"role": "assistant", "name": who, "content": f"Using TRICK-A I found Poland {i}.5 for round {i}"
                     if i % 3 == 0 else f"note {i} from {who}"})
    (tmp_path / "trace.json").write_text(json.dumps(msgs))
    (tmp_path / "cfg.json").write_text(json.dumps({"families": {"alpha-task": "alpha-task"}, "bootstrap_reps": 0,
                                                   "techniques": [{"name": "TRICK-A", "pattern": "TRICK-A"}]}))
    run = pipeline.run_all(tmp_path / "trace.json", tmp_path / "run", config=str(tmp_path / "cfg.json"))
    s = json.loads((run.path / "summary.json").read_text())
    assert s["funnel"]["posts_with_family"] == 1 and s["A3"][0]["at_risk_basis"] == "all_active"
    assert "At-risk set: every agent active after the technique's first post, because no post has a task family" \
        in (run.path / "report.md").read_text()
    assert "the agents at risk (active after its first post)" in s["headline"][2]


def test_configured_families_leave_unmatched_posts_familyless_and_say_why_a1_is_missing(familyless):
    run = familyless["unmatched"]
    assert set(run.read("events")["family"]) == {""} and len(run.read("exposures_merged")) == 0
    text = _text(run)
    assert "## A1. Provenance split\n\n_Skipped: needs agent-rounds (an agent's answer to an item in a task family); " \
           "5 answer claims extracted, 0 agent-rounds: none has a task family: their posts match no configured " \
           "`families` pattern (widen the patterns, or use `--llm`)._" in text
    assert "default_family" not in text            # it does not apply once families are configured
    f = json.loads((run.path / "summary.json").read_text())["funnel"]
    assert (f["posts"], f["posts_with_family"], f["answer_claims"], f["answer_claims_no_family"], f["agent_rounds"]) == (5, 0, 5, 5, 0)
    assert f["catch_all_family"] is None and "Catch-all" not in text


def test_freeform_chat_explains_every_skipped_section_and_counts_url_chains(familyless):
    run = familyless["freeform"]
    text = _text(run)
    assert "## Pipeline funnel" in text and "| answer claims | 0 |" in text
    assert "## A1. Provenance split\n\n_Skipped: needs agent-rounds (an agent's answer to an item in a task family); " \
           "no answer claims were extracted" in text
    assert "## A2. Does availability cause copying?\n\n_Skipped: needs more than 20 agent-rounds; this run has 0._" in text
    assert "## A3. Technique diffusion\n\n_Skipped: needs configured techniques" in text
    assert "No task value was carried by two or more agents, but 2 URLs were, over 4 hops (3 same-page relay, 1 " \
           "cross-page attributed to the originator;" in text
    assert "No fact was carried" not in text


def test_a1_skip_names_what_kept_claims_from_agent_rounds():
    from swarmprov import report
    from swarmprov.adapters.base import AdapterConfig

    def answers(*pairs):
        return pd.DataFrame({"event_type": "answer", "family": [f for f, _ in pairs], "item": [i for _, i in pairs]})
    hint, none = report._family_hint(AdapterConfig(name="chat"), roster=False), pd.DataFrame()
    assert hint == "none has a task family (set `families` or `default_family`, or use `--llm`)"
    pred = answers(("f", "Poland")).assign(event_type="prediction")
    assert report._a1_gap(pred, none, hint).startswith("no answer claims were extracted")
    assert report._a1_gap(answers(("", "Poland")), none, hint) == f"1 answer claim extracted, 0 agent-rounds: {hint}"
    assert report._a1_gap(answers(("f", None), ("f", None)), none, hint) == \
        "2 answer claims extracted, 0 agent-rounds: none names an item (add `extra_items`, or use `--llm`)"
    assert report._a1_gap(answers(("f", None), ("", "Poland")), none, hint) == \
        "2 answer claims extracted, 0 agent-rounds: none names both an item and a task family"
    assert report._a1_gap(answers(("f", "Poland")), pd.DataFrame({"t_report": [pd.NaT] * 2}), hint) == \
        "none of the 2 agent-rounds has a timestamp, so nothing can be ordered against them"


def test_a1_skip_suggests_default_family_only_where_it_applies():
    from swarmprov import report
    from swarmprov.adapters.base import AdapterConfig
    gdp = AdapterConfig(name="chat", families={"gdp": "GDP"}, default_family="misc")
    for cfg, roster, says in ((gdp, False, "their posts match no configured `families` pattern (widen the patterns"),
                              (gdp, True, "match no configured `families` pattern and fall in no roster or shared goal"),
                              (AdapterConfig(name="chat"), True, "`default_family` does not apply with a roster (set "
                                                                 "`families`, or use `--llm`)")):
        hint = report._family_hint(cfg, roster)
        assert says in hint and "set `families` or `default_family`" not in hint, (cfg.families, roster)


def test_post_families_apply_the_default_only_without_families_or_roster():
    from swarmprov import adapters, identity
    from swarmprov.adapters.base import AdapterConfig
    ev = pd.DataFrame({"channel_family": ["", "", "task-b"], "channel": ["main"] * 3,
                       "text": ["GDP of Poland", "a free note", "x"]})
    chat = adapters.get("chat").config()
    assert chat.default_family == "all"
    assert list(identity.post_families(ev, chat)) == ["all", "all", "task-b"]
    assert list(identity.post_families(ev, chat, roster=True)) == ["", "", "task-b"]
    configured = AdapterConfig.from_dict({"families": {"gdp": "GDP"}}, base=adapters.get("chat").config())
    assert list(identity.post_families(ev, configured)) == ["gdp", "", "task-b"]
    for name in ("wiki", "corpus", "village", "roster", "swarmtraces"):
        assert adapters.get(name).config().default_family is None, name


def test_a3_pool_is_unchanged_when_tags_carry_families():
    """The wiki's at-risk sets: a family-bearing swarm keeps the family-restricted pool."""
    from swarmprov.adapters.base import Technique
    from swarmprov.analysis import diffusion
    t = pd.Timestamp("2026-06-20T05:00:00Z")
    tags = pd.DataFrame({"event_id": [f"e{m}" for m in range(5)], "ts": [t + pd.Timedelta(minutes=m) for m in range(5)],
                         "agent_merged": ["A", "B", "C", "D", "B"],
                         "family": ["f", "f", "", "f", "f"],
                         "techniques": ["blob", "", "", "", "blob"]})
    for tech in (Technique("blob", "x"), Technique("blob", "x", ("f",))):
        r = diffusion.analyze(tags, [tech], "agent_merged")["blob"]
        assert (r["at_risk"], r["adopters"]) == (2, 1)            # B and D; C posted with no family
    r = diffusion.analyze(tags.assign(family=""), [Technique("blob", "x")], "agent_merged")["blob"]
    assert (r["at_risk"], r["adopters"]) == (3, 1)                # no post has a family: C is at risk too
