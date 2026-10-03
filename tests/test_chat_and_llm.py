import json

import pandas as pd

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
