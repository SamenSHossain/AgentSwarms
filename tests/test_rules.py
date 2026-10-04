import pandas as pd
import pytest

from swarmprov import identity, rules
from swarmprov.adapters.base import AdapterConfig
from swarmprov.adapters.wiki import FAMILIES


def _extract(texts, family="oecd-equity", items=None):
    cfg = AdapterConfig(name="t", families=FAMILIES)
    ev = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(len(texts))],
        "ts": pd.date_range("2026-06-20", periods=len(texts), freq="min", tz="UTC"),
        "channel": "dse/X", "author_raw": "OpenAIOECDNov27", "author_alt": "", "text": texts,
        "channel_family": family,
    })
    fam = identity.post_families(ev, cfg)
    ag = identity.resolve(ev, fam, cfg)
    im = rules.ItemMatcher()
    for f, s in (items or {}).items():
        im.add(f, set(s))
    claims, tags = rules.extract(ev, fam, ag, cfg, im)
    return claims, tags, ag


@pytest.mark.parametrize("raw,norm", [
    ("16.40%", "16.4"), ("20,369", "20369"), ("874,322;951,258", "874322/951258"),
    ("18.2/13.5", "18.2/13.5"), ("9.70", "9.7"),
])
def test_norm_value(raw, norm):
    assert rules.norm_value(raw) == norm


@pytest.mark.parametrize("seg,expect", [
    ("681115,693859,702337", "681115"),        # comma list, not a thousands number
    ("2015-20 = 457639,460507", "457639"),     # skip the year range
    ("12:12:43/44 then 9.9", "9.9"),           # clock time with /ss suffix is not a value
    ("#3 Business 5,269", "5,269"),            # '#3' is a round marker
])
def test_first_value(seg, expect):
    assert rules.first_value(seg) == expect


def test_answer_report_fields():
    cl, tags, _ = _extract(["Nov27 R3 CONFIRMED: Poland arrived exactly 19:43:07 = R2 deadline 18:14:31 +1h28m36; "
                            "timer 56s; answered 16.40% instantly. R3 deadline 19:44:03. -- OpenAIOECDNov27"])
    a = cl[cl.event_type == "answer"].iloc[0]
    assert (a["item"], a["value_norm"], a["episode"], a["latency_class"], a["timer_s"]) == \
        ("Poland", "16.4", 3, "instant", 56.0)
    assert bool(tags.iloc[0]["is_answer"])


def test_latency_from_clock_times_and_wrong_flag():
    cl, _, _ = _extract(["R1 Czech Republic arrived task clock Mar26 03:14:08, timer 12m18s; answered 9.70% at 03:20:36.",
                         "R1 Croatia prompt 20:52:46, timer 19m11s; answered wrong before OWID discovery."],
                        family="oecd-equity", items={"oecd-equity": {"Croatia"}})
    a = cl[cl.event_type == "answer"].sort_values("event_id")
    assert a.iloc[0]["latency_class"] == "delayed" and a.iloc[0]["latency_s"] == 388
    assert bool(a.iloc[1]["wrong_flag"]) and pd.isna(a.iloc[1]["value_norm"])


def test_questions_are_not_answers():
    cl, _, _ = _extract(["Did anyone who answered Poland correctly receive follow-up rounds?"])
    assert (cl.empty or (cl.event_type != "answer").all())


def test_state_code_qualifier_not_an_item():
    im = rules.ItemMatcher()
    im.add("pov", {"Saginaw"})
    assert [i for _, i in im.find("pov", "R4 expected Saginaw MI 21.8")] == ["Saginaw"]
    assert [i for _, i in im.find("pov", "Our WV arrived 03:44:10")] == ["West Virginia"]


def test_sequence_report_claims():
    cl, _, _ = _extract(["G5 CONFIRMED: Montana = 8553. Prompt observed by Apr20 cohort."],
                        family="datausa-grocery-workforce")
    seq = cl[cl.event_type == "sequence"]
    assert (seq["episode"].iloc[0], seq["item"].iloc[0]) == (5, "Montana")


def test_identity_cohort_merge():
    cfg = AdapterConfig(name="t")
    ev = pd.DataFrame({"author_raw": ["OpenAIOECDNov27", "Nov27Scout", "MayTwoObserverFreshX"],
                       "text": ["x", "y", "May02 R3 CONFIRMED: ..."], "channel_family": ["f", "f", "f"],
                       "channel": "c"})
    ag = identity.resolve(ev, pd.Series(["f", "f", "f"]), cfg).set_index("author_raw")
    assert ag.loc["OpenAIOECDNov27", "agent_merged"] == ag.loc["Nov27Scout", "agent_merged"] == "Nov27|f"
    assert ag.loc["MayTwoObserverFreshX", "cohort"] == "May02"   # from the post's leading token
    assert ag.loc["OpenAIOECDNov27", "agent_strict"] != ag.loc["Nov27Scout", "agent_strict"]
