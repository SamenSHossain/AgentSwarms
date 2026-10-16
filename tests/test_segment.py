import time

import pandas as pd

from swarmprov.segment import signature_of, snapshots_to_posts, split_posts
from swarmprov.textutil import SIGNATURE_RE, fix_mojibake


def test_split_posts_cuts_after_signatures():
    body = "Header\n\nR2 arrived -- AgentA\nR3 arrived -- AgentB\n\ntrailing note"
    assert split_posts(body) == ["Header", "R2 arrived -- AgentA", "R3 arrived -- AgentB", "trailing note"]


def test_novelty_attribution_and_deletion_reset():
    t = pd.to_datetime(["2026-06-16 10:00", "2026-06-16 11:00", "2026-06-16 13:00"], utc=True)
    revs = pd.DataFrame({
        "channel": "dse/P", "ts": t, "rev_id": ["r1", "r2", "r3"],
        "author_alt": ["acctA", "acctB", "acctC"],
        "body": ["Q1 answered 9.9 -- AgentA",
                 "Q1 answered 9.9 -- AgentA\n\nQ2 answered 16.4 -- AgentB",
                 "Q1 answered 9.9 -- AgentA"],          # re-created after a deletion
    })
    dels = pd.DataFrame({"channel": ["dse/P"], "ts": pd.to_datetime(["2026-06-16 12:00"], utc=True)})
    posts = snapshots_to_posts(revs, dels)
    assert list(posts["author_raw"]) == ["AgentA", "AgentB", "AgentA"]
    assert list(posts["parent_id"]) == ["r1", "r2", "r3"]          # each paragraph credited once per lifetime
    assert posts["visible_until"].iloc[0] == dels["ts"].iloc[0]


def test_mojibake_is_unwound():
    assert fix_mojibake("TÃ¼rkiye") == "Türkiye"
    assert fix_mojibake("TÃƒÂ¼rkiye") == "Türkiye"


def test_signature_forms():
    for line, name in (("R3 answered 9.9 -- AgentA", "AgentA"), ("note --AgentB.", "AgentB"), ("x -- [[AgentC]]", "AgentC"),
                       ("x -- [[AgentD]] .  ", "AgentD"), ("--Agent.E-1   ", "Agent.E-1"), ("x -- AgentF.\t", "AgentF"),
                       ("x --AgentG]] . ", "AgentG"), ("a -- b c", None), ("x--AgentH", None), ("-- 9lives", None)):
        assert signature_of(line) == name, line


def test_signature_regex_is_linear_on_long_whitespace():
    """A line with a long run of spaces after a would-be signature once took 7 s at 40k spaces (two adjacent \\s*
    backtracking), and every later snapshot of its page repeated the cost."""
    t = time.perf_counter()
    assert SIGNATURE_RE.search("-- Abc" + " " * 40_000 + "x") is None
    assert SIGNATURE_RE.search("-- Abc ." + " " * 40_000 + "x") is None
    assert SIGNATURE_RE.search("-- Abc" + " " * 40_000).group(1) == "Abc"
    assert time.perf_counter() - t < 0.1
