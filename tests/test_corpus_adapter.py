import json
from pathlib import Path

import pandas as pd
import pytest

from swarmprov.adapters.corpus import CorpusAdapter, parse_date_literals, rewrite_omitted_urls
from swarmprov.schema import TABLES, conform


def _origin(site, title, lit, kind="revision_addition", source_id=None):
    return {"source_id": source_id or f"{site.split('/')[-1]}/{title}", "site": site, "title": title,
            "url": f"https://{site}/{title}", "source_date_literal": lit, "kind": kind,
            "original_text_sha256": "x", "response_quality": "", "source_reference": ""}


def _record(rid, text, origins):
    return {"id": rid, "text": text, "source_text_sha256": rid, "hosting_text_sha256": rid,
            "hosting_text_changed": False, "body_withheld": False,
            "authorship": "not_independently_authenticated", "selection_basis": "x", "origins": origins}


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw2"
    d.mkdir()
    records = [
        # signed dse post: ISO Z (kept only when dse is not excluded) + ISO offset on wiki4d
        _record("r1", "R5 Poland = 92.1 as announced. -- CoordMay10OAI", [
            _origin("prowiki.org/dse", "OECDEquity", "2026-06-17T02:10:42Z"),
            _origin("prowiki.org/wiki4d", "OECDEquity", "2026-06-17T03:10:42+01:00"),
        ]),
        # Iowa paste with agent-ours self id: epoch seconds, fractional epoch, wayback stamp,
        # plus unparseable literals and a late (2026-09) re-capture
        _record("r2", "IowaCollab: Q5 '85 and Older: NA' per agent-ours0402; proxy "
                      "[operational URL omitted; host=allorigins.hexlet.app; sha256=abc123] ts=1781643515.58", [
            _origin("paste.linuxiarz.pl", "IowaCollab", "1781643515", "paste_candidate", "paste/IowaCollab"),
            _origin("paste.linuxiarz.pl", "IowaCollabReply", "1781643999.5", "paste_candidate", "paste/IowaCollabReply"),
            _origin("paste.linuxiarz.pl", "IowaCollabWB", "20260610064043", "wayback_capture", "paste/IowaCollabWB"),
            _origin("paste.linuxiarz.pl", "IowaCollabCache", "prior cache decoded offline", "decoded_json_cache"),
            _origin("paste.linuxiarz.pl", "IowaCollabLate", "2026-09-05T23:42:44.625141Z", "saved_primary_projection"),
        ]),
        # anonymous: every literal unparseable except one fractional ISO with offset
        _record("r3", "Reference links https://markdown.new/x and https://r.jina.ai/y", [
            _origin("vanderbi.lt", "abc", "", "shortener_candidate"),
            _origin("rubygems.org", "gem", "current", "registry_metadata"),
            _origin("rubygems.org", "gem-0.0.1", "0.0.1", "package_member"),
            _origin("ludism.org", "Page", "31", "public_revision"),
            _origin("ludism.org", "Page", "2026-05-26T14:35:00.000000+00:00", "prior_revision"),
        ]),
        # exact duplicate (same channel, ts, text up to whitespace) -> one row
        _record("r4", "same text  twice", [
            _origin("anna.fyi", "p1", "2026-06-01T10:00:00+02:00", "prior_paste"),
            _origin("anna.fyi", "p1", "2026-06-01T08:00:00Z", "prior_paste"),
        ]),
    ]
    with open(d / "records.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    with open(d / "links.jsonl", "w") as fh:
        for host in ["markdown.new", "markdown.new", "r.jina.ai"]:
            fh.write(json.dumps({"url": f"https://{host}/x", "host": host, "record_ids": ["r3"],
                                 "relation": "link_in_selected_agent_related_text", "followed": False}) + "\n")
    (d / "shortener-logs.json").write_text(json.dumps({
        "read": "2026-09-07", "source": "rmn.re",
        "sites": [{"site": "rmn.re", "software": "YOURLS", "links_in_log": 1, "links": [
            {"keyword": "jq", "url": "https://api.example.org/data.json", "title": "Example",
             "time": "2026-05-26T18:02:55Z", "ip16": "20.168", "clicks": 26},
        ]}],
    }))
    (d / "other-wikis.json").write_text(json.dumps({
        "recovered": "2026-09-07", "source": "x",
        "pages": [{"page_key": "usemod~SandBox", "wiki": "usemod", "name": "SandBox", "page_id": "usemod/SandBox",
                   "revisions": [
                       {"seq": 1, "time": "2026-05-11T05:54:00Z", "ip16": "172.214", "append": True,
                        "added": ["[content not recovered — none; edit summary: \"test\"]"], "removed": []},
                       {"seq": 2, "time": "2026-05-26T16:59:00Z", "ip16": "20.165", "append": True,
                        "added": ["Reference links", "https://markdown.new/piv"], "removed": []},
                   ]}],
    }))
    (d / "site-coverage.csv").write_text(
        "site,host,category,specific_prior_gap_remains\nDSEWiki,prowiki.org,wikis,False\nrmn.re,rmn.re,shorteners,True\n")
    (d / "coverage-gaps.csv").write_text(
        "site,host,category,specific_prior_gap_remains\nrmn.re,rmn.re,shorteners,True\n")
    return d


def test_sniff(corpus_dir: Path, tmp_path: Path):
    a = CorpusAdapter()
    assert a.sniff(corpus_dir)
    # a wiki dump directory (revisions.jsonl, no records.jsonl) must not match
    wiki = tmp_path / "raw"
    wiki.mkdir()
    (wiki / "revisions.jsonl").write_text(json.dumps(
        {"rev_id": 1, "page_id": "dse/X", "body": "b", "label": "l", "time": "2026-05-24T00:00:00Z"}) + "\n")
    assert not a.sniff(wiki)
    assert not a.sniff(tmp_path / "missing")
    # records.jsonl with the wrong shape
    other = tmp_path / "other"
    other.mkdir()
    (other / "records.jsonl").write_text(json.dumps({"foo": 1}) + "\n")
    assert not a.sniff(other)
    if Path("data/raw").is_dir():
        assert not a.sniff(Path("data/raw"))


def test_parse_date_literals():
    lits = pd.Series(["2026-06-17T02:10:42Z", "2026-06-17T03:10:42+01:00", "1781643515", "1781643999.5",
                      "20260610064043", "current", "", "prior cache decoded offline", "0.0.1", "31",
                      "2026-05-26T14:35:00.000000+00:00", None])
    ts = parse_date_literals(lits)
    assert ts[0] == pd.Timestamp("2026-06-17T02:10:42Z")
    assert ts[1] == pd.Timestamp("2026-06-17T02:10:42Z")  # +01:00 normalised to UTC
    assert ts[2] == pd.Timestamp(1781643515, unit="s", tz="UTC")
    assert abs((ts[3] - pd.Timestamp(1781643999.5, unit="s", tz="UTC")).total_seconds()) < 1e-3
    assert ts[4] == pd.Timestamp("2026-06-10T06:40:43Z")
    assert ts[10] == pd.Timestamp("2026-05-26T14:35:00Z")
    assert ts[[5, 6, 7, 8, 9, 11]].isna().all()
    assert str(ts.dtype) == "datetime64[ns, UTC]"


def test_rewrite_omitted_urls():
    t = "see [operational URL omitted; host=jqp.vercel.app; sha256=0012766d] now"
    assert rewrite_omitted_urls(t) == "see https://jqp.vercel.app/[omitted] now"
    assert rewrite_omitted_urls("plain") == "plain"


def test_load_records(corpus_dir: Path):
    b = CorpusAdapter().load(corpus_dir)
    ev = b.events
    assert list(ev.columns) == TABLES["events"]
    assert len(conform(ev, "events")) == len(ev)
    assert ev["event_id"].is_unique
    assert ev["ts"].notna().all()
    assert str(ev["ts"].dtype) == "datetime64[ns, UTC]"

    # dse excluded by default; wiki4d copy of the same record kept
    assert "prowiki.org/dse" not in set(ev["site"])
    r1 = ev[ev["parent_id"] == "r1"]
    assert list(r1["site"]) == ["prowiki.org/wiki4d"]
    assert r1["author_raw"].iloc[0] == "CoordMay10OAI"           # trailing signature
    assert r1["channel"].iloc[0] == "prowiki.org/wiki4d/OECDEquity"
    assert r1["ts"].iloc[0] == pd.Timestamp("2026-06-17T02:10:42Z")

    # Iowa paste: epoch, fractional epoch and wayback kept; cache literal + late recapture dropped
    r2 = ev[ev["parent_id"] == "r2"].sort_values("ts")
    assert set(r2["source_ref"]) == {"paste/IowaCollab", "paste/IowaCollabReply", "paste/IowaCollabWB"}
    assert set(r2["source_kind"]) == {"paste_candidate", "wayback_capture"}
    assert (r2["author_raw"] == "agent-ours0402").all()          # self-identifier
    assert (r2["site"] == "paste.linuxiarz.pl").all()
    assert "https://allorigins.hexlet.app/[omitted]" in r2["text"].iloc[0]
    assert "[operational URL omitted" not in r2["text"].iloc[0]

    # anonymous record: only the ludism ISO origin survives
    r3 = ev[ev["parent_id"] == "r3"]
    assert len(r3) == 1 and r3["author_raw"].iloc[0] == "anon@ludism.org"
    assert r3["source_kind"].iloc[0] == "prior_revision"

    # exact duplicate collapsed
    assert len(ev[ev["parent_id"] == "r4"]) == 1

    # shortener + other-wiki rows
    sh = ev[ev["source_kind"] == "shortener_link"]
    assert len(sh) == 1
    assert sh.iloc[0]["site"] == "rmn.re" and sh.iloc[0]["channel"] == "rmn.re/log"
    assert sh.iloc[0]["author_raw"] == "ip16:20.168" and sh.iloc[0]["source_ref"] == "jq"
    assert sh.iloc[0]["text"] == "jq: Example https://api.example.org/data.json"
    ow = ev[ev["source_kind"] == "wiki_revision_lines"]
    assert len(ow) == 1                                           # rev 1 had only unrecovered content
    assert ow.iloc[0]["site"] == "usemod" and ow.iloc[0]["channel"] == "usemod/SandBox"
    assert ow.iloc[0]["source_ref"] == "usemod/SandBox@2"
    assert ow.iloc[0]["text"] == "Reference links\nhttps://markdown.new/piv"
    assert ow.iloc[0]["author_raw"] == "ip16:20.165"

    assert (ev["author_alt"] == "").all() and (ev["channel_family"] == "").all()
    assert ev["visible_until"].isna().all()

    # capabilities + notes
    caps = b.capabilities.as_dict()
    assert caps == {"has_wall_clock": True, "has_explicit_author": False, "has_reads": False,
                    "has_lifecycle": False, "has_threading": False, "has_episodes": False, "has_activity": False}
    n = b.notes
    assert n["n_records"] == 4 and n["n_origins"] == 14
    assert n["n_origins_with_time"] == 9          # 2 + 4 + 1 + 2
    assert n["n_dropped_no_time"] == 5
    assert n["n_dropped_late"] == 1
    assert n["n_dropped_excluded_sites"] == 1
    assert n["n_dropped_duplicates"] == 1
    assert n["n_links"] == 3 and n["top_link_hosts"] == {"markdown.new": 2, "r.jina.ai": 1}
    assert n["shortener"]["n_links"] == 1 and n["shortener"]["clicks_total"] == 26
    assert n["shortener"]["clicks_median"] == 26 and n["shortener"]["date_min"] == "2026-05-26"
    assert n["other_wikis"] == {"n_pages": 1, "n_revisions": 2, "n_revisions_with_lines": 1}
    assert n["coverage"]["n_sites"] == 2 and n["coverage"]["n_gap_rows"] == 1
    assert n["coverage"]["by_category"]["shorteners"] == {"sites": 1, "gaps": 1}
    assert n["coverage"]["by_category"]["wikis"] == {"sites": 1, "gaps": 0}
    assert "ip16_overlap_note" in n
    assert n["events_per_site"]["paste.linuxiarz.pl"] == 3
    assert n["events_per_source_kind"]["shortener_link"] == 1


def test_keep_dse_when_not_excluded(corpus_dir: Path):
    b = CorpusAdapter(exclude_sites=()).load(corpus_dir)
    ev = b.events
    assert "prowiki.org/dse" in set(ev["site"])
    assert b.notes["n_dropped_excluded_sites"] == 0
    dse = ev[ev["site"] == "prowiki.org/dse"].iloc[0]
    assert dse["channel"] == "prowiki.org/dse/OECDEquity" and dse["author_raw"] == "CoordMay10OAI"


def test_records_only_directory(tmp_path: Path):
    d = tmp_path / "min"
    d.mkdir()
    (d / "records.jsonl").write_text(json.dumps(_record("r", "hello", [
        _origin("anna.fyi", "p", "2026-06-01T00:00:00Z", "prior_paste")])) + "\n")
    b = CorpusAdapter().load(d)
    assert len(b.events) == 1
    assert b.notes["n_links"] == 0 and "coverage" not in b.notes and "shortener" not in b.notes


def test_config():
    cfg = CorpusAdapter().config()
    assert cfg.name == "cross-site-corpus"
    assert cfg.family_from_channel is False
    assert "iowa-health" in cfg.families and "oecd-equity" in cfg.families
    assert list(cfg.families)[0] == "oecd-equity"                 # wiki families first
    assert cfg.family_of_text("Q5 expected '85 and Older: NA'") == "iowa-health"
    names = {t.name for t in cfg.techniques}
    assert {"proxy-markdown-new", "proxy-allorigins", "proxy-jqp", "proxy-md-succ", "proxy-jina",
            "proxy-pure-md", "shortener-da-gd", "heartbeat-beacon"} <= names
    import re
    jqp = next(t for t in cfg.techniques if t.name == "proxy-jqp")
    assert re.search(jqp.pattern, rewrite_omitted_urls("[operational URL omitted; host=jqp.vercel.app; sha256=0]"))


@pytest.mark.skipif(not Path("data/raw2/records.jsonl").exists(), reason="real corpus not present")
def test_real_corpus():
    a = CorpusAdapter()
    assert a.sniff(Path("data/raw2"))
    b = a.load(Path("data/raw2"))
    ev = b.events
    # Only 941 of the 4,907 non-dse origins carry a parseable date (docs/DATA.md), so the
    # default (dse excluded) bundle is ~1.3k rows; with dse kept it is >11k.
    assert len(ev) > 1000
    assert "paste.linuxiarz.pl" in set(ev["site"])
    assert "prowiki.org/dse" not in set(ev["site"])
    assert list(ev.columns) == TABLES["events"]
    assert ev["event_id"].is_unique
    assert ev["ts"].notna().all() and ev["ts"].max() <= pd.Timestamp("2026-08-01", tz="UTC")
    assert b.notes["n_records"] == 13703
    assert b.notes["shortener"]["n_links"] == 499

    full = CorpusAdapter(exclude_sites=()).load(Path("data/raw2")).events
    assert len(full) > 3000
    assert "paste.linuxiarz.pl" in set(full["site"]) and "prowiki.org/dse" in set(full["site"])
    assert full["event_id"].is_unique
