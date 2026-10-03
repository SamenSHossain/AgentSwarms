from pathlib import Path

import pandas as pd
import pytest

from swarmprov.adapters.base import Technique
from swarmprov.analysis import crosssite

ROOT = Path(__file__).resolve().parents[1]
T0 = pd.Timestamp("2026-06-01 00:00", tz="UTC")
TECHS = [Technique("alpha", r"alpha\.proxy"), Technique("beta", r"\bbeta\b")]


def _events() -> pd.DataFrame:
    rows = []

    def add(site, hours, text, author="a1"):
        rows.append({"site": site, "ts": T0 + pd.Timedelta(hours=hours), "text": text, "author_raw": author,
                     "channel": f"{site}/Page{len(rows)}", "source_kind": "revision", "family": ""})

    # dse: 12 posts over 10 days; alpha originates here at h=0, beta at h=48
    for i in range(12):
        add("dse", i * 20, f"plain post {i}", author=f"d{i % 3}")
    add("dse", 0, "use alpha.proxy for fetch", author="d0")
    add("dse", 48, "beta works too", author="d1")
    add("dse", 60, "beta again", author="d2")
    # paste site: 8 posts; adopts alpha at h=6 (two authors), beta at h=96
    for i in range(8):
        add("paste.linuxiarz.pl", 1 + i * 24, f"iowa collab {i}", author=f"p{i % 2}")
    add("paste.linuxiarz.pl", 6, "ALPHA.PROXY mirror", author="p0")
    add("paste.linuxiarz.pl", 7, "alpha.proxy again", author="p1")
    add("paste.linuxiarz.pl", 96, "try beta", author="p1")
    # wiki4d: 6 posts, adopts alpha at h=30 only
    for i in range(6):
        add("prowiki.org/wiki4d", 2 + i * 30, f"wiki4d note {i}", author="w0")
    add("prowiki.org/wiki4d", 30, "alpha.proxy noted", author="w0")
    # tiny shortener site (2 events) adopts alpha but is below the at-risk floor
    add("rmn.re", 3, "alpha.proxy link", author="")
    add("rmn.re", 200, "another", author="")
    df = pd.DataFrame(rows)
    df["event_id"] = [f"e{i}" for i in range(len(df))]
    return df


def test_technique_spread_rows_and_origin():
    ev = _events()
    sp = crosssite.technique_spread(ev, TECHS, min_site_events=5)
    assert list(sp.columns) == crosssite.SPREAD_COLS
    # alpha first (earlier origin), rows within a technique ordered by lag
    assert sp["technique"].tolist() == ["alpha", "alpha", "alpha", "beta", "beta"]
    alpha = sp[sp["technique"] == "alpha"]
    assert alpha["site"].tolist() == ["dse", "paste.linuxiarz.pl", "prowiki.org/wiki4d"]
    assert (alpha["origin_site"] == "dse").all()
    assert alpha["hours_after_origin"].tolist() == pytest.approx([0.0, 6.0, 30.0])
    paste = alpha[alpha["site"] == "paste.linuxiarz.pl"].iloc[0]
    assert paste["n_posts"] == 2 and paste["n_authors"] == 2  # case-insensitive match
    # rmn.re adopted but has only 2 events: filtered out as not at risk
    assert "rmn.re" not in set(sp["site"])
    assert "rmn.re" in set(crosssite.technique_spread(ev, TECHS, min_site_events=1)["site"])
    beta = sp[sp["technique"] == "beta"]
    assert beta["site"].tolist() == ["dse", "paste.linuxiarz.pl"]
    assert beta["hours_after_origin"].tolist() == pytest.approx([0.0, 48.0])


def test_technique_site_summary():
    ev = _events()
    sp = crosssite.technique_spread(ev, TECHS)
    sm = crosssite.technique_site_summary(sp, ev, TECHS).set_index("technique")
    a, b = sm.loc["alpha"], sm.loc["beta"]
    assert a["origin_site"] == "dse" and a["first_seen"] == T0
    assert a["n_sites_adopted"] == 3 and a["n_sites_at_risk"] == 3
    assert a["median_hours_to_site_adoption"] == pytest.approx(18.0)  # median of 6 and 30
    assert a["sites_within_24h"] == 1
    assert b["n_sites_adopted"] == 2 and b["sites_within_24h"] == 0
    assert b["median_hours_to_site_adoption"] == pytest.approx(48.0)
    # a technique nobody used is absent
    sm2 = crosssite.technique_site_summary(sp, ev, TECHS + [Technique("gamma", r"zzzz")])
    assert "gamma" not in set(sm2["technique"])


def test_site_derived_from_channel_when_missing():
    ev = _events().drop(columns="site")
    sp = crosssite.technique_spread(ev, TECHS)
    assert set(sp["site"]) >= {"dse", "paste.linuxiarz.pl"}


def test_surface_groups():
    g = crosssite.surface_group
    assert g("dse") == "main wiki (dse)"
    assert g("probier") == g("publictestwiki") == g("prowiki.org/wiki4d") == g("wikiservice.at/user/milk") == "other wikis"
    assert g("paste.linuxiarz.pl") == g("anna.fyi") == g("pastebin.k4be.pl") == "pastebins"
    assert g("rmn.re") == g("vanderbi.lt") == g("is.gd") == "shorteners"
    assert g("rubygems.org") == "other"
    assert crosssite.short_site("prowiki.org/wiki4d") == "wiki4d"


def test_figures_render(tmp_path):
    ev = _events()
    sp = crosssite.technique_spread(ev, TECHS)
    p1 = crosssite.technique_spread_figure(sp, tmp_path / "spread.png")
    assert p1 and Path(p1).exists() and Path(p1).stat().st_size > 1000
    assert crosssite.technique_spread_figure(sp.iloc[0:0], tmp_path / "none.png") is None
    p2 = crosssite.timeline(ev, tmp_path / "timeline.png",
                            markers=[(pd.Timestamp("2026-06-03 10:00"), "something"),
                                     (pd.Timestamp("2026-06-05 05:10", tz="UTC"), "else")])
    assert Path(p2).exists() and Path(p2).stat().st_size > 1000


def test_coverage_summary_real_file():
    csv = ROOT / "data" / "raw2" / "site-coverage.csv"
    if not csv.exists():
        pytest.skip("site-coverage.csv not present")
    s = crosssite.coverage_summary(csv)
    assert s["n_sites"] == 143
    assert s["n_gap_rows"] == 58
    assert s["relays_unvisited"] == 23
    assert s["by_category"]["sites"].sum() == 143
    assert s["by_category"].loc["wikis", "sites"] == 30
    assert any("427" in c for c in s["caveats"])
    assert any("Discord" in c and str(s["discord_urls"]) in c for c in s["caveats"])
    md = crosssite.coverage_markdown(s)
    assert "| category |" in md and "427" in md
