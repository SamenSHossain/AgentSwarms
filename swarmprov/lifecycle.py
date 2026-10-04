"""Channel lifecycle beyond "when was it deleted": deletion sweeps, recreations
after deletion, the request-log probes, and what each means for exposure.

The wiki dump's ``events.jsonl`` carries more than the delete rows the
segmenter needs: first-recreation edges (a deletion -> the first later save on
the page), whether the deleted page ever had a published revision
(``page_held``), successive delete/recreate cycles (``round_id``), and 101
script-injection probe requests.  This module turns those into checks and
report sections; every number is computed from the run's tables.
"""

from __future__ import annotations

import pandas as pd

from .segment import split_posts
from .textutil import fix_mojibake, norm_key

SWEEP_GAP_MIN = 30


def sweeps(deletes: pd.DataFrame, gap_min: float = SWEEP_GAP_MIN) -> pd.DataFrame:
    """Cluster deletions: a gap longer than ``gap_min`` minutes starts a new sweep."""
    d = deletes.sort_values("ts")
    if not len(d):
        return pd.DataFrame(columns=["start", "end", "n", "pages"])
    new = d["ts"].diff().dt.total_seconds().div(60).gt(gap_min).fillna(True)
    sid = new.cumsum()
    return (d.groupby(sid).agg(start=("ts", "min"), end=("ts", "max"), n=("ts", "size"), pages=("channel", "nunique"))
            .reset_index(drop=True))


def recreation_check(revs: pd.DataFrame, lifecycle: pd.DataFrame, posts: pd.DataFrame,
                     cutoff: pd.Timestamp | None = None) -> dict:
    """Compare the dump's first-recreation edges with the pipeline's own rule.

    Pipeline rule: each deletion -> the first revision saved on the same channel
    strictly after it.  Also splits the posts attributed to recreation revisions
    into *restored* (their text stood on the page before the deletion) and
    *fresh*.  ``revs`` needs channel, ts, rev_id, body; ``cutoff`` is the dump's
    eligibility cutoff, so later pipeline edges count as outside its window."""
    dels = lifecycle[lifecycle["action"] == "delete"]
    recs = lifecycle[lifecycle["action"] == "recreate"]
    by_ch = {ch: g.sort_values("ts") for ch, g in revs.groupby("channel")}
    pipe: dict[tuple[str, str], pd.Timestamp] = {}
    for ch, g in dels.groupby("channel"):
        rv = by_ch.get(ch)
        if rv is None:
            continue
        for eid, ts in zip(g["event_id"], g["ts"]):
            later = rv[rv["ts"] > ts]
            if len(later):
                pipe[(str(eid), str(later["rev_id"].iloc[0]))] = later["ts"].iloc[0]
    dump = {(str(e), str(r)) for e, r in zip(recs["related_event_id"], recs["revision_ref"]) if isinstance(r, str) and r}
    overlap = dump & set(pipe)
    pipe_only = set(pipe) - dump
    outside = {k for k in pipe_only if cutoff is not None and pipe[k] > cutoff}
    out = {
        "dump_edges": int(len(recs)), "dump_edges_with_revision": len(dump),
        "dump_edges_without_revision": int(len(recs) - len(dump)),
        "pipeline_edges": len(pipe), "overlap": len(overlap),
        "pipeline_only": len(pipe_only), "pipeline_only_after_cutoff": len(outside),
        "dump_only": len(dump - set(pipe)),
        "cutoff": str(cutoff) if cutoff is not None else None,
    }
    # restored vs fresh posts on the recreation revisions
    first_del: dict[str, pd.Timestamp] = {}
    dts = dict(zip(dels["event_id"].astype(str), dels["ts"]))
    for e, r in zip(recs["related_event_id"], recs["revision_ref"]):
        if isinstance(r, str) and r and str(e) in dts:
            first_del[r] = min(first_del.get(r, dts[str(e)]), dts[str(e)])
    on_rec = posts[posts["parent_id"].astype(str).isin(first_del)]
    restored = 0
    rows = []
    for ch, g in on_rec.groupby("channel"):
        rv = by_ch.get(ch)
        for r in g.itertuples(index=False):
            before = rv[rv["ts"] < first_del[str(r.parent_id)]] if rv is not None else rv
            keys = set()
            if before is not None:
                for body in before["body"]:
                    keys.update(norm_key(fix_mojibake(p)) for p in split_posts(body))
            is_restored = norm_key(r.text) in keys
            restored += is_restored
            rows.append({"event_id": r.event_id, "channel": ch, "restored": is_restored})
    out.update({"posts_on_recreations": int(len(on_rec)), "restored_posts": int(restored),
                "fresh_posts": int(len(on_rec) - restored)})
    out["restored_event_ids"] = [x["event_id"] for x in rows if x["restored"]]
    return out


def deletion_summary(lifecycle: pd.DataFrame, events: pd.DataFrame | None, check: dict | None) -> dict:
    dels = lifecycle[lifecycle["action"] == "delete"]
    if not len(dels):
        return {}
    recs = lifecycle[lifecycle["action"] == "recreate"]
    sw = sweeps(dels)
    big = sw.sort_values("n", ascending=False).iloc[0]
    held = dels["page_held"].map(lambda v: v is True or v == "True") if "page_held" in dels else pd.Series(False, index=dels.index)
    s = {
        "n": int(len(dels)), "pages": int(dels["channel"].nunique()),
        "actors": sorted(a for a in dels["actor"].dropna().astype(str).unique() if a),
        "first": dels["ts"].min(), "last": dels["ts"].max(),
        "held": int(held.sum()), "unheld": int((~held).sum()) if "page_held" in dels and dels["page_held"].notna().any() else None,
        "n_sweeps": int(len(sw)), "sweep_median": float(sw["n"].median()),
        "sweeps_15": int(len(sweeps(dels, 15))), "sweeps_60": int(len(sweeps(dels, 60))),
        "largest": {"n": int(big["n"]), "pages": int(big["pages"]), "start": big["start"], "end": big["end"]},
        "n_recreate": int(len(recs)),
        "recreate_with_revision": int(recs["revision_ref"].map(lambda v: isinstance(v, str) and bool(v)).sum()) if len(recs) else 0,
    }
    if len(recs):
        dts = dict(zip(dels["event_id"].astype(str), dels["ts"]))
        gaps = [(t - dts[str(e)]).total_seconds() / 3600 for e, t in zip(recs["related_event_id"], recs["ts"]) if str(e) in dts]
        s["recreate_gap_median_h"] = float(pd.Series(gaps).median()) if gaps else None
    if events is not None and len(events):
        last_write = events["ts"].max()
        s["last_write"] = last_write
        s["deleted_before_last_write"] = int((dels["ts"] <= last_write).sum())
        vu = events["visible_until"]
        s["posts"] = int(len(events))
        s["posts_on_deleted_pages"] = int(vu.notna().sum())
        s["posts_deleted_while_writing"] = int((vu <= last_write).sum())
    if check:
        s["check"] = check
    return s


def deletion_section(s: dict, pct) -> list[str]:
    if not s:
        return []
    P: list[str] = []
    who = ", ".join(s["actors"]) or "unknown actors"
    held = (f": {s['held']:,} hit pages with a published revision, {s['unheld']:,} hit pages the dump never published"
            if s.get("unheld") is not None else "")
    P.append(f"{s['n']:,} deletions by {who} between {s['first']:%Y-%m-%d} and {s['last']:%Y-%m-%d} named "
             f"{s['pages']:,} pages{held}. Grouped into sweeps (a gap over {SWEEP_GAP_MIN} min starts a new one, a convention: "
             f"{s['sweeps_15']} sweeps at 15 min, {s['sweeps_60']} at 60): {s['n_sweeps']} sweeps, median {s['sweep_median']:.0f} deletions; "
             f"the largest removed {s['largest']['pages']:,} pages between {s['largest']['start']:%Y-%m-%d %H:%M} and "
             f"{s['largest']['end']:%H:%M} UTC.")
    if "posts" in s:
        P.append(f"{s['deleted_before_last_write']:,} deletions ({pct(s['deleted_before_last_write'] / s['n'])}) happened before the "
                 f"last post was written ({s['last_write']:%Y-%m-%d %H:%M}); {s['posts_on_deleted_pages']:,} of {s['posts']:,} posts "
                 f"({pct(s['posts_on_deleted_pages'] / s['posts'])}) sit on pages that were eventually deleted, "
                 f"{s['posts_deleted_while_writing']:,} of them deleted while the swarm was still writing.")
    if s["n_recreate"]:
        gap = f", median {s['recreate_gap_median_h']:.1f} h after the deletion" if s.get("recreate_gap_median_h") is not None else ""
        P.append(f"Recreations: {s['n_recreate']} first-recreation edges in the source ({s['recreate_with_revision']} with a stored "
                 f"revision, {s['n_recreate'] - s['recreate_with_revision']} without{gap}).")
    c = s.get("check")
    if c:
        P.append(f"The pipeline's own rule (each deletion -> the first later revision on the page) finds {c['pipeline_edges']} edges: "
                 f"{c['overlap']} shared with the source, {c['pipeline_only']} not in it"
                 + (f" ({c['pipeline_only_after_cutoff']} after the source's cutoff {c['cutoff'][:10]})" if c.get("cutoff") else "")
                 + f", {c['dump_only']} source edges missed. Of {c['posts_on_recreations']} posts on recreation revisions, "
                 f"{c['restored_posts']} restore text that stood on the page before the deletion and {c['fresh_posts']} are new.")
    return ["## Deletions and recreations\n", " ".join(P) + "\n"]


def probe_summary(probes: pd.DataFrame) -> dict:
    p = probes
    if not len(p):
        return {}
    by_day = p["ts"].dt.strftime("%Y-%m-%d").value_counts()
    by_ip = p["ip16"].astype(str).value_counts()
    cot = p["cotimed_label"].dropna() if "cotimed_label" in p else pd.Series(dtype=object)
    payload = p.loc[p["request_action"].astype(str).str.contains("<", regex=False), "request_action"].astype(str).tolist()
    return {
        "n": int(len(p)), "sites": sorted(set(p["site"].dropna().astype(str))),
        "first": p["ts"].min(), "last": p["ts"].max(),
        "n_ip16": int(by_ip.size), "n_success": int(p["success_observed"].map(lambda v: v is True or v == "True").sum()),
        "actions": p["request_action"].astype(str).map(lambda a: a if "<" not in a else "<payload>").value_counts().to_dict(),
        "top_day": by_day.index[0], "top_day_n": int(by_day.iloc[0]),
        "top_ip16": by_ip.index[0], "top_ip16_n": int(by_ip.iloc[0]),
        "n_cotimed": int(len(cot)), "cotimed_labels": sorted(set(cot.astype(str))),
        "payloads": payload[:3],
    }


def probe_section(s: dict) -> list[str]:
    if not s:
        return []
    P: list[str] = []
    acts = ", ".join(f"{k} {v}" for k, v in sorted(s["actions"].items(), key=lambda kv: -kv[1]))
    P.append(f"The request log contributes {s['n']} script-injection probe requests against {', '.join(s['sites']) or 'the wiki'} "
             f"({s['first']:%Y-%m-%d} to {s['last']:%Y-%m-%d}) from {s['n_ip16']} /16 prefixes; {s['n_success']} succeeded. "
             f"By request action: {acts}. {s['top_day_n']} fall on {s['top_day']}, {s['top_ip16_n']} from one prefix ({s['top_ip16']}).")
    if s["n_cotimed"]:
        P.append(f"{s['n_cotimed']} probes land within 1 s of a save from a prefix that holds a single stored account "
                 f"({', '.join(s['cotimed_labels'])}); they are *co-timed with* that account, not attributed to it: a shared /16 "
                 f"names an address block, and the collector may have flagged the account's own edit-form traffic.")
    else:
        P.append("No probe lands within 1 s of a save from a prefix with a single stored account, so none is co-timed with an agent.")
    if s["payloads"]:
        P.append(f"The only visible payload is `{s['payloads'][0]}`; the other rows record the action and parameter name, not the payload.")
    P.append("Probing leaves no text footprint in the revision corpus, so it cannot appear under Techniques.")
    return ["## Probing\n", " ".join(P) + "\n"]
