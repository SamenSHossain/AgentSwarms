"""Turn page snapshots (wiki revisions) into posts.

A wiki revision stores the whole page, so the unit of authorship is not the
revision but the paragraph it added.  Line diffs over-attribute: pages that
are re-saved with drifting encodings change every line on every save.  We
instead split each snapshot into paragraphs (and further at signature line
ends), and a paragraph is new the first time its normalised text appears on
that page since the page was last deleted.

The paragraph's author is its trailing signature when there is one, else the
account that saved the revision.
"""

from __future__ import annotations

import re

import pandas as pd

from .textutil import SIGNATURE_RE, fix_mojibake, norm_key, stable_id


def split_posts(body: str) -> list[str]:
    """Split a page snapshot into posts: blank-line blocks, cut after signatures."""
    if "\\n" in body:
        body = body.replace("\\n", "\n")
    posts: list[str] = []
    for block in re.split(r"\n\s*\n", body):
        cur: list[str] = []
        for line in block.split("\n"):
            cur.append(line)
            if SIGNATURE_RE.search(line):
                posts.append("\n".join(cur).strip())
                cur = []
        if cur and "\n".join(cur).strip():
            posts.append("\n".join(cur).strip())
    return [p for p in posts if p]


def signature_of(post: str) -> str | None:
    last = post.rstrip().split("\n")[-1]
    m = SIGNATURE_RE.search(last)
    return m.group(1) if m else None


def snapshots_to_posts(revs: pd.DataFrame, deletions: pd.DataFrame | None = None,
                       boilerplate: set[str] | None = None) -> pd.DataFrame:
    """revs columns: channel, ts, rev_id, author_alt, body.  deletions: channel, ts."""
    boilerplate = {norm_key(b) for b in (boilerplate or set())}
    dels: dict[str, list[pd.Timestamp]] = {}
    if deletions is not None and len(deletions):
        for ch, g in deletions.groupby("channel"):
            dels[ch] = sorted(g["ts"])

    rows = []
    for ch, g in revs.sort_values(["channel", "ts", "rev_id"]).groupby("channel", sort=False):
        seen: set[str] = set()
        ch_dels = dels.get(ch, [])
        di = 0
        for r in g.itertuples(index=False):
            # A deletion between revisions empties the page: content re-posted
            # afterwards is visible again and counts as new.
            while di < len(ch_dels) and ch_dels[di] <= r.ts:
                seen = set()
                di += 1
            visible_until = ch_dels[di] if di < len(ch_dels) else pd.NaT
            for i, post in enumerate(split_posts(r.body)):
                k = norm_key(post)
                if not k or k in seen or k in boilerplate:
                    continue
                seen.add(k)
                text = fix_mojibake(post)
                sig = signature_of(text)
                rows.append({
                    "event_id": stable_id(r.rev_id, i, k[:200]),
                    "ts": r.ts,
                    "channel": ch,
                    "author_raw": sig or r.author_alt,
                    "author_alt": r.author_alt,
                    "text": text,
                    "parent_id": r.rev_id,
                    "visible_until": visible_until,
                })
    return pd.DataFrame(rows)
