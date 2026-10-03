"""Cross-site view: technique spread between surfaces, an activity timeline and
the collectors' coverage inventory.

Inputs are the canonical ``events`` table (wiki run concatenated with the
cross-site corpus; ``site`` is the surface a post lives on), the adapter's
``Technique`` list and ``site-coverage.csv``.  Every "first seen on site X"
below means first seen *in the captured material* (see docs/DATA.md).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import plotting
from ..adapters.base import Technique
from ..plotting import plt

# ---------------------------------------------------------------- surfaces

MAIN_WIKI = "dse"
WIKI_SITES = {"probier", "fractal", "dorfwiki", "publictestwiki", "uncyclopedia", "usemod", "prowiki.org/wiki4d"}
SHORTENERS = {"rmn.re", "vanderbi.lt", "uoft.me", "goto.unm.edu", "is.gd", "tinyurl.com", "v.gd", "da.gd",
              "url.popcat.xyz", "u.ethz.ch"}
PASTE_HINTS = ("paste", "pastebin", "anna.fyi", "k4be")
SURFACE_GROUPS = ["main wiki (dse)", "other wikis", "pastebins", "shorteners", "other"]
TIMELINE_START, TIMELINE_END = pd.Timestamp("2026-05-10"), pd.Timestamp("2026-07-05")


def surface_group(site: str) -> str:
    s = (site or "").lower()
    if s == MAIN_WIKI:
        return SURFACE_GROUPS[0]
    if s in SHORTENERS:
        return SURFACE_GROUPS[3]
    if any(h in s for h in PASTE_HINTS):
        return SURFACE_GROUPS[2]
    if s in WIKI_SITES or "wikiservice.at" in s or "wiki" in s:
        return SURFACE_GROUPS[1]
    return SURFACE_GROUPS[4]


# the wiki dump and the cross-site corpus name the same four wikis differently
SITE_ALIASES = {"prowiki.org/dse": "dse", "wikiservice.at/probier": "probier", "wikiservice.at/fractal": "fractal",
                "wikiservice.at/dorfwiki": "dorfwiki", "dorfwiki.org": "dorfwiki"}


def canonical_site(site: str) -> str:
    return SITE_ALIASES.get(str(site), str(site))


def short_site(site: str) -> str:
    s = canonical_site(site)
    for prefix in ("prowiki.org/", "wikiservice.at/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _with_site(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    if "site" not in ev.columns or ev["site"].isna().all():
        ev["site"] = ev["channel"].astype(str).str.split("/").str[0]
    ev["site"] = ev["site"].fillna("").astype(str).map(canonical_site)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    return ev


def _hours(delta) -> float:
    return float(pd.Timedelta(delta).total_seconds() / 3600)


# ---------------------------------------------------------------- 1. spread

SPREAD_COLS = ["technique", "site", "origin_site", "first_seen", "n_posts", "n_authors", "hours_after_origin"]


def technique_spread(events: pd.DataFrame, techniques: list[Technique], min_site_events: int = 5) -> pd.DataFrame:
    """One row per (technique, site) where an at-risk site (>= ``min_site_events``
    events in total) posted text matching the technique regex (case-insensitive).

    ``hours_after_origin`` is measured from the technique's global first post
    across *all* sites; ``origin_site`` is where that post lives.  Rows are
    ordered by the technique's global first appearance, then by lag."""
    ev = _with_site(events)
    text = ev["text"].fillna("").astype(str)
    site_n = ev.groupby("site").size()
    at_risk = set(site_n[site_n >= min_site_events].index)
    rows = []
    for tech in techniques:
        hit = ev[text.str.contains(tech.pattern, case=False, regex=True, na=False)].sort_values("ts")
        if hit.empty:
            continue
        t0, origin = hit["ts"].iloc[0], hit["site"].iloc[0]
        for site, g in hit[hit["site"].isin(at_risk)].groupby("site"):
            first = g["ts"].min()
            authors = g["author_raw"].fillna("").astype(str)
            rows.append({
                "technique": tech.name, "site": site, "origin_site": origin, "first_seen": first,
                "n_posts": int(len(g)), "n_authors": int(authors[authors != ""].nunique()),
                "hours_after_origin": _hours(first - t0), "_t0": t0,
            })
    if not rows:
        return pd.DataFrame(columns=SPREAD_COLS)
    df = pd.DataFrame(rows).sort_values(["_t0", "hours_after_origin", "site"]).drop(columns="_t0")
    return df[SPREAD_COLS].reset_index(drop=True)


def technique_site_summary(spread_df: pd.DataFrame, events: pd.DataFrame, techniques: list[Technique],
                           min_site_events: int = 5) -> pd.DataFrame:
    """One row per technique that appeared at least once.

    n_sites_at_risk counts at-risk sites (including the origin) with any event
    at or after the technique's first appearance; median / within-24h figures
    are over the *other* adopting sites, so the origin's 0 h does not pull them down."""
    ev = _with_site(events)
    site_n = ev.groupby("site").size()
    at_risk = site_n[site_n >= min_site_events].index
    last_by_site = ev.groupby("site")["ts"].max()
    rows = []
    for tech in techniques:
        g = spread_df[spread_df["technique"] == tech.name] if len(spread_df) else spread_df
        text = ev["text"].fillna("").astype(str)
        hit = ev[text.str.contains(tech.pattern, case=False, regex=True, na=False)]
        if hit.empty:
            continue
        t0 = hit["ts"].min()
        origin = hit.loc[hit["ts"].idxmin(), "site"]
        others = g[g["site"] != origin]["hours_after_origin"] if len(g) else pd.Series(dtype=float)
        rows.append({
            "technique": tech.name, "origin_site": origin, "first_seen": t0,
            "n_sites_adopted": int(g["site"].nunique()) if len(g) else 0,
            "n_sites_at_risk": int((last_by_site.reindex(at_risk) >= t0).sum()),
            "median_hours_to_site_adoption": float(others.median()) if len(others) else float("nan"),
            "sites_within_24h": int((others <= 24).sum()),
            "n_posts": int(len(hit)),
        })
    cols = ["technique", "origin_site", "first_seen", "n_sites_adopted", "n_sites_at_risk",
            "median_hours_to_site_adoption", "sites_within_24h", "n_posts"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("first_seen").reset_index(drop=True)


# ---------------------------------------------------------------- 2. spread figure

def _clusters(xs_px: np.ndarray, gap_px: float) -> list[list[int]]:
    """Indices (in x order) grouped so that consecutive markers closer than ``gap_px`` share a label block."""
    out: list[list[int]] = []
    for i in np.argsort(xs_px, kind="stable"):
        if out and xs_px[i] - xs_px[out[-1][-1]] < gap_px:
            out[-1].append(int(i))
        else:
            out.append([int(i)])
    return out


def _wrap(names: list[str], width: int = 26) -> str:
    lines, cur = [], ""
    for n in names:
        cand = n if not cur else f"{cur} · {n}"
        if cur and len(cand) > width:
            lines.append(cur)
            cur = n
        else:
            cur = cand
    lines.append(cur)
    return "\n".join(lines)


def technique_spread_figure(spread_df: pd.DataFrame, path, top: int = 6) -> str | None:
    """Strip chart: one row per technique (``top`` by sites adopted), one marker
    per adopting site at hours after the technique's first appearance.  The
    origin site sits at 0 h as a hollow marker; the x axis is symlog so that
    0 h and 600 h share one readable scale.  Markers that fall close together
    share one label block (sites in time order), placed above or below the row
    so that blocks never overlap."""
    if spread_df is None or spread_df.empty:
        return None
    n_sites = spread_df.groupby("technique")["site"].nunique()
    order = spread_df.drop_duplicates("technique")["technique"].tolist()  # already in first-seen order
    keep = sorted(order, key=lambda t: (-n_sites[t], order.index(t)))[:top]
    keep = [t for t in order if t in keep]  # chronological top-N
    df = spread_df[spread_df["technique"].isin(keep)].copy()
    df["x"] = df["hours_after_origin"].clip(lower=0)

    plotting.setup()
    fig, ax = plt.subplots(figsize=(10, 1.3 + 1.05 * len(keep)))
    ypos = {t: i for i, t in enumerate(reversed(keep))}
    xmax = max(24.0, float(df["x"].max()) * 1.35)
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set_xlim(-0.35, xmax)
    ax.set_ylim(-0.6, len(keep) - 0.4)
    ticks = [t for t in [0, 1, 6, 24, 72, 168, 336, 720, 1440] if t <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0" if t == 0 else (f"{t // 24}d" if t >= 24 else f"{t}h") for t in ticks])
    ax.grid(axis="y", visible=False)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()))
    fig.canvas.draw()
    ax_px = ax.get_window_extent()
    char_px = 5.2 * (fig.dpi / 72)   # ~8pt text width per character, in pixels
    line_dy = 0.17                   # data units per text line (row pitch is 1.0)

    for tech, g in df.groupby("technique", sort=False):
        y = ypos[tech]
        g = g.sort_values("x").reset_index(drop=True)
        ax.plot([0, g["x"].max()], [y, y], color=plotting.GRID, lw=3, solid_capstyle="round", zorder=1)
        is_origin = g["site"] == g["origin_site"]
        ax.scatter(g.loc[~is_origin, "x"], [y] * int((~is_origin).sum()), s=70, color=plotting.SERIES[0],
                   edgecolor=plotting.SURFACE, linewidth=1.5, zorder=3)
        ax.scatter(g.loc[is_origin, "x"], [y] * int(is_origin.sum()), s=90, facecolor=plotting.SURFACE,
                   edgecolor=plotting.SERIES[1], linewidth=2.2, zorder=4)
        xs_px = ax.transData.transform(np.column_stack([g["x"], np.full(len(g), y)]))[:, 0]
        # placed blocks per side (above=True) and stacking level: (left_px, right_px, n_lines)
        occupied: dict[tuple[bool, int], list[tuple[float, float, int]]] = {}
        for idx in _clusters(xs_px, gap_px=26):
            names = [short_site(s) for s in g.loc[idx, "site"]]
            label = _wrap(names)
            lines = label.split("\n")
            w = max(len(l) for l in lines) * char_px
            cx = float(np.mean(xs_px[idx]))
            # keep the block inside the axes horizontally
            if cx - w / 2 < ax_px.x0 + 4:
                ha, left = "left", max(cx - 10, ax_px.x0 + 4)
            elif cx + w / 2 > ax_px.x1 - 4:
                ha, left = "right", min(cx + 10, ax_px.x1 - 4) - w
            else:
                ha, left = "center", cx - w / 2
            right = left + w

            def busy(side, lvl):
                return any(left < r + 6 and right > l - 6 for l, r, _ in occupied.get((side, lvl), []))

            # first free slot, nearest the row first: above 0, below 0, above 1, below 1, ...
            above, level = next(((sd, lv) for lv in range(4) for sd in (True, False) if not busy(sd, lv)), (True, 4))
            occupied.setdefault((above, level), []).append((left, right, len(lines)))
            # offset = base gap + height of every block stacked beneath on this side
            dy = 0.16 + sum(max(n for _, _, n in occupied.get((above, lv), [(0, 0, 1)])) * line_dy + 0.04
                            for lv in range(level))
            xd = ax.transData.inverted().transform((left if ha == "left" else right if ha == "right" else cx, 0))[0]
            ax.text(xd, y + (dy if above else -dy), label, fontsize=8, color=plotting.INK_2, ha=ha,
                    va="bottom" if above else "top", zorder=5, clip_on=False, linespacing=1.15)
    ax.set_xlabel("Hours after the technique's first post anywhere (symlog)")
    ax.set_title("Where techniques travelled: first post per surface", loc="left", pad=14)
    ax.scatter([], [], s=90, facecolor=plotting.SURFACE, edgecolor=plotting.SERIES[1], linewidth=2.2,
               label="origin surface (0 h)")
    ax.scatter([], [], s=70, color=plotting.SERIES[0], label="later surface, first matching post")
    ax.legend(loc="lower right", ncol=1, borderaxespad=0.6)
    fig.tight_layout()
    return plotting.save(fig, path)


# ---------------------------------------------------------------- 3. timeline

def timeline(events: pd.DataFrame, path, markers: list[tuple[pd.Timestamp, str]] = ()) -> str:
    """Posts per day, one small multiple per surface group, shared day axis
    (2026-05-10 .. 2026-07-05).  A row switches to a log y axis when its busiest
    day is >= 50x its quietest non-empty day; empty days draw no bar."""
    ev = _with_site(events)
    ev["group"] = ev["site"].map(surface_group)
    ev["day"] = ev["ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    days = pd.date_range(TIMELINE_START, TIMELINE_END, freq="D")
    counts = ev.groupby(["group", "day"]).size().unstack(0).reindex(days).fillna(0)
    for g in SURFACE_GROUPS:
        if g not in counts.columns:
            counts[g] = 0.0
    n_sites = ev.groupby("group")["site"].nunique()

    plotting.setup()
    fig, axes = plt.subplots(len(SURFACE_GROUPS), 1, figsize=(11, 7), sharex=True)
    for ax, g in zip(axes, SURFACE_GROUPS):
        c = counts[g]
        pos = c[c > 0]
        total = int(c.sum())
        use_log = len(pos) > 0 and pos.max() / max(pos.min(), 1) >= 50
        if len(pos):
            if use_log:
                ax.set_yscale("log")
                ax.set_ylim(0.7, pos.max() * 2.5)
                ax.bar(pos.index, pos.values, width=0.85, color=plotting.SERIES[0], bottom=0.7, zorder=2)
            else:
                ax.set_ylim(0, pos.max() * 1.35)
                ax.bar(pos.index, pos.values, width=0.85, color=plotting.SERIES[0], zorder=2)
            peak_day = pos.idxmax()
            ax.text(0.005, 0.93, f"peak {int(pos.max()):,} on {peak_day:%b %d}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color=plotting.INK_2)
        else:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "no dated posts", transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color=plotting.INK_2)
        label = f"{g}\n{total:,} posts, {int(n_sites.get(g, 0))} site{'s' if n_sites.get(g, 0) != 1 else ''}"
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, labelpad=8)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
        if use_log:
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.tick_params(axis="y", labelsize=8)
        ax.text(0.995, 0.92, "log" if use_log else "", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=plotting.INK_2)
    # labels sit in a strip above the top row; markers closer than ~5 days share a column, so each
    # gets its own vertical level (0, 1, 2 ...) and the levels restart once the gap is wide enough
    whens = []
    for when, _ in markers:
        when = pd.Timestamp(when)
        whens.append(when.tz_convert("UTC").tz_localize(None) if when.tzinfo else when)
    order = sorted(range(len(markers)), key=lambda k: whens[k])
    level, prev = 0, None
    levels = {}
    for k in order:
        level = 0 if prev is None or (whens[k] - prev) > pd.Timedelta(days=5) else level + 1
        levels[k], prev = level, whens[k]
    for k, (when, text) in enumerate(markers):
        for ax in axes:
            ax.axvline(whens[k], color=plotting.INK_2, lw=1, ls="--", zorder=3)
        axes[0].annotate(text, (whens[k], 1.0), xycoords=("data", "axes fraction"),
                         xytext=(3, 3 + 11 * levels[k]), textcoords="offset points", fontsize=8,
                         color=plotting.INK, ha="left", va="bottom", annotation_clip=False)
    if markers:
        fig.subplots_adjust(top=0.9 - 0.012 * max(levels.values()))
    axes[-1].set_xlim(TIMELINE_START - pd.Timedelta(hours=12), TIMELINE_END + pd.Timedelta(hours=12))
    axes[-1].xaxis.set_major_locator(plotting.matplotlib.dates.WeekdayLocator(byweekday=0))
    axes[-1].xaxis.set_major_formatter(plotting.matplotlib.dates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("Day (UTC), 2026")
    fig.suptitle("Activity across surfaces: posts per day", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return plotting.save(fig, path)


# ---------------------------------------------------------------- 4. coverage

def coverage_summary(site_coverage_csv_path) -> dict:
    """Digest of the collectors' site inventory (``site-coverage.csv``)."""
    d = pd.read_csv(site_coverage_csv_path)
    gap = d["specific_prior_gap_remains"].astype(str).str.lower().isin(["true", "1"])
    # a count repeated verbatim across many rows is an inventory total, not a per-site figure
    vc = d.loc[d["selected_distinct_texts"] > 0, "selected_distinct_texts"].value_counts()
    shared_val, shared_n = (int(vc.index[0]), int(vc.iloc[0])) if len(vc) and vc.iloc[0] >= 5 else (None, 0)
    has_text = (d["selected_distinct_texts"] > 0) & (d["selected_distinct_texts"] != (shared_val if shared_val else -1))
    d = d.assign(_gap=gap, _text=has_text)
    by_cat = d.groupby("category").agg(
        sites=("site", "size"), gaps=("_gap", "sum"), sites_with_text=("_text", "sum"),
        discord_urls=("discord_urls", "sum"), fresh_failures=("fresh_read_failures_or_redirects", "sum"),
    ).astype(int).sort_values("sites", ascending=False)
    unsearched = (d["prior_status"] == "remote_inventory_only_gap") | \
                 (d["compilation_status"] == "no_selected_agent_text_in_compilation")
    n_discord = int(d["discord_urls"].sum())
    n_relays = int((d["category"] == "relays").sum())
    n_short = int((d["category"] == "shorteners").sum())
    caveats = []
    if shared_val is not None:
        caveats.append(f"{shared_n} rows share selected_distinct_texts = {shared_val}; that is a shared inventory "
                       f"count, not a per-site text count, so the column must not be summed.")
    caveats.append(f"relays and shorteners were deliberately not visited, so no read data exists "
                   f"({n_relays} relays, {n_short} shorteners).")
    caveats.append(f"{n_discord} Discord-linked URLs but no Discord messages are in the corpus.")
    caveats.append(f"{int(unsearched.sum())} of {len(d)} surfaces were never searched or yielded no agent text; "
                   f"absence there is not evidence of absence.")
    caveats.append("every row's scope is 'selected artifacts; whole-site completeness not established'.")
    return {
        "n_sites": int(len(d)), "by_category": by_cat, "n_gap_rows": int(gap.sum()),
        "n_unsearched": int(unsearched.sum()), "relays_unvisited": n_relays, "shorteners": n_short,
        "discord_urls": n_discord, "shared_inventory_value": shared_val, "shared_inventory_rows": shared_n,
        "caveats": caveats,
    }


def _md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Local copy of report.md_table (report imports the analyses; avoid the cycle)."""
    if df is None or len(df) == 0:
        return "_(none)_\n"
    d = df.reset_index() if index else df
    cols = [str(c).replace("|", "\\|") for c in d.columns]

    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{v:.2f}"
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d %H:%M")
        return str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for rec in d.to_dict("records"):
        lines.append("| " + " | ".join(fmt(rec[c]) for c in d.columns) + " |")
    return "\n".join(lines) + "\n"


def coverage_markdown(summary: dict) -> str:
    s = summary
    L = [f"{s['n_sites']} surfaces are inventoried; {s['n_gap_rows']} carry a specific unresolved gap and "
         f"{s['n_unsearched']} were never searched or yielded no agent text.  Everything downstream "
         f"(exposure shares, cross-site first-seen times) is bounded by what was captured.\n",
         _md_table(s["by_category"]),
         "Caveats:\n"]
    L += [f"- {c}" for c in s["caveats"]]
    return "\n".join(L) + "\n"
