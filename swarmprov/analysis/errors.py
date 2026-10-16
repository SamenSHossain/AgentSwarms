"""A4 - error propagation: competing values for the same slot, and who won.

A *value slot* is (family, item) and its variants are values (16.40 vs 16.38).
A *sequence slot* is (family, episode) and its variants are items - e.g. the
cracked-seed prediction "G5 = Maryland" vs the observed "G5 = Montana".

For each slot we track, per agent, the first time it carried each variant,
and report: first appearance of every variant, the eventual winner (the most
common variant among the last third of carriers; none on a tie), when the
winner first held >= 50% of new carriers in a 3-hour bin and kept it, and a
status: *held* (the first variant still wins), *overturned* (a later variant
won and kept that majority on enough late carriers) or *contested*.  Two
variants first seen at the same moment leave no first variant: contested.

Analyst reference values never change the status; they only label the outcome.
Posts without a timestamp take no part.  With ``bin_h=None`` (a sequence clock,
which has no hours) the whole run is one bin, so a later winner must hold the
majority of all the slot's carriers.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .. import plotting
from ..claims import value_key
from ..plotting import plt
from ..rules import norm_value

# majority-bin width (hours), the share of carriers that counts as "late", and the fewest late carriers
# that can overturn the first variant
BIN_H, TAIL_SHARE, MIN_TAIL = 3.0, 1 / 3, 5
SPARSE_BIN = 3   # the figure draws a bin with fewer new carriers than this faint


def _status(first, winner, t50, n_tail: int, min_tail: int) -> str:
    if winner is None or first is None:
        return "contested"            # the late carriers tie, or no variant came first
    if winner == first:
        return "held"
    if t50 is None:
        return "contested"            # the winner never kept a majority of new carriers
    return "contested (small n)" if n_tail < min_tail else "overturned"


def _slot_stats(carriers: pd.DataFrame, slot: str, bin_h: float | None = BIN_H, tail_share: float = TAIL_SHARE,
                min_tail: int = MIN_TAIL) -> dict:
    """carriers: one row per (agent, variant) with first ts."""
    c = carriers.sort_values("ts")
    agent_col = [col for col in c.columns if col not in ("ts", "variant")][0]
    overlap = int((c.groupby(agent_col)["variant"].nunique() > 1).sum())
    tail = c["variant"].iloc[int(len(c) * (1 - tail_share)):].value_counts()
    n_tail = int(tail.sum())
    tie = len(tail) > 1 and tail.iloc[0] == tail.iloc[1]
    winner = None if tie else tail.index[0]
    firsts = c.groupby("variant")["ts"].min().sort_values()
    first_tie = len(firsts) > 1 and firsts.iloc[0] == firsts.iloc[1]
    first = None if first_tie else firsts.index[0]
    t0 = c["ts"].min()
    c = c.assign(h=(c["ts"] - t0).dt.total_seconds() / 3600)
    bins = np.arange(0, c["h"].max() + bin_h, bin_h) if bin_h else np.array([0.0, c["h"].max() + 1.0])
    new = (c.assign(b=pd.cut(c["h"], bins=bins, include_lowest=True, labels=bins[:-1]))
            .groupby(["b", "variant"], observed=False).size().unstack(fill_value=0))
    frac = new.div(new.sum(axis=1).replace(0, np.nan), axis=0)
    w = frac.get(winner) if winner is not None else None
    t50 = None
    if w is not None:
        ok = (w >= 0.5) | w.isna()
        for i in range(len(ok)):
            if w.iloc[i] >= 0.5 and ok.iloc[i:].all():
                t50 = max(t0 + pd.Timedelta(hours=float(frac.index[i])), firsts[winner])
                break
    t_first = firsts[winner] if winner is not None else None
    later = t50 is not None and first is not None and winner != first
    return {
        "slot": slot, "n_carriers": len(c), "variants": firsts.index.tolist(), "overlap_agents": overlap,
        "first_seen": {k: v for k, v in firsts.items()}, "winner": winner, "first_variant": first,
        "first_tie": bool(first_tie),
        "tail_counts": {k: int(v) for k, v in tail.items()}, "n_tail": n_tail, "tie": bool(tie),
        "status": _status(first, winner, t50, n_tail, min_tail),
        "t_winner_first": t_first, "t_winner_majority": t50,
        "hours_to_overtake": (t50 - t_first).total_seconds() / 3600 if later else np.nan,
        "share_by_bin": frac, "new_by_bin": new, "bin_edges_h": bins, "t0": t0, "carriers": c[["ts", "variant"]],
    }


def reference_keys(values: dict) -> dict[tuple[str, str], str]:
    """Config ``reference_values`` {"family|item": value} -> {(family, item): value as text}."""
    out = {}
    for k, v in values.items():
        fam, _, item = str(k).partition("|")
        out[(fam.strip(), item.strip())] = str(v).strip()
    return out


def unused_references(refs: dict, items: set, disputed: set, tabled: set) -> tuple[list[str], list[str]]:
    """Reference keys naming no item or disputed slot (a mistyped key surfaces here), and those for
    disputed slots left out of the table."""
    def keys(pairs):
        return sorted(f"{f}|{i}" for f, i in pairs)
    return keys(set(refs) - disputed - items), keys(set(refs) & (disputed - tabled))


def _key(v) -> str:
    s = str(v).strip()
    return value_key(norm_value(s.lstrip("$€£"))) or s.casefold()


def match_reference(variants: list[str], ref) -> str | None:
    """The variant a reference value names: the same text, else the only variant with the same
    comparison key ('1,481' and '$1,481', '16.40%' and '16.4'); None if it names none."""
    ref = str(ref).strip()
    if ref in variants:
        return ref
    hits = [v for v in variants if _key(v) == _key(ref)]
    return hits[0] if len(hits) == 1 else None


def _rounds_to(a: str, b: str) -> bool:
    """One number is the other rounded to fewer decimals (9.9 and 9.91)."""
    try:
        x, y = Decimal(a), Decimal(b)
    except InvalidOperation:
        return False
    if not (x.is_finite() and y.is_finite()):
        return False
    if x.as_tuple().exponent < y.as_tuple().exponent:
        x, y = y, x
    ex = x.as_tuple().exponent
    return ex > y.as_tuple().exponent and y.quantize(Decimal(1).scaleb(ex), ROUND_HALF_UP) == x


def outcome(s: dict, ref) -> str:
    """The slot against an analyst reference value; the status itself is never changed.  A held or
    overturned slot is judged on the variant left standing (the winner if overturned, else the first);
    a contested one is unresolved, noting whether the reference came first."""
    ref = match_reference(s["variants"], ref)
    if ref is None:
        return "reference not among variants"
    first = s["first_variant"]
    if s["status"] not in ("held", "overturned"):
        if all(_rounds_to(v, ref) for v in s["variants"] if v != ref):
            return "precision variant: unresolved"
        return "unresolved" if first is None else f"unresolved ({'truth' if first == ref else 'error'} first)"
    final = s["winner"] if s["status"] == "overturned" else first
    others = [final] if final != ref else [v for v in s["variants"] if v != ref]
    if all(_rounds_to(v, ref) for v in others):
        return "precision variant: reference " + ("prevailed" if final == ref else "did not prevail")
    if final == ref:
        return "truth held" if first == ref else "error corrected"
    return "truth displaced by error" if first == ref else "error persisted"


def value_slots(mentions: pd.DataFrame, agent_col: str, min_agents: int = 3,
                min_share: float = 0.1, **rule) -> list[dict]:
    """``rule``: bin_h / tail_share / min_tail, as in ``_slot_stats`` (also for the other slot finders)."""
    m = mentions[(mentions["family"] != "url") & mentions["ts"].notna()]
    out = []
    for (fam, item), g in m.groupby(["family", "item"]):
        first = g.sort_values("ts").drop_duplicates([agent_col, "value_key"])
        counts = first.groupby("value_key")[agent_col].nunique()
        # same order of magnitude as the modal value: '0', '15' next to a 95,897 answer are timers/counters
        try:
            mode = abs(float(counts.idxmax())) or 1.0
            counts = counts[[0.1 <= abs(float(k)) / mode <= 10 for k in counts.index]]
        except ValueError:
            pass
        strong = counts[(counts >= min_agents) & (counts >= min_share * counts.sum())]
        if len(strong) < 2:
            continue
        car = first[first["value_key"].isin(strong.index)].rename(columns={"value_key": "variant"})
        st = _slot_stats(car[["ts", agent_col, "variant"]], f"{fam} | {item}", **rule)
        st.update(family=fam, item=item, kind="value", agents_per_variant=strong.to_dict())
        out.append(st)
    return sorted(out, key=lambda s: -s["n_carriers"])


def sequence_slots(claims: pd.DataFrame, agent_col: str, min_agents: int = 2,
                   min_share: float = 0.1, min_overlap: int = 2, **rule) -> list[dict]:
    """Competing items for the same round.  Requires >= ``min_overlap`` agents who
    carried both variants (they discussed the conflict); otherwise two variants
    with disjoint carriers are just different task versions, not a dispute."""
    # strict "round marker -> item" reports only: rounds inferred for free-text predictions are too loose
    c = claims[(claims["event_type"] == "sequence") & claims["episode"].notna()
               & claims["item"].notna() & (claims["family"] != "") & claims["ts"].notna()]
    out = []
    for (fam, ep), g in c.groupby(["family", "episode"]):
        first = g.sort_values("ts").drop_duplicates([agent_col, "item"])
        counts = first.groupby("item")[agent_col].nunique()
        strong = counts[(counts >= min_agents) & (counts >= min_share * counts.sum())]
        if len(strong) < 2:
            continue
        car = first[first["item"].isin(strong.index)].rename(columns={"item": "variant"})
        st = _slot_stats(car[["ts", agent_col, "variant"]], f"{fam} | R{int(ep)}", **rule)
        if st["overlap_agents"] < min_overlap:
            continue
        st.update(family=fam, item=f"R{int(ep)}", kind="sequence", agents_per_variant=strong.to_dict())
        out.append(st)
    return sorted(out, key=lambda s: -s["n_carriers"])


def regex_slots(events: pd.DataFrame, agents: pd.DataFrame, disputes: list[dict], agent_col: str,
                **rule) -> list[dict]:
    """Analyst-declared disputes: {"family", "slot", "context": regex, "variants": {name: regex}}.
    An agent carries a variant from its first post (in the family) matching both the
    context and the variant pattern - for claims no generic parser can attribute,
    e.g. 'RNG prep Maryland 52,395' said in a G5 context."""
    amap = agents.set_index("author_raw")[agent_col]
    out = []
    for d in disputes:
        ev = events[(events["family"].fillna("") == d["family"]) & events["ts"].notna()]
        ev = ev[ev["text"].str.contains(d.get("context", "."), regex=True)]
        rows = []
        for name, rx in d["variants"].items():
            hit = ev[ev["text"].str.contains(rx, regex=True)]
            for r in hit.itertuples(index=False):
                rows.append({"ts": r.ts, agent_col: amap.get(r.author_raw, r.author_raw), "variant": name})
        if not rows:
            continue
        car = pd.DataFrame(rows).sort_values("ts").drop_duplicates([agent_col, "variant"])
        if car["variant"].nunique() < 2:
            continue
        st = _slot_stats(car[["ts", agent_col, "variant"]], f"{d['family']} | {d['slot']}", **rule)
        st.update(family=d["family"], item=d["slot"], kind="declared",
                  agents_per_variant=car.groupby("variant")[agent_col].nunique().to_dict())
        out.append(st)
    return out


def pick_featured(vslots: list[dict], sslots: list[dict], featured: list[tuple[str, str]], k: int = 4,
                  declared: list[dict] = ()) -> list[dict]:
    chosen = list(declared)[:k]
    for fam, item in featured:
        ep = re.fullmatch(r"[RG#]?(\d)", item)
        pool = sslots if ep else vslots
        key = f"R{ep.group(1)}" if ep else item
        for s in pool:
            if s["family"] == fam and s["item"] == key and s not in chosen:
                chosen.append(s)
    for s in vslots:
        if len(chosen) >= k:
            break
        if s not in chosen:
            chosen.append(s)
    return chosen[:k]


def axis_range(ts: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """A slot's time axis, padded by 2%: from the first carrier, or from the first one past the earliest 3% when
    those would take more than a third of the axis (a stray early post)."""
    ts = ts.sort_values()
    start, end = ts[ts >= ts.quantile(0.03)].iloc[0], ts.iloc[-1]
    if start - ts.iloc[0] <= 0.5 * (end - start):
        start = ts.iloc[0]
    pad = max(0.02 * (end - start), pd.Timedelta(minutes=1))
    return start - pad, end + pad


def plotted_variants(s: dict) -> list:
    """The (at most three) variants the figure draws, in order of first appearance: the first variant, the winner or
    every variant tied for the lead of the late carriers, then the next ones to appear."""
    tail = s["tail_counts"]
    lead = [s["winner"]] if s["winner"] is not None else [v for v, k in tail.items() if k == max(tail.values())]
    keep = list(dict.fromkeys([s["variants"][0], *lead, *s["variants"]]))[:3]
    return sorted(keep, key=s["variants"].index)


def figure(slots: list[dict], path) -> str | None:
    """One column per slot, drawing ``plotted_variants``.  Top: cumulative distinct carriers of each variant over wall
    time; every carrier counts, so a line ends at the variant's agents, and those left of a clipped axis
    (``axis_range``) are counted in the legend with the first one's time, as are the winner and when its lasting
    majority began (a dashed line).  Bottom: the winner's share of the new carriers in each majority bin (every
    plotted variant's when there is no winner), empty bins left out and sparse ones faint."""
    if not slots:
        return None
    plotting.setup()
    n = len(slots)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 6.8), sharex="col", squeeze=False,
                             gridspec_kw={"height_ratios": [2.6, 1]})
    for ax, axb, s in zip(axes[0], axes[1], slots):
        car, win, t50 = s["carriers"], s["winner"], s["t_winner_majority"]
        left, right = axis_range(car["ts"])
        edges = s["t0"] + pd.to_timedelta(s["bin_edges_h"], unit="h")
        new = s["new_by_bin"].sum(axis=1).to_numpy()
        dense, sparse = new >= SPARSE_BIN, (new > 0) & (new < SPARSE_BIN)
        for i, v in enumerate(plotted_variants(s)):
            t = car.loc[car["variant"] == v, "ts"].sort_values()
            k = s["agents_per_variant"][v]
            ax.step(list(t) + [car["ts"].max()], list(range(1, len(t) + 1)) + [len(t)], where="post",
                    color=plotting.SERIES[i],
                    label=f"{v} ({k} agent{'s' * (k != 1)})" + (", winner" if v == win else ""))
            if win is None or v == win:
                share = s["share_by_bin"][v].to_numpy()
                for keep, alpha in ((dense, 1.0), (sparse, 0.3)):   # above the grid and the 50% line
                    axb.hlines(share[keep], edges[:-1][keep], edges[1:][keep], color=plotting.SERIES[i], lw=2.5,
                               alpha=alpha, zorder=3)
        if t50 is not None:
            ax.axvline(t50, color=plotting.INK_2, lw=1, ls="--",
                       label=f"winner's lasting majority of new carriers from {t50:%m-%d %H:%M}")
            axb.axvline(t50, color=plotting.INK_2, lw=1, ls="--")
        notes = ["no winner: the late carriers tie"] if win is None else (
            [f"{win} never held a lasting majority of new carriers"] if t50 is None else [])
        early = int((car["ts"] < left).sum())
        if early:
            notes.append(f"{early} earlier carrier{'s' * (early > 1)} before {left:%m-%d %H:%M}, "
                         f"{'first ' * (early > 1)}at {car['ts'].min():%m-%d %H:%M}")
        handles = ax.get_legend_handles_labels()[0] + [Line2D([], [], ls="", label=t) for t in notes]
        if (sparse & (edges[1:] > left) & (edges[:-1] < right)).any():
            handles.append(Line2D([], [], color=plotting.INK_2, lw=2.5, alpha=0.3,
                                  label=f"share in a bin with fewer than {SPARSE_BIN} new carriers"))
        axb.axhline(0.5, color=plotting.NEUTRAL, lw=1)
        axb.set_ylim(-0.05, 1.05)      # room below 0% so a 0% segment is not half hidden by the axis
        axb.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_xlim(left, right)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
        ax.set_title(s["slot"].replace("datausa-", "").replace(" | ", "\n") + f"\n{s['status']}", fontsize=10)
        axb.tick_params(axis="x", rotation=30)
        axb.xaxis.set_major_locator(plotting.matplotlib.dates.AutoDateLocator(minticks=3, maxticks=6))
        axb.xaxis.set_major_formatter(plotting.matplotlib.dates.DateFormatter("%m-%d %H:%M"))
        axb.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.62), fontsize=8, ncol=1)
    axes[0][0].set_ylabel("Agents carrying the value (cumulative)")
    axes[1][0].set_ylabel(f"Share of new carriers\nper {BIN_H:g} h bin")
    fig.suptitle("Competing values: did a later value overturn the first?", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return plotting.save(fig, path)
