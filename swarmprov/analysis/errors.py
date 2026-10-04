"""A4 - error propagation: competing values for the same slot, and who won.

A *value slot* is (family, item) and its variants are values (16.40 vs 16.38).
A *sequence slot* is (family, episode) and its variants are items - e.g. the
cracked-seed prediction "G5 = Maryland" vs the observed "G5 = Montana".

For each slot we track, per agent, the first time it carried each variant,
and report: first appearance of every variant, the eventual winner (majority
among the last third of carriers), and when the winner first held >= 50% of
carriers in a rolling window and kept it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import plotting
from ..plotting import plt


def _slot_stats(carriers: pd.DataFrame, slot: str, bin_h: float = 3.0) -> dict:
    """carriers: one row per (agent, variant) with first ts."""
    c = carriers.sort_values("ts")
    agent_col = [col for col in c.columns if col not in ("ts", "variant")][0]
    overlap = int((c.groupby(agent_col)["variant"].nunique() > 1).sum())
    tail = c.iloc[int(len(c) * 2 / 3):]
    winner = tail["variant"].value_counts().idxmax()
    firsts = c.groupby("variant")["ts"].min().sort_values()
    t0 = c["ts"].min()
    c = c.assign(h=(c["ts"] - t0).dt.total_seconds() / 3600)
    bins = np.arange(0, c["h"].max() + bin_h, bin_h)
    share = (c.assign(b=pd.cut(c["h"], bins=bins, include_lowest=True, labels=bins[:-1]))
              .groupby(["b", "variant"], observed=False).size().unstack(fill_value=0))
    frac = share.div(share.sum(axis=1).replace(0, np.nan), axis=0)
    w = frac.get(winner)
    t50 = None
    if w is not None:
        ok = (w >= 0.5) | w.isna()
        for i in range(len(ok)):
            if w.iloc[i] >= 0.5 and ok.iloc[i:].all():
                t50 = max(t0 + pd.Timedelta(hours=float(frac.index[i])), firsts[winner])
                break
    return {
        "slot": slot, "n_carriers": len(c), "variants": firsts.index.tolist(), "overlap_agents": overlap,
        "first_seen": {k: v for k, v in firsts.items()}, "winner": winner,
        "first_variant": firsts.index[0], "corrected": firsts.index[0] != winner,
        "t_winner_first": firsts[winner], "t_winner_majority": t50,
        "hours_to_overtake": ((t50 - firsts[winner]).total_seconds() / 3600) if t50 is not None else np.nan,
        "share_by_bin": frac, "t0": t0, "carriers": c[["ts", "variant"]],
    }


def value_slots(mentions: pd.DataFrame, agent_col: str, min_agents: int = 3,
                min_share: float = 0.1) -> list[dict]:
    m = mentions[mentions["family"] != "url"]
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
        st = _slot_stats(car[["ts", agent_col, "variant"]], f"{fam} | {item}")
        st.update(family=fam, item=item, kind="value", agents_per_variant=strong.to_dict())
        out.append(st)
    return sorted(out, key=lambda s: -s["n_carriers"])


def sequence_slots(claims: pd.DataFrame, agent_col: str, min_agents: int = 2,
                   min_share: float = 0.1, min_overlap: int = 2) -> list[dict]:
    """Competing items for the same round.  Requires >= ``min_overlap`` agents who
    carried both variants (they discussed the conflict); otherwise two variants
    with disjoint carriers are just different task versions, not a dispute."""
    # strict "round marker -> item" reports only: rounds inferred for free-text predictions are too loose
    c = claims[(claims["event_type"] == "sequence") & claims["episode"].notna()
               & claims["item"].notna() & (claims["family"] != "")]
    out = []
    for (fam, ep), g in c.groupby(["family", "episode"]):
        first = g.sort_values("ts").drop_duplicates([agent_col, "item"])
        counts = first.groupby("item")[agent_col].nunique()
        strong = counts[(counts >= min_agents) & (counts >= min_share * counts.sum())]
        if len(strong) < 2:
            continue
        car = first[first["item"].isin(strong.index)].rename(columns={"item": "variant"})
        st = _slot_stats(car[["ts", agent_col, "variant"]], f"{fam} | R{int(ep)}")
        if st["overlap_agents"] < min_overlap:
            continue
        st.update(family=fam, item=f"R{int(ep)}", kind="sequence", agents_per_variant=strong.to_dict())
        out.append(st)
    return sorted(out, key=lambda s: -s["n_carriers"])


def regex_slots(events: pd.DataFrame, agents: pd.DataFrame, disputes: list[dict], agent_col: str) -> list[dict]:
    """Analyst-declared disputes: {"family", "slot", "context": regex, "variants": {name: regex}}.
    An agent carries a variant from its first post (in the family) matching both the
    context and the variant pattern - for claims no generic parser can attribute,
    e.g. 'RNG prep Maryland 52,395' said in a G5 context."""
    amap = agents.set_index("author_raw")[agent_col]
    out = []
    for d in disputes:
        ev = events[events["family"].fillna("") == d["family"]]
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
        st = _slot_stats(car[["ts", agent_col, "variant"]], f"{d['family']} | {d['slot']}")
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


def figure(slots: list[dict], path) -> str | None:
    """Cumulative distinct carriers of each variant over wall time, one panel per slot."""
    if not slots:
        return None
    plotting.setup()
    n = len(slots)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.6), squeeze=False)
    for ax, s in zip(axes[0], slots):
        car = s["carriers"]
        order = [v for v in s["variants"]][:3]
        lo = car["ts"].quantile(0.03)
        for i, v in enumerate(order):
            t = car.loc[car["variant"] == v, "ts"].sort_values()
            t = t[t >= lo]
            if t.empty:
                continue
            ax.step(list(t) + [car["ts"].max()], list(range(1, len(t) + 1)) + [len(t)], where="post",
                    color=plotting.SERIES[i], label=f"{v} ({s['agents_per_variant'].get(v, len(t))} agents)")
        ax.set_title(s["slot"].replace("datausa-", "").replace(" | ", "\n"), fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.xaxis.set_major_formatter(plotting.matplotlib.dates.DateFormatter("%m-%d %H:%M"))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), fontsize=8, ncol=1)
    axes[0][0].set_ylabel("Agents carrying the value (cumulative)")
    fig.suptitle("Competing values: did the correction overtake the error?", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return plotting.save(fig, path)
