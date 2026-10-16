"""A2 - does availability cause copying?  Two-way fixed effects on agent-rounds.

    Y_ir = beta * D_ir + alpha_agent + gamma_item + e,   SEs clustered by agent

Item FE absorb "same question, same answer"; agent FE absorb ability.  D
varies within agent (early rounds rarely exposed) and within item (first
mover vs follower).  Identifying assumption: an agent's position in the run
order (its assigned date) is unrelated to its ability.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import plotting
from ..plotting import plt

OUTCOMES = {
    "y_instant": ("Answered instantly (<= 2 s claimed)", None),
    "y_instant|known": ("Answered instantly, rounds with stated latency", "latency_known"),
    "y_consensus": ("Answer matches the consensus value", None),
    # an extra row, not the headline: the later majority A4 finds, on the slots it calls overturned only
    "y_winner": ("Answer matches the A4 winner (overturned slots only)", None),
}
OPTIONAL = ("y_winner",)   # estimated only where some round has a value (a run with an overturned slot)
# Treatment = consensus value public before the report (independent of the agent's own answer).
TREATMENTS = {"D_cons": "Consensus public before report", "D_cons_w60m": "Consensus public >= 1 h before"}


def _within_variation(df: pd.DataFrame, d: str) -> int:
    return int((df.groupby("agent")[d].nunique() > 1).sum())


MIN_SWITCHERS = 10  # agents whose exposure status varies across their rounds
# how A2 reads a row (``reading``): the table's notes, the key findings and the report.html tile
IDENTIFIED, NO_EFFECT, WEAK, NOT_IDENTIFIED = "identified", "no effect detected", "weakly identified", "not identified"


def reading(k: int, ci_lo, ci_hi, min_switchers: int = MIN_SWITCHERS) -> str:
    """A row with ``k`` agents that switch exposure status: not identified under ``min_switchers`` or with no
    estimate, weakly identified under twice that, else identified where its 95% CI excludes 0 and no effect detected
    where it covers 0."""
    if k < min_switchers or ci_lo is None or ci_lo != ci_lo:
        return NOT_IDENTIFIED
    if k < 2 * min_switchers:
        return WEAK
    return IDENTIFIED if ci_lo > 0 or ci_hi < 0 else NO_EFFECT


def estimate(ex: pd.DataFrame, min_switchers: int = MIN_SWITCHERS, wall_clock: bool = True) -> pd.DataFrame:
    """Without a wall clock (a message-order run) only the consensus outcome under ``D_cons`` is estimated: the
    >= 1 h treatment needs one, and the latency outcomes, though read from the post text, are left out with it."""
    try:
        import pyfixest as pf
    except ImportError:  # pragma: no cover
        pf = None
    rows = []
    outcomes = OUTCOMES if wall_clock else {"y_consensus": OUTCOMES["y_consensus"]}
    treatments = TREATMENTS if wall_clock else {"D_cons": TREATMENTS["D_cons"]}
    for key, (ylab, subset) in outcomes.items():
        y = key.split("|")[0]
        if key in OPTIONAL and (y not in ex or ex[y].isna().all()):
            continue
        for d, dlab in treatments.items():
            df = ex.dropna(subset=[y]).copy()
            if subset:
                df = df[df[subset] == 1]
            df = df[df.groupby("agent")["agent"].transform("size") > 1]
            df = df[df.groupby("item_fe")["item_fe"].transform("size") > 1]
            row = {"outcome": key, "outcome_label": ylab, "treatment": d, "treatment_label": dlab,
                   "n": len(df), "agents": df["agent"].nunique(), "items": df["item_fe"].nunique(),
                   "agents_with_D_variation": _within_variation(df, d),
                   "mean_y_D0": df.loc[df[d] == 0, y].mean(), "mean_y_D1": df.loc[df[d] == 1, y].mean()}
            if row["agents_with_D_variation"] < min_switchers:
                row["note"] = f"not identified: only {row['agents_with_D_variation']} agents switch exposure status"
            elif pf is not None and len(df) > 20 and df[d].nunique() > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        fit = pf.feols(f"{y} ~ {d} | agent + item_fe", data=df, vcov={"CRV1": "agent"})
                        t = fit.tidy().loc[d]
                        row.update(beta=t["Estimate"], se=t["Std. Error"], p=t["Pr(>|t|)"],
                                   ci_lo=t["2.5%"], ci_hi=t["97.5%"])
                        k = row["agents_with_D_variation"]
                        if reading(k, row["ci_lo"], row["ci_hi"], min_switchers) == WEAK:
                            row["note"] = f"weakly identified: only {k} agents switch (< {2 * min_switchers})"
                    except Exception as e:  # singular designs etc.
                        row["error"] = str(e)[:120]
            rows.append(row)
    out = pd.DataFrame(rows)
    for c in ("beta", "se", "p", "ci_lo", "ci_hi", "note"):
        if c not in out:
            out[c] = np.nan
    return out


# Exploratory: dose response among exposed rounds (D_cons == 1).  The dose is how long the consensus value had been
# public before the report (gap_cons_s, measured to the report, itself an upper bound on arrival); within an agent it
# varies across items even where D_cons does not, but it identifies only the intensive margin.  Outcome -> its caveat:
# y_instant scores a round with no stated latency 0, so how often agents state one enters its slope.
DOSE_OUTCOMES = {"y_instant|known": None, "y_instant": "biased: a round with no stated latency scores 0"}
HEAD_BINS = {"<1h": 1, "1-6h": 6, "6-24h": 24, ">24h": np.inf}   # label -> upper edge in hours
DOSE_REF = "<1h"
SLOPE = "log1p(head h)"
MIN_DOSE_SD = 0.25    # pooled within-agent SD of log1p(head start in hours)
MIN_BIN_AGENTS = 10   # agents with rounds in two or more bins
MIN_INFORMATIVE = 10  # per term, agents whose outcome varies and whose head start (slope) or bin (a bin) does too


def _dose_sample(ex: pd.DataFrame, key: str) -> pd.DataFrame:
    y, subset = key.split("|")[0], OUTCOMES[key][1]
    df = ex[(ex["D_cons"] == 1) & ex["gap_cons_s"].notna()].dropna(subset=[y])
    if subset:
        df = df[df[subset] == 1]
    df = df[df.groupby("agent")["agent"].transform("size") > 1]
    df = df[df.groupby("item_fe")["item_fe"].transform("size") > 1]
    h = df["gap_cons_s"] / 3600
    return df.assign(log_head=np.log1p(h), head_bin=pd.cut(h, [0, *HEAD_BINS.values()], labels=list(HEAD_BINS),
                                                           right=False).astype(str))


def _within_sd(df: pd.DataFrame) -> float:
    dev = df["log_head"] - df.groupby("agent")["log_head"].transform("mean")
    dof = len(df) - df["agent"].nunique()
    return float(np.sqrt((dev ** 2).sum() / dof)) if dof > 0 else 0.0


def _fit(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    import pyfixest as pf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pf.feols(f"{formula} | agent + item_fe", data=df, vcov={"CRV1": "agent"}).tidy()


def _leave_one_out(df: pd.DataFrame, y: str) -> dict:
    """The slope on ``log_head`` refitted without each agent in turn (its range), and the share of its clustered
    variance the two agents with the largest scores carry.  Exact and without refitting: agent effects drop out by
    demeaning within agent, which leaving an agent out does not change, and item effects enter as dummies."""
    g = pd.factorize(df["agent"])[0]
    X = np.column_stack([df["log_head"], pd.get_dummies(df["item_fe"])]).astype(float)
    Y = df[y].to_numpy(float)
    n = np.bincount(g)
    X -= np.stack([np.bincount(g, col) for col in X.T], 1)[g] / n[g, None]
    Y -= (np.bincount(g, Y) / n)[g]
    A, c = X.T @ X, X.T @ Y
    inv = np.linalg.pinv(A, hermitian=True)
    e = Y - X @ (inv @ c)
    rows = [np.flatnonzero(g == k) for k in range(len(n))]
    score = np.array([inv[0] @ (X[r].T @ e[r]) for r in rows]) ** 2
    loo = [(np.linalg.pinv(A - X[r].T @ X[r], hermitian=True) @ (c - X[r].T @ Y[r]))[0] for r in rows]
    return {"loo_lo": min(loo), "loo_hi": max(loo), "var_share_top2": np.sort(score)[-2:].sum() / score.sum()}


def estimate_dose(ex: pd.DataFrame, min_sd: float = MIN_DOSE_SD, min_bin_agents: int = MIN_BIN_AGENTS,
                  min_informative: int = MIN_INFORMATIVE) -> pd.DataFrame:
    """Per outcome of ``DOSE_OUTCOMES``, on the exposed rounds: the slope on log1p(head start h) and each ``HEAD_BINS``
    coefficient against ``DOSE_REF``, agent + item FE, SEs clustered by agent.  One row per (outcome, term), with the
    term's rounds, mean outcome, ``agents_spanning`` (for the slope, agents whose head start varies; for a bin, agents
    with a round in it and in another bin) and ``agents_informative`` (those of them whose outcome varies too).  An
    outcome under ``min_sd`` or ``min_bin_agents`` is not identified; so is a term with fewer than ``min_informative``
    informative agents, and every bin when the reference is.  An estimate informed by fewer than twice that (a bin: in
    it or in the reference) is noted weak.  The slope carries its leave-one-agent-out range (``loo_lo``, ``loo_hi``) and
    the share of its clustered variance its two largest agents carry (``var_share_top2``)."""
    rows, weak = [], 2 * min_informative
    for key in DOSE_OUTCOMES:
        y = key.split("|")[0]
        df = _dose_sample(ex, key)
        varies = df.groupby("agent")[y].nunique() > 1
        moves = df.groupby("agent")["log_head"].nunique() > 1
        multi = df[df.groupby("agent")["head_bin"].transform("nunique") >= 2]
        base = {"outcome": key, "outcome_label": OUTCOMES[key][0], "n": len(df), "agents": df["agent"].nunique(),
                "items": df["item_fe"].nunique(), "within_sd": _within_sd(df), "bin_agents": multi["agent"].nunique()}
        terms = {SLOPE: {"n_term": len(df), "mean_y": df[y].mean(), "agents_spanning": int(moves.sum()),
                         "agents_informative": int((moves & varies).sum())}}
        for b in HEAD_BINS:
            span = multi.loc[multi["head_bin"] == b, "agent"].unique()
            terms[b] = {"n_term": int((df["head_bin"] == b).sum()), "mean_y": df.loc[df["head_bin"] == b, y].mean(),
                        "agents_spanning": len(span), "agents_informative": int(varies.reindex(span).sum())}
        note = (f"not identified: only {len(df)} exposed rounds after dropping single-round agents and items"
                if len(df) <= 20 else
                f"not identified: within-agent SD of the log head start is {base['within_sd']:.2f} (< {min_sd})"
                if base["within_sd"] < min_sd else
                f"not identified: only {base['bin_agents']} agents have rounds in two or more head-start bins"
                if base["bin_agents"] < min_bin_agents else None)
        if note:
            for v in terms.values():
                v["note"] = note
        else:
            for term, v in terms.items():
                s, k = v["agents_spanning"], v["agents_informative"]
                if k < min_informative:
                    vary = "varies" if k == 1 else "vary"
                    v["note"] = (f"not identified: only {k} of the {s} agents whose head start varies also {vary} in "
                                 "the outcome" if term == SLOPE else
                                 f"not identified: only {s} agents have a round here and in another bin"
                                 if s < min_informative else
                                 f"not identified: only {k} of the {s} agents with a round here and in another bin "
                                 f"{vary} in the outcome")
            if "note" in terms[DOSE_REF]:
                for b in HEAD_BINS:
                    terms[b].setdefault("note", f"not identified: the {DOSE_REF} reference bin is not identified")
            fits = [(f"{y} ~ log_head", {"log_head": SLOPE}),
                    (f"{y} ~ i(head_bin, ref='{DOSE_REF}')", {f"head_bin::{b}": b for b in HEAD_BINS if b != DOSE_REF})]
            for formula, names in fits:
                names = {n: term for n, term in names.items() if "note" not in terms[term]}
                if not names:
                    continue
                try:
                    t = _fit(df, formula)
                except Exception as e:  # singular designs etc.
                    for term in names.values():
                        terms[term]["note"] = f"not estimated: {str(e)[:120]}"
                    continue
                for name, term in names.items():
                    if name not in t.index:   # pyfixest drops a term the fixed effects absorb
                        terms[term]["note"] = "not estimated: collinear with the agent and item fixed effects"
                    elif not t.loc[name, "Std. Error"] > 1e-6:   # within pyfixest's demeaning tolerance: an exact fit
                        terms[term]["note"] = "not estimated: the model fits the outcome exactly"
                    else:
                        r = t.loc[name]
                        terms[term].update(beta=r["Estimate"], se=r["Std. Error"], p=r["Pr(>|t|)"], ci_lo=r["2.5%"],
                                           ci_hi=r["97.5%"])
            if "beta" in terms[SLOPE]:
                terms[SLOPE] |= _leave_one_out(df, y)
            k_ref = terms[DOSE_REF]["agents_informative"]
            for term, v in terms.items():
                k = v["agents_informative"]
                if "beta" in v and term == SLOPE and k < weak:
                    v["note"] = f"weak: only {k} agents inform it (< {weak})"
                elif "beta" in v and term != SLOPE and min(k, k_ref) < weak:
                    v["note"] = (f"shape only: {k} agents inform it (< {weak})" if k <= k_ref else
                                 f"shape only: {k_ref} agents inform the {DOSE_REF} reference (< {weak})")
        rows += [base | {"term": term} | v for term, v in terms.items()]
    out = pd.DataFrame(rows)
    for c in ("beta", "se", "p", "ci_lo", "ci_hi", "loo_lo", "loo_hi", "var_share_top2", "note"):
        if c not in out:
            out[c] = np.nan
    return out


def stated_latency_slope(ex: pd.DataFrame) -> dict:
    """Why ``y_instant``'s dose slope is biased: on its sample, how the share of rounds that state a latency at all moves
    with log1p(head start h), agent + item FE.  Empty when every round states one (the two outcomes then agree)."""
    df = _dose_sample(ex, "y_instant")
    if len(df) <= 20 or df["latency_known"].nunique() < 2:
        return {}
    try:
        t = _fit(df, "latency_known ~ log_head").loc["log_head"]
    except Exception:  # singular designs etc.
        return {}
    return {"beta": float(t["Estimate"]), "se": float(t["Std. Error"]), "p": float(t["Pr(>|t|)"]), "n": len(df)}


def figure(est: pd.DataFrame, path, min_switchers: int = MIN_SWITCHERS) -> str:
    """Every row of ``est``: the estimate and its 95% CI, or 'not identified' at zero with no bar.  Each label gives the
    agents whose exposure status switches, the only ones that identify beta; an estimate from fewer than twice
    ``min_switchers`` is grey."""
    plotting.setup()
    fig, ax = plt.subplots(figsize=(8, 1.5 + 0.55 * len(est)))
    for i, r in enumerate(est.itertuples()):
        k = int(r.agents_with_D_variation)
        c = (plotting.NEUTRAL if k < 2 * min_switchers else
             plotting.SERIES[0] if r.treatment == "D_cons" else plotting.SERIES[1])
        if pd.notna(r.beta):
            ax.errorbar(r.beta, i, xerr=[[r.beta - r.ci_lo], [r.ci_hi - r.beta]], fmt="o", color=c,
                        capsize=3, lw=2, markersize=7)
            ax.text(r.ci_hi + 0.01, i, f"{r.beta:+.2f}  (n={r.n})", va="center", fontsize=8, color=plotting.INK_2)
        else:
            ax.text(0, i, f"not identified ({k} agents switch)" if pd.notna(r.note) else "not estimated",
                    ha="center", va="center", fontsize=8, style="italic", color=plotting.INK_2,
                    bbox={"boxstyle": "round,pad=0.2", "fc": plotting.SURFACE, "ec": "none"})
    ax.axvline(0, color=plotting.NEUTRAL, lw=1)
    if est["beta"].isna().all():
        ax.set_xlim(-1, 1)
    ax.set_yticks(range(len(est)))
    ax.set_yticklabels([f"{r.outcome_label}\n[{r.treatment_label}], {int(r.agents_with_D_variation)} switchers"
                        for r in est.itertuples()], fontsize=8)
    ax.set_ylim(len(est) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Effect of exposure on the outcome (share, -1..1); agent + item FE; 95% CI clustered by agent")
    ax.set_title(f"Only agents whose exposure status switches identify an effect: under {min_switchers} switchers "
                 f"a row is not identified,\nunder {2 * min_switchers} its estimate is grey (weakly identified)",
                 loc="left", fontsize=9, fontweight="normal", color=plotting.INK_2)
    fig.suptitle("Does availability cause copying?", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return plotting.save(fig, path)


def dose_figure(dose: pd.DataFrame, path, min_informative: int = MIN_INFORMATIVE) -> str:
    """One panel per outcome of ``dose``: each head-start bin's effect against ``DOSE_REF`` with its 95% CI (the
    reference hollow at zero), 'not identified' at zero with no bar where a bin is.  As in ``figure``, a weak estimate
    (one ``estimate_dose`` gives a note: fewer than twice ``min_informative`` agents inform it or the reference) is
    grey; each panel's title gives the slope and its leave-one-agent-out range, each bin's label its rounds and
    informative agents."""
    plotting.setup()
    outs = list(dict.fromkeys(dose["outcome"]))
    fig, axes = plt.subplots(1, len(outs), figsize=(4.8 * len(outs), 4.9), sharey=True, squeeze=False)
    for k, (ax, key) in enumerate(zip(axes[0], outs)):
        d = dose[dose["outcome"] == key].set_index("term")
        for i, b in enumerate(HEAD_BINS):
            r = d.loc[b]
            if b == DOSE_REF and pd.isna(r["note"]):
                ax.plot(i, 0, "o", mfc=plotting.SURFACE, mec=plotting.SERIES[k], mew=2, markersize=7)
                ax.annotate("reference", (i, 0), xytext=(0, -10), textcoords="offset points", ha="center", va="top",
                            fontsize=8, color=plotting.INK_2)
            elif pd.notna(r["beta"]):
                c = plotting.NEUTRAL if pd.notna(r["note"]) else plotting.SERIES[k]
                ax.errorbar(i, r["beta"], yerr=[[r["beta"] - r["ci_lo"]], [r["ci_hi"] - r["beta"]]], fmt="o", color=c,
                            capsize=3, lw=2, markersize=7)
                ax.annotate(f"{r['beta']:+.2f}", (i, r["beta"]), xytext=(8, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=8, color=plotting.INK_2)
            else:
                ax.text(i, 0, "not\nidentified", ha="center", va="center", fontsize=8, style="italic",
                        color=plotting.INK_2, bbox={"boxstyle": "round,pad=0.2", "fc": plotting.SURFACE, "ec": "none"})
        s = d.loc[SLOPE]
        slope = (f"slope {s['beta']:+.3f} per log1p(h), 95% CI {s['ci_lo']:+.3f} to {s['ci_hi']:+.3f}\n"
                 f"leaving one agent out: {s['loo_lo']:+.3f} to {s['loo_hi']:+.3f}"
                 if pd.notna(s["beta"]) else "slope not identified")
        label = OUTCOMES[key][0][0].lower() + OUTCOMES[key][0][1:]
        ax.set_title(f"{key}: {label}\n{slope}" + (f"\n({DOSE_OUTCOMES[key]})" if DOSE_OUTCOMES.get(key) else ""),
                     loc="left", fontsize=9, fontweight="normal", color=plotting.INK)
        ax.axhline(0, color=plotting.NEUTRAL, lw=1)
        ax.set_xticks(range(len(HEAD_BINS)))
        ax.set_xticklabels([f"{b}\n{int(d.loc[b, 'n_term'])} rounds\n{int(k)} agent{'s' * (k != 1)}"
                            for b, k in d.loc[list(HEAD_BINS), "agents_informative"].items()], fontsize=8)
        ax.set_xlim(-0.6, len(HEAD_BINS) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("How long the consensus had been public before the report")
    if dose["beta"].isna().all():
        axes[0][0].set_ylim(-1, 1)
    axes[0][0].set_ylabel(f"Effect on the outcome vs {DOSE_REF} (share)")
    fig.suptitle("Exploratory: dose response among exposed rounds", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.91, "Given the answer was public, does reuse grow with how long it had been? Agent + item FE, "
             "95% CI clustered by agent. The agents under each bin\nhave a round in it and in another bin and an "
             f"outcome that varies: under {min_informative} the bin is not identified, and its estimate is grey when "
             f"it or the reference has\nunder {2 * min_informative}. Whether exposure itself matters is not estimated "
             "here.", ha="left", va="top", fontsize=9, color=plotting.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    return plotting.save(fig, path)
