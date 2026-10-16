"""Uncertainty: cluster-bootstrap intervals, and Greenwood's interval for a Kaplan-Meier median.

The bootstrap resamples whole clusters with replacement (A1 resamples items: one first poster supplies most of an
item's answers, so answers to one item are not independent) and takes the 2.5th and 97.5th percentiles.  It is
vectorised: a statistic sees every resample at once as a matrix of row weights (how often each row's cluster was
drawn), so a mean is one matrix product.  The intervals cover sampling noise across clusters only, not
identity-mapping or extraction error.
"""

from __future__ import annotations

import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd

LEVEL = 0.95
# fewer clusters than this give no interval: three items have only 10 distinct resamples
MIN_CLUSTERS = 5
# rows x resamples held in memory at once
CHUNK_CELLS = 4_000_000


def cluster_bootstrap(df: pd.DataFrame, cluster_col: str | None, stat_fn, B: int = 1000, seed: int = 0,
                      min_clusters: int = MIN_CLUSTERS) -> tuple[np.ndarray, np.ndarray]:
    """The 95% percentile interval of ``stat_fn(df, w)`` over ``B`` resamples of whole clusters of ``cluster_col``
    (None: every row is its own cluster, the plain bootstrap).  ``w`` is a (resamples, rows) weight matrix and
    ``stat_fn`` returns one value per resample, or a (resamples, m) array for m statistics at once.  Returns
    (lo, hi), NaN with ``B`` = 0 or fewer than ``min_clusters`` clusters."""
    if cluster_col is None:
        k, code = len(df), np.arange(len(df))
    else:
        members = df.groupby(cluster_col, sort=True).indices   # cluster -> its row positions
        k, code = len(members), np.empty(len(df), dtype=np.intp)
        for c, rows in enumerate(members.values()):
            code[rows] = c
    if not B or k < min_clusters:
        shape = np.shape(stat_fn(df, np.ones((1, len(df)))))[1:]
        return np.full(shape, np.nan), np.full(shape, np.nan)
    rng = np.random.default_rng(seed)
    step = max(1, CHUNK_CELLS // max(k, len(df)))
    stats = []
    for start in range(0, B, step):
        b = min(step, B - start)
        draws = rng.integers(0, k, size=(b, k)) + k * np.arange(b)[:, None]
        counts = np.bincount(draws.ravel(), minlength=b * k).reshape(b, k)
        stats.append(np.asarray(stat_fn(df, counts[:, code])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # a statistic undefined in every resample stays NaN
        lo, hi = np.nanpercentile(np.concatenate(stats), [50 * (1 - LEVEL), 50 * (1 + LEVEL)], axis=0)
    return lo, hi


def wmean(x, w: np.ndarray) -> np.ndarray:
    """The mean of ``x`` in each resample (rows of ``w``), NaN rows of ``x`` skipped."""
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    if not ok.all():
        x, w = x[ok], w[:, ok]
    tot = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, w @ x / tot, np.nan)


def wmedian(x, w: np.ndarray) -> np.ndarray:
    """The median of ``x`` in each resample, as pandas takes it: NaN skipped, an even count's two middle values
    averaged (row i counted ``w[:, i]`` times)."""
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    if not ok.any():
        return np.full(len(w), np.nan)
    order = np.argsort(x[ok], kind="stable")
    xs, cum = x[ok][order], np.cumsum(w[:, ok][:, order], axis=1)
    n = cum[:, -1]
    lo = (cum > ((n - 1) // 2)[:, None]).argmax(axis=1)
    hi = (cum > (n // 2)[:, None]).argmax(axis=1)
    return np.where(n > 0, (xs[lo] + xs[hi]) / 2, np.nan)


def km_greenwood(durations, events) -> pd.DataFrame:
    """Kaplan-Meier survival from t = 0, with Greenwood's standard error ``se`` and the log-log 95% band
    ``lo``/``hi`` (survfit's ``conf.type="log-log"``) at each event time.  ``events`` is 1 for an event, 0 for a
    censored duration."""
    d, e = np.asarray(durations, dtype=float), np.asarray(events, dtype=int)
    t = np.unique(d[e == 1])
    at_risk = (d[None, :] >= t[:, None]).sum(axis=1)
    n_ev = ((d[None, :] == t[:, None]) & (e[None, :] == 1)).sum(axis=1)
    s = np.cumprod(1 - n_ev / at_risk)
    z = NormalDist().inv_cdf(0.5 + LEVEL / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.cumsum(n_ev / (at_risk * (at_risk - n_ev)))   # infinite once everyone still at risk has the event
        v = z * np.sqrt(g) / np.abs(np.log(s))
        se = np.where(s > 0, s * np.sqrt(g), 0.0)
        lo, hi = np.where(s > 0, s ** np.exp(v), 0.0), np.where(s > 0, s ** np.exp(-v), 0.0)
    return pd.DataFrame({"t": np.concatenate([[0.0], t]), "survival": np.concatenate([[1.0], s]),
                         "se": np.concatenate([[0.0], se]), "lo": np.concatenate([[1.0], lo]),
                         "hi": np.concatenate([[1.0], hi])})


def km_median(km: pd.DataFrame) -> tuple[float, float, float]:
    """The median time to the event (the first time survival reaches 0.5) and its interval: the first times the
    lower and the upper band reach 0.5.  NaN where a curve never does: no median, or an open upper bound."""
    def first(col: str) -> float:
        hit = km.loc[km[col] <= 0.5, "t"]
        return float(hit.iloc[0]) if len(hit) else float("nan")
    return first("survival"), first("lo"), first("hi")
