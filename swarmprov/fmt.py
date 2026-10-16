"""Formatting shared by every writer: strict JSON values and percentages."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd


def json_safe(obj):
    """``obj`` as strict JSON values: NaN, infinities and NaT become null, timestamps ISO text, numpy scalars
    Python numbers, tuples lists."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def dumps(obj, **kw) -> str:
    """``json_safe(obj)`` as strict JSON text (``allow_nan=False``; anything else not JSON as its text)."""
    return json.dumps(json_safe(obj), allow_nan=False, default=str, **kw)


def pct(x) -> str:
    """A share as a whole percentage, "n/a" when missing."""
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"
