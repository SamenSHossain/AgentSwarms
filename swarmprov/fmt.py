"""Formatting shared by every writer: strict JSON values, percentages, console-safe text and progress lines."""

from __future__ import annotations

import json
import math
import re
import time
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


# what a terminal acts on: C0 controls but tab and newline (escape sequences, a bell), DEL and C1, and the bidi
# overrides and isolates that reorder what follows them
CONSOLE_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def console(s: str) -> str:
    """``s`` safe to print: each control or bidi-override character as its escape (``\\x1b``, ``\\u202e``)."""
    return CONSOLE_RE.sub(lambda m: f"\\x{ord(m[0]):02x}" if ord(m[0]) < 0x100 else f"\\u{ord(m[0]):04x}", s)


def log(msg: str) -> None:
    """A stage's progress line, timestamped."""
    print(f"[swarmprov {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def say(*parts) -> None:
    """``print`` for lines that carry agent-written names, channels or keys (``console``)."""
    print(*(console(str(p)) for p in parts))
