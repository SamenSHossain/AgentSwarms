"""The shared helpers every writer and command uses: strict JSON and percentages (``fmt``), the explorer's script
JSON and the CLI's range validators."""

import argparse
import json
import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from swarmprov import cli, fmt, html


def _strict(text: str):
    return json.loads(text, parse_constant=lambda c: pytest.fail(f"not strict JSON: {c}"))


def test_dumps_writes_strict_json():
    obj = {"nan": float("nan"), "inf": np.float64("inf"), "-inf": -math.inf, "nat": pd.NaT, "none": None,
           "ts": pd.Timestamp("2026-06-16 08:00", tz="UTC"), "dt": datetime(2026, 6, 16, tzinfo=timezone.utc),
           "i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True), "t": (1, (np.int8(2),)), "s": {1, 2} - {1}}
    assert _strict(fmt.dumps(obj)) == {"nan": None, "inf": None, "-inf": None, "nat": None, "none": None,
                                       "ts": "2026-06-16T08:00:00+00:00", "dt": "2026-06-16T00:00:00+00:00",
                                       "i": 3, "f": 0.5, "b": True, "t": [1, [2]], "s": "{2}"}   # the rest as text
    assert fmt.json_safe({"x": [np.float64("nan")]}) == {"x": [None]}


def test_pct_reads_missing_as_na():
    assert (fmt.pct(0.25), fmt.pct(None), fmt.pct(float("nan")), fmt.pct(1)) == ("25%", "n/a", "n/a", "100%")


def test_the_explorer_data_nulls_infinities_instead_of_failing():
    assert html._script_json({"gap": [math.inf, 1], "who": "</script>"}) == '{"gap":[null,1],"who":"<\\/script>"}'


@pytest.mark.parametrize("parse, ok, bad, expected", [
    (cli._hours, ("0.5", "48"), ("0", "-1", "inf", "nan", "x"), "a positive, finite number of hours"),
    (cli._share, ("0", "1", "0.25"), ("-0.1", "1.5", "nan", "x"), "a share from 0 to 1"),
    (cli._tolerance, ("0", "1e-4"), ("1", "-1e-6", "nan", "x"), "a relative tolerance from 0 up to (not including) 1"),
])
def test_range_validators_take_their_range_and_name_it(parse, ok, bad, expected):
    assert [parse(v) for v in ok] == [float(v) for v in ok]
    for v in bad:
        with pytest.raises(argparse.ArgumentTypeError, match=re.escape(f"expected {expected}, got '{v}'")):
            parse(v)
