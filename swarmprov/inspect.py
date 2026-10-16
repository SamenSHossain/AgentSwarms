"""``swarmprov inspect``: what the adapters and the chat reader make of an input, before a run.

Prints each adapter's sniff in detection order (with the error a sniff raised).  Unless another adapter claims the
input, it then shows what the chat reader finds: the first record's keys, the key each canonical field resolved to,
coverage over a sample (and the framework roles, when messages carry them), and a ``{"fields": ...}`` snippet for
``--config`` guessed from key names and values.  It only suggests: nothing is applied or written.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from . import adapters, remote
from .adapters import framework
from .adapters.base import FIELD_NAMES, AdapterConfig, check_fields, config_snippet
from .adapters.chat import FORMATS, URLISH, _pick, _resolve, _text, _to_ts, read_messages

SAMPLE = 500   # records the coverage and the guesses look at
# the fields every analysis needs -> words their source key tends to contain
GUESS = {"ts": ("time", "date"), "author": ("name", "agent", "user", "speaker", "sender", "handle"),
         "text": ("text", "content", "output", "message", "body")}


def _fits(field: str, values: list) -> bool:
    """Most sampled values look like the field: a readable time, a short name, text."""
    vals = [v for v in values if v not in (None, "")]
    if len(vals) < len(values) / 2:
        return False
    if field == "ts":
        return sum(pd.notna(_to_ts(v)) for v in vals) >= len(vals) / 2
    if field == "author":
        return all(isinstance(v, str) and len(v) <= 100 and not URLISH.search(v) for v in vals)
    return all(isinstance(v, (str, list)) for v in vals)


def guess_fields(records: list[dict], fields: list[str], taken: set) -> dict[str, str]:
    """A source key for each of ``fields`` among the first record's keys, by name and by its sampled values."""
    out = {}
    for f in fields:
        k = next((k for k in records[0] if k not in taken and any(w in k.lower() for w in GUESS[f])
                  and _fits(f, [m.get(k) for m in records])), None)
        if k:
            out[f] = k
            taken.add(k)
    return out


def inspect(path, config: str | None = None, fields: dict | None = None) -> int:
    fields = check_fields({**((AdapterConfig.read_file(config).get("fields") or {}) if config else {}), **(fields or {})})
    path = remote.resolve(path)
    print(f"input: {path}" + (f"\nfields given: {', '.join(f'{k}={v}' for k, v in fields.items())}" if fields else ""))
    print("\nadapters, in detection order:")
    picked = None
    for a, r in adapters.sniffs(path, fields):
        first = r is True and picked is None
        picked = a if first else picked
        status = "yes" if r is True else "no" if r is False else f"error: {type(r).__name__}: {r}"
        print(f"  {a.name:<12} {status}" + ("   <- auto-detection picks this one" if first else ""))
    if picked is not None and picked.name != "chat":
        print(f"\nThe `{picked.name}` adapter reads its own schema; field mapping applies to the chat adapter only.")
        return 0
    if picked is None:
        print("\nNo adapter recognises it; this is what the chat reader finds.")
    try:
        msgs, _ = read_messages(Path(path))
    except Exception as e:
        print(f"\nThe chat reader cannot read it: {e}")
        return 1
    if not msgs:
        print(f"\nNo message records found; the chat adapter reads {FORMATS}.")
        return 1
    sample = msgs[:SAMPLE]
    recs, n = [m for m, _, _ in sample], len(sample)
    print(f"\nfirst record's keys: {', '.join(recs[0])}")
    print(f"\nfields over the first {n:,} of {len(msgs):,} records (canonical <- source key, records holding it):")
    found = {}
    for f in FIELD_NAMES:
        hits = Counter(k for k, v in (_resolve(m, f, fields) for m in recs) if v is not None)
        found[f] = hits
        how = " [mapped]" if f in fields else ""
        keys = ", ".join(f"{k} ({c:,})" for k, c in hits.most_common(3))
        if not hits and f == "channel" and any(c for _, c, _ in sample):
            keys = "the container (a {room: [...]} key, an export's channel name, a file name)"
        print(f"  {f:<10} <- {keys or '(none)'}{how}")
    n_ts = sum(pd.notna(_to_ts(_pick(m, "ts", fields))) for m in recs)
    who = [_pick(m, "author", fields) or _pick(m, "author_alt", fields) for m in recs]
    roles = [framework.role(m, w) for m, w in zip(recs, who)]
    n_auth = sum(bool(w) or r in ("task", "tool") for w, r in zip(who, roles))   # ingest names these `task` / the caller
    n_text = sum(bool(_text(_pick(m, "text", fields)).strip()) for m in recs)
    channels = {str(_pick(m, "channel", fields) or c or "main") for m, c, _ in sample}
    print(f"\ncoverage: timestamps read {n_ts / n:.0%}, authors known (not `unknown`) {n_auth / n:.0%}, "
          f"text non-empty {n_text / n:.0%}, {len(channels):,} channel{'s' if len(channels) > 1 else ''}")
    kinds = Counter(roles)
    if set(kinds) != {None}:
        print(f"framework roles: {kinds['agent']:,} agent turns, {kinds['task']:,} task messages (system or user: author "
              f"`task`), {kinds['tool']:,} tool results (the calling agent's own reads, not posts)"
              + (f", {kinds[None]:,} with no role (ordinary posts)" if kinds[None] else ""))
    weak = [f for f in GUESS if f not in fields and sum(found[f].values()) < n / 2]
    taken = set(fields.values()) | {k for h in found.values() for k in h}
    guessed = guess_fields(recs, weak, taken)
    if [f for f in weak if f not in guessed]:
        print(f"\nNo key looks like the {' or the '.join(f for f in weak if f not in guessed)}.")
    if fields or guessed:
        print("\nsuggested --config (guessed from key names and values: check it before use; nothing is applied):")
        print(config_snippet(fields={**fields, **guessed}))
    elif not weak:
        print("\nNothing to suggest: ts, author and text resolve through the built-in aliases.")
    return 0
