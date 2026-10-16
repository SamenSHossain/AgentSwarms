"""``swarmprov inspect``: what the adapters and the chat reader make of an input, before a run.

Prints each adapter's sniff in detection order (with the error a sniff raised).  Unless another adapter claims the
input (an AI Village directory's message table is still read by the chat reader, so it is shown), it then shows what
the chat reader finds: the first record's keys, the key each canonical field resolved to, coverage over a sample (and
the framework roles, when messages carry them), and a ``{"fields": ...}`` snippet for ``--config`` guessed from key
names and values.  It only suggests: nothing is applied or written.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from . import adapters, remote
from .adapters import framework
from .adapters.base import FIELD_NAMES, AdapterConfig, check_fields, config_snippet
from .adapters.chat import FORMATS, GUESS, SAMPLE, _pick, _resolve, _text, _to_ts, guess_fields, read_messages
from .adapters.village import discover


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
        table = discover(Path(path)).get("messages") if picked.name == "village" else None
        if table is None:
            print(f"\nThe `{picked.name}` adapter reads its own schema; field mapping applies to the chat reader only.")
            return 0
        print(f"\nThe `village` adapter reads its message table, {table.name}, with the chat reader (field mapping "
              "applies to it); this is what that reader finds.")
        path = table
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
    recs, n = [m for m, *_ in sample], len(sample)
    print(f"\nfirst record's keys: {', '.join(recs[0])}")
    print(f"\nfields over the first {n:,} of {len(msgs):,} records (canonical <- source key, records holding it):")
    found = {}
    for f in FIELD_NAMES:
        hits = Counter(k for k, v in (_resolve(m, f, fields) for m in recs) if v is not None)
        found[f] = hits
        how = " [mapped]" if f in fields else ""
        keys = ", ".join(f"{k} ({c:,})" for k, c in hits.most_common(3))
        if not hits and f == "channel" and any(c for _, c, *_ in sample):
            keys = "the container (a {room: [...]} key, an export's channel name, a file name)"
        print(f"  {f:<10} <- {keys or '(none)'}{how}")
    n_ts = sum(pd.notna(_to_ts(_pick(m, "ts", fields))) for m in recs)
    who = [_pick(m, "author", fields) or _pick(m, "author_alt", fields) for m in recs]
    roles = [framework.role(m, w) for m, w in zip(recs, who)]
    n_auth = sum(bool(w) or r in ("task", "tool") for w, r in zip(who, roles))   # ingest names these `task` / the caller
    n_text = sum(bool(_text(_pick(m, "text", fields)).strip()) for m in recs)
    channels = {str(_pick(m, "channel", fields) or c or "main") for m, c, *_ in sample}
    print(f"\ncoverage: timestamps read {n_ts / n:.0%}, authors known (not `unknown`) {n_auth / n:.0%}, "
          f"text non-empty {n_text / n:.0%}, {len(channels):,} channel{'s' if len(channels) > 1 else ''}")
    kinds = Counter(roles)
    if set(kinds) != {None}:
        cnt = lambda k, word: f"{kinds[k]:,} {word}" + "s" * (kinds[k] != 1)
        print(f"framework roles: {cnt('agent', 'agent turn')}, {cnt('task', 'task message')} (system or user: author "
              f"`task`), {cnt('tool', 'tool result')} (the calling agent's own reads, not posts)"
              + (f", {cnt('inner', 'inner event')} (AutoGen thoughts, call requests, streaming chunks: no other agent "
                 "receives them, so ingest drops them)" if kinds["inner"] else "")
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
