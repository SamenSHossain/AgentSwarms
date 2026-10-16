"""Generic chat-transcript adapter (AI Village, Slack/Discord exports, framework logs, spreadsheets).

Accepts JSON (a list of messages, ``{"messages": [...]}``, or ``{room: [...]}``),
JSONL or CSV/TSV with a header row, each optionally gzipped, and two kinds of
directory: a Slack export (``users.json`` and ``channels.json`` beside one folder
of day files per channel) and any other folder of JSON/JSONL files, joined with
each file's name as the channel of messages that name none.  Field names are
guessed from common aliases (``ALIASES``); map any of them with ``fields`` in the
config or ``--field author=source`` (a dotted key such as ``author.name`` reaches
into nested objects), and ``swarmprov inspect`` shows what each resolved to.
Each message is one post visible to everyone in its channel from its timestamp.

Framework logs (OpenAI message lists, AutoGen, LangGraph state) often carry no
timestamps.  When fewer than 90% of the messages have one and the file is a
single message list, the whole file switches to a *sequence clock*: message i is
posted at ``SEQUENCE_EPOCH + i`` seconds, and the real timestamps are discarded
rather than mixed in.  A ``{room: [...]}`` file with two rooms or more, and any
directory, concatenates lists whose relative order is unknown, so it keeps its
missing timestamps and the time-based analyses skip those posts.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from decimal import Decimal
from pathlib import Path

import pandas as pd

from ..textutil import stable_id
from .base import Adapter, AdapterConfig, Bundle, Capabilities

ALIASES = {
    "ts": ["timestamp", "time", "created_at", "createdAt", "ts", "date", "datetime", "sent_at"],
    # "source" (AutoGen's speaker) comes last: elsewhere it often names a data source
    "author": ["author", "sender", "agent", "agent_name", "name", "user", "from", "speaker", "username", "source"],
    "author_alt": ["agent_id", "author_id", "sender_id", "user_id", "model"],
    "text": ["content", "text", "message", "body", "msg"],
    "channel": ["channel", "room", "thread", "conversation", "chat", "chat_id", "room_id", "chat_room_id", "channel_id"],
    "id": ["id", "message_id", "uuid", "event_id"],
    "parent": ["parent_id", "reply_to", "in_reply_to", "thread_ts", "reference.messageId"],   # the last is Discord's
}
URLISH = re.compile(r"^[a-z][a-z0-9+.-]*://|^www\.|/", re.I)   # a `source` naming a URL or path is not a speaker
FORMATS = ("a JSON array of messages, {\"messages\": [...]}, {room: [...]}, JSONL (one message per line) or CSV/TSV "
           "with a header row, each optionally gzipped; or a directory: a Slack export, or JSON/JSONL files")

WRAPPER_KEYS = ("messages", "events", "transcript", "items", "data")
TABULAR = (".csv", ".tsv", ".txt")
DIR_FILES = (".json", ".jsonl", ".json.gz", ".jsonl.gz")
EPOCH_RE = re.compile(r"^\d{9,13}(\.\d+)?$")   # Slack's "1777885200.000100"; shorter digit runs are ids, not times
SEQUENCE_EPOCH = pd.Timestamp("2000-01-01", tz="UTC")
EPOCH_RANGE = (pd.Timestamp("1990-01-01", tz="UTC"), pd.Timestamp("2100-01-01", tz="UTC"))   # a time outside it is no time
MIN_TIMED = 0.9   # share of messages that must carry a timestamp for the file to have a wall clock
WARN_MIN_POSTS = 10   # one author or one channel is worth a warning above this many posts
WARN_NO_AUTHOR = 0.5  # share of posts naming no author above which the warning names the record's keys
WARN_UNTIMED = 0.1    # share of posts without a readable timestamp above which the run warns
SLACK_MARKUP = re.compile(r"<([^<>|]+)(?:\|([^<>]*))?>")


def _is_tabular(path: Path) -> bool:
    name = path.name[:-3] if path.suffix == ".gz" else path.name
    return name.endswith(TABULAR)


def _read_any(path: Path):
    try:
        if _is_tabular(path):
            # a spreadsheet export: comma or tab separated, delimiter sniffed; everything read as text
            return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False).to_dict("records")
        return _read_json(path)
    except (ValueError, csv.Error, UnicodeDecodeError) as e:   # JSONDecodeError and pandas' parser errors are ValueErrors
        raise ValueError(f"{path.name} cannot be read ({e}); the chat adapter reads {FORMATS}") from None


def _read_json(path: Path):
    gz = path.suffix == ".gz"
    inner = path.name[:-3] if gz else path.name
    with (gzip.open if gz else open)(path, "rt", encoding="utf-8") as fh:
        if inner.endswith(".jsonl"):
            # line-delimited first, so a one-record file is still one message ...
            try:
                rows = [json.loads(line) for line in fh if line.strip()]
            except json.JSONDecodeError:
                rows = None
            if rows is not None and not (len(rows) == 1 and _is_document(rows[0])):
                return rows
            fh.seek(0)  # ... but a JSON document saved under a .jsonl name still loads
            return json.load(fh)
        head = fh.read(1)
        fh.seek(0)
        if head in "[{":
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                fh.seek(0)
        return [json.loads(line) for line in fh if line.strip()]


def _is_document(obj) -> bool:
    """A whole transcript on one JSONL line: an array, or a wrapper object around one."""
    return isinstance(obj, list) or (isinstance(obj, dict) and any(isinstance(obj.get(k), list) for k in WRAPPER_KEYS))


def _n_lists(obj) -> int:
    """How many message lists ``_flatten`` joins; with one, file position is an order."""
    if isinstance(obj, list):
        return int(any(isinstance(m, dict) for m in obj))
    if not isinstance(obj, dict):
        return 0
    if any(isinstance(obj.get(k), list) for k in WRAPPER_KEYS):
        return _n_lists(next(obj[k] for k in WRAPPER_KEYS if isinstance(obj.get(k), list)))
    return sum(_n_lists(v) for v in obj.values())


def _flatten(obj, channel: str = "") -> list[tuple[dict, str]]:
    """(message, the channel its container names): a ``{room: [...]}`` key, a wrapper's ``channel`` or
    ``channel.name`` (a Discord export, whose guild and channel metadata stay out of the messages), a file name."""
    if isinstance(obj, list):
        return [(m, channel) for m in obj if isinstance(m, dict)]
    if isinstance(obj, dict):
        for key in WRAPPER_KEYS:
            if isinstance(obj.get(key), list):
                ch = obj.get("channel")
                ch = ch.get("name") if isinstance(ch, dict) else ch
                return _flatten(obj[key], ch if isinstance(ch, str) and ch else channel)
        out = []
        for k, v in obj.items():
            if isinstance(v, (list, dict)):
                out += _flatten(v, k)
        return out
    return []


def _get(m: dict, key: str):
    """``m[key]``, or a dotted path into nested objects (``author.name``, ``reference.messageId``)."""
    if key in m or "." not in key:
        return m.get(key)
    v = m
    for part in key.split("."):
        v = v.get(part) if isinstance(v, dict) else None
    return v


def _value(v):
    if isinstance(v, dict):  # e.g. {"author": {"name": ...}}
        v = v.get("name") or v.get("id") or json.dumps(v)
    return None if v is None or v == "" else v


def _resolve(m: dict, field: str, mapping: dict) -> tuple[str | None, object]:
    """(source key, value) of a canonical field: the mapped key, else the first alias that holds a value."""
    if field in mapping:
        return mapping[field], _value(_get(m, mapping[field]))
    for k in ALIASES[field]:
        v = _value(_get(m, k))
        if v is not None and not (k == "source" and URLISH.search(str(v))):
            return k, v
    return None, None


def _pick(m: dict, field: str, mapping: dict) -> object:
    return _resolve(m, field, mapping)[1]


def _to_ts(v) -> pd.Timestamp:
    """ISO text, an epoch number or an epoch string (seconds, or milliseconds above 1e11).  A result outside
    ``EPOCH_RANGE`` (a mis-scaled number, a 1970 placeholder) is NaT."""
    try:
        if isinstance(v, str) and EPOCH_RE.match(v.strip()):
            d = Decimal(v.strip())   # exact: a float would turn Slack's .000100 into .000099897
            t = pd.to_datetime(int(d * (10**3 if d > 10**11 else 10**6)), unit="us", utc=True, errors="coerce")
        elif isinstance(v, (int, float)):
            t = pd.to_datetime(v, unit="ms" if v > 1e11 else "s", utc=True, errors="coerce")
        else:
            t = pd.to_datetime(v, utc=True, errors="coerce") if v is not None else pd.NaT
    except (OverflowError, ValueError):   # a number past any datetime
        return pd.NaT
    return t if isinstance(t, pd.Timestamp) and EPOCH_RANGE[0] <= t < EPOCH_RANGE[1] else pd.NaT


def _site(path: Path) -> str:
    name = path.name
    while name.endswith((".gz", ".json", ".jsonl", *TABULAR)):
        name = name.rsplit(".", 1)[0]
    return name


def _text(v) -> str:
    if isinstance(v, list):  # content blocks
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in v)
    return "" if v is None else str(v)


def _is_slack(path: Path) -> bool:
    return path.is_dir() and (path / "users.json").is_file() and (path / "channels.json").is_file()


def _slack_text(text: str, users: dict) -> str:
    """Slack markup as plain text: ``<url|label>`` -> url, ``<@U..>`` -> @name, ``<#C..|general>`` -> #general."""
    def sub(m):
        target, label = m.group(1), m.group(2)
        if target.startswith("@"):
            return "@" + users.get(target[1:], label or target[1:])
        if target.startswith("#"):
            return "#" + (label or target[1:])
        if target.startswith("!"):
            return label or "@" + target[1:]
        return target   # a link: the URL is the fact, the label only its display text
    return SLACK_MARKUP.sub(sub, text)


def _slack(path: Path) -> list[tuple[dict, str, str]]:
    """A Slack export's messages in ``ts`` order, channel = folder.  User ids become ``users.json`` names, and a
    message's id is its ``ts``, so a reply's ``thread_ts`` names its thread's first message."""
    users = {u["id"]: u.get("real_name") or u.get("name") or u["id"]
             for u in _read_any(path / "users.json") if isinstance(u, dict) and u.get("id")}
    out = []
    for f in sorted(path.glob("*/*.json")):
        for j, (m, _) in enumerate(_flatten(_read_any(f))):
            uid, ts, thread = m.get("user") or m.get("bot_id") or "", m.get("ts"), m.get("thread_ts")
            out.append(({"ts": ts, "author": users.get(uid) or m.get("username") or uid, "author_id": uid,
                         "text": _slack_text(m.get("text") or "", users), "id": ts,
                         "parent_id": thread if thread != ts else None}, f.parent.name, f"{f.parent.name}/{f.name}:{j}"))
    return sorted(out, key=lambda r: Decimal(r[0]["ts"]) if EPOCH_RE.match(str(r[0]["ts"])) else Decimal("Infinity"))


def _dir_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.name.endswith(DIR_FILES))


def read_messages(path: Path) -> tuple[list[tuple[dict, str, str]], bool]:
    """(message, container channel, source ref) in reading order, and whether they form one message list (a
    single list's file position is an order; a directory's files have none between them)."""
    path = Path(path)
    if _is_slack(path):
        return _slack(path), False
    if path.is_dir():
        return [(m, ch, f"{f.name}:{j}") for f in _dir_files(path)
                for j, (m, ch) in enumerate(_flatten(_read_any(f), _site(f)))], False
    doc = _read_any(path)
    return [(m, ch, f"{path.name}:{i}") for i, (m, ch) in enumerate(_flatten(doc))], _n_lists(doc) == 1


def _keys(keys) -> str:
    return ", ".join(f"`{k}`" for k in keys)


def _warnings(ev: pd.DataFrame, n_no_author: int, n_ts_values: int, bad_ts, keys: list) -> list[str]:
    """Signs that a field was not found: posts with no author, one author or one channel for all, unread times."""
    n, out = len(ev), []
    if n_no_author:
        many = n_no_author > 1
        out.append(f"{n_no_author:,} post{'s name' if many else ' names'} no author (no author field was recognised), so "
                   f"{'they are' if many else 'it is'} pooled as one agent, `unknown`, which can look like a hub or a relay "
                   "source in A5." + (f" The first record's keys are {_keys(keys)}; name the author's key with "
                                      "`--field author=<key>`." if n_no_author > WARN_NO_AUTHOR * n else ""))
    elif n > WARN_MIN_POSTS and ev["author_raw"].nunique() == 1:
        out.append(f"All {n:,} posts have one author, `{ev['author_raw'].iloc[0]}`, so no answer can be exposed to another "
                   f"agent's post and A5 has no structure to find. The first record's keys are {_keys(keys)}; if another "
                   "one names the speaker, use `--field author=<key>`.")
    n_bad, n_missing = n_ts_values - int(ev["ts"].notna().sum()), n - n_ts_values
    if n_bad + n_missing > WARN_UNTIMED * n:
        parts = [f"{n_missing:,} of {n:,} posts have no timestamp field (the first record's keys are {_keys(keys)}); if "
                 "one holds the time, map it with `--field ts=<key>`."] if n_missing else []
        if n_bad:
            parts.append(f"{n_bad:,} of {n:,} timestamps could not be read as a time (ISO text, or epoch seconds or "
                         f"milliseconds from 1990 to 2100), e.g. `{bad_ts}`.")
        out.append(" ".join(parts))
    if n > WARN_MIN_POSTS and ev["channel"].nunique() == 1:
        out.append(f"All {n:,} posts are in one channel, `{ev['channel'].iloc[0]}`, so each counts as visible to every "
                   "agent and every relay hop as same-channel. If the source has rooms or threads, name their key with "
                   "`--field channel=<key>`.")
    return out


class ChatAdapter(Adapter):
    name = "chat"

    def __init__(self, mapping: dict | None = None):
        self.mapping = mapping or {}

    def sniff(self, path: Path) -> bool:
        path = Path(path)
        if _is_slack(path):
            return True
        if path.is_dir():
            msgs = next((m for m in (_flatten(_read_any(f)) for f in _dir_files(path)) if m), [])
        elif path.is_file() and (path.name.endswith((".json", ".jsonl", ".gz")) or _is_tabular(path)):
            msgs = _flatten(_read_any(path))
        else:
            return False
        return bool(msgs) and all(_pick(m, "text", self.mapping) is not None for m, _ in msgs[:5])

    def load(self, path: Path) -> Bundle:
        path = Path(path)
        msgs, one_list = read_messages(path)
        rows, pos, n_no_author, n_ts_values, bad_ts = [], [], 0, 0, None
        for i, (m, container, ref) in enumerate(msgs):
            text = _text(_pick(m, "text", self.mapping))
            if not text.strip():
                continue
            mid = _pick(m, "id", self.mapping) or stable_id(path, i)
            alt = str(_pick(m, "author_alt", self.mapping) or "")
            author = _pick(m, "author", self.mapping) or alt
            n_no_author += not author
            raw_ts, parent = _pick(m, "ts", self.mapping), _pick(m, "parent", self.mapping)
            ts = _to_ts(raw_ts)
            n_ts_values += raw_ts is not None
            if bad_ts is None and raw_ts is not None and pd.isna(ts):
                bad_ts = raw_ts
            rows.append({
                "event_id": str(mid),
                "ts": ts,
                "channel": str(_pick(m, "channel", self.mapping) or container or "main"),
                "author_raw": str(author or "unknown"),  # an id beats "unknown"
                "author_alt": alt,
                "text": text,
                "parent_id": None if parent is None else str(parent),
                "visible_until": pd.NaT,
                "channel_family": "",
                "source_ref": ref,
                "site": _site(path),
                "source_kind": "message",
            })
            pos.append(i)
        if not rows:
            raise ValueError(f"{path}: no messages with text found; " + (
                f"the first record's keys are {_keys(msgs[0][0])}: name the text's key with `--field text=<key>` "
                f"(`swarmprov inspect {path}` shows what each field resolved to)" if msgs else f"the chat adapter reads {FORMATS}"))
        ev = pd.DataFrame(rows)
        n_timed = int(ev["ts"].notna().sum())
        has_ts = n_timed >= MIN_TIMED * len(ev)
        clock = "wall" if has_ts else ("sequence" if one_list else "none")
        warns = _warnings(ev, n_no_author, n_ts_values, bad_ts, list(msgs[0][0]))   # before positions replace times
        if clock == "sequence":
            ev["ts"] = SEQUENCE_EPOCH + pd.to_timedelta(pos, unit="s")
        caps = Capabilities(has_wall_clock=bool(has_ts), has_order=clock != "none", has_explicit_author=True,
                            has_reads=False, has_lifecycle=False, has_threading=bool(ev["parent_id"].notna().any()))
        notes = {"n_messages": len(ev), "n_channels": int(ev["channel"].nunique()),
                 "n_authors": int(ev["author_raw"].nunique()), "clock": clock, "n_timestamps": n_timed,
                 "n_no_author": n_no_author, "warnings": warns}
        if clock == "sequence":
            notes["timestamps_discarded"] = n_timed
        return Bundle(events=ev, capabilities=caps, notes=notes)

    def config(self) -> AdapterConfig:
        return AdapterConfig(name="chat", default_family="all")
