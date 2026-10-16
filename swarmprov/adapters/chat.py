"""Generic chat-transcript adapter (AI Village, Slack/Discord exports, framework logs, spreadsheets).

Accepts JSON (a list of messages, ``{"messages": [...]}``, or ``{room: [...]}``),
JSONL or CSV/TSV with a header row, each optionally gzipped, and two kinds of
directory: a Slack export (``users.json`` and ``channels.json`` beside one folder
of day files per channel) and any other folder of JSON/JSONL files, joined with
each file's name as the channel of messages that name none.  Field names are
guessed from common aliases (``ALIASES``); map any of them with ``fields`` in the
config or ``--field author=source`` (a dotted key such as ``author.name`` reaches
into nested objects), and ``swarmprov inspect`` shows what each resolved to.
Each message is one post, public from its timestamp (a channel is a page for A5's
relay rule; only an AI Village room table restricts who can read a room), except in
framework traces (``framework``: messages with an OpenAI/Anthropic ``role`` or a
LangChain/AutoGen ``type``), where task messages (author ``task``) and tool results
(the calling agent's, also written as ``reads``) are kept with ``source_kind``
``task`` / ``tool_result`` and are not public posts, and AutoGen's inner events
(thoughts, streaming chunks, call requests) are dropped.

Framework logs (OpenAI message lists, AutoGen, LangGraph state) often carry no
timestamps.  When fewer than 90% of the messages have one and the file is a
single message list, the whole file switches to a *sequence clock*: message i is
posted at ``SEQUENCE_EPOCH + i`` seconds, and the real timestamps are discarded
rather than mixed in.  A ``{room: [...]}`` file with two rooms or more, and any
directory, concatenates lists whose relative order is unknown, so it keeps its
missing timestamps and the time-based analyses skip those posts.  When such lists
reuse each other's ids (``1, 2, ...`` in every file), each id and parent is prefixed
with its list's name (``planning/2``) so a reply finds its parent in its own list.
"""

from __future__ import annotations

import csv
import gzip
import html
import json
import re
import zlib
from decimal import Decimal
from pathlib import Path

import pandas as pd

from ..textutil import stable_id
from . import framework
from .base import Adapter, AdapterConfig, Bundle, Capabilities

ALIASES = {
    "ts": ["timestamp", "time", "created_at", "createdAt", "ts", "date", "datetime", "sent_at"],
    "author": ["author", "sender", "agent", "agent_name", "name", "user", "from", "speaker", "username"],
    "author_alt": ["agent_id", "author_id", "sender_id", "user_id", "model"],
    "text": ["content", "text", "message", "body", "msg"],
    "channel": ["channel", "room", "thread", "conversation", "chat", "chat_id", "room_id", "chat_room_id", "channel_id"],
    "id": ["id", "message_id", "uuid", "event_id"],
    "parent": ["parent_id", "reply_to", "in_reply_to", "thread_ts", "reference.messageId"],   # the last is Discord's
}
# AutoGen's speaker, read only when no author alias and no id fallback holds a value: elsewhere `source`
# often names a data source (web, api), and one naming a URL or a path is never a speaker
SPEAKER = "source"
URLISH = re.compile(r"^[a-z][a-z0-9+.-]*://|^www\.|/", re.I)
FORMATS = ("a JSON array of messages, {\"messages\": [...]}, {room: [...]}, JSONL (one message per line) or CSV/TSV "
           "with a header row, each optionally gzipped; or a directory: a Slack export, or JSON/JSONL files")
TABLES = "a table is a JSON array (or an object holding one), JSONL or CSV/TSV with a header row, each optionally gzipped"
# the fields every analysis needs -> words their source key tends to contain (``inspect`` and the ts warning guess by them)
GUESS = {"ts": ("time", "date"), "author": ("name", "agent", "user", "speaker", "sender", "handle"),
         "text": ("text", "content", "output", "message", "body")}

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
SAMPLE = 500          # records ``inspect`` and the timestamp warning's guess look at
# Slack markup, or one of the three entities Slack escapes in text and links
SLACK_MARKUP = re.compile(r"<([^<>|]+)(?:\|([^<>]*))?>|&(?:amp|lt|gt);")
SLACK_SYSTEM = {"channel_join", "channel_leave", "channel_topic", "channel_purpose", "channel_name", "channel_archive",
                "channel_unarchive", "bot_add", "bot_remove", "pinned_item", "unpinned_item"}   # not anyone's post


def _is_tabular(path: Path) -> bool:
    name = path.name[:-3] if path.suffix == ".gz" else path.name
    return name.endswith(TABULAR)


def _open(path: Path):
    """Text, gunzipped by suffix; ``utf-8-sig`` drops the byte-order mark Excel's "CSV UTF-8" writes."""
    return (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8-sig")


def _sep(path: Path) -> str:
    """The delimiter the header row uses most (a tie or none goes to the suffix's: tab for .tsv, else comma)."""
    with _open(path) as fh:
        head = fh.readline()
    order = "\t,;|" if ".tsv" in path.name else ",\t;|"
    return max(order, key=head.count)


def _read_any(path: Path, expected: str = f"the chat adapter reads {FORMATS}"):
    try:
        if _is_tabular(path):   # a spreadsheet export; everything read as text
            return pd.read_csv(path, sep=_sep(path), dtype=str, keep_default_na=False,
                               encoding="utf-8-sig").to_dict("records")
        return _read_json(path)
    # JSONDecodeError and pandas' parser errors are ValueErrors; a broken or mis-named .gz raises the last three
    except (ValueError, csv.Error, UnicodeDecodeError, gzip.BadGzipFile, EOFError, zlib.error) as e:
        raise ValueError(f"{path.name} cannot be read ({e}); {expected}") from None


def _read_json(path: Path):
    inner = path.name[:-3] if path.suffix == ".gz" else path.name
    with _open(path) as fh:
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
        return [(framework.unwrap(m), channel) for m in obj if isinstance(m, dict)]
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
    """(source key, value) of a canonical field: the mapped key, else the first alias that holds a value.  The
    author falls back to ``SPEAKER`` only when ``author_alt`` (an agent id) is empty too."""
    if field in mapping:
        return mapping[field], _value(_get(m, mapping[field]))
    for k in ALIASES[field]:
        v = _value(_get(m, k))
        if v is not None:
            return k, v
    if field == "author" and _pick(m, "author_alt", mapping) is None:
        v = _value(m.get(SPEAKER))
        if v is not None and not URLISH.search(str(v)):
            return SPEAKER, v
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


def _site(path: Path) -> str:
    name = path.name
    while name.endswith((".gz", ".json", ".jsonl", *TABULAR)):
        name = name.rsplit(".", 1)[0]
    return name


def _text(v) -> str:
    if isinstance(v, list):  # content blocks; a tool result's block keeps its text under "content"
        return "\n".join(_text(b.get("text", b.get("content"))) if isinstance(b, dict) else str(b) for b in v)
    return "" if v is None else str(v)


def _is_slack(path: Path) -> bool:
    return path.is_dir() and (path / "users.json").is_file() and (path / "channels.json").is_file()


def _slack_text(text: str, users: dict) -> str:
    """Slack markup as plain text: ``<url|label>`` -> url, ``<@U..>`` -> @name, ``<#C..|general>`` -> #general, and
    ``&amp;`` ``&lt;`` ``&gt;`` unescaped (in a link too, so a URL with a query string matches its copies)."""
    def sub(m):
        if m.group(1) is None:
            return html.unescape(m.group(0))
        target, label = html.unescape(m.group(1)), m.group(2) and html.unescape(m.group(2))
        if target.startswith("@"):
            return "@" + users.get(target[1:], label or target[1:])
        if target.startswith("#"):
            return "#" + (label or target[1:])
        if target.startswith("!"):
            return label or "@" + target[1:]
        return target   # a link: the URL is the fact, the label only its display text
    return SLACK_MARKUP.sub(sub, text)


def _slack(path: Path) -> list[tuple[dict, str, str, str]]:
    """A Slack export's messages in ``ts`` order, channel = folder (also the list name).  User ids become
    ``users.json`` names, and a message's id is its ``ts``, so a reply's ``thread_ts`` names its thread's first
    message.  Joins, topic changes and other system messages are no one's post and are skipped."""
    users = {u["id"]: u.get("real_name") or u.get("name") or u["id"]
             for u in _read_any(path / "users.json") if isinstance(u, dict) and u.get("id")}
    out = []
    for f in sorted(path.glob("*/*.json")):
        for j, (m, _) in enumerate(_flatten(_read_any(f))):
            if m.get("subtype") in SLACK_SYSTEM:
                continue
            uid, ts, thread = m.get("user") or m.get("bot_id") or "", m.get("ts"), m.get("thread_ts")
            out.append(({"ts": ts, "author": users.get(uid) or m.get("username") or uid, "author_id": uid,
                         "text": _slack_text(m.get("text") or "", users), "id": ts,
                         "parent_id": thread if thread != ts else None},
                        f.parent.name, f"{f.parent.name}/{f.name}:{j}", f.parent.name))
    return sorted(out, key=lambda r: Decimal(r[0]["ts"]) if EPOCH_RE.match(str(r[0]["ts"])) else Decimal("Infinity"))


def _dir_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.name.endswith(DIR_FILES))


def read_messages(path: Path) -> tuple[list[tuple[dict, str, str, str]], bool]:
    """(message, container channel, source ref, list name) in reading order, and whether they form one message list
    (a single list's file position is an order; a directory's files have none between them).  A list is named by its
    container, or in a directory file holding several lists by the file and its container."""
    path = Path(path)
    if _is_slack(path):
        return _slack(path), False
    if path.is_dir():
        out = []
        for f in _dir_files(path):
            doc = _read_any(f)
            several = _n_lists(doc) > 1
            out += [(m, ch, f"{f.name}:{j}", f"{_site(f)}/{ch}" if several else ch)
                    for j, (m, ch) in enumerate(_flatten(doc, _site(f)))]
        return out, False
    doc = _read_any(path)
    return [(m, ch, f"{path.name}:{i}", ch) for i, (m, ch) in enumerate(_flatten(doc))], _n_lists(doc) == 1


def _keys(keys) -> str:
    return ", ".join(f"`{k}`" for k in keys)


def no_author_warning(n: int, keys: list | None = None) -> str:
    """Posts that name no author; with the first record's ``keys`` when they are most of the posts."""
    many = n > 1
    return (f"{n:,} post{'s name' if many else ' names'} no author (no author field was recognised), so "
            f"{'they are' if many else 'it is'} pooled as one agent, `unknown`, which can look like a hub or a relay "
            "source in A5." + (f" The first record's keys are {_keys(keys)}; name the author's key with "
                               "`--field author=<key>`." if keys else ""))


def _warnings(ev: pd.DataFrame, n_no_author: int, n_ts_values: int, bad_ts, records: list[dict],
              mapping: dict) -> list[str]:
    """Signs that a field was not found: posts with no author, one author or one channel for all, unread times."""
    n, out, keys = len(ev), [], list(records[0])
    posts = ev[ev["source_kind"] == "message"]   # the agents' own: task messages and tool results have set authors
    if n_no_author:
        out.append(no_author_warning(n_no_author, keys if n_no_author > WARN_NO_AUTHOR * len(posts) else None))
    elif len(posts) > WARN_MIN_POSTS and posts["author_raw"].nunique() == 1:
        out.append(f"All {len(posts):,} posts have one author, `{posts['author_raw'].iloc[0]}`, so no answer can be exposed to another "
                   f"agent's post and A5 has no structure to find. The first record's keys are {_keys(keys)}; if another "
                   "one names the speaker, use `--field author=<key>`.")
    n_bad, n_missing = n_ts_values - int(ev["ts"].notna().sum()), n - n_ts_values
    if n_bad + n_missing > WARN_UNTIMED * n:
        # a log with no time field at all runs on message order and the report's clock note says so: it is worth a
        # warning only when some key holds times
        guess = guess_fields(records, ["ts"], set()).get("ts") if n_missing == n and "ts" not in mapping else None
        parts = []
        if guess:
            parts.append(f"No post has a timestamp field the chat aliases know, but `{guess}` holds times: map it with "
                         f"`--field ts={guess}`.")
        elif n_missing and (n_missing < n or "ts" in mapping):
            parts.append(f"{n_missing:,} of {n:,} posts have no timestamp field (the first record's keys are "
                         f"{_keys(keys)}); if one holds the time, map it with `--field ts=<key>`.")
        if n_bad:
            parts.append(f"{n_bad:,} of {n:,} timestamps could not be read as a time (ISO text, or epoch seconds or "
                         f"milliseconds from 1990 to 2100), e.g. `{bad_ts}`.")
        if parts:
            out.append(" ".join(parts))
    if n > WARN_MIN_POSTS and ev["channel"].nunique() == 1:
        out.append(f"All {n:,} posts are in one channel, `{ev['channel'].iloc[0]}`, so A5's relay rule takes every hop as "
                   "same-page (the parent is the latest earlier carrier anywhere) and `family_from_channel` has no "
                   "channel names to use. If the source has rooms or threads, name their key with `--field channel=<key>`.")
    return out


class ChatAdapter(Adapter):
    name = "chat"
    reads_fields = True

    def __init__(self, mapping: dict | None = None):
        self.mapping = mapping or {}

    def sniff(self, path: Path) -> bool:
        path = Path(path)
        if _is_slack(path):
            return True
        if path.is_dir():   # any file of messages: load joins them all and skips records without text
            return any(self._messages(_flatten(_read_any(f))) for f in _dir_files(path))
        if path.is_file() and (path.name.endswith((".json", ".jsonl", ".gz")) or _is_tabular(path)):
            return self._messages(_flatten(_read_any(path)))
        return False

    def _messages(self, msgs: list) -> bool:
        """Every sampled message has text, or is a framework turn that only calls tools (OpenAI's `content: null`)."""
        texts = [_pick(m, "text", self.mapping) is not None for m, _ in msgs[:5]]
        return any(texts) and all(t or framework.call_ids(m) for t, (m, _) in zip(texts, msgs))

    def load(self, path: Path) -> Bundle:
        path = Path(path)
        msgs, one_list = read_messages(path)
        rows, pos, lists, n_no_author, n_ts_values, bad_ts = [], [], [], 0, 0, None
        calls, n_unmatched, n_inner = framework.Calls(), 0, 0
        for i, (m, container, ref, lst) in enumerate(msgs):
            alt = str(_pick(m, "author_alt", self.mapping) or "")
            author = _pick(m, "author", self.mapping) or alt
            kind = framework.role(m, author)
            # before the text check: a call-only turn has no text but opens calls
            if kind == "agent" or kind == "inner" and framework.call_ids(m):
                calls.open(str(author or "unknown"), framework.call_ids(m))
            elif kind == "tool":   # an AutoGen event's speaker ran the tool; an OpenAI tool's `name` is the function's
                author = calls.answer(framework.result_ids(m), framework.autogen_type(m) and author) or "tool"
            elif kind == "task":
                author = "task"
            if kind == "inner":   # no other agent receives it
                n_inner += 1
                continue
            text = _text(_pick(m, "text", self.mapping) or (framework.result_text(m) if kind == "tool" else None))
            if not text.strip():
                continue
            n_unmatched += kind == "tool" and author == "tool"
            mid = _pick(m, "id", self.mapping)
            n_no_author += not author
            raw_ts, parent = _pick(m, "ts", self.mapping), _pick(m, "parent", self.mapping)
            ts = _to_ts(raw_ts)
            n_ts_values += raw_ts is not None
            if bad_ts is None and raw_ts is not None and pd.isna(ts):
                bad_ts = raw_ts
            rows.append({
                "event_id": str(stable_id(path, i) if mid is None else mid),
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
                "source_kind": {"task": "task", "tool": "tool_result"}.get(kind, "message"),
            })
            pos.append(i)
            lists.append(lst)
        if not rows:
            raise ValueError(f"{path}: no messages with text found; " + (
                f"the first record's keys are {_keys(msgs[0][0])}: name the text's key with `--field text=<key>` "
                f"(`swarmprov inspect {path}` shows what each field resolved to)" if msgs else f"the chat adapter reads {FORMATS}"))
        ev = pd.DataFrame(rows)
        lists = pd.Series(lists, index=ev.index)
        if lists.groupby(ev["event_id"]).nunique().gt(1).any():   # lists reuse ids (1, 2, ... in every file)
            ev["event_id"] = lists + "/" + ev["event_id"]
            ev["parent_id"] = [None if pd.isna(q) else f"{l}/{q}" for l, q in zip(lists, ev["parent_id"])]
        n_timed = int(ev["ts"].notna().sum())
        has_ts = n_timed >= MIN_TIMED * len(ev)
        clock = "wall" if has_ts else ("sequence" if one_list else "none")
        warns = _warnings(ev, n_no_author, n_ts_values, bad_ts, [m for m, *_ in msgs[:SAMPLE]],
                          self.mapping)   # before positions replace times
        if n_unmatched:
            many = n_unmatched > 1
            warns.append(f"{n_unmatched:,} tool result{'s' if many else ''} answer{'' if many else 's'} no call id seen "
                         "in the file, and neither an AutoGen `source` nor a single agent with calls pending names the "
                         f"caller, so {'they are' if many else 'it is'} attributed to `tool`, not to the agent that called "
                         f"the tool, and count{'' if many else 's'} as no agent's own lookup.")
        if clock == "sequence":
            ev["ts"] = SEQUENCE_EPOCH + pd.to_timedelta(pos, unit="s")
        tools = ev[(ev["source_kind"] == "tool_result") & (ev["author_raw"] != "tool")]
        reads = pd.DataFrame({"agent_raw": tools["author_raw"], "ts": tools["ts"], "channel": tools["channel"],
                              "event_id": tools["event_id"], "method": "tool_result"}).reset_index(drop=True)
        caps = Capabilities(has_wall_clock=bool(has_ts), has_order=clock != "none", has_explicit_author=True,
                            has_reads=bool(len(reads)), has_lifecycle=False,
                            has_threading=bool(ev["parent_id"].notna().any()))
        notes = {"n_messages": len(ev), "n_channels": int(ev["channel"].nunique()),
                 "n_authors": int(ev["author_raw"].nunique()), "clock": clock, "n_timestamps": n_timed,
                 "n_no_author": n_no_author, "warnings": warns}
        kinds = ev["source_kind"].value_counts()
        if kinds.get("task") or kinds.get("tool_result") or n_inner:
            notes.update(n_task=int(kinds.get("task", 0)), n_tool_results=int(kinds.get("tool_result", 0)),
                         n_tool_unmatched=n_unmatched, n_inner_dropped=n_inner)
        if clock == "sequence":
            notes["timestamps_discarded"] = n_timed
        return Bundle(events=ev, reads=reads if len(reads) else None, capabilities=caps, notes=notes)

    def config(self) -> AdapterConfig:
        return AdapterConfig(name="chat", default_family="all")
