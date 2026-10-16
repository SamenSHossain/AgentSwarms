"""Generic chat-transcript adapter (AI Village, Slack/Discord exports, framework logs).

Accepts JSON (a list of messages, ``{"messages": [...]}``, or ``{room: [...]}``)
or JSONL.  Field names are guessed from common aliases; override any of them
with ``ChatAdapter(mapping={"ts": "created", "author": "agent_name", ...})``.
Each message is one post visible to everyone in its channel from its timestamp.

Framework logs (OpenAI message lists, AutoGen, LangGraph state) often carry no
timestamps.  When fewer than 90% of the messages have one and the file is a
single message list, the whole file switches to a *sequence clock*: message i is
posted at ``SEQUENCE_EPOCH + i`` seconds, and the real timestamps are discarded
rather than mixed in.  A ``{room: [...]}`` file with two rooms or more concatenates
lists whose relative order is unknown, so it keeps its missing timestamps and the
time-based analyses skip those posts.
"""

from __future__ import annotations

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
    "author": ["author", "sender", "agent", "agent_name", "name", "user", "from", "speaker", "username"],
    "author_alt": ["agent_id", "author_id", "sender_id", "user_id", "model"],
    "text": ["content", "text", "message", "body", "msg"],
    "channel": ["channel", "room", "thread", "conversation", "chat", "chat_id", "room_id", "chat_room_id", "channel_id"],
    "id": ["id", "message_id", "uuid", "event_id"],
    "parent": ["parent_id", "reply_to", "in_reply_to", "thread_ts"],
}


WRAPPER_KEYS = ("messages", "events", "transcript", "items", "data")
EPOCH_RE = re.compile(r"^\d{9,13}(\.\d+)?$")   # Slack's "1777885200.000100"; shorter digit runs are ids, not times
SEQUENCE_EPOCH = pd.Timestamp("2000-01-01", tz="UTC")
EPOCH_RANGE = (pd.Timestamp("1990-01-01", tz="UTC"), pd.Timestamp("2100-01-01", tz="UTC"))   # a time outside it is no time
MIN_TIMED = 0.9   # share of messages that must carry a timestamp for the file to have a wall clock


def _read_any(path: Path):
    if path.is_dir():
        raise ValueError(f"{path} is a directory; the chat adapter reads one transcript file "
                         f"(e.g. hf://datasets/<owner>/<name>/<file>.jsonl.gz)")
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


def _flatten(obj, channel: str = "") -> list[dict]:
    if isinstance(obj, list):
        out = []
        for m in obj:
            if isinstance(m, dict):
                m = dict(m)
                if channel:
                    m.setdefault("channel", channel)
                out.append(m)
        return out
    if isinstance(obj, dict):
        for key in WRAPPER_KEYS:
            if isinstance(obj.get(key), list):
                return _flatten(obj[key], channel)
        out = []
        for k, v in obj.items():
            if isinstance(v, (list, dict)):
                out += _flatten(v, k)
        return out
    return []


def _pick(m: dict, field: str, mapping: dict) -> object:
    if field in mapping:
        return m.get(mapping[field])
    for k in ALIASES[field]:
        if k in m and m[k] not in (None, ""):
            v = m[k]
            if isinstance(v, dict):  # e.g. {"author": {"name": ...}}
                v = v.get("name") or v.get("id") or json.dumps(v)
            return v
    return None


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
    while name.endswith((".gz", ".json", ".jsonl")):
        name = name.rsplit(".", 1)[0]
    return name


def _text(v) -> str:
    if isinstance(v, list):  # content blocks
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in v)
    return "" if v is None else str(v)


class ChatAdapter(Adapter):
    name = "chat"

    def __init__(self, mapping: dict | None = None):
        self.mapping = mapping or {}

    def sniff(self, path: Path) -> bool:
        path = Path(path)
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".gz"):
            return False
        msgs = _flatten(_read_any(path))[:20]
        return bool(msgs) and all(_pick(m, "text", self.mapping) is not None for m in msgs[:5])

    def load(self, path: Path) -> Bundle:
        doc = _read_any(Path(path))
        msgs = _flatten(doc)
        rows, pos, n_no_author = [], [], 0
        for i, m in enumerate(msgs):
            text = _text(_pick(m, "text", self.mapping))
            if not text.strip():
                continue
            mid = _pick(m, "id", self.mapping) or stable_id(path, i)
            alt = str(_pick(m, "author_alt", self.mapping) or "")
            author = _pick(m, "author", self.mapping) or alt
            n_no_author += not author
            rows.append({
                "event_id": str(mid),
                "ts": _to_ts(_pick(m, "ts", self.mapping)),
                "channel": str(_pick(m, "channel", self.mapping) or "main"),
                "author_raw": str(author or "unknown"),  # an id beats "unknown"
                "author_alt": alt,
                "text": text,
                "parent_id": _pick(m, "parent", self.mapping),
                "visible_until": pd.NaT,
                "channel_family": "",
                "source_ref": f"{Path(path).name}:{i}",
                "site": _site(Path(path)),
                "source_kind": "message",
            })
            pos.append(i)
        if not rows:
            raise ValueError(f"{path}: no messages with text found (pass mapping={{'text': <field>}} ?)")
        ev = pd.DataFrame(rows)
        n_timed = int(ev["ts"].notna().sum())
        has_ts = n_timed >= MIN_TIMED * len(ev)
        clock = "wall" if has_ts else ("sequence" if _n_lists(doc) == 1 else "none")
        if clock == "sequence":
            ev["ts"] = SEQUENCE_EPOCH + pd.to_timedelta(pos, unit="s")
        caps = Capabilities(has_wall_clock=bool(has_ts), has_order=clock != "none", has_explicit_author=True,
                            has_reads=False, has_lifecycle=False, has_threading=bool(ev["parent_id"].notna().any()))
        notes = {"n_messages": len(ev), "n_channels": int(ev["channel"].nunique()),
                 "n_authors": int(ev["author_raw"].nunique()), "clock": clock, "n_timestamps": n_timed,
                 "n_no_author": n_no_author}
        if clock == "sequence":
            notes["timestamps_discarded"] = n_timed
        return Bundle(events=ev, capabilities=caps, notes=notes)

    def config(self) -> AdapterConfig:
        return AdapterConfig(name="chat", default_family="all")
