"""Generic chat-transcript adapter (AI Village, Slack/Discord exports, framework logs).

Accepts JSON (a list of messages, ``{"messages": [...]}``, or ``{room: [...]}``)
or JSONL.  Field names are guessed from common aliases; override any of them
with ``ChatAdapter(mapping={"ts": "created", "author": "agent_name", ...})``.
Each message is one post visible to everyone in its channel from its timestamp.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from ..textutil import stable_id
from .base import Adapter, AdapterConfig, Bundle, Capabilities

ALIASES = {
    "ts": ["timestamp", "time", "created_at", "createdAt", "ts", "date", "datetime", "sent_at"],
    "author": ["author", "sender", "agent", "agent_name", "name", "user", "from", "speaker", "username"],
    "text": ["content", "text", "message", "body", "msg"],
    "channel": ["channel", "room", "thread", "conversation", "chat", "chat_id", "room_id"],
    "id": ["id", "message_id", "uuid", "event_id"],
    "parent": ["parent_id", "reply_to", "in_reply_to", "thread_ts"],
}


WRAPPER_KEYS = ("messages", "events", "transcript", "items", "data")


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
    """A whole transcript on one line: an array, or a wrapper object around one."""
    return isinstance(obj, list) or (isinstance(obj, dict) and any(isinstance(obj.get(k), list) for k in WRAPPER_KEYS))


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
    if v is None:
        return pd.NaT
    if isinstance(v, (int, float)):
        unit = "ms" if v > 1e11 else "s"
        return pd.to_datetime(v, unit=unit, utc=True)
    return pd.to_datetime(v, utc=True, errors="coerce")


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
        msgs = _flatten(_read_any(Path(path)))
        rows = []
        for i, m in enumerate(msgs):
            text = _text(_pick(m, "text", self.mapping))
            if not text.strip():
                continue
            mid = _pick(m, "id", self.mapping) or stable_id(path, i)
            rows.append({
                "event_id": str(mid),
                "ts": _to_ts(_pick(m, "ts", self.mapping)),
                "channel": str(_pick(m, "channel", self.mapping) or "main"),
                "author_raw": str(_pick(m, "author", self.mapping) or "unknown"),
                "author_alt": "",
                "text": text,
                "parent_id": _pick(m, "parent", self.mapping),
                "visible_until": pd.NaT,
                "channel_family": "",
                "source_ref": f"{Path(path).name}:{i}",
                "site": _site(Path(path)),
                "source_kind": "message",
            })
        if not rows:
            raise ValueError(f"{path}: no messages with text found (pass mapping={{'text': <field>}} ?)")
        ev = pd.DataFrame(rows)
        has_ts = ev["ts"].notna().mean() > 0.9
        caps = Capabilities(has_wall_clock=has_ts, has_explicit_author=True, has_reads=False,
                            has_lifecycle=False, has_threading=bool(ev["parent_id"].notna().any()))
        notes = {"n_messages": len(ev), "n_channels": int(ev["channel"].nunique()),
                 "n_authors": int(ev["author_raw"].nunique())}
        return Bundle(events=ev, capabilities=caps, notes=notes)

    def config(self) -> AdapterConfig:
        return AdapterConfig(name="chat")
