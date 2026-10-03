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


def _read_any(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        head = fh.read(1)
        fh.seek(0)
        if head in "[{":
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                fh.seek(0)
        return [json.loads(line) for line in fh if line.strip()]


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
        for key in ("messages", "events", "transcript", "items", "data"):
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
            })
        ev = pd.DataFrame(rows)
        has_ts = len(ev) > 0 and ev["ts"].notna().mean() > 0.9
        caps = Capabilities(has_wall_clock=has_ts, has_explicit_author=True, has_reads=False,
                            has_lifecycle=False, has_threading=bool(len(ev) and ev["parent_id"].notna().any()))
        notes = {"n_messages": len(ev), "n_channels": int(ev["channel"].nunique()) if len(ev) else 0,
                 "n_authors": int(ev["author_raw"].nunique()) if len(ev) else 0}
        return Bundle(events=ev, capabilities=caps, notes=notes)

    def config(self) -> AdapterConfig:
        return AdapterConfig(name="chat")
