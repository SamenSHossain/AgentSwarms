"""Adapter for the *Swarm traces* redacted payload release.

Input: the publisher's public evidence file ``redacted.jsonl`` (or
``redacted.jsonl.gz``) — a directory holding it, the file itself, or a ``.zip``.
Each row is one recovered record with the fields:

* ``id``        stable record id (``R0000001``), unique, file order;
* ``cite``      ``id:hash`` citation handle, unique;
* ``kind``      ``payload`` / ``recovered_text`` / ``response``;
* ``parent_id`` the ``payload`` a response or recovered text was reconstructed
                from, or null for a root record;
* ``time_utc``  null on every row;
* ``tags``      a ``;``-joined label string, empty on almost every row;
* ``text``      the recovered content, with sensitive spans replaced by
                bracketed placeholders (``[CREDENTIAL n]``, ``[ENCODED BLOB n]`` …).

This release is a **reconstruction corpus, not a timed multi-agent transcript**:
it carries no wall clock and no agent identity (both are redacted), so
swarmprov's provenance analyses (A1–A6) cannot run on it — there is no "when"
to order by and no "who" to attribute to. The adapter therefore classifies the
file, records the reconstruction tree (each ``payload`` and the responses and
recovered text rebuilt from it), the tag families, text de-duplication and how
thoroughly the release is redacted, and leaves the events table empty so the
report is a reconstruction summary that states plainly what is not computable.
It never executes, decodes or interprets a payload; every number is a count of
records and fields.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterator

import pandas as pd

from .base import Adapter, AdapterConfig, Bundle, Capabilities

REQUIRED_KEYS = {"id", "cite", "kind", "parent_id", "time_utc", "tags", "text"}
DATA_NAMES = ("redacted.jsonl", "redacted.jsonl.gz")
# A redaction placeholder: [REDACTED:category:000001], [CREDENTIAL 1], [ENCODED BLOB 207876], ...
PLACEHOLDER_RE = re.compile(r"\[[^\[\]]{1,80}\]")
_DIGITS_RE = re.compile(r"\d+")


def _open(root: Path, name: str) -> io.TextIOBase | None:
    """Open ``name`` as text inside a directory or a zip; handle gzip members."""
    if root.is_file() and zipfile.is_zipfile(root):
        z = zipfile.ZipFile(root)
        for n in z.namelist():
            if Path(n).name == name:
                raw = z.open(n)
                return io.TextIOWrapper(gzip.open(raw) if n.endswith(".gz") else raw, encoding="utf-8")
        return None
    folder = root if root.is_dir() else root.parent
    p = folder / name
    if not p.is_file():
        return None
    return io.TextIOWrapper(gzip.open(p), encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8")


def _data_file(path: Path) -> tuple[str, str] | None:
    """Return (kind, name) of the swarmtraces file reachable from ``path``.

    ``kind`` is 'self' when ``path`` is the data file, else the member name to
    open inside the directory or zip.
    """
    if path.is_file() and not zipfile.is_zipfile(path):
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            return ("self", path.name)
        return None
    for name in DATA_NAMES:
        if _open(path, name) is not None:
            return ("member", name)
    return None


def _reader(path: Path) -> Iterator[dict]:
    hit = _data_file(path)
    if hit is None:
        return iter(())
    if hit[0] == "self":
        fh: io.TextIOBase = (io.TextIOWrapper(gzip.open(path), encoding="utf-8")
                             if path.suffix == ".gz" else open(path, encoding="utf-8"))
    else:
        fh = _open(path, hit[1])  # type: ignore[assignment]
    return (json.loads(line) for line in fh if line.strip())


def _first_record(path: Path) -> dict | None:
    for r in _reader(path):
        return r if isinstance(r, dict) else None
    return None


def _category(placeholder: str) -> str:
    """The redaction *category* of a placeholder, with the instance number removed.

    ``[REDACTED:destination:000002]`` -> ``destination``;
    ``[SERVICE 2 URL 1]`` -> ``SERVICE URL``; ``[ENCODED BLOB 207876]`` -> ``ENCODED BLOB``.
    Used only to report how the release is redacted, never to recover a value.
    """
    inner = placeholder[1:-1].strip()
    if inner.upper().startswith("REDACTED:"):
        parts = inner.split(":")
        return parts[1].strip() if len(parts) > 1 and parts[1].strip() else "REDACTED"
    inner = _DIGITS_RE.sub("", inner)
    inner = re.sub(r"[\s_\-]+", " ", inner).strip(" -")
    return inner or "(unlabelled)"


class SwarmTracesAdapter(Adapter):
    name = "swarmtraces"

    def sniff(self, path: Path) -> bool:
        path = Path(path)
        try:
            if _data_file(path) is None:
                return False
            first = _first_record(path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile):
            return False
        return isinstance(first, dict) and REQUIRED_KEYS <= set(first)

    def load(self, path: Path) -> Bundle:
        path = Path(path)
        kinds: Counter = Counter()
        tag_families: Counter = Counter()
        redaction: Counter = Counter()
        child_of: dict[str, str] = {}          # child id -> parent id
        id_kind: dict[str, str] = {}
        ids: set[str] = set()
        cites: set[str] = set()
        n = n_tagged = n_redacted = dup_cite = cite_prefix_ok = 0
        text_keys: set[str] = set()
        n_text_repeat = 0
        text_chars: Counter = Counter()        # kind -> total chars (for a mean length)

        for r in _reader(path):
            n += 1
            rid = str(r.get("id"))
            kind = str(r.get("kind"))
            kinds[kind] += 1
            id_kind[rid] = kind
            if rid in ids:
                pass  # duplicate id: reported via uniqueness check below
            ids.add(rid)
            cite = r.get("cite")
            if isinstance(cite, str):
                if cite in cites:
                    dup_cite += 1
                cites.add(cite)
                if cite.split(":", 1)[0] == rid:
                    cite_prefix_ok += 1
            parent = r.get("parent_id")
            if isinstance(parent, str) and parent:
                child_of[rid] = parent
            tags = r.get("tags")
            if isinstance(tags, str) and tags.strip():
                n_tagged += 1
                tag_families[tags.split(";", 1)[0].strip()] += 1
            text = r.get("text") or ""
            text_chars[kind] += len(text)
            if text in text_keys:            # exact bodies, to match the publisher's distinct-text count
                n_text_repeat += 1
            else:
                text_keys.add(text)
            seen_here = set()
            for m in PLACEHOLDER_RE.findall(text):
                cat = _category(m)
                redaction[cat] += 1
                seen_here.add(cat)
            if seen_here:
                n_redacted += 1

        # reconstruction tree: payload (root) -> response / recovered_text (child)
        edge_kinds: Counter = Counter()
        fanout: Counter = Counter()
        orphans = 0
        n_children_by_parent: Counter = Counter()
        for child, parent in child_of.items():
            if parent in id_kind:
                edge_kinds[f"{id_kind[child]} <- {id_kind[parent]}"] += 1
                n_children_by_parent[parent] += 1
            else:
                orphans += 1
        for parent, c in n_children_by_parent.items():
            fanout[min(c, 10)] += 1
        n_roots = n - len(child_of)

        notes = {
            "swarmtraces": {
                "n_records": n,
                "by_kind": dict(kinds.most_common()),
                "mean_text_len": {k: round(text_chars[k] / kinds[k], 1) for k in kinds},
                "tree": {
                    "roots": int(n_roots),
                    "children": int(len(child_of)),
                    "child_of_parent_kind": dict(edge_kinds.most_common()),
                    "parents_with_children": int(len(n_children_by_parent)),
                    "fanout_dist": {str(k): int(v) for k, v in sorted(fanout.items())},
                    "orphan_children": int(orphans),
                    "max_depth": self._max_depth(child_of),
                },
                "tags": {
                    "n_tagged": int(n_tagged),
                    "families": dict(tag_families.most_common(15)),
                    "n_families": int(len(tag_families)),
                },
                "text": {"distinct": int(len(text_keys)), "repeated": int(n_text_repeat)},
                "redaction": {
                    "records_with_placeholder": int(n_redacted),
                    "placeholder_occurrences": int(sum(redaction.values())),
                    "top_categories": dict(redaction.most_common(15)),
                    "n_categories": int(len(redaction)),
                },
                "integrity": {
                    "ids_unique": len(ids) == n,
                    "cites_unique": dup_cite == 0,
                    "cite_prefixes_match_id": cite_prefix_ok == n,
                },
            }
        }
        caps = Capabilities(has_wall_clock=False, has_explicit_author=False, has_reads=False,
                            has_lifecycle=False, has_threading=True)  # parent_id is a reconstruction link, not a timeline
        from ..schema import empty
        return Bundle(events=empty("events"), capabilities=caps, notes=notes)

    @staticmethod
    def _max_depth(child_of: dict[str, str], cap: int = 64) -> int:
        best = 0
        for start in child_of:
            d, cur, seen = 0, start, set()
            while cur in child_of and cur not in seen and d < cap:
                seen.add(cur)
                cur = child_of[cur]
                d += 1
            best = max(best, d)
        return best

    def config(self) -> AdapterConfig:
        # No families or techniques: this release carries no provenance to extract.
        return AdapterConfig(name="swarmtraces")
