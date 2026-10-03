"""Small text helpers shared by the adapters and the rule extractor."""

from __future__ import annotations

import hashlib
import re


def fix_mojibake(s: str, rounds: int = 4) -> str:
    """Undo repeated UTF-8-read-as-Latin-1 corruption ("TÃ¼rkiye" -> "Türkiye").

    Wiki pages that are re-saved by many agents accumulate one layer of
    corruption per save, so this unwinds up to ``rounds`` layers and stops as
    soon as a layer does not decode cleanly.
    """
    for _ in range(rounds):
        if "Ã" not in s and "Â" not in s:
            break
        try:
            fixed = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                fixed = s.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                break
        if fixed == s:
            break
        s = fixed
    return s


def norm_key(s: str) -> str:
    """Comparison key that ignores whitespace, case and non-ASCII noise."""
    s = s.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


def stable_id(*parts: object) -> str:
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return h[:16]


# A trailing "-- Name" signature (UseMod / MediaWiki habit), optionally [[Name]].
SIGNATURE_RE = re.compile(r"(?:^|\s)--\s?\[{0,2}([A-Za-z][A-Za-z0-9_\-\.]{1,60}?)\]{0,2}\s*\.?\s*$")
