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


# lone surrogates (a truncated emoji's JSON escape) cannot be stored as UTF-8; XML (GraphML) forbids the C0 controls
# but tab, newline and return, and U+FFFE/U+FFFF; names and ids also lose DEL and the C1 controls, which drive
# terminals, and turn tab, newline and return into a space, so a name printed in a line cannot start another
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffe\uffff]")
BREAK_RE = re.compile(r"[\t\n\r]+")


def clean(s: str, kind: str = "id") -> str:
    """``s`` with lone surrogates as U+FFFD; a name or id (``kind="id"``) gets a space for a tab or line break and
    loses every other control character, ``"text"`` gets a space for each XML-illegal one and keeps C1 (how a
    mis-decoded "–" reads); any other kind keeps them."""
    s = SURROGATE_RE.sub("\ufffd", s)
    if kind == "id":
        return CONTROL_RE.sub("", BREAK_RE.sub(" ", s))
    return XML_ILLEGAL_RE.sub(" ", s) if kind == "text" else s


def stable_id(*parts: object) -> str:
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return h[:16]


# A trailing "-- Name" signature (UseMod / MediaWiki habit), optionally [[Name]].  The tail is "\s*(?:\.\s*)?$", not
# "\s*\.?\s*$": the same strings, but two adjacent \s* backtrack quadratically on a long run of spaces.
SIGNATURE_RE = re.compile(r"(?:^|\s)--\s?\[{0,2}([A-Za-z][A-Za-z0-9_\-\.]{1,60}?)\]{0,2}\s*(?:\.\s*)?$")
