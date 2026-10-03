"""Rule-based extraction: answer reports, predictions, tags and citations.

The engine is source-agnostic; the patterns that are specific to a source
(families, cohort and episode markers, techniques) come from AdapterConfig.
The built-in phrases cover the templated reports agents write when they
coordinate on timed question sequences, e.g.::

    R3 CONFIRMED: Poland arrived 19:43:07; timer 56s; answered 16.40% instantly.

Every claim row carries ``extractor="rule"`` so LLM / gold rows can be merged
and compared later.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pandas as pd

from .adapters.base import AdapterConfig
from .gazetteer import COUNTRIES, US_STATES, canonical_item
from .textutil import stable_id

NUM = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
ONE_NUM = rf"(?<![\w.:+#]){NUM}(?![\w:])"
VALUE_RE = re.compile(rf"{ONE_NUM}(?:\s*[;/]\s*{ONE_NUM})*")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:/\d{2})?\b")
ANSWER_RE = re.compile(
    r"(?<!will )(?<!to )(?<!please )\b(answered|submitted|replied|responded)\b(?!\s+(?:first|asap|immediately\s+on))",
    re.I)
ANSWER_EQ_RE = re.compile(r"\b(?:Answer(?:ed)?|CONFIRMED|confirmed)\s*:?\s*(?P<item>[A-Z][\w\.' ]{1,40}?)\s*=\s*(?:'''|\*\*)?(?P<value>" + NUM + r"(?:\s*[;/]\s*" + NUM + r")*)")
INSTANT_RE = re.compile(
    r"instantly|immediately|same[- ]second|(?:at|on|upon) receipt|\bat \+?[01]s\b|\+1s\b|within (?:1|one) ?s(?:ec(?:ond)?)?\b|\bin 1s\b|\(1s\)|<\s?2s",
    re.I)
WRONG_RE = re.compile(r"answered wrong|wrong answer|\bguess(?:ed)?\b|incorrect(?:ly)?|answered\s+\S+\s+\(?wrong", re.I)
DISPUTE_RE = re.compile(r"\b(?:wrong|correction|actually|not\s+" + NUM + r"|beware|mismatch|conflict)", re.I)
ARRIVE_RE = re.compile(r"\b(arrived|prompt(?:ed)?(?: observed)?|appeared|came in)\b", re.I)
ARRIVE_TIME_RE = re.compile(
    r"(?:arrived|prompt|activated)[^\d\n]{0,30}?(?:[A-Z][a-z]{2,8}\s?\d{1,2}\s+)?(\d{1,2}:\d{2}:\d{2})", re.I)
CLOCK_RE = re.compile(r"\b(\d{1,2}:\d{2}:\d{2})\b")
# words allowed between a round marker and the item it names: "G5 CONFIRMED: **Montana**"
SEQ_LEAD_RE = re.compile(
    r"\s*(?:-|:|=|is|was|CONFIRMED|confirmed|actual(?:ly)?|arrived)?\s*[:=]?\s*(?:'''|\*\*)?")
# "R2 deadline", "R4 due": a marker naming some other round, not the one being reported
OTHER_ROUND_RE = re.compile(r"\s*(?:deadline|due|cooldown|ETA|projected|expected|prep|would|will)\b", re.I)
ANSWER_AT_RE = re.compile(r"\bat\s+(\d{1,2}:\d{2}:\d{2}|:\d{2})\b|\bat\s+\+(\d+)s\b", re.I)
TIMER_RE = re.compile(
    r"(?:timer|window|deadline in)\s*(?:of\s*)?(\d+m\d{0,2}s?|\d+\s?s(?:ec)?)\b"
    r"|(\d+m\d{0,2}s?|\d+\s?s(?:ec)?)\s*(?:timer|window|answer window)", re.I)
TIMER_AFTER_TIME_RE = re.compile(r"\d{1,2}:\d{2}:\d{2}\)?,?\s*\(?(\d+m\d{2}s?|\d{1,3}s)\b")
PREDICT_RE = re.compile(
    r"\b(expected|expect|predicts?|predicted|projected|projects|likely|suspect|hypothesis|prep|cached|ready)\b", re.I)
CONFIRM_RE = re.compile(r"\bCONFIRMED\b|\bconfirmed\b")
CORRECTION_RE = re.compile(r"\bCORRECTION\b|\bcorrection\b|\bactually\b|\bnow known\b|\bnot\s+" + NUM, re.I)
REQUEST_RE = re.compile(r"\bplease\s+(?:post|relay|signal|confirm|report|reply|share|add)\b|\bASAP\b|\burgent", re.I)
INDEPENDENT_RE = re.compile(r"independent(?:ly)?|own (?:lookup|query|computation|calculation)|re-?derived|reproduc(?:ed|ible)|verified (?:it )?(?:myself|ourselves|directly)", re.I)
CITE_RES = [
    re.compile(r"@([A-Za-z][\w\-]{2,60})"),
    re.compile(r"\b([A-Z][\w\-]{2,60})'s\s+(?:report|value|post|projection|confirmation|relay|mapping|signal|offset|evidence|table|numbers?)", re.I),
    re.compile(r"\b(?:per|via|thanks?,?|thank you,?|from)\s+@?([A-Z][a-z]{2}\d{2}[\w\-]*|[A-Z][\w]*\d[\w]*)"),
]
STOP_ITEMS = {
    "Confirmed", "Live", "Prompt", "Answer", "Answered", "Deadline", "Timer", "Next", "Our", "The",
    "Due", "Expected", "Please", "Values", "Value", "Exact", "Arrived", "Cooldown", "Update", "Slow",
    "Fast", "Tier", "Task", "Clock", "Final", "Signal", "Relay", "Round", "Note", "UTC", "Wall", "Live",
    "Notice", "Projected", "Predicted", "Likely", "Unconfirmed", "Same", "Cohort", "No", "Yes",
    "State", "States", "ETA", "Item", "Question", "Query", "Then", "Now", "Also", "Our", "Their",
}
STOP_LOWER = {w.lower() for w in STOP_ITEMS}
FUNCTION_WORDS = {
    "at", "is", "was", "are", "answered", "projects", "projected", "task", "sent", "for", "done",
    "activated", "exactly", "creation", "scaffold", "due", "definitely", "exists", "the", "to", "on",
    "in", "by", "with", "from", "after", "before", "then", "now", "still", "arrived", "prompt",
}
CORRECT_RE = re.compile(r"answered\s+(?:exact(?:ly)?\s+)?correct(?:ly)?|\bcorrect answer\b|\bscored\b", re.I)
LATENCY_IN_RE = re.compile(r"\b(?:in|after|within)\s+(\d+(?:\.\d+)?)\s?s(?:ec(?:onds?)?)?\b", re.I)
STATE_CODE_RE = re.compile(r"\b([A-Z]{2})(?:\s+(?:arrived|prompt|confirmed|CONFIRMED|answered)\b|\s*[:=]?\s+\d)")
AMBIGUOUS_CODES = {"IN", "OR", "ME", "OK", "HI", "ID"}
QUESTION_RE = re.compile(r"\b(?:did|has|have|does|do|anyone|who)\b[^.?\n]{0,80}\banswered\b[^.?\n]*\?", re.I)


YEAR_RE = re.compile(r"(?:19[5-9]\d|20[0-4]\d)")


def first_value(seg: str) -> str | None:
    """First value-like token in ``seg``, skipping bare years (2015) and clock times."""
    seg = TIME_RE.sub(" ", seg)
    for m in VALUE_RE.finditer(seg):
        tok = m.group(0)
        if YEAR_RE.fullmatch(tok.strip()):
            continue
        if re.search(r"(?:19|20)\d\d-$", seg[: m.start()]):
            continue  # the "20" in a "2015-20" year range
        # "2015-20 = ..." style year ranges
        if YEAR_RE.fullmatch(tok.split("-")[0].strip()) and seg[m.end(): m.end() + 1] == "-":
            continue
        return tok
    return None


def _secs(tok: str | None) -> float | None:
    if not tok:
        return None
    tok = tok.lower().replace(" ", "").replace("sec", "s")
    m = re.fullmatch(r"(?:(\d+)m)?(\d+)?s?", tok)
    if not m or not (m.group(1) or m.group(2)):
        return None
    return float(int(m.group(1) or 0) * 60 + int(m.group(2) or 0))


def _clock(tok: str) -> int:
    parts = [int(p) for p in tok.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def norm_value(raw: str | None) -> str | None:
    """'16.40%' -> '16.4', '20,369' -> '20369', '874,322;951,258' -> '874322/951258'."""
    if not raw:
        return None
    out = []
    for tok in re.split(r"\s*[;/]\s*", raw.strip()):
        tok = tok.replace(",", "").replace("%", "").strip()
        try:
            f = float(tok)
        except ValueError:
            continue
        out.append(("%.4f" % f).rstrip("0").rstrip("."))
    return "/".join(out) or None


class ItemMatcher:
    """Finds item mentions (states, countries, learned task items) in text."""

    def __init__(self, extra: dict[str, list[str]] | None = None):
        names = set(US_STATES.values()) | set(COUNTRIES)
        self.global_names = names
        self.by_family: dict[str, set[str]] = defaultdict(set)
        for fam, items in (extra or {}).items():
            self.by_family[fam] |= set(items)
        self._cache: dict[str, re.Pattern] = {}
        self._lower: dict[str, dict[str, str]] = {}

    def add(self, family: str, items: set[str]) -> None:
        self.by_family[family] |= items
        self._cache.pop(family, None)

    def pattern(self, family: str) -> re.Pattern:
        if family not in self._cache:
            names = sorted(self.global_names | self.by_family.get(family, set()), key=len, reverse=True)
            alt = "|".join(re.escape(n) for n in names)
            # G4-KY style state codes
            self._cache[family] = re.compile(
                rf"(?<![\w])(?:{alt})(?:s|es)?(?![\w])|(?<=[RG#][1-9]-)[A-Z]{{2}}\b", re.I)
            self._lower[family] = {n.lower(): n for n in names}
        return self._cache[family]

    def find(self, family: str, text: str) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        ends: list[int] = []
        pat = self.pattern(family)
        lower = self._lower[family]
        for m in pat.finditer(text):
            tok = m.group(0)
            if tok.lower() not in lower:
                for suf in ("es", "s"):
                    if tok.lower().endswith(suf) and tok.lower()[: -len(suf)] in lower:
                        tok = lower[tok.lower()[: -len(suf)]]
                        break
            if len(tok) == 2 and tok.upper() not in US_STATES:
                continue
            if len(tok) == 2 and not tok.isupper():
                continue
            out.append((m.start(), canonical_item(tok)))
            ends.append(m.end())
        for m in STATE_CODE_RE.finditer(text):
            if m.group(1) in US_STATES and m.group(1) not in AMBIGUOUS_CODES:
                # "Saginaw MI": the code qualifies the preceding item, it is not an item itself
                if any(0 <= m.start() - e <= 2 for e in ends):
                    continue
                if any(p == m.start() for p, _ in out):
                    continue
                out.append((m.start(), US_STATES[m.group(1)]))
        return sorted(out)


def learn_items(events: pd.DataFrame, families: pd.Series, cfg: AdapterConfig,
                min_authors: int = 2) -> dict[str, set[str]]:
    """Learn task items: capitalised phrases right after a round marker, e.g. 'R2 Business arrived'."""
    rx = re.compile(
        r"(?<![A-Za-z0-9])(?:R|G|#)[1-9]\s*(?:CONFIRMED|confirmed|LIVE)?\s*[:\-]?\s*"
        r"(?:'''|\*\*)?([A-Z][A-Za-z\.']+(?:\s+(?:&\s+|and\s+|of\s+)?[A-Za-z][A-Za-z\.']+){0,3}?)(?:'''|\*\*)?\s*"
        r"(?:arrived|prompt|confirmed|CONFIRMED|due|=|\(|-\s*\d|\d)")
    seen: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for text, fam, author in zip(events["text"], families, events["author_raw"]):
        if not fam:
            continue
        for m in rx.finditer(text):
            phrase = m.group(1).strip()
            words = phrase.split()
            while words and words[-1].lower() in STOP_LOWER:
                words.pop()
            while words and words[0].lower() in STOP_LOWER:
                words.pop(0)
            # "Saginaw MI", "Merced County CA" -> "Saginaw", "Merced": qualifiers are not part of the item
            while len(words) > 1 and (words[-1].upper() in US_STATES and words[-1].isupper()
                                      or words[-1] in ("County", "Parish", "Co.")):
                words.pop()
            if words and words[0] == "New" and len(words) == 1:
                continue
            if not words or len(" ".join(words)) < 3:
                continue
            if any(w.lower() in FUNCTION_WORDS for w in words) or any(w.lower() in STOP_LOWER for w in words[1:]):
                continue  # "Poland at", "Hungary answered at": a clause, not an item name
            phrase = " ".join(words)
            if phrase.isupper() and canonical_item(phrase) == phrase:
                continue  # acronyms / shouting that is not a known place
            seen[fam][canonical_item(phrase)].add(author)
    return {fam: {p for p, a in d.items() if len(a) >= min_authors} for fam, d in seen.items()}


def _episode_before(text: str, pos: int, rx: re.Pattern) -> tuple[int | None, int]:
    best, best_pos = None, 0
    for m in rx.finditer(text, 0, pos):
        if OTHER_ROUND_RE.match(text, m.end()):
            continue
        best, best_pos = int(m.group(1)), m.start()
    return best, best_pos


SEQ_SPLIT_RE = re.compile(r"\s*(?:->|→|=>|>)\s*")
SEQ_CHAIN_RE = re.compile(r"[A-Z][\w .&']{1,30}(?:\s*(?:->|→|=>)\s*[A-Z?][\w .&']{0,30}){2,}")


def learn_sequences(events: pd.DataFrame, families: pd.Series, items: "ItemMatcher",
                    min_support: int = 2) -> dict[tuple[str, str], int]:
    """Item -> position from chains like 'MA -> CT -> MI -> WV' or 'GA -> AR -> NV -> KY'."""
    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for text, fam in zip(events["text"], families):
        if not fam or ("->" not in text and "→" not in text and "=>" not in text):
            continue
        for chain in SEQ_CHAIN_RE.findall(text):
            parts = SEQ_SPLIT_RE.split(chain)
            for pos, tok in enumerate(parts, start=1):
                found = items.find(fam, tok.strip() + " ")
                if found:
                    votes[(fam, found[0][1])][pos] += 1
    out = {}
    for k, c in votes.items():
        pos, n = c.most_common(1)[0]
        if n >= min_support and n / sum(c.values()) >= 0.6:
            out[k] = pos
    return out


def _latency(scope: str, clause: str, wider: str = "", thr: float = 2.0) -> tuple[str | None, float | None]:
    """scope: text from the round marker to 'answered'; clause: the answer clause
    after it (up to the sentence end); wider: preceding text, used only to find
    the arrival time when the scope has none."""
    if INSTANT_RE.search(clause):
        m_s = ANSWER_AT_RE.search(clause)
        if m_s and m_s.group(2):
            return "instant", float(m_s.group(2))
        return "instant", None
    m_in = LATENCY_IN_RE.search(clause[:40])
    if m_in:
        v = float(m_in.group(1))
        return ("instant" if v <= thr else "delayed"), v
    m_at = ANSWER_AT_RE.search(clause)
    if m_at:
        if m_at.group(2):
            v = float(m_at.group(2))
            return ("instant" if v <= thr else "delayed"), v
        m_arr = ARRIVE_TIME_RE.search(scope) or CLOCK_RE.search(scope) or ARRIVE_TIME_RE.search(wider)
        if m_arr:
            t_arr = _clock(m_arr.group(1))
            tok = m_at.group(1)
            t_ans = t_arr - t_arr % 60 + int(tok[1:]) if tok.startswith(":") else _clock(tok)
            d = (t_ans - t_arr) % 86400
            if d > 43200:
                d = 0
            return ("instant" if d <= thr else "delayed"), float(d)
    if INSTANT_RE.search(scope[-60:]):
        return "instant", None
    return None, None


def extract(events: pd.DataFrame, families: pd.Series, agents: pd.DataFrame,
            cfg: AdapterConfig, items: ItemMatcher) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (claims, tags)."""
    ep_rx = re.compile(cfg.episode_regex)
    amap = agents.set_index("author_raw")
    tech_rx = {t.name: re.compile(t.pattern, re.I) for t in cfg.techniques}
    claims, tags = [], []
    for ev, fam in zip(events.itertuples(index=False), families):
        text = ev.text
        a = amap.loc[ev.author_raw] if ev.author_raw in amap.index else None
        base = {
            "event_id": ev.event_id, "ts": ev.ts,
            "agent_strict": a["agent_strict"] if a is not None else ev.author_raw,
            "agent_merged": a["agent_merged"] if a is not None else ev.author_raw,
            "family": fam,
        }
        n_answers = 0
        # 1) "... answered 16.40% instantly" reports
        for m in ANSWER_RE.finditer(text):
            sent_start = max(text.rfind("\n", 0, m.start()), text.rfind(". ", 0, m.start())) + 1
            if QUESTION_RE.search(text[sent_start: m.end() + 120]):
                continue
            line_start = text.rfind("\n", 0, m.start()) + 1
            episode, ep_pos = _episode_before(text, m.start(), ep_rx)
            if episode is not None and ep_pos < line_start:
                episode = None  # a marker on an earlier line belongs to another report
            if episode is None or m.start() - ep_pos > 300:
                line_start = max(text.rfind("\n", 0, m.start()), text.rfind(". ", 0, m.start()))
                ep_pos = line_start + 1
                if episode is not None and m.start() - ep_pos > 300:
                    episode = None
            scope = text[ep_pos:m.start()]
            after = text[m.end(): m.end() + 120]
            cut = re.search(r"\.\s|\n|\(|\)|;\s*(?=[A-Z#])|\bat\b|\bbut\b|\bbefore\b|\bthen\b|\bafter\b|,\s*correct\s+\d", after)
            vphrase = after[: cut.start()] if cut else after[:60]
            value_raw = first_value(vphrase)
            in_scope = items.find(fam, scope + text[m.start():m.end()])
            in_value = items.find(fam, vphrase)
            item = in_scope[-1][1] if in_scope else (in_value[0][1] if in_value else None)
            if item is None and episode is None:
                back = text[max(0, ep_pos - 200):ep_pos]
                bc = items.find(fam, back)
                item = bc[-1][1] if bc else None
            clause_end = re.search(r"\.\s|\n|;\s*(?=[A-Z#])", text[m.end():])
            clause = text[m.end(): m.end() + (clause_end.start() if clause_end else 80)][:120]
            lat_class, lat_s = _latency(scope, clause, text[max(0, ep_pos - 200): ep_pos],
                                        cfg.instant_threshold_s)
            tm = TIMER_RE.search(scope) or TIMER_RE.search(after[:60])
            timer = _secs(next((g for g in tm.groups() if g), None)) if tm else None
            if timer is None and (tm2 := TIMER_AFTER_TIME_RE.search(scope)):
                timer = _secs(tm2.group(1))
            window = text[ep_pos: m.end() + 80]
            if item is None and value_raw is None:
                continue
            claims.append({**base,
                "claim_id": stable_id(ev.event_id, "ans", m.start()),
                "episode": episode, "item": item, "value_raw": value_raw,
                "value_norm": norm_value(value_raw), "event_type": "answer",
                "latency_class": lat_class, "latency_s": lat_s, "timer_s": timer,
                "wrong_flag": bool(WRONG_RE.search(window)),
                "correct_flag": bool(CORRECT_RE.search(text[m.start(): m.end() + 30])),
                "extractor": "rule"})
            n_answers += 1
        # 2) "Answer: Hungary = 9.90%" / "G5 CONFIRMED: Montana = 8553"
        if n_answers == 0:
            for m in ANSWER_EQ_RE.finditer(text):
                item_c = items.find(fam, m.group("item"))
                episode, _ = _episode_before(text, m.start() + 1, ep_rx)
                if episode is None:
                    me = ep_rx.search(text[m.start(): m.start() + 20])
                    episode = int(me.group(1)) if me else None
                claims.append({**base,
                    "claim_id": stable_id(ev.event_id, "eq", m.start()),
                    "episode": episode,
                    "item": item_c[-1][1] if item_c else canonical_item(m.group("item")),
                    "value_raw": m.group("value"), "value_norm": norm_value(m.group("value")),
                    "event_type": "answer",
                    "latency_class": _latency(text[m.start():m.end()], text[m.end():m.end() + 60], "",
                                              cfg.instant_threshold_s)[0],
                    "latency_s": None, "timer_s": None, "wrong_flag": False, "correct_flag": False,
                    "extractor": "rule"})
        # 3) predictions: "expected Slovak Republic 14.59%", "predicts G5 Maryland (52,395)"
        for m in PREDICT_RE.finditer(text):
            seg = text[m.end(): m.end() + 70]
            found = items.find(fam, seg)
            if not found:
                continue
            p, item = found[0]
            if p > 30:
                continue
            pv = first_value(seg[p: p + 45])
            episode, _ = _episode_before(text, m.end() + p, ep_rx)
            claims.append({**base,
                "claim_id": stable_id(ev.event_id, "pred", m.start()),
                "episode": episode, "item": item,
                "value_raw": pv, "value_norm": norm_value(pv),
                "event_type": "prediction", "latency_class": None, "latency_s": None,
                "timer_s": None, "wrong_flag": False, "correct_flag": False, "extractor": "rule"})
        # 4) sequence reports: "G5 CONFIRMED: Montana", "predicts G5 **Maryland**", "R4 = Slovak Republic"
        seen_seq = set()
        for m in ep_rx.finditer(text):
            if OTHER_ROUND_RE.match(text, m.end()):
                continue
            seg = text[m.end(): m.end() + 40]
            lead = SEQ_LEAD_RE.match(seg)
            found = items.find(fam, seg)
            if not found or found[0][0] > lead.end() + 1:
                continue
            key = (int(m.group(1)), found[0][1])
            if key in seen_seq:
                continue
            seen_seq.add(key)
            claims.append({**base,
                "claim_id": stable_id(ev.event_id, "seq", m.start()),
                "episode": key[0], "item": key[1], "value_raw": None, "value_norm": None,
                "event_type": "sequence", "latency_class": None, "latency_s": None,
                "timer_s": None, "wrong_flag": False, "correct_flag": False, "extractor": "rule"})
        cites = sorted({c for rx in CITE_RES for c in rx.findall(text)})
        tags.append({
            "event_id": ev.event_id, "ts": ev.ts, "agent_strict": base["agent_strict"],
            "agent_merged": base["agent_merged"], "family": fam,
            "is_answer": n_answers > 0, "is_arrival": bool(ARRIVE_RE.search(text)),
            "is_confirm": bool(CONFIRM_RE.search(text)),
            "is_prediction": bool(PREDICT_RE.search(text)),
            "is_correction": bool(CORRECTION_RE.search(text)),
            "is_request": bool(REQUEST_RE.search(text)),
            "is_independent": bool(INDEPENDENT_RE.search(text)),
            "techniques": ",".join(n for n, rx in tech_rx.items() if rx.search(text)),
            "cites": ",".join(cites),
        })
    cl = pd.DataFrame(claims)
    return cl, pd.DataFrame(tags)
