"""``report.html``: a report as one self-contained page that opens in a browser with no network.

``md_to_html`` converts exactly the Markdown subset the reports write (headings, pipe tables with ``\\|`` escapes,
images, bullets, bold, italics, inline code), HTML-escaping all text before any formatting: agent text is
untrusted, and the wiki report quotes a script payload.  Figures are inlined as data URIs.  ``build_html`` adds
headline tiles from ``summary.json``, a table of contents, sortable tables and, where the run has the graph
tables, a fact explorer: the relay timelines of the most-carried facts and of A4's featured slots
(``query.fact_lineage``), and each agent's links (``agent_profiles_*`` and the edges).  The explorer's data is
a JSON block the page's own script reads as text, so nothing in it is parsed as HTML.
"""

from __future__ import annotations

import base64
import html
import json
import re
import time
from pathlib import Path

import pandas as pd

from . import query
from .analysis import causal
from .claims import value_key
from .fmt import dumps, json_safe, pct
from .graph import CLOCK_RES_S, KINDS, RELAY_KINDS
from .report import a2_identified, top10_range
from .schema import RunDir, suffix

MAX_FACTS = 200     # task facts carried by two or more agents, most carriers first
MAX_URLS = 3        # URL facts, most carriers first (their trees run to hundreds of carriers)
MAX_LINKS = 6       # agents listed per direction in an agent's links
EXCERPT = 120       # characters of a post kept for the explorer (query's excerpt, centred on the item, cut shorter)
LINK_KINDS = [k for k in KINDS if k != "reply"]   # a reply is an interaction, not a knowledge transfer
PEOPLE_COLS = ["family", "n_posts", "n_answers", "n_no_prior_copy", "n_exposed", "answers_supplied", "relay_in",
               "relay_out", "facts_originated", "agents_reached"]
CAVEAT = ("Inferred paths, not proven ones: each carrier is linked to the latest earlier carrier of the value on the "
          "same page (a solid line from the parent's post), or, with none there, to the value's originator (grey: "
          "the source is unknown). Nothing in the transcript records who read what.")

_IMG = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+)\)$")
_SEP = re.compile(r"^\|(\s*:?-+:?\s*\|)+$")
_CELL = re.compile(r"(?<!\\)\|")
_NUM = re.compile(r"^[-+−]?[\d,]*\.?\d+(%| h| s)?( \(.*\))?$")


def _inline(s: str) -> str:
    """Inline formatting on text that is already HTML-escaped: code spans first (kept literal), then bold, then
    italics with ``*`` or ``_`` (an underscore inside a word, as in ``D_visible``, is no italic)."""
    code: list[str] = []

    def keep(m: re.Match) -> str:
        code.append(m.group(1))
        return f"\x00{len(code) - 1}\x00"
    s = re.sub(r"`([^`]+)`", keep, s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)", r"<em>\1</em>", s)
    return re.sub("\x00(\\d+)\x00", lambda m: f"<code>{code[int(m.group(1))]}</code>", s)


def _cells(row: str) -> list[str]:
    row = row[1:] if row.startswith("|") else row
    row = row[:-1] if row.endswith("|") and not row.endswith("\\|") else row
    return [_inline(html.escape(c.strip())).replace("\\|", "|") for c in _CELL.split(row)]


def _table(rows: list[str]) -> str:
    head = "".join(f"<th>{c}</th>" for c in _cells(rows[0]))
    body = "".join("<tr>" + "".join(f"<td{' class=num' if _NUM.match(re.sub('<[^>]+>', '', c)) else ''}>{c}</td>"
                                    for c in _cells(r)) + "</tr>" for r in rows[1:])
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _figure(alt: str, src: str, base: Path) -> str:
    """An image the report links (``figures/NAME.png`` only) as a data URI; a missing one says so."""
    p, a = (base / src).resolve(), html.escape(alt)
    if p.parent != (base / "figures").resolve() or p.suffix != ".png" or not p.is_file():
        return f'<p class="missing">Figure not found: {html.escape(src)}</p>'
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<figure><img alt="{a}" src="data:image/png;base64,{data}"></figure>'


def _slug(text: str, seen: dict) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>|&[a-z#0-9]+;", "", text).lower()).strip("-") or "section"
    seen[s] = seen.get(s, 0) + 1
    return s if seen[s] == 1 else f"{s}-{seen[s]}"


def md_to_html(text: str, base: Path) -> str:
    """The reports' Markdown as HTML, each ``##`` section a ``<section>`` with an id; ``base`` is the directory the
    report's figure links are relative to."""
    lines, out, para, seen, open_ = text.replace("\x00", "").splitlines(), [], [], {}, False
    i = 0

    def end_para() -> None:
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            end_para()
        elif m := re.match(r"^(#{1,6})\s+(.*)$", line):
            end_para()
            level, body = len(m.group(1)), _inline(html.escape(m.group(2).strip()))
            if level == 2:
                out.append(("</section>" if open_ else "") + f'<section id="{_slug(body, seen)}">')
                open_ = True
            out.append(f"<h{level}>{body}</h{level}>")
        elif m := _IMG.match(line):
            end_para()
            out.append(_figure(m.group(1), m.group(2), base))
        elif line.startswith("|") and i + 1 < len(lines) and _SEP.match(lines[i + 1].strip()):
            end_para()
            rows = [line]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            out.append(_table(rows))
            continue
        elif line.startswith("- "):
            end_para()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"<li>{_inline(html.escape(lines[i].strip()[2:]))}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue
        else:
            para.append(html.escape(line))
        i += 1
    end_para()
    return "\n".join(out) + ("</section>" if open_ else "")


def _t(ts) -> float | int | None:
    """A time as epoch seconds (a sequence clock's positions too: 2000-01-01 + i s)."""
    if ts is None or pd.isna(ts):
        return None
    t = round(pd.Timestamp(ts).timestamp(), 3)
    return int(t) if t.is_integer() else t


def _only(df: pd.DataFrame, facts: pd.DataFrame, value: str) -> pd.DataFrame:
    """The rows of ``df`` that belong to one of ``facts`` (family, item, value key), in their order."""
    keys = pd.MultiIndex.from_frame(facts[["family", "item", "value_key"]])
    return df[pd.MultiIndex.from_arrays([df["family"], df["item"], df[value]]).isin(keys)]


def explorer_data(run: RunDir, mapping: str = "merged", summary: dict | None = None) -> dict | None:
    """The fact explorer's data, or None for a run without the graph tables: the task facts (at most ``MAX_FACTS``)
    and URL facts (``MAX_URLS``) carried by two or more agents, most carriers first, and A4's featured slots, each
    fact's hops from ``query.fact_lineage``, the posts they cite once per item (excerpt and page), and each agent's
    links (``_people``)."""
    need = (f"chains_{mapping}", f"edges_{mapping}", "events", "mentions")
    if not all(run.has(n) for n in need):
        return None
    tables = query.lineage_tables(run, mapping)
    ch, edges = tables["chains"], tables["edges"]
    rep = ch[ch["n_agents"] > 1].sort_values(["n_agents", "max_depth", "t0"], ascending=[False, False, True])
    feat = [r for r in ((summary or {}).get("A4") or {}).get("rows", []) if r.get("featured")]
    disputes = run.read(f"disputes_{mapping}") if feat and run.has(f"disputes_{mapping}") else pd.DataFrame()
    slot_rows = disputes[disputes["slot"].isin([r["slot"] for r in feat])] if len(disputes) else disputes
    picked = pd.concat([rep[rep["family"] != "url"].head(MAX_FACTS), rep[rep["family"] == "url"].head(MAX_URLS)])
    on_slots = ch.merge(slot_rows.loc[slot_rows["kind"] == "value", ["family", "item"]].drop_duplicates(),
                        on=["family", "item"]) if len(slot_rows) else ch.iloc[:0]
    # fact_lineage reads only its fact's rows, so the tables shrink to the facts traced here
    keep = pd.concat([picked, on_slots]).drop_duplicates(["family", "item", "value_key"])
    tables |= {"chains": keep, "mentions": _only(tables["mentions"], keep, "value_key"),
               "edges": _only(edges[edges["kind"].isin(RELAY_KINDS)], keep, "value_norm")}
    agents, chans, posts, facts, index = {}, {}, {}, [], {}

    def a(label) -> int:
        return agents.setdefault(label, len(agents))

    def post(eid, excerpt, channel, item) -> int:
        """A post once per item it carries: query's excerpt is centred on the fact's item, and a digest's two items
        can lie further apart than the excerpt is long."""
        if not isinstance(eid, str):
            return -1
        if (eid, item) not in posts:
            c = chans.setdefault(channel, len(chans)) if isinstance(channel, str) else -1
            cut = excerpt if len(excerpt or "") <= EXCERPT else excerpt[:EXCERPT].rstrip() + "…"
            posts[eid, item] = (len(posts), cut, c)
        return posts[eid, item][0]

    def add(f: dict, kind: str) -> int:
        key = (f["family"], f["item"], f["value"])
        if key not in index:
            o = f["origin"]
            index[key] = len(facts)
            facts.append({"kind": kind, "family": f["family"], "item": f["item"], "value": f["value"],
                          "carriers": f["carriers"], "depth": f["max_depth"], "span": f["span_s"],
                          "o": [a(o["agent"]), _t(o.get("ts")),
                                post(o.get("event_id"), o.get("excerpt"), o.get("channel"), f["item"])],
                          "h": [[a(h["src"]), a(h["dst"]), int(h["kind"] == "relay"), _t(h["ts"]), _t(h["src_ts"]),
                                 int(h["within_clock_res"]), post(h["event_id"], h["excerpt"], h["channel"], f["item"]),
                                 h["depth"]]
                                for h in f["hops"]]})
        return index[key]
    for r in picked.itertuples(index=False):
        for f in query.fact_lineage(run, r.family, r.item, r.value_key, mapping, tables):
            add(f, "url" if r.family == "url" else "task")
    slots = []
    for r in feat:
        g = slot_rows[slot_rows["slot"] == r["slot"]].sort_values(["t_first", "agent"]) if len(slot_rows) else slot_rows
        if not len(g):
            continue
        fam, item, kind = g["family"].iloc[0], g["item"].iloc[0], g["kind"].iloc[0]
        lin = {f["value"]: f for f in query.fact_lineage(run, fam, item, None, mapping, tables)} if kind == "value" else {}
        variants = []
        for v, gv in g.groupby("variant", sort=False):
            f = lin.get(value_key(v)) or lin.get(v)
            variants.append({"value": v, "fact": add(f, "task") if f else None,
                             "c": [[a(x), _t(t)] for x, t in zip(gv["agent"], gv["t_first"])]})
        slots.append({"slot": r["slot"], "kind": kind, "status": r["status"], "first": r.get("first variant"),
                      "winner": r.get("winner"), "outcome": r.get("outcome"), "variants": variants})
    people = _people(run, mapping, edges, a, set(agents))
    prof = run.profile() if (run.path / "profile.json").exists() else {}
    return json_safe({"clock": prof.get("clock") or "wall", "res": CLOCK_RES_S, "mapping": mapping,
                          "agents": list(agents), "channels": list(chans),
                          "posts": [[e, x, c] for (e, _), (_, x, c) in posts.items()],
                          "facts": facts, "slots": slots, "people": people})


def _people(run: RunDir, mapping: str, edges: pd.DataFrame, a, named: set) -> dict:
    """The agents the page names (``named``: the explorer's carriers), those with an answer and the first public
    sources: each one's profile counts (``agent_profiles_*``, as in the report's Agents section) and the agents it
    learned from and supplied, with edges per kind (the ``MAX_LINKS`` with most edges each way, and how many
    there are)."""
    out = {"cols": PEOPLE_COLS, "kinds": LINK_KINDS, "families": [], "rows": []}
    if not run.has(f"agent_profiles_{mapping}"):
        return out
    prof = run.read(f"agent_profiles_{mapping}")
    prof = prof[prof["label"].isin(named) | (prof["n_answers"] > 0) | (prof["answers_supplied"] > 0)]
    e = edges[edges["kind"].isin(LINK_KINDS) & (edges["src"].isin(prof["label"]) | edges["dst"].isin(prof["label"]))]
    n = e.groupby(["src", "dst", "kind"]).size().unstack("kind").reindex(columns=LINK_KINDS).fillna(0).astype(int)
    n = n.assign(total=n.sum(axis=1)).reset_index()

    def side(me: str, other: str) -> tuple[pd.Series, dict]:
        g = n.sort_values([me, "total", other], ascending=[True, False, True])
        top = g[g.groupby(me).cumcount() < MAX_LINKS]
        lists: dict = {}
        for m, o, *k in zip(top[me], top[other], *(top[c] for c in LINK_KINDS)):
            lists.setdefault(m, []).append([a(o), *map(int, k)])
        return g.groupby(me).size(), lists
    (n_in, into), (n_out, to) = side("dst", "src"), side("src", "dst")
    fams: dict = {}
    for p in prof.sort_values("label").to_dict("records"):
        lab, fam = p["label"], p["family"]
        out["rows"].append([a(lab), fams.setdefault(fam, len(fams)) if isinstance(fam, str) and fam else -1,
                            *[int(p[c]) for c in PEOPLE_COLS[1:]],
                            int(n_in.get(lab, 0)), into.get(lab, []), int(n_out.get(lab, 0)), to.get(lab, [])])
    out["families"] = list(fams)
    return out


def _script_json(obj) -> str:
    """JSON safe inside a script element: no ``</`` can close it and no ``<!--`` can open an escaped state."""
    return dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\u0021--")


def tiles(summary: dict) -> list[tuple[str, str, str]]:
    """The hero row, (value, label, what it means), from the summary values the key findings quote."""
    out = []
    if a1 := summary.get("A1"):
        k = a1.get("n_self_originated") or 0   # absent from a summary written before self-originated answers
        out += [(f"{a1['n_answers']:,}", "answers", f"by {a1['n_agents']:,} agents"),
                (pct(a1["share_exposed"]), "already public before the report", "a floor: only captured posts are searched"),
                (f"≤ {a1['n_independent'] + k:,}", "independent lookups",
                 f"{a1['n_independent']:,} with nothing public, {k:,} first posted by the agent itself" if k
                 else "nothing carrying the answer was public")]
    if rng := top10_range(summary.get("A5") or {}):
        out.append((rng[0], "top-10 source share", f"of {rng[1]:,} exposed answers, across attribution rules"))
    if (tv := (summary.get("A6") or {}).get("task values")) and tv.get("max_hops") is not None:
        out.append((f"{tv['max_hops']:,}", "longest task-value relay chain (hops)", "inferred paths, not proven ones"))
    if summary.get("A2"):
        main, by = a2_identified(summary["A2"])
        null = sorted(by[causal.NO_EFFECT], key=lambda r: -r["agents_with_D_variation"])
        if main and null and not by[causal.IDENTIFIED]:
            r, rest = null[0], len(main) - len(null)
            row = (f"{r['outcome']} under {r['treatment']}: {r['agents_with_D_variation']:,} agents switch, 95% CI "
                   f"{r['ci_lo']:+.2f} to {r['ci_hi']:+.2f}")
            # as the key finding opens: one row's null does not answer the question when most rows are not identified
            out.append(("mostly not identified", "does availability cause copying?",
                        f"{rest} of {len(main)} rows; no effect detected for {row}") if 2 * rest > len(main) else
                       (causal.NO_EFFECT, "does availability cause copying?",
                        row + (f"; {rest} of {len(main)} rows not identified" if rest else "")))
        elif main:
            out.append((causal.IDENTIFIED if by[causal.IDENTIFIED] else causal.NOT_IDENTIFIED,
                        "does availability cause copying?", "A2: two-way fixed effects on agent-rounds"))
    return out


def build_html(run: RunDir, mapping: str = "merged") -> Path | None:
    """``report.html`` (``report_strict.html``) beside the report it renders, or None when there is none."""
    sfx, t0 = suffix(mapping), time.perf_counter()
    md = run.path / f"report{sfx}.md"
    if not md.exists():
        return None
    sp = run.path / f"summary{sfx}.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    text = md.read_text(encoding="utf-8")   # a title line, then the source line, then the sections
    title, _, text = text.partition("\n\n") if text.startswith("# ") else ("# swarmprov report", "", text)
    lede, _, text = ("", "", text) if text.startswith("#") else text.partition("\n\n")
    title, body = title[2:].strip(), md_to_html(text, run.path)
    data = explorer_data(run, mapping, summary) if summary.get("A1") or summary.get("A5") else None
    if data and (data["facts"] or data["slots"]):
        at = body.find("</section>") + len("</section>") if body.startswith('<section id="key-findings">') else 0
        body = body[:at] + EXPLORER + body[at:]
    else:
        data = None
    toc = "".join(f'<li><a href="#{i}">{t}</a></li>' for i, t in re.findall(r'<section id="([^"]+)">\s*<h2>(.*?)</h2>', body))
    tile_html = "".join(f'<div class="tile"><div class="v{" word" if v[-1].isalpha() else ""}">{html.escape(v)}</div>'
                        f'<div class="l">{html.escape(lab)}</div><div class="n">{html.escape(note)}</div></div>'
                        for v, lab, note in tiles(summary))
    page = (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f'<header class="hero"><h1>{_inline(html.escape(title))}</h1>'
            + (f'<p class="lede">{_inline(html.escape(" ".join(lede.split())))}</p>' if lede else "")
            + (f'<div class="tiles">{tile_html}</div>' if tile_html else "")
            + f'</header><div class="layout"><nav class="toc" aria-label="Contents"><details><summary>Contents'
            f"</summary><ol>{toc}</ol></details></nav><main>{body}</main></div>"
            + (f'<script type="application/json" id="fx-data">{_script_json(data)}</script>' if data else "")
            + f"<script>{JS}</script></body></html>\n")
    out = run.path / f"report{sfx}.html"
    out.write_text(page, encoding="utf-8")
    print(f"[swarmprov] wrote {out} ({len(page.encode()) / 1e6:.1f} MB, {time.perf_counter() - t0:.1f} s)")
    return out


def main(run_path: str, mapping: str = "merged") -> int:
    """The ``html`` command: render a run's report again (after editing it, or for a run from before it was built).
    A directory holding the report and no ``profile.json`` (a committed ``results/`` one) renders too."""
    md = f"report{suffix(mapping)}.md"
    if not (Path(run_path) / md).is_file():
        RunDir.open(run_path)   # no run at all: NotARun
        print(f"{run_path}: no {md} to render (run `swarmprov report {run_path}"
              + (f" --mapping {mapping}" if suffix(mapping) else "") + "` first)")
        return 1
    build_html(RunDir.open(run_path, need=md), mapping)
    return 0


EXPLORER = f"""<section id="fact-explorer"><h2>Fact explorer</h2>
<p>How one value travelled: pick a fact (an item and value two or more agents posted) or one of A4's featured
disputes. Each row is a carrier, in order of its first post of the value; hover or tap a row, or step through
rows with the arrow keys, to see the post that carried it, its page and the delay from its inferred parent.</p>
<p class="caveat"><strong>{CAVEAT}</strong></p>
<div class="fx"><div class="fx-pick"><label for="fx-q">Find a fact</label>
<input id="fx-q" type="search" placeholder="item, value or family" autocomplete="off">
<ul id="fx-list" class="pick" aria-label="Facts"></ul></div>
<div class="fx-main"><div id="fx-head" class="fx-head"></div><div id="fx-legend" class="legend"></div>
<div id="fx-detail" class="detail" aria-live="polite"></div><div id="fx-chart" class="chart"></div>
<details class="tv"><summary>The rows as a table</summary><div id="fx-table" class="tbl"></div></details></div></div>
<h3>Agents</h3>
<p>Whom an agent learned from (edges into it) and whom it supplied (edges out of it), by edge kind: same-page
relay, cross-page hop (source unknown, credited to the originator), exposure (first public source of an answer)
and citation; replies are left out, as interactions rather than transfers. The counts come from the agent
profiles behind the report's Agents section.</p>
<div class="ag"><label for="ag-q">Find an agent</label>
<input id="ag-q" type="search" placeholder="agent label, or click a carrier above" autocomplete="off">
<ul id="ag-list" class="pick" aria-label="Agents"></ul><div id="ag-card" class="card"></div></div></section>"""

CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
--line:#e1e0d9;--axis:#c3c2b7;--wash:rgba(42,120,214,.10);--code:#f0efec;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
--xch:#b9b8b3;--ring:#fcfcfb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--axis:#383835;--wash:rgba(57,135,229,.16);--code:#262624;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--xch:#5c5b57;--ring:#1a1a19}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;
--muted:#898781;--line:#2c2c2a;--axis:#383835;--wash:rgba(57,135,229,.16);--code:#262624;--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--xch:#5c5b57;--ring:#1a1a19}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.hero{max-width:1240px;margin:0 auto;padding:28px 16px 8px}
.hero h1{font-size:28px;line-height:1.2;margin:0 0 8px}
.lede{color:var(--ink-2);margin:0 0 18px;max-width:80ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .v{font-size:28px;font-weight:600;line-height:1.15}
.tile .v.word{font-size:20px;line-height:1.4}
.tile .l{font-size:14px;margin-top:4px}
.tile .n{font-size:12px;color:var(--muted);margin-top:2px}
.layout{max-width:1240px;margin:0 auto;padding:0 16px 48px;display:grid;grid-template-columns:220px minmax(0,1fr);gap:28px}
.toc{position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto;padding-top:20px;font-size:13px}
.toc summary{font-weight:600;cursor:pointer;margin-bottom:6px}
.toc ol{margin:0;padding:0;list-style:none}
.toc li{margin:3px 0}
.toc a{color:var(--ink-2);text-decoration:none}
.toc a:hover{color:var(--ink);text-decoration:underline}
main{min-width:0}
section{border-top:1px solid var(--line);padding-top:6px}
h2{font-size:21px;margin:18px 0 10px;scroll-margin-top:12px}
h3{font-size:16px;margin:20px 0 8px}
p,li{max-width:92ch;overflow-wrap:anywhere}
code{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code);padding:1px 4px;border-radius:4px}
figure{margin:14px 0}
figure img{display:block;max-width:100%;height:auto;background:#fcfcfb;border-radius:10px;border:1px solid var(--line)}
.missing{color:var(--muted);font-style:italic}
.tbl{overflow-x:auto;margin:10px 0 14px;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;font-size:13px;width:max-content;min-width:100%}
th,td{padding:6px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
td{overflow-wrap:break-word;max-width:34ch}
tr:last-child td{border-bottom:0}
th{font-weight:600;color:var(--ink-2);background:var(--surface);cursor:pointer;user-select:none;white-space:nowrap}
th[aria-sort=ascending]::after{content:" \\25B2";font-size:10px}
th[aria-sort=descending]::after{content:" \\25BC";font-size:10px}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.caveat{border-left:3px solid var(--s2);padding:4px 0 4px 12px;color:var(--ink-2)}
.caveat strong{font-weight:500}
.fx{display:grid;grid-template-columns:260px minmax(0,1fr);gap:18px;margin-top:12px}
.fx-pick label,.ag label{display:block;font-size:13px;color:var(--ink-2);margin-bottom:4px}
input[type=search]{width:100%;font:inherit;padding:7px 10px;border:1px solid var(--axis);border-radius:8px;
background:var(--surface);color:var(--ink)}
.pick{list-style:none;margin:8px 0 0;padding:0;max-height:430px;overflow:auto;border:1px solid var(--line);
border-radius:10px;background:var(--surface)}
.pick:empty{display:none}
.pick button{display:block;width:100%;text-align:left;font:inherit;font-size:13px;color:var(--ink);background:none;
border:0;border-bottom:1px solid var(--line);padding:7px 10px;cursor:pointer}
.pick button:hover,.pick button[aria-current=true]{background:var(--wash)}
.pick small{display:block;color:var(--muted);font-size:12px}
.fx-head{font-weight:600}
.fx-head small{display:block;font-weight:400;color:var(--ink-2)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;color:var(--ink-2);margin:8px 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.chart{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.chart svg{display:block;width:100%;touch-action:pan-y}
.chart svg:focus-visible{outline:2px solid var(--s1);outline-offset:-2px}
.detail{min-height:4.8em;font-size:13px;margin:10px 0;padding:10px 12px;background:var(--surface);
border:1px solid var(--line);border-radius:10px}
.detail .q{color:var(--ink-2);font-style:italic;margin-top:4px;overflow-wrap:anywhere}
.detail button,.card button{font:inherit;color:var(--s1);background:none;border:0;padding:0;cursor:pointer;
text-decoration:underline}
.tv summary{cursor:pointer;font-size:13px;color:var(--ink-2)}
.ag{max-width:760px}
.card{margin-top:12px}
.card .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card ul{margin:4px 0;padding-left:18px;font-size:13px}
.card .stats{font-size:13px;color:var(--ink-2)}
@media (max-width:860px){.layout{grid-template-columns:minmax(0,1fr);gap:0}
.toc{position:static;max-height:none;padding-top:8px}.toc details:not([open]) ol{display:none}
.fx{grid-template-columns:minmax(0,1fr)}.pick{max-height:220px}.card .cols{grid-template-columns:minmax(0,1fr)}}
@media (max-width:520px){.hero h1{font-size:22px}.tile .v{font-size:22px}.tiles{grid-template-columns:1fr 1fr}}
"""

JS = r"""(() => {
"use strict";
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, text) => { const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text != null) e.textContent = text; return e; };
const sv = (tag, attrs) => { const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v); return e; };

// sortable tables: click (or Enter on) a header; numbers by value, ISO dates and text in order, empty cells last
const num = s => { const m = s.replace(/,/g, "").replace(/−/g, "-").match(/^[-+]?\d*\.?\d+(e[-+]?\d+)?/i);
  return m ? parseFloat(m[0]) : NaN; };
function sortable(table) {
  if (!table.tHead) return;
  const ths = [...table.tHead.rows[0].cells];
  ths.forEach((th, i) => {
    th.tabIndex = 0; th.setAttribute("aria-sort", "none");
    const go = () => {
      const dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
      ths.forEach(h => h.setAttribute("aria-sort", "none")); th.setAttribute("aria-sort", dir);
      const body = table.tBodies[0], rows = [...body.rows];
      const key = rows.map(r => r.cells[i] ? r.cells[i].textContent.trim() : "");
      const iso = key.every(k => k === "" || /^\d{4}-\d\d-\d\d/.test(k));
      const n = key.map(num), numeric = !iso && key.every((k, j) => k === "" || !isNaN(n[j]));
      const order = rows.map((r, j) => j).sort((a, b) => {
        if (key[a] === "" || key[b] === "") return (key[a] === "") - (key[b] === "") || a - b;
        const c = numeric ? n[a] - n[b] : key[a].localeCompare(key[b]);
        return (dir === "ascending" ? c : -c) || a - b; });
      order.forEach(j => body.appendChild(rows[j]));
    };
    th.addEventListener("click", go);
    th.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  });
}
document.querySelectorAll("main table").forEach(sortable);
if (matchMedia("(min-width: 861px)").matches) document.querySelector(".toc details").open = true;

const src = document.getElementById("fx-data");
if (!src) return;
const D = JSON.parse(src.textContent);
const SEQ = D.clock === "sequence", T0 = 946684800;
const A = D.agents, P = D.posts, chan = p => p && p[2] >= 0 ? D.channels[p[2]] : "";
const pad2 = x => String(x).padStart(2, "0");
const when = t => { if (t == null) return "?"; if (SEQ) return "message " + Math.round(t - T0);
  const d = new Date(t * 1000); return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())} ` +
    `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())} UTC`; };
const dur = s => { if (s == null || isNaN(s)) return "?"; if (SEQ) { const k = Math.round(s); return k + (k === 1 ? " message" : " messages"); }
  const a = Math.abs(s); return a < 120 ? s.toFixed(0) + " s" : a < 7200 ? (s / 60).toFixed(1) + " min" : (s / 3600).toFixed(1) + " h"; };
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COL = ["--s1", "--s2", "--s3"];
const pl = (k, w) => `${k} ${w}${k === 1 ? "" : "s"}`;

// the timeline model: rows (a carrier each, in order of first post) and hops between rows
function model(sel) {
  const rows = [], hops = [];
  const addFact = (f, color, variant) => {
    const at = {}, push = r => { at[r.agent] = rows.length; rows.push(r); }, mine = [];
    push({agent: f.o[0], t: f.o[1], post: f.o[2], origin: true, color, variant});
    [...f.h].sort((x, y) => x[3] - y[3]).forEach(h => {
      if (at[h[1]] != null) return;
      push({agent: h[1], t: h[3], post: h[6], color, variant, hop: hops.length});
      // under a sequence clock positions are not seconds, so no hop is a clock-resolution tie
      const hop = {src: h[0], dst: h[1], relay: h[2] === 1, t: h[3], st: h[4], tie: !SEQ && h[5] === 1, depth: h[7], color};
      hops.push(hop); mine.push(hop);
    });
    mine.forEach(h => { h.srow = at[h.src]; h.drow = at[h.dst]; });   // rows of this fact, before sorting
  };
  if (sel.type === "fact") addFact(D.facts[sel.i], "--s1", null);
  else D.slots[sel.i].variants.forEach((v, k) => {
    const color = COL[k] || "--muted";
    if (v.fact != null) addFact(D.facts[v.fact], color, v.value);
    else v.c.forEach((c, j) => rows.push({agent: c[0], t: c[1], post: -1, origin: j === 0, color, variant: v.value}));
  });
  const order = rows.map((r, j) => j).sort((a, b) => (rows[a].t - rows[b].t) || a - b), pos = {};
  order.forEach((j, k) => pos[j] = k);
  hops.forEach(h => { h.srow = pos[h.srow]; h.drow = pos[h.drow]; });
  return {rows: order.map(j => rows[j]), hops};
}

const chart = document.getElementById("fx-chart"), detail = document.getElementById("fx-detail");
let cur = null, M = null, active = -1;

function ticks(lo, hi, n) {
  if (SEQ) { const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    const step = [1, 2, 5, 10].map(x => x * p).find(x => x >= raw) || raw; const out = [];
    for (let t = Math.ceil((lo - T0) / step) * step; t <= hi - T0; t += step) out.push([t + T0, String(t)]); return out; }
  const steps = [60, 300, 900, 3600, 10800, 21600, 43200, 86400, 172800, 604800, 2592000];
  const step = steps.find(s => (hi - lo) / s <= n) || steps[steps.length - 1], out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) { const d = new Date(t * 1000);
    out.push([t, `${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}` + (step < 86400 ? ` ${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}` : "")]); }
  return out;
}

function draw() {
  if (!M) return;
  const rows = M.rows, n = rows.length, W = Math.max(280, chart.clientWidth);
  const LW = W >= 640 ? 190 : 0, rh = n <= 40 ? 18 : n <= 100 ? 12 : Math.max(3, 900 / n);
  const top = 10, H = top + n * rh + 34, x0 = LW + 10, x1 = W - 18;
  const ts = []; rows.forEach(r => r.t != null && ts.push(r.t)); M.hops.forEach(h => h.st != null && ts.push(h.st));
  let lo = Math.min(...ts), hi = Math.max(...ts); if (!(hi > lo)) { lo -= 60; hi += 60; }
  const X = t => x0 + (t - lo) / (hi - lo) * (x1 - x0), Y = k => top + (k + 0.5) * rh;
  const svg = sv("svg", {viewBox: `0 0 ${W} ${H}`, height: H, tabindex: 0, role: "img",
    "aria-label": `Timeline of ${pl(n, "carrier")}; arrow keys step through them`});
  const g = sv("g");
  svg.append(sv("rect", {x: 0, y: 0, width: W, height: H, fill: css("--surface")}));
  for (const [t, lab] of ticks(lo, hi, Math.max(2, Math.floor((x1 - x0) / 110)))) {
    svg.append(sv("line", {x1: X(t), x2: X(t), y1: top, y2: top + n * rh, stroke: css("--line"), "stroke-width": 1}));
    const tx = sv("text", {x: X(t), y: H - 12, "text-anchor": "middle", "font-size": 11, fill: css("--muted")}); tx.textContent = lab; svg.append(tx);
  }
  svg.append(sv("line", {x1: x0, x2: x1, y1: top + n * rh, y2: top + n * rh, stroke: css("--axis"), "stroke-width": 1}));
  const band = sv("rect", {x: 0, width: W, height: Math.max(rh, 2), fill: css("--wash"), visibility: "hidden"});
  svg.append(band);
  const dense = rh < 8, r0 = dense ? Math.max(1.6, rh / 2.4) : 4;
  M.hops.forEach(h => { if (h.srow == null || h.drow == null) return;
    const ln = sv("line", {x1: X(h.st), y1: Y(h.srow), x2: X(h.t), y2: Y(h.drow), "stroke-linecap": "round",
      stroke: h.relay ? css(h.color) : css("--xch"), "stroke-width": h.relay ? (dense ? 1.2 : 2) : 1, opacity: h.relay ? 0.9 : 0.8});
    g.append(ln); h.el = ln;
    if (h.relay) g.append(sv("circle", {cx: X(h.st), cy: Y(h.srow), r: dense ? 1 : 2, fill: css(h.color)}));
  });
  rows.forEach((r, k) => {
    if (r.t == null) return;
    const cx = X(r.t), cy = Y(k);
    if (r.hop != null && M.hops[r.hop].tie) g.append(sv("circle", {cx, cy, r: r0 + 3, fill: "none", stroke: css("--ink"), "stroke-width": 1}));
    g.append(sv("circle", {cx, cy, r: r.origin ? r0 + 1.5 : r0, fill: r.origin ? css("--ink") : css(r.color),
      stroke: css("--ring"), "stroke-width": dense ? 0.8 : 2}));
    if (LW && rh >= 12) { const tx = sv("text", {x: LW, y: cy + 3.5, "text-anchor": "end", "font-size": 11, fill: css("--ink-2")});
      const s = A[r.agent]; tx.textContent = s.length > 28 ? s.slice(0, 27) + "…" : s; g.append(tx); }
  });
  svg.append(g);
  const pick = e => { const b = svg.getBoundingClientRect(), y = (e.clientY - b.top) * H / b.height;
    const k = Math.floor((y - top) / rh); if (k >= 0 && k < n) setActive(k); };
  svg.addEventListener("pointermove", pick); svg.addEventListener("pointerdown", pick);
  svg.addEventListener("keydown", e => { if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault();
    setActive(Math.min(n - 1, Math.max(0, active + (e.key === "ArrowDown" ? 1 : -1)))); } });
  M.band = band; M.Y = Y; M.rh = rh;
  chart.replaceChildren(svg);
  if (active >= 0) setActive(active);
}

function agentLink(i) { const b = el("button", {type: "button"}, A[i]); b.addEventListener("click", () => showAgent(i, true)); return b; }

function setActive(k) {
  active = k; const r = M.rows[k]; if (!r) return;
  M.band.setAttribute("y", M.Y(k) - Math.max(M.rh, 2) / 2); M.band.setAttribute("visibility", "visible");
  M.hops.forEach(h => h.el && h.el.setAttribute("stroke-width", h.relay ? (M.rh < 8 ? 1.2 : 2) : 1));
  const h = r.hop != null ? M.hops[r.hop] : null, p = r.post >= 0 ? P[r.post] : null;
  if (h && h.el) h.el.setAttribute("stroke-width", 3.5);
  const line1 = el("div"); line1.append(agentLink(r.agent));
  const v = r.variant != null ? ` ${r.variant}` : "";
  if (h) { line1.append(h.relay ? ` posted${v} after a same-page post by ` : ` posted${v}; cross-page hop, source unknown, attributed to the originator `);
    line1.append(agentLink(h.src)); }
  else line1.append(r.origin ? ` first posted${v}` + (r.variant != null ? "" : " (the originator)") : ` carried${v}`);
  const bits = [when(r.t)];
  if (h) bits.push(`${dur(h.t - h.st)} after the parent's post`, `depth ${h.depth}`);
  if (h && h.tie) bits.push(`within the ${D.res} s clock resolution: direction unreliable`);
  if (p) bits.push(chan(p), `post ${p[0]}`);
  detail.replaceChildren(line1, el("div", {}, bits.filter(Boolean).join(" · ")));
  if (p && p[1]) detail.append(el("div", {class: "q"}, `“${p[1]}”`));
}

function table() {
  const t = el("table"), hr = el("tr");
  ["#", "carrier", "first post", "variant", "inferred parent", "hop", "delay", "page", "post"].forEach(c => hr.append(el("th", {}, c)));
  const th = el("thead"); th.append(hr); t.append(th); const tb = el("tbody");
  M.rows.forEach((r, k) => { const h = r.hop != null ? M.hops[r.hop] : null, p = r.post >= 0 ? P[r.post] : null, tr = el("tr");
    [k + 1, A[r.agent], when(r.t), r.variant ?? "", h ? A[h.src] : "", h ? (h.relay ? "same-page relay" : "cross-page hop") + (h.tie ? " (tie)" : "") : r.origin ? "origin" : "",
     h ? dur(h.t - h.st) : "", chan(p), p ? p[0] : ""].forEach(c => tr.append(el("td", {}, String(c))));
    tb.append(tr); });
  t.append(tb); sortable(t); document.getElementById("fx-table").replaceChildren(t);
}

function legend(sel) {
  const L = document.getElementById("fx-legend"), sw = (draw, text) => { const s = el("span"); const g = sv("svg", {width: 22, height: 12});
    draw(g); s.append(g, document.createTextNode(text)); return s; };
  const line = c => g => g.append(sv("line", {x1: 1, x2: 21, y1: 6, y2: 6, stroke: css(c), "stroke-width": 2, "stroke-linecap": "round"}));
  const dot = (c, r) => g => g.append(sv("circle", {cx: 11, cy: 6, r, fill: css(c)}));
  const items = [];
  const both = c => g => { line(c)(g); dot(c, 4)(g); };
  // a slot whose carriers were matched by pattern has no hops, so no hop or tie to explain
  const hops = sel.type === "fact" || D.slots[sel.i].variants.some(v => v.fact != null);
  if (sel.type === "slot") D.slots[sel.i].variants.forEach((v, k) => items.push(sw(both(COL[k] || "--muted"),
    `${v.value}: carriers` + (v.fact != null ? " and same-page relays" : ""))));
  else items.push(sw(line("--s1"), "same-page relay, from the parent's post (small dot)"));
  if (hops) items.push(sw(line("--xch"), "cross-page hop: source unknown, from the originator"));
  items.push(sw(dot("--ink", 5), "originator"));
  if (hops && !SEQ) items.push(sw(g => g.append(sv("circle", {cx: 11, cy: 6, r: 5, fill: "none", stroke: css("--ink")})),
    `within ${D.res} s of the parent: direction unreliable`));
  L.replaceChildren(...items);
}

function select(sel) {
  cur = sel; active = -1; M = model(sel);
  const head = document.getElementById("fx-head");
  if (sel.type === "fact") { const f = D.facts[sel.i], xs = f.h.filter(h => h[2] === 0).length, ties = f.h.filter(h => h[5] === 1).length;
    head.replaceChildren(document.createTextNode(f.kind === "url" ? f.value : `${f.family} | ${f.item} | ${f.value}`),
      el("small", {}, `${pl(f.carriers, "carrier")}, ${pl(f.h.length, "hop")} (${f.h.length - xs} same-page relay, ${xs} cross-page` +
        (SEQ ? "" : `, ${ties} within the clock resolution`) + `), max depth ${f.depth}, span ${dur(f.span)}`)); }
  else { const s = D.slots[sel.i];
    head.replaceChildren(document.createTextNode(s.slot), el("small", {}, `A4 ${s.kind} slot, ${s.status}: first ${s.first ?? "none (two first seen together)"}, winner ${s.winner ?? "none (a tie)"}` +
      (s.outcome ? `; against the analyst's reference: ${s.outcome}` : "") +
      (s.variants.some(v => v.fact == null) ? ". Carriers matched by pattern: no relay hops are inferred for them" : ""))); }
  legend(sel); draw(); table();
  detail.replaceChildren(el("div", {}, "Hover or tap a row for the post that carried it."));
  document.querySelectorAll("#fx-list button").forEach(b => b.setAttribute("aria-current", b.dataset.k === `${sel.type}${sel.i}` ? "true" : "false"));
}

const items = [...D.slots.map((s, i) => ({type: "slot", i, label: s.slot, sub: `A4 dispute · ${s.status}`})),
  ...D.facts.map((f, i) => ({type: "fact", i, label: f.kind === "url" ? f.value : `${f.family} | ${f.item} | ${f.value}`,
    sub: `${f.kind === "url" ? "URL" : "task value"} · ${pl(f.carriers, "carrier")} · max depth ${f.depth}`}))];
const list = document.getElementById("fx-list"), q = document.getElementById("fx-q");
function fill() { const s = q.value.trim().toLowerCase();
  const hit = items.filter(it => !s || it.label.toLowerCase().includes(s)).slice(0, 80);
  list.replaceChildren(...hit.map(it => { const li = el("li"), b = el("button", {type: "button", "data-k": `${it.type}${it.i}`}, it.label);
    b.append(el("small", {}, it.sub)); b.addEventListener("click", () => select(it)); li.append(b); return li; }));
  if (cur) list.querySelectorAll("button").forEach(b => b.setAttribute("aria-current", b.dataset.k === `${cur.type}${cur.i}` ? "true" : "false")); }
q.addEventListener("input", fill); fill();
const first = D.facts.length ? {type: "fact", i: 0} : {type: "slot", i: 0};
select(first);
let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(draw, 120); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { legend(cur); draw(); });

// agents
const PR = D.people, byAgent = {}; PR.rows.forEach((r, k) => byAgent[r[0]] = k);
const C = Object.fromEntries(PR.cols.map((c, k) => [c, k + 1]));
const KN = {relay: "same-page relay", relay_xchannel: "cross-page", exposure: "exposure", citation: "citation"};
const card = document.getElementById("ag-card"), aq = document.getElementById("ag-q"), al = document.getElementById("ag-list");
function linkList(title, total, links) {
  const box = el("div"); box.append(el("strong", {}, `${title} (${total})`));
  const ul = el("ul"); links.forEach(l => { const li = el("li"); li.append(agentLink(l[0]));
    li.append(document.createTextNode(" — " + PR.kinds.map((k, j) => l[j + 1] ? `${l[j + 1]} ${KN[k] || k}` : "").filter(Boolean).join(", ")));
    ul.append(li); });
  if (total > links.length) ul.append(el("li", {}, `and ${total - links.length} more`));
  if (!total) ul.append(el("li", {}, "none"));
  box.append(ul); return box;
}
function showAgent(i, scroll) {
  const k = byAgent[i]; aq.value = A[i]; al.replaceChildren();
  if (k == null) { card.replaceChildren(el("p", {}, `${A[i]} is not among the agents listed here (the carriers above, agents with an answer and first public sources).`)); return; }
  const r = PR.rows[k], v = c => r[C[c]] ?? 0, n = PR.cols.length;
  const stats = `${v("n_posts")} posts; ${v("n_answers")} answers (no earlier public copy ${v("n_no_prior_copy")}, exposed ${v("n_exposed")}); ` +
    `first public source for ${v("answers_supplied")} answers; same-page relays ${v("relay_in")} in, ${v("relay_out")} out; ` +
    `task values first posted ${v("facts_originated")}, carried by ${v("agents_reached")} other agents`;
  const cols = el("div", {class: "cols"}); cols.append(linkList("Learned from", r[n + 1], r[n + 2]), linkList("Supplied", r[n + 3], r[n + 4]));
  const fam = r[C.family] >= 0 ? `family ${PR.families[r[C.family]]}; ` : "";
  card.replaceChildren(el("h4", {}, A[i]), el("div", {class: "stats"}, fam + stats), cols);
  if (scroll) card.scrollIntoView({block: "nearest"});
}
aq.addEventListener("input", () => { const s = aq.value.trim().toLowerCase(); if (!s) { al.replaceChildren(); return; }
  const exact = PR.rows.find(r => A[r[0]].toLowerCase() === s); if (exact) { showAgent(exact[0]); return; }
  const hit = PR.rows.filter(r => A[r[0]].toLowerCase().includes(s)).slice(0, 12);
  al.replaceChildren(...hit.map(r => { const li = el("li"), b = el("button", {type: "button"}, A[r[0]]);
    b.addEventListener("click", () => showAgent(r[0])); li.append(b); return li; })); });
card.append(el("p", {class: "stats"}, `${PR.rows.length} agents: the carriers above, every agent with an answer and every first public source. Type a label, or click an agent in the fact detail.`));
})();
"""
