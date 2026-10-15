#!/usr/bin/env python3
"""The handful of facts every script in here needs to agree on.

Not a framework — just the places where two scripts asking the same question and getting
different answers would quietly corrupt something: what counts as a finished review, when two
links are the same paper, and how ``_data/queue.yml`` is read and written.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "_papers"
DATA = ROOT / "_data"
QUEUE = DATA / "queue.yml"
DROPPED = DATA / "dropped.yml"
RUBRIC = DATA / "rubric.yml"


# --------------------------------------------------------------------------- scalars


def as_date(value) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def normalise_url(url: str | None) -> str | None:
    """Collapse the ways the same paper gets linked, so sources dedupe against each other.

    arXiv is the case that actually bites: /abs/2308.07037, /pdf/2308.07037v2 and
    arxiv.org/abs/2308.07037/ are all one paper. OpenReview is the same story with ?id=.
    """
    if not url:
        return None
    u = url.strip().lower().rstrip("/")
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    arxiv = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", u)
    if arxiv:
        return "arxiv:" + re.sub(r"v\d+$", "", arxiv.group(1)).removesuffix(".pdf")
    forum = re.search(r"openreview\.net/(?:forum|pdf|attachment)\?id=([^&]+)", u)
    if forum:
        return "openreview:" + forum.group(1)
    return u


def normalise_title(title: str | None) -> str | None:
    """Titles are the fallback key, so strip everything that varies between sources."""
    if not title:
        return None
    t = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return t or None


# --------------------------------------------------------------------------- papers


def read_front_matter(path: Path) -> tuple[dict, str]:
    """Split a Jekyll document into its YAML front matter and its body."""
    text = path.read_text(encoding="utf-8")
    # Split on the fence lines, not on any "---": a comment like `# --- notes` inside the
    # front matter would otherwise end it early.
    match = re.match(r"---[ \t]*\n(.*?)^---[ \t]*$\n?(.*)", text, re.S | re.M)
    if not match:
        raise ValueError(f"{path} has no front matter")
    fm, body = match.groups()
    return yaml.safe_load(fm) or {}, body


def is_published(fm: dict) -> bool:
    """Visible to Jekyll and to the scoreboard. Drafts are neither."""
    return not fm.get("draft") and fm.get("published") is not False


def load_rubric() -> dict:
    """The review rubric — axes, scale, rating rule. _data/rubric.yml is the user's to edit."""
    if not RUBRIC.exists():
        return {}
    return yaml.safe_load(RUBRIC.read_text(encoding="utf-8")) or {}


def answered(value) -> bool:
    """Blank, whitespace and list-of-blanks all mean "not written yet"."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(answered(v) for v in value)
    if isinstance(value, dict):
        return any(answered(v) for v in value.values())
    return True


def cell(value):
    """A table cell as written. YAML reads a bare yes/no as a boolean, so turn it back."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return value


def axis_entry(fm: dict, key: str) -> dict:
    return ((fm.get("review") or {}).get(key)) or {}


def axis_score(fm: dict, axis: dict, rubric: dict) -> int | None:
    """The axis score if it's a whole number on the scale, else None."""
    scale = rubric.get("scale") or {}
    value = axis_entry(fm, axis["key"]).get("score")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if scale.get("min", 1) <= value <= scale.get("max", 4) else None


def table_rows(fm: dict, axis: dict) -> list[dict]:
    """The rows of an axis's table that have every required column filled in."""
    table = axis.get("table")
    if not table:
        return []
    rows = axis_entry(fm, axis["key"]).get(table["key"]) or []
    required = table.get("required") or [c["key"] for c in table["columns"]]
    return [
        row for row in rows
        if isinstance(row, dict) and all(answered(cell(row.get(col))) for col in required)
    ]


def compute_rating(fm: dict, rubric: dict | None = None) -> float | None:
    """The rating is arithmetic on the axis scores, never written by hand.

    The mean score, rescaled from the axis scale onto 0..rating.max. The lowest possible score on
    any axis caps it at rating.floor_cap, so one fatal flaw can't be averaged away by the rest.
    None until every axis is scored.
    """
    if rubric is None:
        rubric = load_rubric()
    axes = rubric.get("axes") or []
    scores = [axis_score(fm, axis, rubric) for axis in axes]
    if not scores or any(s is None for s in scores):
        return None
    scale, cfg = rubric.get("scale") or {}, rubric.get("rating") or {}
    lo, hi, top = scale.get("min", 1), scale.get("max", 4), cfg.get("max", 10)
    value = round((sum(scores) / len(scores) - lo) / (hi - lo) * top, 1)
    if lo in scores and cfg.get("floor_cap") is not None:
        value = min(value, float(cfg["floor_cap"]))
    return value


def missing_core(fm: dict, rubric: dict | None = None) -> list[str]:
    """What the minimal review still lacks before it counts as finished.

    The core is the verdict and the reasons for it: a summary; every axis scored, and answered —
    in prose, or for an axis with a table, at least one complete row; and at least one plus and
    one minus somewhere. A score on its own isn't a review — it's a number nobody can check.
    """
    if rubric is None:
        rubric = load_rubric()
    axes = rubric.get("axes") or []
    missing = [] if answered(fm.get("summary")) else ["summary"]
    for axis in axes:
        key = axis["key"]
        if axis_score(fm, axis, rubric) is None:
            missing.append(f"review.{key}.score")
        if axis.get("table"):
            if not table_rows(fm, axis):
                missing.append(f"review.{key}.{axis['table']['key']}")
        elif not answered(axis_entry(fm, key).get("answer")):
            missing.append(f"review.{key}.answer")
    for side in ("plus", "minus"):
        if not any(answered(axis_entry(fm, a["key"]).get(side)) for a in axes):
            missing.append(f"a {side}")
    return missing


def is_finished(fm: dict, rubric: dict | None = None) -> bool:
    """A review is done when it is published and its core is complete — see `missing_core`.

    Until then it is "under review", which is deliberately not the same as a bad review, and a
    draft isn't on the site at all. This is the single definition of done: the queue drops
    these, the deadline check is satisfied by them, and only these get a verdict.
    """
    return is_published(fm) and not missing_core(fm, rubric)


def load_paper_status() -> dict[str, dict]:
    """Every folder under _papers/, by slug, with just enough to judge its state."""
    status = {}
    rubric = load_rubric()
    for index in sorted(PAPERS.glob("*/index.md")):
        fm, _ = read_front_matter(index)
        status[index.parent.name] = {
            "slug": index.parent.name,
            "title": fm.get("title"),
            "ref": fm.get("ref") or None,
            "authors": fm.get("authors") or None,
            "year": fm.get("year") or None,
            "tags": list(fm.get("tags") or []),
            "read": as_date(fm.get("read")),
            "rating": compute_rating(fm, rubric),
            "published": is_published(fm),
            "finished": is_finished(fm, rubric),
        }
    return status


# --------------------------------------------------------------------------- the queue


def load_queue() -> list[dict]:
    if not QUEUE.exists():
        return []
    return yaml.safe_load(QUEUE.read_text(encoding="utf-8")) or []


def queue_header() -> str:
    """The comment block at the top of queue.yml, so rewriting the data keeps the notes."""
    if not QUEUE.exists():
        return ""
    lines = []
    for line in QUEUE.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("- ") or (line.strip() and not line.startswith("#")):
            break
        lines.append(line)
    return "".join(lines)


def write_queue(entries: list[dict], header: str | None = None) -> None:
    body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=10_000)
    QUEUE.write_text((header if header is not None else queue_header()) + body, encoding="utf-8")


# --------------------------------------------------------------------------- tombstones

DROPPED_HEADER = """\
# Papers that left the reading queue without being reviewed — skipped twice, or dropped on
# sight. INTERNAL; nothing on the site reads it.
#
# This file exists so they stay gone. The Zotero and Keep exports still contain them, so
# without a record here the next `python scripts/import_queue.py` would hand them straight
# back. Delete a line to let a paper into the draw again.
#
#   why: struck-out  — ran out of strikes in scripts/next_paper.py
#   why: dropped     — removed on sight
"""


def load_dropped() -> list[dict]:
    if not DROPPED.exists():
        return []
    return yaml.safe_load(DROPPED.read_text(encoding="utf-8")) or []


def dropped_keys(entries: list[dict] | None = None) -> tuple[set[str], set[str]]:
    """(urls, titles), normalised, for matching against a fresh import."""
    entries = load_dropped() if entries is None else entries
    urls, titles = set(), set()
    for entry in entries:
        for key, bucket in ((normalise_url(entry.get("url")), urls),
                            (normalise_title(entry.get("title")), titles)):
            if key:
                bucket.add(key)
        alias = normalise_title(entry.get("alias"))
        if alias:
            titles.add(alias)
    return urls, titles


def record_dropped(entry: dict, why: str, when: dt.date) -> None:
    """Append one tombstone, keeping just enough to recognise the paper again."""
    entries = load_dropped()
    stone = {
        "title": entry.get("title"),
        "url": entry.get("url"),
        "why": why,
        "when": when.isoformat(),
        "strikes": int(entry.get("strikes") or 0),
    }
    if entry.get("alias"):
        stone["alias"] = entry["alias"]
    entries.append(stone)
    body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=10_000)
    DROPPED.write_text(DROPPED_HEADER + "\n" + body, encoding="utf-8")
