#!/usr/bin/env python3
"""Start a new review.

Creates ``_papers/<slug>/index.md`` from ``TEMPLATE.md`` and rescores the journal. The
review block comes from ``_data/rubric.yml``, so every axis is already there, blank.

    python scripts/new_paper.py neural-collapse \\
        --title "Prevalence of Neural Collapse" --ref https://arxiv.org/abs/2008.08186 \\
        --authors "Papyan, Han & Donoho" --year 2020 --tags shouldnt-happen,deep-learning

    # Or pull the metadata straight out of the reading queue by matching the title:
    python scripts/new_paper.py neural-collapse --from-queue "neural collapse"

    # Add one of the longer pages beside an existing review (see _data/parts.yml):
    python scripts/new_paper.py neural-collapse --part tutorial

Nothing here talks to the network — _data/queue.yml is the local copy of your queue. Note
that this does not start the two-week clock; `python scripts/next_paper.py` is what accepts a
paper and writes its deadline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import string
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "_papers"
TEMPLATE = ROOT / "TEMPLATE.md"
PART_TEMPLATE = ROOT / "TEMPLATE-part.md"
DATA = ROOT / "_data"
QUEUE = DATA / "queue.yml"


# Flipping both of these — and filling in `read` — is what publishes a review.
DRAFT_BLOCK = """\
# Scaffolded, not written. Invisible to Jekyll and to the scoreboard until you flip these
# two and fill in `read` with the date you actually finished it.
draft: true
published: false"""


def find_in_queue(needle: str) -> dict:
    entries = yaml.safe_load(QUEUE.read_text(encoding="utf-8")) or []
    needle = needle.lower()
    hits = [e for e in entries if needle in (e.get("title") or "").lower()]
    if not hits:
        raise SystemExit(f"nothing in the queue matches {needle!r}")
    if len(hits) > 1:
        listing = "\n".join(f"  - {h['title']}" for h in hits[:10])
        raise SystemExit(f"{len(hits)} queue entries match {needle!r}:\n{listing}")
    return hits[0]


def review_block() -> str:
    """One blank entry per rubric axis, with its question above it as a comment."""
    rubric = yaml.safe_load((DATA / "rubric.yml").read_text(encoding="utf-8")) or {}
    scale = rubric.get("scale") or {}
    lines = ["review:"]
    for axis in rubric.get("axes") or []:
        lines += [f"  # {axis['label']}: {axis['ask']}", f"  {axis['key']}:",
                  f"    score:        # {scale.get('min', 1)}–{scale.get('max', 4)}"]
        table = axis.get("table")
        if table:
            lines.append(f"    {table['key']}:")
            for i, column in enumerate(table["columns"]):
                hint = f"        # {' | '.join(column['values'])}" if column.get("values") else ""
                lines.append(f"      {'- ' if i == 0 else '  '}{column['key']}:{hint}")
        else:
            lines.append("    answer:")
        lines += ["    plus:", "      -", "    minus:", "      -"]
    return "\n".join(lines)


def scaffold_part(slug: str, key: str) -> int:
    """Add tutorial.md / discussion.md / thoughts.md beside an existing review."""
    parts = yaml.safe_load((DATA / "parts.yml").read_text(encoding="utf-8")) or []
    part = next((p for p in parts if p["key"] == key), None)
    if part is None:
        raise SystemExit(f"no part {key!r} in _data/parts.yml — one of: "
                         + ", ".join(p["key"] for p in parts))
    index = PAPERS / slug / "index.md"
    if not index.exists():
        raise SystemExit(f"{index.relative_to(ROOT)} doesn't exist — start the review first")
    target = index.parent / f"{key}.md"
    if target.exists():
        raise SystemExit(f"{target.relative_to(ROOT)} already exists")
    fm = re.match(r"---[ \t]*\n(.*?)^---[ \t]*$", index.read_text(encoding="utf-8"), re.S | re.M)
    title = (yaml.safe_load(fm.group(1)) if fm else {}).get("title") or slug
    target.write_text(
        string.Template(PART_TEMPLATE.read_text(encoding="utf-8")).substitute(
            slug=slug,
            part=key,
            title=json.dumps(f"{part['label']} · {title}", ensure_ascii=False),
            prompt=f"<!-- {part['blurb']} -->",
        ),
        encoding="utf-8",
    )
    print(f"created {target.relative_to(ROOT)} — served at /papers/{slug}/{key}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slug", help="short lowercase folder name, e.g. 'bfn'")
    parser.add_argument("--title")
    parser.add_argument("--authors", default="")
    parser.add_argument("--year", default="")
    parser.add_argument("--venue", default="arXiv")
    parser.add_argument("--ref", default="", help="link to the paper")
    parser.add_argument(
        "--depth",
        default="read",
        choices=["read", "rebuilt", "extended"],
        help="how carefully you read it — drives the XP (default: read)",
    )
    parser.add_argument("--tags", default="", help="comma-separated tag slugs; see _data/tags.yml")
    parser.add_argument("--read", default="", help="date finished, YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--draft",
        action="store_true",
        help="scaffold only: draft/unpublished, and no `read` date invented for you",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="skip the rescore (for callers that will run scripts/score.py themselves)",
    )
    parser.add_argument(
        "--from-queue",
        "--from-backlog",
        metavar="TITLE",
        dest="from_queue",
        help="fill metadata from a _data/queue.yml entry matching this title fragment",
    )
    parser.add_argument(
        "--part",
        metavar="KEY",
        help="add a longer page (tutorial, discussion, thoughts) beside an existing review",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.slug):
        raise SystemExit("slug must be lowercase letters, digits and hyphens")
    if args.part:
        return scaffold_part(args.slug, args.part)

    folder = PAPERS / args.slug
    target = folder / "index.md"
    if target.exists():
        raise SystemExit(f"{target.relative_to(ROOT)} already exists")

    title, authors, year, ref = args.title, args.authors, args.year, args.ref
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    if args.from_queue:
        entry = find_in_queue(args.from_queue)
        title = title or entry.get("title") or ""
        authors = authors or entry.get("authors") or ""
        year = year or (entry.get("year") or "")
        ref = ref or entry.get("url") or ""
        for tag in entry.get("tags") or []:
            if tag not in tags and tag != "other":
                tags.append(tag)
        print(f"from the queue: {entry['title']}")

    if not title:
        raise SystemExit("need a --title (or --from-queue)")

    # Titles routinely contain ':' — unquoted, YAML reads that as a mapping and the front
    # matter fails to parse.
    def quoted(value):
        return json.dumps(str(value), ensure_ascii=False) if value not in (None, "") else ""

    filled = string.Template(TEMPLATE.read_text(encoding="utf-8")).substitute(
        slug=args.slug,
        title=quoted(title),
        authors=quoted(authors),
        year=year,
        venue=args.venue,
        ref=ref,
        # A draft hasn't been read yet, and a wrong `read` date quietly corrupts the
        # streak — so it is left blank for the one person who knows it.
        read=args.read or ("" if args.draft else dt.date.today().isoformat()),
        depth=args.depth,
        tags=", ".join(tags),
        review=review_block(),
    )

    if args.draft:
        filled, swaps = re.subn(
            r"(?m)^related: \[\][ \t]*$",
            "related: []\n" + DRAFT_BLOCK,
            filled,
            count=1,
        )
        if not swaps:
            raise SystemExit("TEMPLATE.md has no `related: []` line to anchor the draft flags to")

    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(filled, encoding="utf-8")
    print(f"created {target.relative_to(ROOT)}")

    if not args.no_score:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "score.py")], check=False)
    print(f"\nnow score and answer each axis, then: python scripts/score.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
