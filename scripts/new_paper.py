#!/usr/bin/env python3
"""Start a new review.

Creates ``_papers/<slug>/index.md`` from ``TEMPLATE.md`` and rescores the journal.

    python scripts/new_paper.py neural-collapse \\
        --title "Prevalence of Neural Collapse" --ref https://arxiv.org/abs/2008.08186 \\
        --authors "Papyan, Han & Donoho" --year 2020 --tags shouldnt-happen,deep-learning

    # Or pull the metadata straight off the backlog by matching the title:
    python scripts/new_paper.py neural-collapse --from-backlog "neural collapse"

Nothing here talks to the network — the backlog is the local copy of your queue.
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
BACKLOG = ROOT / "_data" / "backlog.yml"


def find_in_backlog(needle: str) -> dict:
    entries = yaml.safe_load(BACKLOG.read_text(encoding="utf-8")) or []
    needle = needle.lower()
    hits = [e for e in entries if needle in (e.get("title") or "").lower()]
    if not hits:
        raise SystemExit(f"nothing on the backlog matches {needle!r}")
    if len(hits) > 1:
        listing = "\n".join(f"  - {h['title']}" for h in hits[:10])
        raise SystemExit(f"{len(hits)} backlog entries match {needle!r}:\n{listing}")
    return hits[0]


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
        choices=["skim", "read", "rebuilt", "extended"],
        help="how carefully you read it — drives the XP (default: read)",
    )
    parser.add_argument("--tags", default="", help="comma-separated tag slugs; see _data/tags.yml")
    parser.add_argument("--read", default="", help="date finished, YYYY-MM-DD (default: today)")
    parser.add_argument("--from-backlog", metavar="TITLE", help="fill metadata from a backlog entry")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.slug):
        raise SystemExit("slug must be lowercase letters, digits and hyphens")

    folder = PAPERS / args.slug
    target = folder / "index.md"
    if target.exists():
        raise SystemExit(f"{target.relative_to(ROOT)} already exists")

    title, authors, year, ref = args.title, args.authors, args.year, args.ref
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    if args.from_backlog:
        entry = find_in_backlog(args.from_backlog)
        title = title or entry.get("title") or ""
        authors = authors or entry.get("authors") or ""
        year = year or (entry.get("year") or "")
        ref = ref or entry.get("url") or ""
        if entry.get("tag") and entry["tag"] not in tags and entry["tag"] != "other":
            tags.append(entry["tag"])
        print(f"from backlog: {entry['title']}")

    if not title:
        raise SystemExit("need a --title (or --from-backlog)")

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
        read=args.read or dt.date.today().isoformat(),
        depth=args.depth,
        tags=", ".join(tags),
    )

    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(filled, encoding="utf-8")
    print(f"created {target.relative_to(ROOT)}")

    subprocess.run([sys.executable, str(ROOT / "scripts" / "score.py")], check=False)
    print(f"\nnow write the review, then: python scripts/score.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
