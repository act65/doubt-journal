#!/usr/bin/env python3
"""Draw the next paper to read out of _data/queue.yml.

A weighted draw, not a shuffle: recent interests come up more often, but every paper in the
queue keeps a real chance, so the thing you wrote down in 2018 and forgot can still surface.
The weights live in _data/sampling.yml and `--explain` shows the arithmetic.

    python scripts/next_paper.py                 # draw one and decide
    python scripts/next_paper.py --tag rl        # draw from one tag only
    python scripts/next_paper.py --explain       # show how the weight was built
    python scripts/next_paper.py --top 20        # the heaviest entries, no draw
    python scripts/next_paper.py --due           # what you have accepted and when it is due
    python scripts/next_paper.py --seed 7        # reproducible draw

Accepting a paper starts the clock: the review is due `review_deadline.days` later (see
_data/gamify.yml) and every finished week past that costs XP until its review is finished
(journal.is_finished). Skipping costs the paper a strike, and at `max_strikes` it leaves the queue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import journal
from journal import DATA, ROOT, as_date, normalise_title, normalise_url

SAMPLING = DATA / "sampling.yml"
GAMIFY = DATA / "gamify.yml"


# --------------------------------------------------------------------------- weighting


def weigh(entry: dict, cfg: dict, today: dt.date) -> tuple[float, list[tuple[str, float]]]:
    """The weight of one entry, plus a readable trace of where it came from."""
    interest_cfg = cfg.get("tag_interest") or {}
    boost_cfg = cfg.get("boost") or {}
    tags = entry.get("tags") or []

    why = []
    if tags:
        interest = max(float(interest_cfg.get(t, 1.0)) for t in tags)
        top = max(tags, key=lambda t: float(interest_cfg.get(t, 1.0)))
        why.append((f"interest ({top})", interest))
    else:
        interest = float(boost_cfg.get("untagged", 1.0))
        why.append(("interest (untagged)", interest))

    source_cfg = cfg.get("source_interest") or {}
    factor = float(source_cfg.get(entry.get("source"), 1.0))
    if factor != 1.0:
        interest *= factor
        why.append((f"source ({entry.get('source')})", factor))

    added = as_date(entry.get("added"))
    halflife = float(cfg.get("halflife_days") or 540)
    floor = float(cfg.get("floor") or 0.05)
    if added and halflife > 0:
        age = max((today - added).days, 0)
        recency = max(floor, 0.5 ** (age / halflife))
        why.append((f"recency ({age} days since last written down)", recency))
    else:
        recency = floor
        why.append(("recency (undated)", recency))

    weight = interest * recency

    strikes = int(entry.get("strikes") or 0)
    if strikes:
        penalty = float(cfg.get("strike_penalty") or 0.35) ** strikes
        weight *= penalty
        why.append((f"{strikes} strike{'' if strikes == 1 else 's'}", penalty))

    for flag in ("curated", "starred"):
        if entry.get(flag):
            factor = float(boost_cfg.get(flag, 1.0))
            weight *= factor
            why.append((flag, factor))
    if entry.get("flagged") and not entry.get("starred"):
        # A starred alert holding k papers starred all of them at once, so the star is worth
        # 1/k here. An alert with one paper in it never gets `flagged` — it gets `starred`.
        # And if the paper is starred outright, that already says it: don't pay twice.
        siblings = max(int(entry.get("siblings") or 1), 1)
        factor = 1 + (float(boost_cfg.get("starred", 1.0)) - 1) / siblings
        weight *= factor
        why.append((f"flagged (1 of {siblings} in a starred alert)", factor))
    if entry.get("slug"):
        factor = float(boost_cfg.get("in_progress", 1.0))
        weight *= factor
        why.append(("in progress", factor))
    if entry.get("weight"):
        factor = float(entry["weight"])
        weight *= factor
        why.append(("manual weight", factor))

    return weight, why


# --------------------------------------------------------------------------- the queue


def prune_finished(queue: list[dict], status: dict) -> list[dict]:
    """Drop anything that has since been reviewed. Keeps the queue honest without a flag."""
    done_slugs = {s for s, p in status.items() if p["finished"]}
    done_urls = {normalise_url(p["ref"]) for p in status.values() if p["finished"]} - {None}
    done_titles = {normalise_title(p["title"]) for p in status.values() if p["finished"]} - {None}
    kept, dropped = [], []
    for entry in queue:
        if (
            entry.get("slug") in done_slugs
            or normalise_url(entry.get("url")) in done_urls
            or normalise_title(entry.get("title")) in done_titles
        ):
            dropped.append(entry)
        else:
            kept.append(entry)
    for entry in dropped:
        print(f"  reviewed, removed from the queue: {entry.get('title')}")
    return kept


STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "on", "in", "to", "with", "without",
    "is", "are", "was", "be", "as", "at", "by", "from", "into", "via", "that", "this",
    "its", "your", "our", "their", "it", "we", "you", "can", "do", "does", "how", "what",
    "why", "when", "towards", "toward", "using", "use", "new", "more", "less", "than",
}


def slugify(title: str, used: set[str]) -> str:
    """A short folder name from a title. Only ever a suggestion — you are asked to confirm.

    Short tokens go first, because they are usually the maths: 'From r to Q*: Your Language
    Model is Secretly a Q-Function' is better off as language-model-secretly than as
    from-r-q-your.
    """
    import re

    words = re.sub(r"[^a-z0-9\s-]", "", (title or "").lower()).replace("-", " ").split()
    keep = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if len(keep) < 2:
        keep = [w for w in words if w not in STOPWORDS] or words
    base = "-".join(keep[:4]) or "paper"
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    return slug


# --------------------------------------------------------------------------- presentation


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }".replace(" ", "")


def describe(entry: dict, status: dict) -> str:
    lines = [f"  {entry.get('title') or '(no title — ' + (entry.get('url') or '?') + ')'}"]
    bits = [b for b in (entry.get("authors"), str(entry["year"]) if entry.get("year") else None) if b]
    bits.append(entry.get("source", "?"))
    if entry.get("starred"):
        bits.append("★")
    if entry.get("curated"):
        bits.append("curated")
    lines.append("  " + " · ".join(bits))
    if entry.get("alert"):
        lines.append(f"  found by your alert for \u201c{entry['alert']}\u201d")
    if entry.get("flagged"):
        n = entry.get("siblings") or "several"
        lines.append(f"  you starred that alert — one of its {n} papers was the one you wanted")
    if entry.get("url"):
        lines.append(f"  {entry['url']}")
    else:
        lines.append("  no link — try `python scripts/resolve_queue.py --retry`")
    if entry.get("tags"):
        lines.append("  tags: " + ", ".join(entry["tags"]))
    slug = entry.get("slug")
    if slug and slug in status:
        state = "published, not finished" if status[slug]["published"] else "draft"
        lines.append(f"  half-written already: _papers/{slug}/ ({state})")
    strikes = int(entry.get("strikes") or 0)
    if strikes:
        lines.append(f"  skipped {strikes} time{'' if strikes == 1 else 's'} before")
    return "\n".join(lines)


def show_due(queue: list[dict], status: dict, today: dt.date) -> None:
    taken = [e for e in queue if e.get("taken")]
    if not taken:
        print("nothing accepted — the clock isn't running on anything")
        return
    print(f"{len(taken)} paper{'' if len(taken) == 1 else 's'} accepted:\n")
    for entry in sorted(taken, key=lambda e: str(e["taken"].get("due"))):
        due = as_date(entry["taken"].get("due"))
        days = (due - today).days if due else None
        if days is None:
            when = "no due date"
        elif days < 0:
            when = f"OVERDUE by {-days} day{'' if days == -1 else 's'}"
        elif days == 0:
            when = "due today"
        else:
            when = f"{days} day{'' if days == 1 else 's'} left"
        slug = entry["taken"].get("slug") or entry.get("slug") or "?"
        note = "" if slug in status else "  (no folder — deleted?)"
        print(f"  {when:<22} {slug:<34} {(entry.get('title') or '')[:46]}{note}")


# --------------------------------------------------------------------------- actions


def accept(entry: dict, status: dict, days: int, today: dt.date, dry: bool) -> None:
    """Take the paper on: make sure a folder exists, then start the clock."""
    slug = entry.get("slug")
    if slug and slug in status:
        print(f"\n  _papers/{slug}/ already exists — finish what's there.")
    else:
        suggested = slugify(entry.get("title") or "", set(status))
        slug = suggested
        if sys.stdin.isatty():
            typed = input(f"  slug [{suggested}]: ").strip()
            slug = typed or suggested
        if dry:
            print(f"  would create _papers/{slug}/index.md")
        else:
            cmd = [
                sys.executable, str(ROOT / "scripts" / "new_paper.py"), slug,
                "--title", entry.get("title") or slug,
                "--ref", entry.get("url") or "",
                "--authors", entry.get("authors") or "",
                "--year", str(entry.get("year") or ""),
                "--tags", ",".join(entry.get("tags") or []),
                # A scaffold, not a claim to have read it: no `read` date is invented and
                # the folder stays invisible until it is published on purpose.
                "--draft",
                # The rescore happens once, after the deadline is in the queue.
                "--no-score",
            ]
            sys.stdout.flush()
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("  new_paper.py failed — the queue was left alone", file=sys.stderr)
                raise SystemExit(result.returncode)

    due = today + dt.timedelta(days=days)
    entry["slug"] = slug
    entry["taken"] = {"slug": slug, "on": today.isoformat(), "due": due.isoformat()}
    print(f"\n  accepted. The review is due {due.isoformat()} ({days} days).")
    print("  Score and answer every rubric axis, fill in `read`, flip draft/published —")
    print("  until all of that is done it counts as unfinished, and after the due date")
    print("  every finished week costs XP.")


def skip(entry: dict, cfg: dict) -> bool:
    """Add a strike. Returns True when that was the last one."""
    limit = int(cfg.get("max_strikes") or 2)
    entry["strikes"] = int(entry.get("strikes") or 0) + 1
    if entry["strikes"] >= limit:
        print(f"\n  strike {entry['strikes']} of {limit} — dropped from the queue.")
        return True
    print(f"\n  strike {entry['strikes']} of {limit}. Skip it again and it's gone.")
    return False


MENU = """
  [r] review it   start the clock, {days} days to finish the review
  [s] skip        strike {next_strike} of {max_strikes}{gone}
  [d] drop        remove it from the queue now
  [q] quit        change nothing
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", help="only draw from papers carrying this tag")
    parser.add_argument(
        "--source", help="only draw from this source (papers/backlog/keep/scholar/zotero)"
    )
    parser.add_argument(
        "--flagged",
        action="store_true",
        help="only draw from papers in Scholar alerts you starred",
    )
    parser.add_argument("--seed", type=int, help="seed the draw, to repeat one")
    parser.add_argument("--explain", action="store_true", help="show how the weight was built")
    parser.add_argument("--top", type=int, metavar="N", help="list the N heaviest entries and stop")
    parser.add_argument("--due", action="store_true", help="list accepted papers and their deadlines")
    parser.add_argument("--action", choices=["r", "s", "d"], help="answer the prompt up front")
    parser.add_argument("--dry-run", action="store_true", help="draw, but write nothing")
    args = parser.parse_args()

    today = dt.date.today()
    cfg = yaml.safe_load(SAMPLING.read_text(encoding="utf-8")) or {}
    gamify = yaml.safe_load(GAMIFY.read_text(encoding="utf-8")) or {}
    days = int((gamify.get("review_deadline") or {}).get("days") or 14)

    status = journal.load_paper_status()
    queue = journal.load_queue()
    if not queue:
        print("the queue is empty — run: python scripts/import_queue.py --help", file=sys.stderr)
        return 1

    # Papers reviewed since the last run leave the queue now, whatever else this run does —
    # otherwise a read-only `--top` would announce the removal without recording it.
    before = len(queue)
    queue = prune_finished(queue, status)
    if len(queue) != before and not args.dry_run:
        journal.write_queue(queue)

    if args.due:
        show_due(queue, status, today)
        return 0

    pool = [e for e in queue if not e.get("taken")]
    if args.tag:
        pool = [e for e in pool if args.tag in (e.get("tags") or [])]
    if args.source:
        pool = [e for e in pool if e.get("source") == args.source]
    if args.flagged:
        pool = [e for e in pool if e.get("flagged")]
    if not pool:
        print("nothing to draw from with those filters", file=sys.stderr)
        return 1

    weights = {id(e): weigh(e, cfg, today) for e in pool}

    if args.top:
        ranked = sorted(pool, key=lambda e: -weights[id(e)][0])[: args.top]
        total = sum(w for w, _ in weights.values())
        print(f"heaviest {len(ranked)} of {len(pool)}:\n")
        for entry in ranked:
            w = weights[id(entry)][0]
            print(f"  {w:7.3f}  {100 * w / total:5.2f}%  {(entry.get('title') or '')[:68]}")
        return 0

    rng = random.Random(args.seed)
    drawn = rng.choices(pool, weights=[weights[id(e)][0] for e in pool], k=1)[0]
    weight, why = weights[id(drawn)]
    total = sum(w for w, _ in weights.values())
    rank = 1 + sum(1 for w, _ in weights.values() if w > weight)

    print(f"\ndrawn from {len(pool)} papers "
          f"({100 * weight / total:.2f}% chance, {ordinal(rank)} heaviest)\n")
    print(describe(drawn, status))
    if args.explain:
        print("\n  weight " + " x ".join(f"{f:g}" for _, f in why) + f" = {weight:.4f}")
        for label, factor in why:
            print(f"    x {factor:<7.3f} {label}")

    limit = int(cfg.get("max_strikes") or 2)
    strikes = int(drawn.get("strikes") or 0)
    choice = args.action
    if choice is None:
        if not sys.stdin.isatty():
            print("\n  (not a terminal — nothing changed; pass --action r|s|d)")
            return 0
        print(MENU.format(
            days=days,
            next_strike=strikes + 1,
            max_strikes=limit,
            gone=" — and out of the queue" if strikes + 1 >= limit else "",
        ))
        choice = (input("  > ").strip().lower() or "q")[0]

    tombstone = None
    if choice == "r":
        accept(drawn, status, days, today, args.dry_run)
    elif choice == "s":
        if skip(drawn, cfg):
            queue = [e for e in queue if e is not drawn]
            tombstone = "struck-out"
    elif choice == "d":
        queue = [e for e in queue if e is not drawn]
        tombstone = "dropped"
        print("\n  dropped from the queue.")
    else:
        print("\n  nothing changed.")
        return 0

    if args.dry_run:
        print("  --dry-run: _data/queue.yml was not touched")
        return 0
    journal.write_queue(queue)
    if tombstone:
        # Without this the next import would find it in the Zotero export and hand it back.
        journal.record_dropped(drawn, tombstone, today)
        print(f"  recorded in _data/dropped.yml so it stays out")
    print(f"  _data/queue.yml updated — {len(queue)} papers left in the queue")
    # The deadline is now on record, so the scoreboard can account for it.
    sys.stdout.flush()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "score.py")], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
