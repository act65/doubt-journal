#!/usr/bin/env python3
"""Fill in the gaps the exports left behind: titles without links, links without titles.

This is the only script in the repo that talks to the network. It asks the arXiv API — by ID
where a queue entry already has an arXiv link, by title where it doesn't — and writes whatever
it can confirm back into ``_data/queue.yml``.

    python scripts/resolve_queue.py                      # everything still missing
    python scripts/resolve_queue.py --source keep        # just the Keep lists
    python scripts/resolve_queue.py --limit 50           # one bite at a time
    python scripts/resolve_queue.py --retry              # try the ones that failed before

A title match is only accepted when the title arXiv returns is essentially the one we asked
for, because a plausible-but-wrong link is worse than no link. Entries that can't be confirmed
are marked `unresolved: true` and skipped next time unless you pass --retry.

arXiv is not the only place a paper lives. `--pmlr v97` loads one PMLR volume's index — v97
is ICML 2019, which is where the oldest Keep list came from — and matches the leftovers against
it in a single request.

OpenReview sits behind a browser challenge, so its forum links cannot be resolved here; those
entries keep their URL and stay titleless until someone pastes a title in by hand.
"""

from __future__ import annotations

import argparse
import difflib
import html
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "_data" / "queue.yml"

API = "https://export.arxiv.org/api/query"
UA = "doubt-journal/0.1 (reading-queue importer; +https://github.com/act65/doubt-journal)"
ATOM = "{http://www.w3.org/2005/Atom}"

# arXiv asks for one request every three seconds. Being a good citizen is cheaper than
# getting blocked halfway through 600 lookups.
COURTESY_SECONDS = 3.1
ID_BATCH = 50
# How close the returned title has to be before the link is believed.
MATCH_RATIO = 0.93

ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#v]+(?:v\d+)?)", re.I)


def arxiv_id(url: str | None) -> str | None:
    m = ARXIV_ID.search(url or "")
    if not m:
        return None
    return re.sub(r"v\d+$", "", m.group(1)).removesuffix(".pdf")


def normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def fetch(params: dict) -> list[dict]:
    """One arXiv query -> [{id, title, authors, year}]. Retries the transient failures."""
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=40) as response:
                raw = response.read()
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == 3:
                print(f"    ! giving up on this query: {exc}", file=sys.stderr)
                return []
            time.sleep(COURTESY_SECONDS * (attempt + 2))
    else:
        return []

    try:
        tree = ET.fromstring(raw)
    except ET.ParseError as exc:
        print(f"    ! unparseable response: {exc}", file=sys.stderr)
        return []

    out = []
    for entry in tree.findall(f"{ATOM}entry"):
        ident = (entry.findtext(f"{ATOM}id") or "").strip()
        title = re.sub(r"\s+", " ", (entry.findtext(f"{ATOM}title") or "").strip())
        if not ident or not title:
            continue
        names = [
            (a.findtext(f"{ATOM}name") or "").strip()
            for a in entry.findall(f"{ATOM}author")
        ]
        published = (entry.findtext(f"{ATOM}published") or "")[:4]
        out.append(
            {
                "id": re.sub(r"v\d+$", "", ident.rsplit("/", 1)[-1]),
                "url": "https://arxiv.org/abs/" + re.sub(r"v\d+$", "", ident.rsplit("/", 1)[-1]),
                "title": title,
                "authors": names,
                "year": int(published) if published.isdigit() else None,
            }
        )
    return out


def surnames(names: list[str]) -> str | None:
    """Match the house style used elsewhere: up to three surnames, then 'et al.'"""
    tail = [n.split()[-1] for n in names if n.strip()]
    if not tail:
        return None
    if len(tail) == 1:
        return tail[0]
    if len(tail) == 2:
        return f"{tail[0]} & {tail[1]}"
    if len(tail) == 3:
        return f"{tail[0]}, {tail[1]} & {tail[2]}"
    return f"{tail[0]} et al."


def titles_by_id(entries: list[dict]) -> int:
    """Entries that have an arXiv link but no title. Cheap: 50 IDs per request."""
    wanted = {}
    for entry in entries:
        ident = arxiv_id(entry.get("url"))
        if ident and not entry.get("title"):
            wanted.setdefault(ident, []).append(entry)
    if not wanted:
        return 0

    ids = list(wanted)
    filled = 0
    for start in range(0, len(ids), ID_BATCH):
        chunk = ids[start : start + ID_BATCH]
        print(f"  arxiv ids {start + 1}-{start + len(chunk)} of {len(ids)}")
        for found in fetch({"id_list": ",".join(chunk), "max_results": len(chunk)}):
            for entry in wanted.get(found["id"], []):
                entry["title"] = found["title"]
                entry["authors"] = entry.get("authors") or surnames(found["authors"])
                entry["year"] = entry.get("year") or found["year"]
                entry.pop("unresolved", None)
                filled += 1
        time.sleep(COURTESY_SECONDS)
    return filled


def urls_by_title(entries: list[dict], limit: int | None) -> tuple[int, int]:
    """Entries that have a title but no link. One request each, and only believed on a
    near-exact title match."""
    todo = [e for e in entries if e.get("title") and not e.get("url")]
    if limit:
        todo = todo[:limit]
    found = missed = 0
    for i, entry in enumerate(todo, 1):
        title = entry["title"]
        results = fetch(
            {
                "search_query": f'ti:"{title}"',
                "max_results": 4,
                "sortBy": "relevance",
            }
        )
        want = normalise(title)
        best = None
        for candidate in results:
            ratio = difflib.SequenceMatcher(None, want, normalise(candidate["title"])).ratio()
            if ratio >= MATCH_RATIO and (best is None or ratio > best[0]):
                best = (ratio, candidate)
        if best:
            _, hit = best
            entry["url"] = hit["url"]
            # Keep the title we searched on. Re-running the importer re-reads the raw export,
            # and this is the only thread back to the line it came from.
            if normalise(hit["title"]) != want:
                entry["alias"] = title
            entry["title"] = hit["title"]
            entry["authors"] = entry.get("authors") or surnames(hit["authors"])
            entry["year"] = entry.get("year") or hit["year"]
            entry.pop("unresolved", None)
            found += 1
            mark = "->"
        else:
            entry["unresolved"] = True
            missed += 1
            mark = " ?"
        print(f"  [{i}/{len(todo)}] {mark} {title[:70]}")
        time.sleep(COURTESY_SECONDS)
    return found, missed


# --------------------------------------------------------------------------- PMLR

PMLR = "https://proceedings.mlr.press"
# One record per paper in a volume index: title, author list, link to the abstract page.
PMLR_ENTRY = re.compile(
    r'<p class="title">(?P<title>.*?)</p>.*?'
    r'<span class="authors">(?P<authors>.*?)</span>.*?'
    r'\[<a href="(?P<url>[^"]*?\.html)">abs</a>\]',
    re.S,
)


def pmlr_volume(volume: str) -> list[dict]:
    """Every paper in one PMLR volume, e.g. 'v97' for ICML 2019. One request."""
    url = f"{PMLR}/{volume}/"
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read().decode("utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"  ! could not load {url}: {exc}", file=sys.stderr)
        return []

    papers = []
    for m in PMLR_ENTRY.finditer(raw):
        title = html.unescape(re.sub(r"<[^>]+>", "", m.group("title"))).strip()
        people = [
            a.strip()
            for a in html.unescape(m.group("authors").replace("&nbsp;", " ")).split(",")
            if a.strip()
        ]
        if title:
            papers.append({"title": title, "authors": people, "url": m.group("url")})
    return papers


def urls_by_pmlr(entries: list[dict], volume: str) -> tuple[int, int]:
    """Match the entries that arXiv could not place against a PMLR volume index."""
    papers = pmlr_volume(volume)
    if not papers:
        return 0, 0
    print(f"  {len(papers)} papers in PMLR {volume}")
    index = {normalise(p["title"]): p for p in papers}

    todo = [e for e in entries if not e.get("url")]
    found = missed = 0
    for entry in todo:
        want = normalise(entry["title"] or "")
        if not want:
            continue
        best = None
        for key, paper in index.items():
            ratio = difflib.SequenceMatcher(None, want, key).ratio()
            if ratio >= MATCH_RATIO and (best is None or ratio > best[0]):
                best = (ratio, paper)
        if not best:
            missed += 1
            print(f"   ? {entry['title'][:70]}")
            continue
        _, hit = best
        entry["url"] = hit["url"]
        if normalise(hit["title"]) != want:
            entry["alias"] = entry["title"]
        entry["title"] = hit["title"]
        entry["authors"] = entry.get("authors") or surnames(hit["authors"])
        entry["year"] = entry.get("year") or pmlr_year(volume)
        entry.pop("unresolved", None)
        found += 1
        print(f"  -> {hit['title'][:70]}")
    return found, missed


# Volumes this repo has actually needed. Anything else leaves `year` alone.
PMLR_YEARS = {"v97": 2019}


def pmlr_year(volume: str) -> int | None:
    return PMLR_YEARS.get(volume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="only entries from this source (papers/backlog/keep/zotero)")
    parser.add_argument("--limit", type=int, help="stop after this many title lookups")
    parser.add_argument("--retry", action="store_true", help="also retry entries marked unresolved")
    parser.add_argument("--titles-only", action="store_true", help="skip the title->URL lookups")
    parser.add_argument(
        "--pmlr",
        metavar="VOLUME",
        help="also match leftovers against a PMLR volume index, e.g. v97 for ICML 2019",
    )
    args = parser.parse_args()

    header = []
    for line in QUEUE.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("- ") or line.strip() and not line.startswith("#"):
            break
        header.append(line)
    queue = yaml.safe_load(QUEUE.read_text(encoding="utf-8")) or []

    pool = [e for e in queue if not args.source or e.get("source") == args.source]
    if not args.retry:
        pool = [e for e in pool if not e.get("unresolved")]
    print(f"{len(pool)} candidates from {len(queue)} queue entries")

    filled = titles_by_id(pool)
    found = missed = 0
    if not args.titles_only:
        found, missed = urls_by_title(pool, args.limit)
    pmlr_found = pmlr_missed = 0
    if args.pmlr:
        pmlr_found, pmlr_missed = urls_by_pmlr(pool, args.pmlr)

    # The pool holds references into `queue`, so the edits are already in place.
    body = yaml.safe_dump(queue, sort_keys=False, allow_unicode=True, width=10_000)
    QUEUE.write_text("".join(header) + body, encoding="utf-8")

    print(f"\n{filled} titles filled from arXiv IDs")
    print(f"{found} links found by title, {missed} not confidently matched")
    if args.pmlr:
        print(f"{pmlr_found} links found in PMLR {args.pmlr}, {pmlr_missed} not matched")
    still = sum(1 for e in queue if not e.get("url")), sum(1 for e in queue if not e.get("title"))
    print(f"queue now: {still[0]} without a URL, {still[1]} without a title")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
