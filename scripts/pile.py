#!/usr/bin/env python3
"""Draw the pile.

Every published paper becomes one sheet of paper, strewn on a heap and drawn in line art to
``assets/images/pile.svg``. The image is a static file, committed like ``_data/stats.yml`` — no
JavaScript, no build-time rendering, nothing for Jekyll to do but serve it.

    python scripts/pile.py            # rewrite the svg
    python scripts/pile.py --check    # fail if it is stale (score.py --check does this too)

The point of the picture is scale, so the two things it encodes are the two things that scale:
one sheet per paper, and a sheet's *thickness* is the XP its depth earns. A `read` is a single
leaf, a `rebuilt` is a three-leaf wad, an `extended` is five. The heap gets taller by reading
and much taller by rebuilding, which is the same argument ``_data/gamify.yml`` makes in numbers.

Deterministic on purpose. Every jitter is seeded from the paper's slug, so a rerun never
reshuffles the heap and the committed svg only changes when the papers do — otherwise every
`score.py` would produce an unreadable diff.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "assets" / "images" / "pile.svg"

# ---------------------------------------------------------------- the look

# A sheet, in its own coordinates. Roughly A4's aspect so it reads as paper.
SHEET_W, SHEET_H = 168.0, 218.0
MARGIN = 15.0

# How far the plane is tilted away from the viewer. 1.0 would be straight down (no depth at
# all); much below 0.6 and the title text is squashed past legibility.
TILT = 0.70

# Screen pixels of thickness per XP. The whole heap's height is the sum of these.
PER_XP = 3.0

# How much of each sheet the next one may not cover, in screen pixels. A sheet asks for
# exactly its own header height and gets it while the heap is small; once there are more papers
# than SPREAD can seat, the allowance tightens and bylines start going under — which is the
# honest thing for the picture to do rather than growing without bound.
BAND_MAX, BAND_MIN, SPREAD = 52.0, 15.0, 620.0

# How tall one heap may get before the papers spill into another heap beside it, and how many
# heaps there can ever be — past that the sheets just crowd closer.
HEAP_HEIGHT, MAX_HEAPS = 900.0, 8

JITTER_X = 72.0  # sideways scatter — wide, so the heap is a heap and not a cascade
JITTER_X_TIGHT = 34.0  # ...but tighter once there are several heaps, to keep them apart
JITTER_Y = 4.0  # a little slop in the stacking, so it is a heap and not a fan
JITTER_DEG = 16.0  # rotation; more than this and the heap has to spread out to stay legible

PAGE_FILL, PAGE_EDGE = "#fffdf9", "#ded8cd"
SLAB_FILL, SLAB_EDGE = "#f0eae0", "#d8d1c5"
TITLE_INK, AUTHOR_INK = "#33302c", "#8d857a"
BODY_INK = "#ddd7cc"  # the obfuscated text
VERDICT_INK = {"accepted": "#5a3d8c", "rejected": "#aa0000", "pending": "#c3bcb0"}

TITLE_SIZE, AUTHOR_SIZE = 11.0, 8.0
TITLE_LINES = 3
FONT = "Garamond, 'EB Garamond', Georgia, serif"

# Garamond is narrow. Enough to wrap on; nothing here needs to be exact.
EM_WIDTH = 0.46


def _rng(slug: str) -> random.Random:
    """A generator seeded by the slug, so each paper keeps its position for good."""
    return random.Random(int(hashlib.sha256(slug.encode("utf-8")).hexdigest()[:12], 16))


def _wrap(text: str, size: float, width: float, max_lines: int) -> list[str]:
    limit = max(8, int(width / (size * EM_WIDTH)))
    words, lines, line = str(text or "").split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if len(trial) <= limit or not line:
            line = trial
        else:
            lines.append(line)
            line = word
            if len(lines) == max_lines:
                break
    if line and len(lines) < max_lines:
        lines.append(line)
    if not lines:
        return []
    # Ran out of room: mark the truncation rather than pretending the title ended there.
    consumed = sum(len(l.split()) for l in lines)
    if consumed < len(words) or len(lines[-1]) > limit:
        lines[-1] = lines[-1][: max(1, limit - 1)].rstrip(" ,;:") + "…"
    return lines


def _hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Monotone chain. Used to wrap a sheet and its shifted copy into one slab outline."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def half(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    return half(pts)[:-1] + half(reversed(pts))[:-1]


def _esc(text) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _header(paper: dict, verdict: str) -> tuple[list[str], float]:
    """The legible part of a sheet — title, byline, verdict rule — in local page coordinates.

    Returns the elements and the y they end at. The heap's spacing is driven by that height, so
    a three-line title gets more room than a one-line one and nothing important is ever buried.
    """
    text_w = SHEET_W - 2 * MARGIN
    out: list[str] = []

    y = MARGIN + TITLE_SIZE
    for line in _wrap(paper.get("title"), TITLE_SIZE, text_w, TITLE_LINES):
        out.append(
            f'<text x="{MARGIN:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{TITLE_SIZE}" '
            f'fill="{TITLE_INK}">{_esc(line)}</text>'
        )
        y += TITLE_SIZE * 1.18
    y -= TITLE_SIZE * 0.18  # the last baseline needs descender room, not a full line

    byline = " · ".join(str(v) for v in (paper.get("authors"), paper.get("year")) if v)
    line = _wrap(byline, AUTHOR_SIZE, text_w, 1) if byline else []
    if line:
        y += AUTHOR_SIZE + 2
        out.append(
            f'<text x="{MARGIN:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{AUTHOR_SIZE}" '
            f'font-style="italic" fill="{AUTHOR_INK}">{_esc(line[0])}</text>'
        )

    # The verdict, as the one piece of colour on the page.
    y += 7
    out.append(
        f'<rect x="{MARGIN:.1f}" y="{y:.1f}" width="34" height="1.6" fill="{VERDICT_INK[verdict]}"/>'
    )
    return out, y + 3.5


def _sheet(
    paper: dict, xp: int, verdict: str, cx: float, cy: float, theta: float
) -> tuple[list[str], list]:
    """One sheet: the slab of paper underneath it, the page, and what is written on it.

    Local page coordinates run (0,0)-(SHEET_W,SHEET_H) from the page's top-left corner. The
    matrix rotates that page flat on the ground plane and squashes it by TILT, so anything
    drawn in local coordinates — text included — lands on the page in perspective.
    """
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    a, b = cos_t, TILT * sin_t
    c, d = -sin_t, TILT * cos_t
    e = cx - (SHEET_W / 2) * cos_t + (SHEET_H / 2) * sin_t
    f = TILT * (cy - (SHEET_W / 2) * sin_t - (SHEET_H / 2) * cos_t)

    def to_screen(u: float, v: float) -> tuple[float, float]:
        return (a * u + c * v + e, b * u + d * v + f)

    corners = [to_screen(0, 0), to_screen(SHEET_W, 0), to_screen(SHEET_W, SHEET_H), to_screen(0, SHEET_H)]
    thickness = max(1.5, xp * PER_XP)
    slab = _hull(corners + [(x, y + thickness) for x, y in corners])

    rng = _rng(paper["slug"] + "/ink")
    matrix = f"matrix({a:.4f} {b:.4f} {c:.4f} {d:.4f} {e:.2f} {f:.2f})"

    out = [
        f'<polygon points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in slab)}" '
        f'fill="{SLAB_FILL}" stroke="{SLAB_EDGE}" stroke-width="1" vector-effect="non-scaling-stroke"/>',
        f'<g transform="{matrix}">',
        f'<title>{_esc(paper.get("title"))}</title>',
        f'<rect x="0" y="0" width="{SHEET_W}" height="{SHEET_H}" fill="{PAGE_FILL}" '
        f'stroke="{PAGE_EDGE}" stroke-width="1" vector-effect="non-scaling-stroke"/>',
    ]

    header, y = _header(paper, verdict)
    out += header
    y += 8

    # Everything below the fold is deliberately unreadable: two columns of ruled lines, the
    # shape of a paper without the words. Real body text at this size would just be noise.
    text_w = SHEET_W - 2 * MARGIN
    col_w = (text_w - 9) / 2
    for col in range(2):
        x0 = MARGIN + col * (col_w + 9)
        cursor = y
        while cursor < SHEET_H - MARGIN:
            if rng.random() < 0.09:  # a paragraph break
                cursor += 5
                continue
            width = col_w * rng.uniform(0.45, 1.0)
            out.append(
                f'<rect x="{x0:.1f}" y="{cursor:.1f}" width="{width:.1f}" height="1.7" '
                f'rx="0.8" fill="{BODY_INK}"/>'
            )
            cursor += 5.6
    out.append("</g>")
    return out, slab + corners


def render(papers: list[dict], gamify: dict) -> str:
    """Return the whole svg. `papers` is score.py's list; drafts are already filtered out."""
    depth_cfg = gamify.get("depth") or {}
    threshold = gamify.get("accept_threshold", 5)

    def verdict_of(paper) -> str:
        # The same rule as the scoreboard: no verdict until the review is finished, and the
        # rating (already computed by score.load_papers) only decides which one.
        import journal  # noqa: PLC0415 — on sys.path once score is imported

        if not journal.is_finished(paper):
            return "pending"
        return "accepted" if paper["rating"] >= threshold else "rejected"

    def sort_key(paper):
        return (paper.get("read_date") is None, paper.get("read_date"), paper["slug"])

    ordered = sorted(papers, key=sort_key)  # oldest at the bottom of the heap
    total_xp = sum(int((depth_cfg.get(p.get("depth")) or {}).get("xp", 0)) for p in ordered)

    # Two offsets from a sheet's centre, both on screen. Rotating a sheet lifts one corner and
    # drops the other, so a flat step would let the next sheet cut across the previous title.
    # Placing each sheet by its *highest* corner, and measuring the band it needs from the
    # *lower end of its own top edge*, is what makes the title and byline safe at any angle.
    def top_reach(theta: float) -> float:
        return TILT * (abs(math.sin(theta)) * SHEET_W / 2 + abs(math.cos(theta)) * SHEET_H / 2)

    def top_edge_low(theta: float) -> float:
        return TILT * (abs(math.sin(theta)) * SHEET_W / 2 - math.cos(theta) * SHEET_H / 2)

    def lay_out(heaps: int) -> tuple[list[str], list[tuple[float, float]]]:
        per_heap = math.ceil(len(ordered) / heaps)
        scatter = JITTER_X if heaps == 1 else JITTER_X_TIGHT
        # What each sheet may claim of its own header. It asks for all of it and gets it while
        # the heap is short; once a heap seats more than SPREAD can fit, the allowance tightens
        # and bylines start going under. Measured per heap rather than per paper, so it settles
        # at a floor instead of collapsing — the picture grows sideways, not blurrier.
        budget = BAND_MAX if per_heap <= 1 else max(BAND_MIN, min(BAND_MAX, SPREAD / per_heap))
        # Wide enough that neighbouring heaps never overlap, so none can cover another's titles.
        across = 2 * (SHEET_W / 2 + SHEET_H / 2 * math.sin(math.radians(JITTER_DEG)) + scatter) + 14

        body: list[str] = []
        points: list[tuple[float, float]] = []
        exposed = [i * 17.0 for i in range(heaps)]  # per heap: screen y already spoken for
        for i, paper in enumerate(ordered):
            heap = i // per_heap
            rng = _rng(paper["slug"])
            xp = int((depth_cfg.get(paper.get("depth")) or {}).get("xp", 0))
            theta = math.radians(rng.uniform(-JITTER_DEG, JITTER_DEG))
            # Jitter is one-sided: a sheet may sit lower than its slot, never higher, or it
            # would eat into the band the sheet above is relying on.
            centre_y = exposed[heap] + top_reach(theta) + rng.uniform(0, JITTER_Y)
            verdict = verdict_of(paper)
            drawn, corners = _sheet(
                paper, xp, verdict, heap * across + rng.uniform(-scatter, scatter), centre_y / TILT, theta
            )
            body += drawn
            points += corners
            # Leave exactly enough of this sheet showing for its own title and byline to clear.
            header = _header(paper, verdict)[1] * TILT
            exposed[heap] = centre_y + top_edge_low(theta) + min(budget, header)
        return body, points

    # One heap only gets so tall before the picture is a column rather than a pile. Rather than
    # guess what a sheet costs vertically — it depends on the title, the angle and the heap's
    # own crowding — lay the thing out, measure it, and split into more heaps until it fits.
    heaps = 1
    while True:
        body, points = lay_out(heaps)
        height = max((y for _, y in points), default=0.0) - min((y for _, y in points), default=0.0)
        want = math.ceil(height / HEAP_HEIGHT)
        if want <= heaps or heaps >= MAX_HEAPS:
            break
        heaps = max(heaps + 1, want)

    if not points:  # nothing published yet
        points = [(-120, -60), (120, 60)]

    pad = 14
    x0 = min(x for x, _ in points) - pad
    x1 = max(x for x, _ in points) + pad
    y0 = min(y for _, y in points) - pad
    y1 = max(y for _, y in points) + pad

    label = f"{len(ordered)} paper{'' if len(ordered) == 1 else 's'}, {total_xp} XP"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.1f} {y0:.1f} '
        f'{x1 - x0:.1f} {y1 - y0:.1f}" width="{x1 - x0:.0f}" height="{y1 - y0:.0f}" '
        f'role="img" aria-label="A heap of {label}, one sheet per paper.">\n'
        f"<title>The pile — {label}</title>\n"
        "<!-- Generated by scripts/pile.py — do not edit by hand. -->\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


def write(papers, gamify) -> bool:
    """Write the svg if it changed. Returns True if the file on disk was already current."""
    svg = render(papers, gamify)
    current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
    if current == svg:
        return True
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(svg, encoding="utf-8")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero if the svg is stale")
    args = parser.parse_args()

    import yaml  # noqa: PLC0415 — only needed when run on its own

    import score  # noqa: PLC0415 — score imports this module, so import it lazily

    gamify = yaml.safe_load((ROOT / "_data" / "gamify.yml").read_text(encoding="utf-8"))
    parts = yaml.safe_load((ROOT / "_data" / "parts.yml").read_text(encoding="utf-8")) or []
    papers = score.load_papers(parts, score.journal.load_rubric())

    if args.check:
        if render(papers, gamify) != (TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""):
            print("assets/images/pile.svg is stale — run `python scripts/score.py`", file=sys.stderr)
            return 1
        print("the pile is current")
        return 0

    write(papers, gamify)
    print(f"drew {len(papers)} sheets into {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
