# CLAUDE.md

Notes for working in this repo.

## What this is

A reading log that publishes itself. One folder per paper under `_papers/<slug>/`, holding
`index.md` (the review) and any marimo notebooks that rebuild the paper's idea from scratch. A
Jekyll site renders it, and `scripts/score.py` turns the reviews into a scoreboard.

Two audiences, one source: the front matter is for the site and the scorer, the body is for a
human reading the review later.

## Human reviews only

Every word of judgement on this site is written by a human. That is the feature, not a style
preference — an AI-written review would make the journal worthless, however good the prose.

**Never, on the user's behalf:**

- Write or draft any part of a review body — not `## Review`, `## Problem`, `## Overview`,
  `## Additional thoughts` or `## Feedback`, and not the `tldr:` either. Not as a "first draft
  to edit", not as filler, not by paraphrasing what they said in chat into a paragraph.
- Give opinions on a paper: whether it is good, whether the contribution is real, whether the
  experiments hold up, what rating it deserves, which archetype tag it earns. If asked "what do
  you think of this paper?", say this journal is the one place you don't answer that, and offer
  something from the list below instead.
- Set `rating`, `depth` or `read`, or edit the user's prose for anything beyond typos and
  broken markdown/LaTeX they asked you to fix.

A placeholder saying a section isn't written yet is correct; invented prose is not.

**Fair game — the LLM is a research assistant and a programmer, not a reviewer:**

- Literature search: finding the paper, prior work, follow-ups, the related papers worth
  reading next, and adding them to `_data/backlog.yml`.
- Summarising a paper *in chat* to help the user read it — what it claims, where the key
  equation is, what a term means. Factual, attributed to the paper, and never pasted into the
  review.
- Code: notebook implementations, plots, experiments, debugging, the scripts and the site.
  Notebook *code* and plain factual scaffolding (section titles, the notation translation
  table) are fine; the explanatory and interpretive prose in `mo.md` cells is the user's.
- Checking the user's work when asked: does this derivation follow, does this code match
  equation 4, is this claim about the paper accurate. Point at the error; don't rewrite the
  argument around it.
- Front matter mechanics: `title`, `ref`, `year`, `slug`, `notebooks`, `related`, and
  suggesting `topic` tags (archetype tags are a judgement — leave those to the user).

## The split that matters

**Python computes, Jekyll renders.** Every number on the site — XP, rank, streaks, badges,
backlog coverage — is produced by `scripts/score.py` into `_data/stats.yml` and committed. Liquid
templates only read `site.data.stats`; they never do arithmetic. If a number is wrong, fix the
script, not the template.

`_data/stats.yml` and `tags/*.md` are generated. Never hand-edit them; rerun `python
scripts/score.py`. CI reruns `score.py` itself before every build (and every Monday 00:15 UTC on
a schedule), so the site is scored as of the build date — streaks and the idle-week XP penalty
(`idle_penalty` in `_data/gamify.yml`) depend on today. The committed `stats.yml` is a snapshot
that keeps the scoreboard's history readable in the diff.

## The pile (an experiment, not on the site)

`scripts/pile.py` draws every published paper as a sheet of paper on a heap, seen from above,
and writes `assets/images/pile.svg`. Nothing links to it — no page embeds it, `score.py` does
not run it, and the svg is not committed. Run it by hand to look at the idea:

```sh
python scripts/pile.py
```

Two things are encoded, both about scale: one sheet per paper, and a sheet's thickness is the XP
its depth earns, so the heap stands as tall as the reading was deep. It is deterministic —
every position, angle and scribble is seeded from the paper's slug, never from the clock. Keep
it that way if it ever does get published; a random redraw would put thousands of lines of noise
into the diff. Layout is measured rather than guessed: sheets are placed so the next one can
never cover the previous title and byline, and the heap splits into several heaps side by side
once one gets taller than `HEAP_HEIGHT`.

## Adding a paper

```sh
python scripts/new_paper.py <slug> --from-backlog "<title fragment>"
# or fully by hand:
python scripts/new_paper.py <slug> --title "..." --ref <url> --year <year> --tags a,b
```

Then write the review and rerun `python scripts/score.py`.

## Front matter contract

Required: `layout: paper`, `slug` (must equal the folder name), `title`, `permalink:
/papers/<slug>/`, `read` (an ISO date — the streak maths needs a real `YYYY-MM-DD`), `depth`.

- `depth` is one of `read` / `rebuilt` / `extended` and is what earns XP — one read is 1 XP, the
  unit everything else is measured in. Set it honestly; the whole point of the scoring is that
  reading twenty papers should not outscore rebuilding two. There is no `skim`: a paper you only
  skimmed doesn't go in the journal. `rebuilt` implies there is a notebook in the folder.
- `rating` is 0–10 and decides accepted/rejected. **Leave it blank until it has been decided** —
  blank means "under review", which is deliberately different from 0.
- `tags` are slugs from `_data/tags.yml`. A tag not listed there still renders, it just falls
  into the "other" filter row — add it to `tags.yml` when it earns a label.
- `notebooks` is a list of `{file, what}`; the paper page links each one to GitHub.
- `related` is a list of other paper slugs.

Don't invent a `rating` or a `read` date on the user's behalf — both are their judgement, and a
wrong `read` date silently corrupts the streak.

## Review structure

Every review has the same shape. `scripts/score.py` warns when one doesn't.

```
## Review              required — the verdict; what's the contribution, does the paper show it?
## Problem             required — what's being solved and why it's hard; set up the notation
## Overview            required — the paper in my own words, the tutorial I wish I'd had
## Additional thoughts optional — my ideas, extensions, open questions (use `- [ ]`)
## Feedback            optional — what the authors should have done
```

Don't put an `# H1` title in the body; the layout renders `title` from the front matter. Don't
hand-write the rating either — the layout renders it from `rating:`. Break `## Overview` into
`###` subsections that follow the construction rather than the paper's own section order.

All of these sections are the user's to write — see *Human reviews only* above.

## Mining old conversations

`scripts/mine_conversations.py` scans `~/.claude/projects/**/*.jsonl` for paper identifiers and
ranks them by how the paper actually came up. The scoring rule that matters: a mention is
*diluted by how many other papers share the same message*, so a paper named alone counts far
more than one item in a thirty-paper reading list. `solo` mentions drive the depth guess, and
`extended` is never inferred.

Output goes to `.mined/report.yml`, which is **gitignored — it quotes private conversations and
this repo is public**. `--draft <key>` scaffolds `_papers/<slug>/index.md` with `draft: true`
and `published: false`, so it is invisible to both Jekyll and the scoreboard until promoted.

Drafts carry a `MINED EVIDENCE` comment block of raw material. `score.py` warns if that block
survives into a published review. Scrub it before flipping `draft: false`.

## The backlog is internal

`_data/backlog.yml` is the reading queue and is **not published** — no page renders it, and no
tag page lists queued entries. It exists for `new_paper.py --from-backlog` and for the coverage
line `score.py` prints in the terminal. Don't add a page that exposes it.

## Tags, not folders

Grouping is tags only; `_papers/` is deliberately flat. `_data/tags.yml` gives each tag a label
and a `kind`:

- `topic` — what the paper is about
- `archetype` — what kind of contribution it makes (hidden equivalence, shouldn't happen,
  clarity essay, reconception). This is the taxonomy that actually predicts whether a paper is
  worth the time, so prefer tagging it over adding another topic.
- `meta` — anything else worth filtering on

## Notebook conventions

- marimo `.py` format, not `.ipynb`. Never hand-edit the `@app.cell` scaffolding or
  `__generated_with` — let marimo write those.
- Cells return their exported names as a tuple; marimo's dependency graph relies on it. If you
  add a name a later cell uses, add it to the return.
- Prose cells are `mo.md(r"""...""")` with LaTeX. Heavy prose or setup cells use
  `@app.cell(hide_code=True)`.
- Structure is pedagogical: numbered sections or "rungs", each one idea plus a plot. Build from
  a familiar object (a coin, a Gaussian) up to the paper's construction.
- Translate the paper's notation into plain names and keep a translation table near the top.
- JAX on CPU with float64 — each notebook sets `JAX_PLATFORMS=cpu` and
  `jax.config.update("jax_enable_x64", True)` in its own imports cell. Keep that; some of the
  schedules are numerically nasty in float32.
- Plots use matplotlib directly. Don't save figures to disk.

## Site

Jekyll 4.2 with `jekyll-scholar`. `_papers` is a collection; `_posts` holds the older reviews
from before papers had folders and is left alone. Notebooks are excluded from the built site
(`*.py` in `_config.yml`) and linked to GitHub instead.

```sh
bundle exec jekyll serve      # local
```

## Don't

- Don't commit PDFs (gitignored) — put the link in `ref:`.
- Don't hand-edit `_data/stats.yml` or `tags/`.
- Don't add arithmetic to Liquid templates; it belongs in `score.py`.
- Don't refactor notebooks into shared modules. Duplication across notebooks is deliberate;
  each one is self-contained so it can be read alone.
- Don't ignore `*.png` in `.gitignore` — `assets/images/` is tracked.
