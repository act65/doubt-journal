# CLAUDE.md

Notes for working in this repo.

## What this is

A reading log that publishes itself. One folder per paper under `_papers/<slug>/`, holding
`index.md` (the review) and any marimo notebooks that rebuild the paper's idea from scratch. A
Jekyll site renders it, and `scripts/score.py` turns the reviews into a scoreboard.

Two audiences, one source: the front matter is for the site and the scorer, the body is for a
human reading the review later.

## The split that matters

**Python computes, Jekyll renders.** Every number on the site — XP, rank, streaks, badges,
backlog coverage — is produced by `scripts/score.py` into `_data/stats.yml` and committed. Liquid
templates only read `site.data.stats`; they never do arithmetic. If a number is wrong, fix the
script, not the template.

`_data/stats.yml` and `tags/*.md` are generated. Never hand-edit them; rerun `python
scripts/score.py`. CI runs `score.py --check` and fails if they are stale.

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

- `depth` is one of `skim` / `read` / `rebuilt` / `extended` and is what earns XP. Set it
  honestly; the whole point of the scoring is that skimming twenty papers should not outscore
  rebuilding two. `rebuilt` implies there is a notebook in the folder.
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

**Never write the Review section on the user's behalf.** The verdict, the rating and the feedback
are the entire point of the journal being theirs. A placeholder saying it isn't written yet is
correct; invented prose is not.

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
