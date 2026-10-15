# CLAUDE.md

Notes for working in this repo.

## What this is

A reading log that publishes itself. One folder per paper under `_papers/<slug>/`, holding
`index.md` (the minimal review), optional longer pages beside it (`tutorial.md`,
`discussion.md`, `thoughts.md`), and any marimo notebooks that rebuild the paper's idea from scratch. A
Jekyll site renders it, and `scripts/score.py` turns the reviews into a scoreboard.

What to read next is drawn, not chosen: `_data/queue.yml` is a few thousand papers scraped out
of Zotero, Google Keep, Google Scholar alert mail and the repo's own half-written folders, and
`scripts/next_paper.py` samples one with a thumb on the scale for recent interests.

Two audiences, one source: the front matter is for the site and the scorer, the body is for a
human reading the review later.

## Human reviews only

Every word of judgement on this site is written by a human. That is the feature, not a style
preference — an AI-written review would make the journal worthless, however good the prose.

**Never, on the user's behalf:**

- Write or draft any part of a review — not the `index.md` body, not `summary`, `takeaways`, any
  axis `answer`, `plus` or `minus`, any row of the claims or assumptions tables, and not a word
  of `tutorial.md`,
  `discussion.md` or `thoughts.md`. Not as a "first draft to edit", not as filler, not by
  paraphrasing what they said in chat into a paragraph.
- Give opinions on a paper: whether it is good, whether the contribution is real, whether the
  experiments hold up, what rating it deserves, which archetype tag it earns. If asked "what do
  you think of this paper?", say this journal is the one place you don't answer that, and offer
  something from the list below instead.
- Set an axis `score`, `confidence`, `depth` or `read`, or edit the user's prose for anything beyond typos and
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
queue size — is produced by `scripts/score.py` into `_data/stats.yml` and committed. Liquid
templates only read `site.data.stats`; they never do arithmetic. If a number is wrong, fix the
script, not the template.

`_data/stats.yml` and `tags/*.md` are generated. Never hand-edit them; rerun `python
scripts/score.py`. `_data/queue.yml` and `_data/dropped.yml` are machine-managed but *are*
meant to be edited by hand when needed — they are data, not output.

**The scoreboard is private.** XP, rank, streaks, journal badges and the penalties are for the
user only. `stats.html` carries `scoreboard: true` and `_plugins/scoreboard.rb` drops it unless
the site config sets `scoreboard: true`, which only `_config.scoreboard.yml` does (local:
`bundle exec jekyll serve --config _config.yml,_config.scoreboard.yml`). Anything else
scoreboard-ish in a template goes inside `{% if site.scoreboard %}` — XP, but also a paper's
`depth` chip and `read` date, which are about the reader rather than the paper. Public pages may
read the per-paper parts of `stats` — status, ratings, paper badges, tag counts — never XP.

CI does **not** run `score.py`; the public build reads the committed `stats.yml`. So it must be
regenerated and committed with every review change — `.githooks/pre-commit` does that (enabled
with `git config core.hooksPath .githooks`): when a commit touches `_papers/**/*.md`,
`_data/*.yml` or the scorer, it reruns `score.py` and stages `stats.yml` and `tags/`. The public fields depend only on the papers;
the date-dependent ones (streaks, `idle_penalty`, `review_deadline`) only show on the local
scoreboard, which is current whenever `score.py` was last run.

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

Normally you don't choose — you draw:

```sh
python scripts/next_paper.py          # draw one, then [r]eview / [s]kip / [d]rop
```

Accepting scaffolds the folder and starts a two-week clock. By hand:

```sh
python scripts/new_paper.py <slug> --from-queue "<title fragment>"
# or fully by hand:
python scripts/new_paper.py <slug> --title "..." --ref <url> --year <year> --tags a,b
```

Then write the review and rerun `python scripts/score.py`.

Note what `new_paper.py --draft` does and does not do: it leaves `read:` **blank** and sets
`draft: true` / `published: false`. `next_paper.py` always uses it, because accepting a paper
is not the same as having read it and a wrong `read` date silently corrupts the streak. The
`read` date is the user's to fill in when they finish.

## Front matter contract

Required: `layout: paper`, `slug` (must equal the folder name), `title`, `permalink:
/papers/<slug>/`, `read` (an ISO date — the streak maths needs a real `YYYY-MM-DD`), `depth`.

- `depth` is one of `reviewed` / `rebuilt` / `extended` and is what earns XP — one review is 1 XP, the
  unit everything else is measured in. Set it honestly; the whole point of the scoring is that
  reading twenty papers should not outscore rebuilding two. There is no `skim`: a paper you only
  skimmed doesn't go in the journal. `rebuilt` implies there is a notebook in the folder.
- `review` holds one entry per axis in `_data/rubric.yml` — `{score, answer, plus, minus}`, with
  the rigour and scope axes swapping `answer` for a table (`claims`, `assumptions`). Scores are
  1–4. All the user's to write. `confidence` (1–5) sits beside it and isn't part of the rating.
- **There is no `rating:` key.** The rating is computed by `journal.compute_rating`: the mean
  axis score rescaled to 0–10, capped at `floor_cap` if any axis scores 1. It is the verdict —
  there is no accepted/rejected — and it only shows once the review is finished.
  `low_rating` in `gamify.yml` is what Hidden Gem counts as low, nothing more. `rating_before_rubric:` in some
  reviews is the user's old hand rating, kept for comparison; nothing reads it.
- `takeaways` is a list of short markdown items.
- `tags` are slugs from `_data/tags.yml`. A tag not listed there still renders, it just falls
  into the "other" filter row — add it to `tags.yml` when it earns a label.
- `notebooks` is a list of `{file, what}`; the paper page links each one to GitHub.
- `related` is a list of other paper slugs.

Don't invent an axis score or a `read` date on the user's behalf — both are their judgement, and a
wrong `read` date silently corrupts the streak.

## Review structure

A paper's permalink is the **minimal review**, and the longer writing lives on part pages
beside it. Reading order and writing order differ on purpose: the page shows the verdict
first, but you write the axes first, while reading, and the summary last.

```
_papers/<slug>/index.md       /papers/<slug>/            the minimal review — required
_papers/<slug>/tutorial.md    /papers/<slug>/tutorial/   extended summary: problem, setting, construction
_papers/<slug>/discussion.md  /papers/<slug>/discussion/ long-form argument; `## Feedback` for the authors
_papers/<slug>/thoughts.md    /papers/<slug>/thoughts/   extensions, open questions (`- [ ]`)
_papers/<slug>/notes.md       never published            private scratch, pasted chats
```

`index.md` is almost all front matter — `summary`, `confidence`, `review`, `takeaways` — and its
body is an optional short verdict paragraph. The layout renders the score row, then one block
per axis (question, score, answer or table, + and − points), then links whichever part pages
exist. `/rubric/` renders the rubric itself.

- **The rubric lives in `_data/rubric.yml`** and is the user's to edit: the axes, their
  questions, the table columns, the scale, the cap, and the `ideal` paragraph (theirs to
  write — it currently quotes WRITING-HEURISTICS.md §0). The axes are meant to be orthogonal
  and to be about the paper, never the reader — why (the problem), idea (how principled the
  solution is), contribution (the delta to prior work), rigour (claims vs evidence), scope
  (when it applies), clarity (whether why, idea and evidence form one simple narrative). What a paper gave the reader goes
  in `takeaways`. The layout, `/rubric/`, `score.py` and `new_paper.py` all read it. Don't add, drop or
  reword axes on the user's behalf.
- **Paper badges** (`PAPER_BADGES` / `paper_badges` in `score.py`) are rules on one paper's
  axis scores, rating, confidence and tables — Best Paper, Hidden Gem, Jekyll & Hyde, … Only
  finished reviews earn them; `score.py` writes the keys to `stats.reviews[slug].badges` and
  the catalogue to `stats.paper_badges`. They read axes by key, so renaming an axis in
  `rubric.yml` silences the badges that use it. They are public: `/badges/` (`badges.html`)
  lists the catalogue and who earned each, and every stamp on a paper links there. Each has a
  16×16 pixel icon at `assets/badges/<key>.svg`, generated by `python scripts/badge_icons.py`
  from the character grids in that script — edit the grid, never the svg; `--check` flags a
  stale svg or a badge with no grid. `paper-badges.html` falls back to the ✦ star when there's
  no icon, so a new badge works before it has one.
- **YAML reads a bare `yes`/`no` as a boolean**, so `holds: yes` arrives as `true`.
  `journal.cell` and the layout turn it back; quote the `values` lists in `rubric.yml`.
- **Finished** means published, with a summary, every axis scored and answered (a table needs at
  least one complete row), and at least one plus and one minus — `journal.missing_core`. Until
  then it is "under review"; `score.py` writes each paper's status (`reviewed` /
  `under-review`) and rating to `stats.reviews`, and the templates only read it. Only reviewed
  papers are listed — `stats.listed`, best rated first, drives `/papers/` and the tag pages, and
  the tag counts only count them. An unfinished review's page still exists; nothing links it.
- **Part pages** are `layout: paper-part` with `slug` and `part` set; `new_paper.py <slug> --part
  tutorial` scaffolds one. The part names, labels and order are in `_data/parts.yml`.
  `site.papers` contains them, so every listing filters `where_exp: "p", "p.part == nil"`.
  `_plugins/parts.rb` drops a part page whose review isn't published, so a draft's tutorial
  never leaks onto the site on its own.
- **`notes.md`** is excluded in `_config.yml` and carries `published: false`. Pasted chat
  answers go there until the user rewrites them, never on a public page.

Don't put an `# H1` title in any body; the layouts render it. Break `tutorial.md` into `###`
subsections that follow the construction rather than the paper's own section order.

All of it is the user's to write — see *Human reviews only* above.

## The reading queue is internal

`_data/queue.yml` is the database of papers waiting to be read. It is **not published** — no
page renders it, and no tag page lists queued entries. Three scripts own it:

```sh
# (re)build it from the exports
python scripts/import_queue.py --bib ~/all-zotero.bib --keep ~/keep-dump.md \
    --mbox ~/scholar.mbox
python scripts/import_queue.py                      # just re-sync _papers/, keep everything
python scripts/resolve_queue.py                     # fill in missing titles and links
python scripts/resolve_queue.py --pmlr v97          # ...from a PMLR volume instead of arXiv
python scripts/next_paper.py                        # draw one and decide
```

The sources, worst-documented first: Zotero (`--bib`) has no date-added; Keep (`--keep`) has
no dates at all; Scholar alert mail (`--mbox`, Gmail Takeout) has a real `Date:` header, which
makes it the only source where `added` is exact rather than estimated.

Scholar is also by far the biggest — ~12k paper mentions out of 9.8k emails, about 62% of the
draw — and the weakest signal per paper, because almost all of these are *author* alerts
("Yoshua Bengio - new articles"), not topic queries. `source_interest` in
`_data/sampling.yml` is the dial for that; it ships neutral.

An entry holds `title`, `url`, `authors`, `year`, the `source` it came from, the `added` date
driving recency, `strikes`, and `taken` once it has been accepted. Hand-editing is fine;
re-importing preserves strikes, deadlines, deadlines' slugs and looked-up titles, matching on
normalised URL, then title, then the `alias` a lookup left behind.

Things that will bite:

- **`import_queue.py` is safe to rerun, including with no arguments** — the queue folds itself
  back in ahead of the exports. Only `--fresh` starts over.
- **Zotero and Keep record no date-added.** `added` is estimated: Zotero's `urldate`, else the
  publication year; for Keep, the arXiv month or conference year in a link, propagated through
  the file's newest-first ordering. `added_est: true` marks a guess. Don't present these as
  real dates.
- **`added` is the *most recent* sighting**, not the first. A paper written down again this
  year is a current interest, which is the whole point of the recency weight.
- **Dropped papers need `_data/dropped.yml`.** The exports still contain them, so a tombstone
  is the only thing keeping a skipped-out paper from reappearing on the next import. Deleting
  a line there lets a paper back into the draw.
- **A paper leaves the queue when its review is finished** — published, rated, and the core
  review written (see *Review structure*). That
  definition lives in `scripts/journal.py:is_finished` and nowhere else; the queue and the
  scoreboard have to agree on it.
- **Patents, meeting abstracts and blog mirrors are not papers** and never enter the queue —
  see `not_a_paper()`. The rule never overrules a decision, though: anything already
  `curated`, `starred` or `taken` is kept whatever its URL looks like.
- **Only topic tags are guessed.** `import_queue.py` never infers an archetype tag — which
  archetype a paper earns is a judgement about its contribution, so it stays the user's. The
  archetype tags in the queue all came from the hand-curated entries.
- **A star is the best signal in the file, divided by how many papers shared the email.**
  A starred Scholar alert holding exactly one paper is unambiguous, so that paper is
  `starred: true` outright (624 of them). An alert holding k papers starred all k at once, so
  those get `flagged: true` with `siblings: k`, and the boost above 1.0 is divided by k —
  `1 + (starred - 1) / siblings`. Never upgrade a `flagged` entry to `starred` yourself;
  which of the k it was is the user's to say. `next_paper.py --flagged` draws that shortlist.
- **Gmail's own labels are dropped** (Inbox, Unread, Important, Category *, Personal). Only
  labels the user made are kept, in `labels` — here `gscholar` and `Interesting papers`.
  Neither is a signal: `gscholar` is on everything, and `Interesting papers` is deprecated.
  They are recorded because they are what the mail said, and nothing weighs them.
- **Mail in Trash or Spam is skipped.** The user's deletions are the first filter the queue
  gets ("I've already deleted the uninteresting looking ones"), so they are honoured, not
  undone.
- **Label and subject headers are folded.** Gmail wraps long headers, so `Interesting papers`
  arrives as `Interesting\n papers`. Whitespace is collapsed before splitting on commas —
  without that it becomes two labels, neither of which is the real one.
- **`alert` is usually a person, not a subject**, so it is deliberately *not* fed to the tag
  guesser. Don't treat it as a topic.

`scripts/journal.py` is the one shared module: the "is it finished" rule, URL/title
normalisation, and reading and writing the queue. It exists so three scripts can't disagree.

## Sampling and deadlines

`_data/sampling.yml` holds the weights, and nothing on the site reads it:

    weight = interest x recency x strike_penalty^strikes x boosts x the entry's own `weight`

`interest` is the largest `tag_interest` any of the entry's tags earns; `recency` halves every
`halflife_days` down to a `floor` so nothing is ever unreachable. `python scripts/next_paper.py
--explain` prints the arithmetic for a real draw, and `--top N` lists the heaviest entries.

Those weights encode what the user likes. They were seeded from the tags already in `_papers/`
and are theirs to retune — suggesting a change is fine, quietly rebalancing them is not.

Accepting a paper writes `taken: {slug, on, due}`, due `review_deadline.days` later. After
that, every finished week costs `review_deadline.xp`, capped at `max_xp` per paper so one
abandoned draft can't zero the scoreboard, and nothing accepted before `since` is ever charged.
The debt clears when the review is finished, and shows on the next `score.py` run.
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
bundle exec jekyll serve                                            # public site
bundle exec jekyll serve --config _config.yml,_config.scoreboard.yml   # with the scoreboard
```

## Don't

- Don't commit PDFs (gitignored) — put the link in `ref:`.
- Don't hand-edit `_data/stats.yml` or `tags/`.
- Don't invent an axis score, a `read` date, or a `depth` — and don't set `tag_interest` weights
  on the user's behalf either.
- Don't add arithmetic to Liquid templates; it belongs in `score.py`.
- Don't refactor notebooks into shared modules. Duplication across notebooks is deliberate;
  each one is self-contained so it can be read alone.
- Don't ignore `*.png` in `.gitignore` — `assets/images/` is tracked.
