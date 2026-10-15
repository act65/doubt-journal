---
layout: paper
slug: $slug
title: $title
authors: $authors
year: $year
venue: $venue
ref: $ref
permalink: /papers/$slug/
read: $read
# read (1 XP) | rebuilt (3) | extended (5) — see _data/gamify.yml
depth: $depth
tags: [$tags]
# What sent you here — a citation, a problem at work, a claim you didn't believe? One sentence.
question:

# The core review: what /papers/$slug/ shows, top to bottom. The axes and their questions are
# in _data/rubric.yml. A review is finished (and gets a verdict) once it has a summary, every axis
# scored and answered — the rigour and scope tables need at least one complete row — and at
# least one plus and one minus — scripts/journal.py:is_finished.
#
# Write it in the other order: the axes while reading, the summary and takeaways last.

# The whole paper compressed to a sentence or two.
summary:
# 1–5: how sure you are of these scores. Not depth — effort isn't certainty.
confidence:
# No `rating:` — it's computed from the axis scores, and a 1 on any axis caps it.
$review
# One or two things worth remembering.
takeaways:
  -

# Notebooks live in this folder next to the review.
#   uv run marimo edit _papers/$slug/<name>.py
notebooks: []
# Slugs of other papers in _papers/.
related: []
---

<!-- Optional: the verdict in a short paragraph, shown above the strengths and weaknesses.
The longer pages are separate files beside this one, made when you need them:

    python scripts/new_paper.py $slug --part tutorial     # the extended summary
    python scripts/new_paper.py $slug --part discussion   # the long argument, feedback
    python scripts/new_paper.py $slug --part thoughts     # extensions, open questions
-->
