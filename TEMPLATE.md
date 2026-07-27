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
# skim (1 XP) | read (3) | rebuilt (8) | extended (15) — see _data/gamify.yml
depth: $depth
# Out of 10. Leave blank until you've decided — blank means "under review", not zero.
rating:
tags: [$tags]
question: >
  What sent you here — a citation, a problem at work, a claim you didn't believe? One sentence.
tldr: >
  The whole paper compressed to a sentence. If you can't write this yet, that's the thing to go
  work out.
# Notebooks live in this folder next to the review.
#   uv run marimo edit _papers/$slug/<name>.py
notebooks: []
# Slugs of other papers in _papers/.
related: []
---

## Review

<The verdict, in a short paragraph. What is the contribution, and does the paper actually
demonstrate it? This is the bit that justifies the rating — write it last, but keep it first.>

## Problem

<What is being solved, and why it is hard. Set up the notation and the setting here, so the
overview can get straight to the idea.>

## Overview

<The paper in your own words — the tutorial you wish you'd been handed. Break it into `###`
subsections that follow the construction, not the paper's section order. Build up from something
already understood; restate the paper's notation in plain terms and keep a translation table if
the symbols are heavy.>

## Additional thoughts

<Your ideas, not theirs. Extensions, alternative formulations, connections to other papers, and
the open questions you're left with. Use `- [ ]` for anything you might actually chase later.>

- [ ] <What's still unclear.>

## Feedback

<Optional. What the authors should have done — the experiment that's missing, the baseline that
isn't there, the claim that outruns the evidence. Delete this section if the paper is fine.>
