---
layout: paper
slug: inference-via-interpolation
title: "Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference"
authors: "Eysenbach, Myers, Salakhutdinov & Levine"
year: 2024
venue: arXiv
ref: https://arxiv.org/abs/2403.04082
permalink: /papers/inference-via-interpolation/
# skim (1 XP) | read (3) | rebuilt (8) | extended (15) — see _data/gamify.yml
read: 2026-08-05
depth: read
tags: [rl, probabilistic-methods, hidden-equivalence]
question: >
  Assumption 2 reads like the paper's result rather than an assumption — where does it actually
  come from, and does everything really follow from it?
tldr: >
  Choose a contrastive objective whose critic is a Gaussian kernel and whose regulariser is a
  variance budget rather than a unit sphere, and the learned representations become a
  linear-Gaussian state-space model — so planning between a start and a goal collapses to linear
  interpolation in latent space.
# Notebooks live in this folder next to the review.
#   uv run marimo edit _papers/inference-via-interpolation/<name>.py
# The core review — see TEMPLATE.md and _data/rubric.yml. Finished once it has a tldr, every
# axis scored and answered, and at least one plus and one minus — journal.is_finished.
confidence:
review:
  # Why: What problem is being solved, why is it hard, and is it worth solving?
  why:
    score:        # 1–4
    answer:
    plus:
      -
    minus:
      -
  # Idea: Is the solution principled — built from first principles with the right tools, rather than ad hoc?
  idea:
    score:        # 1–4
    answer:
    plus:
      -
    minus:
      -
  # Contribution: What does it add that wasn't known or possible before — the result, and the insight behind it?
  contribution:
    score:        # 1–4
    answer:
    plus:
      -
    minus:
      -
  # Rigour: Can every claim be traced to the evidence for it — and does that evidence hold?
  rigour:
    score:        # 1–4
    claims:
      - claim:
        evidence:
        holds:        # yes | partly | no | unverified
    plus:
      -
    minus:
      -
  # Scope: When does the contribution actually apply — what does it assume, stated and unstated?
  scope:
    score:        # 1–4
    assumptions:
      - assumption:
        stated:        # yes | no
        implications:
    plus:
      -
    minus:
      -
  # Clarity: Is it one simple narrative — a well-motivated problem, leading to minimal experiments and an easily interpretable result?
  clarity:
    score:        # 1–4
    answer:
    plus:
      -
    minus:
      -
takeaways:
  -
notebooks: []
# Slugs of other papers in _papers/.
related:
  - 1000-layer-networks-ssl-rl
---
