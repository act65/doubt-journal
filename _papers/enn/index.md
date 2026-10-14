---
layout: paper
slug: enn
title: Epistemic Neural Networks
authors: Osband et al.
year: 2021
venue: arXiv
ref: https://arxiv.org/abs/2107.08924
permalink: /papers/enn/
read: 2026-07-27
depth: rebuilt
tags: [probabilistic-methods, uncertainty, deep-learning, hidden-equivalence]
question: Why a *joint* over outputs — and how is it learned from single labels?
tldr: >
  Sample one hypothesis $z$, then predict conditionally independently given it — so the
  correlation between predictions at different inputs *is* the epistemic uncertainty, and it gets
  learned from single labels because the shared $z$ is what the data constrains.
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
notebooks:
  - file: epistemic_nn_from_coins.py
    what: >
      Beta–Bernoulli coin. Three priors with the *same* marginal $P(H)=0.5$ but different joints;
      more flips ($\tau=2,3,4$) sharpen the separation; the joint collapses toward a product as
      data arrives. Then the sampling view (= eqn 2), then a randomized-prior ensemble trained on
      single labels only, with the joint measured.
  - file: epistemic_nn_from_gaussian.py
    what: >
      A 15-rung ladder from $y = \mu + \varepsilon$ to a working epinet. Separates aleatoric from
      epistemic, shows why the off-diagonal *is* the epistemic variance, climbs from a number to a
      function via randomized priors, then replaces $K$ full networks with one shared body plus an
      epinet head (eqn 6).
related:
  - bfn
---
