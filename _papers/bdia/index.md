---
layout: paper
slug: bdia
title: Bi-directional Integration Approximation (BDIA)
authors: Zhang et al.
year: 2023
venue: arXiv
ref: https://arxiv.org/abs/2307.10829
permalink: /papers/bdia/
read: 2024-01-01
depth: rebuilt
# Your hand rating from before the rubric. Nothing reads it — the rating is now
# computed from the axis scores. Kept so you can compare; delete it whenever.
rating_before_rubric: 2
tags: [generative-models, numerical-methods, disputed]
question: Can a bi-directional coupled integrator make diffusion inversion exact without doubling the NFEs?
summary: >
  Coupling variables lets you construct reversible integration updates. BDIA extends that idea to
  a bi-directional coupled integrator. The idea may have potential, but this paper does not
  demonstrate it.
scholar:
  bibliography: "bdia.bib"
# The core review — see TEMPLATE.md and _data/rubric.yml. Finished once it has a summary, every
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
related: []
---

Numerical solution of integrals introduce errors. Coupling variables allows us to construct reversible integration updates. BDIA is a method that extends this idea to construct a bi-directional coupled integrator. The idea may have potential, but this paper does not demonstrate it.
