---
layout: paper
slug: bfn
title: Bayesian Flow Networks
authors: Graves, Srivastava, Atkinson, Gomez
year: 2023
venue: arXiv
ref: https://arxiv.org/abs/2308.07037
permalink: /papers/bfn/
read: 2026-07-27
depth: rebuilt
tags: [probabilistic-methods, generative-models, discrete-data, reconception]
question: What is actually fed to the network, and what comes out?
tldr: >
  Keep a belief about the clean data — a posterior built from noisy observations under a naive
  prior — and a network that corrects that naive posterior into a smart estimate using the true
  data prior. Generation bootstraps: guess, observe your own guess, update the belief, repeat.
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
  - file: bfn_tutorial.py
    what: >
      The paper on its own terms: a worked 2D continuous-Gaussian example, walking the five
      distributions ($p_I$, sender, flow, $p_O$, receiver) with equation numbers, the closed-form
      flow, the loss, and the $n$-step sampler.
  - file: bfn_simple.py
    what: >
      The same machinery rebuilt from Bayesian primitives with the paper's notation dropped.
      Continuous data, a deliberately multimodal 3-Gaussian prior, so the "brain" visibly pulls a
      naive estimate toward a mode.
  - file: bfn_discrete.py
    what: >
      The categorical case — where the framework actually earns its keep. Belief is per-dim
      logits, a Bayesian update is `L += y` (addition in log-space), and the output distribution
      is a real categorical rather than a Dirac delta.
related:
  - enn
---
