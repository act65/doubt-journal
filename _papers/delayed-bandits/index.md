---
layout: paper
slug: delayed-bandits
title: Linear Bandits with Stochastic Delayed Feedback
authors: Vernade et al.
year: 2020
venue: ICML
ref: https://arxiv.org/abs/1807.02089
permalink: /papers/delayed-bandits/
read: 2020-06-20
depth: reviewed
tags: [rl, bandits, uncertainty]
summary: >
  A linear bandit where feedback arrives after an unknown delay and a zero is indistinguishable
  from "not converted yet". The paper provides and characterises a baseline: a learner that does
  not exploit any structure in the delays.
# The core review — see TEMPLATE.md and _data/rubric.yml. Finished once it has a summary, every
# axis scored and answered, and at least one plus and one minus — journal.is_finished.
confidence: 4
review:
  # Why: What problem is being solved, why is it hard, and is it worth solving?
  why:
    score: 3
    answer: How do delays in feedback affect your ability to learn efficiently?
    plus:
      - 
    minus:
      - I have always been a little confused by the online setting. 
  # Idea: Is the solution principled — built from first principles with the right tools, rather than ad hoc?
  idea:
    score: 3
    answer: UCB/TS using linear least squares on a fixed window of (possibly yet to be converted) rewards and asking how delays effect regret.
    plus: 
      - Principled and based on UCB/TS.
    minus:
      - It's the dumbest possible extension to the delayed setting. 
  # Contribution: What does it add that wasn't known or possible before — the result, and the insight behind it?
  contribution:
    score: 2       # 1–4
    answer: Adds little meaningful insight.
    plus:
      -
    minus:
      -
  # Rigour: Can every claim be traced to the evidence for it — and does that evidence hold?
  rigour:
    score: 3       # 1–4
    claims:
      - claim: $\mathcal O(d\sqrt{T})$ upper bound on regret of OTFLinUCB
        evidence: theorem 2
        holds: unverified      # yes | partly | no
      - claim: $\frac{1}{\tau_m}$ lower bound on regret
        evidence: theorem 2
        holds: 
    plus:
      -
    minus:
      -
  # Scope: When does the contribution actually apply — what does it assume, stated and unstated?
  scope:
    score: 3       # 1–4
    assumptions:
      - assumption: fixed hindsight window 
        stated: yes
        implications: adds bias LLS into estimates. Lower bound is not general to all delayed bandit problems
      - assumption: delay distribution is unknown
        stated: yes
        implications: harder to exploit structure in the delay distribution
      - assumption: a linear (contextual) model
        stated: yes
        implications: 
    plus:
      -
    minus:
      -
  # Clarity: Is it one simple narrative — a well-motivated problem, leading to minimal experiments and an easily interpretable result?
  clarity:
    score: 4       # 1–4
    answer: test
    plus:
      -
    minus:
      -
takeaways:
  -
notebooks: []
related: []
---
