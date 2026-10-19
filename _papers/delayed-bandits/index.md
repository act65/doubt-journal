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
  A linear bandit where feedback arrives after an unknown delay and no reward is indistinguishable
  from "not observed yet". 
confidence: 3
review:
  # Why: What problem is being solved, why is it hard, and is it worth solving?
  why:
    score: 2
    answer:
      - There's no algorithm for the linear, unknown, delayed bandit setting.
      - In this setting, what are the upper and lower bounds on regret?
    # maybe we need the ability to handle multiple answers here?
    # or handle multiple questions being asked.
    plus:
      - Problems are clearly defined.
    minus:
      - A very specific subproblem. Not particularly general nor practical.
  # Idea: Is the solution principled — built from first principles with the right tools, rather than ad hoc?
  idea:
    score: 2
    answer: UCB/TS using linear least squares on a fixed window of (possibly yet to be converted) rewards. and asking how delays effect regret.
    plus: 
      - Principled and based on UCB/TS.
    minus:
      - It's the dumbest possible extension to the delayed setting. 
  # Contribution: What does it add that wasn't known or possible before — the result, and the insight behind it?
  contribution:
    score: 2       # 1–4
    answer: Incremental.
    plus:
      - new algorithms (OTFLinUCB and OTFLinTS) to handle the linear, unknown, delayed bandit setting.
      - regret upper and lower bounds.
    minus:
      - ...
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
      - claim: new algorithm. 
        evidence: 
        holds: partly
    plus:
      -
    minus:
      -
  # Scope: When does the contribution actually apply — what does it assume, stated and unstated?
  scope:
    score: 2       # 1–4
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
        # im not really sure how common linear bandit problems are. i assume they are not particularly practical?
    plus:
      -
    minus:
      -
  # Clarity: Is it one simple narrative — a well-motivated problem, leading to minimal experiments and an easily interpretable result?
  clarity:
    score: 3       # 1–4
    answer: The linear, unknown, delayed bandit setting is introduced and 
    plus:
      - the problem definition clearly leads to their proofs and experiments 
      - the experiments are minimal
    minus:
      - the results are not easily interpretable. The reader must work to find answers to the 'main' question (how do unknown delays effect regret bounds?).
takeaways:
  -
notebooks:
  - file: pending_zeros.py
    what: OTFLinUCB's worst-case pending-zero width vs reweighting pending zeros by a known or learned delay CDF
related: []
---
