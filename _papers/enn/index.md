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
rating:
tags: [probabilistic-methods, uncertainty, deep-learning, hidden-equivalence]
question: Why a *joint* over outputs — and how is it learned from single labels?
tldr: >
  Sample one hypothesis $z$, then predict conditionally independently given it — so the
  correlation between predictions at different inputs *is* the epistemic uncertainty, and it gets
  learned from single labels because the shared $z$ is what the data constrains.
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

## Review

_Under review — I haven't written the verdict yet._

## Problem

Eqns 1 & 2 define a **joint** distribution over the outputs $\{y_1,\dots,y_\tau\}$ of $\tau$
inputs. Why a joint at all — and since we only ever observe single labels, how is that joint
ever *learned*?

The load-bearing identity:

$$\underbrace{\mathrm{Cov}(y_i,y_j)}_{\text{joint correlation}} = \underbrace{\mathrm{Var}_{\text{posterior}}(\theta)}_{\text{epistemic uncertainty}}$$

The two notebooks are independent routes to the same construction — start with whichever prior
you find more familiar.

## Overview

### What I took away

- Marginals are not enough, and that's the entire motivation. Three models can agree on every
  marginal and disagree completely about what they *don't know* — the disagreement is only
  visible in the joint.
- Aleatoric noise is **independent** across inputs; epistemic uncertainty is **shared**. That
  asymmetry is the whole diagnostic, and it's why a joint over $\tau \geq 2$ inputs recovers
  something a single marginal cannot.
- Training on single labels still shapes the joint, because every label constrains the shared
  $z$-indexed function. The joint structure emerges: collapsed where there is data, correlated
  where there is not.
- The epinet is a cost story, not a conceptual one — rungs 8–11 work with $K$ full networks;
  rungs 12–15 get the same behaviour from one base pass plus a small $z$-conditioned head.

## Additional thoughts

### Open questions

- [ ] How well does the epinet's prior term $\sigma^P = z \cdot h(x)$ scale when the input is
      high-dimensional — do frozen random functions of the raw input still give useful diversity?
- [ ] Ensemble vs epinet: at what point does the shared-body approximation visibly cost joint
      quality?
