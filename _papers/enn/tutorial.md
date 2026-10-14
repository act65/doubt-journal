---
layout: paper-part
slug: enn
part: tutorial
title: "Tutorial · Epistemic Neural Networks"
---

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
