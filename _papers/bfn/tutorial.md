---
layout: paper-part
slug: bfn
part: tutorial
title: "Tutorial · Bayesian Flow Networks"
---

## Problem

A generative model that runs the diffusion idea on the *parameters of a belief* rather than on
noisy data — which means it works on discrete data without any relaxation hack. The paper's
notation ($\theta,\mu,\rho,\gamma,\beta,p_I,p_S,p_R,p_O,\Psi$) hides how simple the mechanism is,
and the five distributions are easy to conflate. The goal was to pin down exactly **what is fed
to the network and what comes out**.

## Overview

### Notation

| paper | here | what it is |
|---|---|---|
| input dist / $\theta$ | `belief = (m, A)`, or `L` for discrete | noisy estimate `m` of $x$ with total precision `A`; logits `L` in the discrete case |
| sender $p_S$ | `observe(x, a)` | draw a noisy measurement of the true $x$ at precision `a` |
| Bayesian update $h$ | `update(belief, y)` | fold a measurement in — precision-weighted mean, or `L += y` |
| network $\Psi$ | `denoise` / "the brain" | corrects the naive posterior using the true data prior |
| output dist $p_O$ | `xhat` | Dirac delta at $\hat x$ (continuous); a genuine categorical (discrete) |
| accuracy schedule $\beta(t)$ | `schedule` | how fast evidence accumulates over time |

### What I took away

- The network's only job is **prior correction**. The belief is a posterior under a *naive*
  (factorized, uninformative) prior; $\Psi$ maps it to what the posterior would have been under
  the *true* data prior. Everything else is exact Bayes.
- The continuous output distribution is a **delta**, so nothing is sampled at the output — all
  the stochasticity lives in the sender. The discrete case is the honest one, and is where the
  construction stops looking like a reparameterized diffusion.
- Float64 matters: the schedule involves $\sigma_1^{-2t}$, and $\gamma$ is tiny near $t=0$.
